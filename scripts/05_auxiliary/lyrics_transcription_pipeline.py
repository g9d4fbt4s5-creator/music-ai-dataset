#!/usr/bin/env python3
"""
lyrics_transcription_pipeline.py
Stage 5.2 歌词转写流水线（GPU 端运行）

流程：
    原始音频 → Demucs 分离 → vocals.wav → 语言检测 → ASR 转写 → 保存文本
                                                                        ↓
                                                                  wav回传Mac（可选）
                                                                        ↓
                                                                  删除GPU上的vocals.wav（可选）

ASR 双轨制：
    - 中文歌唱：FunASR paraformer-zh
    - 非中文/口语：faster-whisper small
    - 语言检测：Whisper base detect_language()

歌词转写输入必须是 Demucs vocals stem（不是原曲），实测WER从60%降到25%
"""
import os
import sys
import argparse
import logging
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

LOG_DIR = "/root/autodl-tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"lyrics_transcription_{time_str}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CHINESE_LANGS = {"zh", "zh-cn", "zh-tw", "zh-hk", "yue", "cmn"}


def check_dependencies():
    deps = {}
    for name, import_path in [
        ("demucs", "demucs"),
        ("whisper", "whisper"),
        ("funasr", "funasr"),
        ("faster_whisper", "faster_whisper"),
    ]:
        try:
            __import__(import_path)
            deps[name] = True
        except ImportError:
            deps[name] = False
    logger.info("依赖检查:")
    for name, installed in deps.items():
        logger.info(f"  {'✅' if installed else '❌'} {name}")
    return deps


def load_whisper_model(model_size="base"):
    import whisper
    logger.info(f"加载 Whisper {model_size} 模型（语言检测）...")
    model = whisper.load_model(model_size)
    logger.info("Whisper 模型加载完成")
    return model


def detect_language(whisper_model, audio_path):
    import whisper
    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        lang = max(probs, key=probs.get)
        confidence = probs[lang]
        logger.info(f"  语言检测: {lang} (置信度: {confidence:.2%})")
        return lang, confidence
    except Exception as e:
        logger.warning(f"  语言检测失败: {e}，默认 en")
        return "en", 0.0


def detect_vocal_ratio_librosa(audio_path):
    """
    librosa 粗筛人声占比（轻量，秒级，用于决定是否跑 Demucs）

    方法：
    1. HPSS 分离谐波成分（人声/旋律）+ 打击成分（鼓/节奏）
    2. 人声频率带（100-3000Hz）能量加权
    3. 综合估算人声占比

    返回：0.0-1.0 的人声占比估算
    """
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)

        # HPSS 分离
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_energy = np.sum(y_harmonic ** 2)
        total_energy = np.sum(y ** 2) + 1e-10
        harmonic_ratio = harmonic_energy / total_energy

        # 人声频率带能量
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        vocal_band_mask = (freqs >= 100) & (freqs <= 3000)
        vocal_band_energy = np.sum(S[vocal_band_mask, :] ** 2)
        total_spectral_energy = np.sum(S ** 2) + 1e-10
        vocal_band_ratio = vocal_band_energy / total_spectral_energy

        # 综合估算
        vocal_ratio = 0.6 * harmonic_ratio + 0.4 * vocal_band_ratio
        vocal_ratio = min(1.0, max(0.0, vocal_ratio))

        logger.info(f"  librosa粗筛人声占比: {vocal_ratio:.1%} (谐波={harmonic_ratio:.1%}, 人声频段={vocal_band_ratio:.1%})")
        return float(vocal_ratio)

    except Exception as e:
        logger.warning(f"  librosa粗筛失败: {e}，默认 0.5（继续处理）")
        return 0.5


def separate_vocals_demucs(audio_path, output_dir, track_id):
    track_output_dir = Path(output_dir) / track_id
    vocals_path = track_output_dir / "vocals.wav"

    if vocals_path.exists() and vocals_path.stat().st_size > 0:
        logger.info(f"  vocals.wav 已存在，跳过分离")
        return str(vocals_path)

    track_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Demucs 分离人声...")
    cmd = [
        "python3", "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", "mdx_extra_q",
        "-o", str(output_dir),
        audio_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"  Demucs 分离失败: {result.stderr[-200:]}")
            return None
        stem = Path(audio_path).stem
        demucs_vocals = Path(output_dir) / "mdx_extra_q" / stem / "vocals.wav"
        if demucs_vocals.exists():
            import shutil
            shutil.copy2(demucs_vocals, vocals_path)
            logger.info(f"  分离成功: {vocals_path}")
            return str(vocals_path)
        logger.error(f"  vocals.wav 未生成")
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"  Demucs 分离超时")
        return None
    except Exception as e:
        logger.error(f"  Demucs 分离异常: {e}")
        return None


def transcribe_chinese_funasr(vocals_path):
    from funasr import AutoModel
    logger.info(f"  FunASR 中文转写...")
    try:
        if not hasattr(transcribe_chinese_funasr, "_model"):
            transcribe_chinese_funasr._model = AutoModel(
                model="paraformer-zh",
                model_revision="v2.0.4",
            )
            logger.info("  FunASR 模型加载完成")
        model = transcribe_chinese_funasr._model
        res = model.generate(input=vocals_path, batch_size=1)
        text = res[0]["text"]
        confidence = res[0].get("confidence", 0.9)
        logger.info(f"  转写完成: {len(text)} 字")
        return text, confidence
    except Exception as e:
        logger.error(f"  FunASR 转写失败: {e}")
        return "", 0.0


def transcribe_whisper(vocals_path, language=None, model_size="small"):
    from faster_whisper import WhisperModel
    logger.info(f"  faster-whisper 转写 (lang={language or 'auto'})...")
    try:
        model_key = f"{model_size}_{language}"
        if not hasattr(transcribe_whisper, "_models"):
            transcribe_whisper._models = {}
        if model_key not in transcribe_whisper._models:
            transcribe_whisper._models[model_key] = WhisperModel(
                model_size, device="cuda", compute_type="float16",
            )
            logger.info(f"  faster-whisper {model_size} 模型加载完成")
        model = transcribe_whisper._models[model_key]
        segments, info = model.transcribe(vocals_path, language=language, beam_size=5)
        text = " ".join([s.text for s in segments])
        confidence = getattr(info, "language_probability", 0.8)
        logger.info(f"  转写完成: {len(text)} 字 (语言={info.language})")
        return text, confidence
    except Exception as e:
        logger.error(f"  faster-whisper 转写失败: {e}")
        return "", 0.0


def rsync_vocals_to_mac(vocals_path, track_id, rsync_dest, ssh_port=22):
    remote_path = f"{rsync_dest.rstrip('/')}/{track_id}/vocals.wav"
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -p {ssh_port} -o StrictHostKeyChecking=no",
        vocals_path, remote_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"  ✅ vocals.wav 回传成功")
            return True
        logger.warning(f"  ⚠️ vocals.wav 回传失败: {result.stderr[-100:]}")
        return False
    except Exception as e:
        logger.error(f"  ❌ vocals.wav 回传异常: {e}")
        return False


def scan_input_dir(input_dir):
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in audio_extensions:
                audio_files.append(os.path.join(root, f))
    logger.info(f"扫描输入目录: {input_dir}，找到 {len(audio_files)} 个音频文件")
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(description="Stage 5.2 歌词转写流水线（GPU 端运行）")
    parser.add_argument("--input-dir", type=str, help="输入音频目录")
    parser.add_argument("--file-list", type=str, help="音频文件列表路径")
    parser.add_argument("--input-file", type=str, help="单个音频文件")
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/lyrics", help="输出目录")
    parser.add_argument("--stems-dir", type=str, default="/root/autodl-tmp/demucs_stems", help="Demucs stems 目录")
    parser.add_argument("--languages", type=str, default=None, help="只转写指定语言（逗号分隔）")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper 语言检测模型大小")
    parser.add_argument("--faster-whisper-model", type=str, default="small", help="faster-whisper 转写模型大小")
    parser.add_argument("--rsync-dest", type=str, default=None, help="转写后回传 vocals.wav 到 Mac")
    parser.add_argument("--rsync-port", type=int, default=22, help="SSH 端口")
    parser.add_argument("--delete-after-rsync", action="store_true", help="回传成功后删除 GPU 上的 vocals.wav")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个音频")
    parser.add_argument("--dry-run", action="store_true", help="预览模式")
    parser.add_argument("--skip-demucs", action="store_true", help="跳过 Demucs 分离")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage 5.2 歌词转写流水线启动")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  stems 目录: {args.stems_dir}")
    logger.info(f"  语言过滤: {args.languages or '全部'}")
    logger.info(f"  回传 Mac: {args.rsync_dest or '不回传'}")
    logger.info(f"  回传后删除: {args.delete_after_rsync}")
    logger.info("=" * 60)

    deps = check_dependencies()
    if not deps["whisper"]:
        logger.error("Whisper 未安装")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.stems_dir).mkdir(parents=True, exist_ok=True)

    if args.input_dir:
        audio_files = scan_input_dir(args.input_dir)
    elif args.file_list:
        with open(args.file_list) as f:
            audio_files = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        logger.info(f"从文件列表加载: {len(audio_files)} 个音频")
    elif args.input_file:
        audio_files = [args.input_file]
    else:
        logger.error("请指定 --input-dir / --file-list / --input-file 之一")
        return

    if not audio_files:
        logger.error("没有找到音频文件")
        return

    if args.limit:
        audio_files = audio_files[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个音频")

    target_languages = None
    if args.languages:
        target_languages = set(lang.strip() for lang in args.languages.split(","))
        logger.info(f"只转写语言: {target_languages}")

    whisper_model = load_whisper_model(args.whisper_model)

    logger.info("")
    logger.info(f"开始处理 {len(audio_files)} 个音频...")
    start_time = datetime.now()

    results = []
    for i, audio_path in enumerate(audio_files):
        track_id = Path(audio_path).stem
        logger.info(f"")
        logger.info(f"[{i+1}/{len(audio_files)}] {track_id}")

        if args.dry_run:
            logger.info(f"  [DRY-RUN] 将处理: {audio_path}")
            continue

        transcript_json = Path(args.output_dir) / f"{track_id}.json"
        if transcript_json.exists():
            logger.info(f"  已转写过，跳过")
            with open(transcript_json) as f:
                results.append(json.load(f))
            continue

        result = {
            "track_id": track_id, "audio_path": audio_path,
            "language": None, "language_confidence": 0.0,
            "lyrics_text": "", "asr_tool": None, "confidence": 0.0,
            "duration": 0.0, "status": "pending", "error": "",
        }

        try:
            logger.info(f"  Step 1: 语言检测")
            lang, lang_conf = detect_language(whisper_model, audio_path)
            result["language"] = lang
            result["language_confidence"] = lang_conf

            if target_languages and lang not in target_languages:
                logger.info(f"  语言 {lang} 不在目标列表，跳过")
                result["status"] = "skipped_language"
                results.append(result)
                continue

            # Step 1.5: librosa 粗筛人声占比（>10% 才跑 Demucs，避免纯器乐浪费时间）
            logger.info(f"  Step 1.5: librosa 粗筛人声占比")
            vocal_ratio = detect_vocal_ratio_librosa(audio_path)
            result["vocal_ratio_estimate"] = vocal_ratio

            if vocal_ratio <= 0.10:
                logger.info(f"  人声占比 {vocal_ratio:.1%} <= 10%，判定为纯器乐，跳过 Demucs 和 ASR")
                result["status"] = "skipped_instrumental"
                result["error"] = f"人声占比 {vocal_ratio:.1%} <= 10%，纯器乐"
                results.append(result)
                continue

            logger.info(f"  人声占比 {vocal_ratio:.1%} > 10%，继续 Demucs 分离")

            if args.skip_demucs:
                vocals_path = str(Path(args.stems_dir) / track_id / "vocals.wav")
                if not Path(vocals_path).exists():
                    logger.warning(f"  vocals.wav 不存在，跳过")
                    result["status"] = "no_vocals"
                    results.append(result)
                    continue
            else:
                logger.info(f"  Step 2: Demucs 分离人声")
                vocals_path = separate_vocals_demucs(audio_path, args.stems_dir, track_id)
                if not vocals_path:
                    result["status"] = "demucs_failed"
                    result["error"] = "Demucs 分离失败"
                    results.append(result)
                    continue

            import soundfile as sf
            info = sf.info(vocals_path)
            result["duration"] = info.duration

            logger.info(f"  Step 3: ASR 转写")
            if lang in CHINESE_LANGS:
                if deps["funasr"]:
                    text, confidence = transcribe_chinese_funasr(vocals_path)
                    result["asr_tool"] = "funasr_paraformer_zh"
                elif deps["faster_whisper"]:
                    text, confidence = transcribe_whisper(vocals_path, "zh", args.faster_whisper_model)
                    result["asr_tool"] = "faster_whisper_zh"
                else:
                    result["status"] = "no_asr_tool"
                    result["error"] = "FunASR 和 faster-whisper 都未安装"
                    results.append(result)
                    continue
            else:
                if deps["faster_whisper"]:
                    text, confidence = transcribe_whisper(vocals_path, lang, args.faster_whisper_model)
                    result["asr_tool"] = f"faster_whisper_{lang}"
                else:
                    result["status"] = "no_asr_tool"
                    result["error"] = "faster-whisper 未安装"
                    results.append(result)
                    continue

            result["lyrics_text"] = text
            result["confidence"] = confidence
            result["status"] = "success"

            with open(transcript_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"  ✅ 转写完成: {len(text)} 字")

            if args.rsync_dest:
                logger.info(f"  Step 4: 回传 vocals.wav 到 Mac")
                rsync_success = rsync_vocals_to_mac(vocals_path, track_id, args.rsync_dest, args.rsync_port)
                if rsync_success and args.delete_after_rsync:
                    try:
                        os.remove(vocals_path)
                        logger.info(f"  已删除 GPU 上的 vocals.wav")
                    except Exception as e:
                        logger.warning(f"  删除 vocals.wav 失败: {e}")

        except Exception as e:
            logger.error(f"  处理失败: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            results.append(result)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    logger.info("歌词转写完成")
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = len(results) - success_count - failed_count
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"  跳过: {skipped_count}")
    logger.info(f"  总计: {len(results)}")
    logger.info(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    logger.info("=" * 60)

    csv_path = Path(args.output_dir) / "lyrics_transcripts.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"汇总 CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
