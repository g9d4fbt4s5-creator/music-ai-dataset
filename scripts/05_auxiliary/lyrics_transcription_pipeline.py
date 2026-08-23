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

# ===================== 配置 =====================
# 自动检测运行环境（Mac本地 vs GPU）
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 日志目录：优先用项目根目录下的logs，GPU上可用/root/autodl-tmp/logs
if os.path.exists("/root/autodl-tmp"):
    LOG_DIR = "/root/autodl-tmp/logs"
else:
    LOG_DIR = str(PROJECT_ROOT / "logs")

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


def _fallback_vocal_ratio_librosa(audio_path):
    """
    [应急降级] librosa 粗筛人声占比（仅在没有 YAMNet 结果时使用）

    ⚠️ 重要：正常流程下 YAMNet 的 has_vocals 是唯一 Demucs 触发器。
    此函数仅用于 YAMNet 未跑、且不想装 TF 环境的应急场景。

    方法：HPSS 谐波分离 + 100-3000Hz 人声频段能量加权
    注意：Jazz 场景下会把萨克斯/钢琴误判为人声，准确率低于 YAMNet

    返回：0.0-1.0 的人声占比估算
    """
    import librosa

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        # HPSS
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(y_harmonic ** 2) / (np.sum(y ** 2) + 1e-10)
        # 100-3000Hz 频段能量
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        vocal_band_mask = (freqs >= 100) & (freqs <= 3000)
        vocal_band_energy = np.sum(S[vocal_band_mask, :] ** 2)
        total_spectral_energy = np.sum(S ** 2) + 1e-10
        vocal_band_ratio = vocal_band_energy / total_spectral_energy
        # 综合估算
        vocal_ratio = 0.6 * harmonic_ratio + 0.4 * vocal_band_ratio
        vocal_ratio = min(1.0, max(0.0, vocal_ratio))
        logger.info(f"  [应急] librosa粗筛人声占比: {vocal_ratio:.1%}")
        return float(vocal_ratio)
    except Exception as e:
        logger.warning(f"  [应急] librosa粗筛失败: {e}，默认 0.5（继续处理）")
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
    # 根据环境自动选择默认路径
    if os.path.exists("/root/autodl-tmp"):
        default_output_dir = "/root/autodl-tmp/lyrics"
        default_stems_dir = "/root/autodl-tmp/demucs_stems"
    else:
        default_output_dir = str(PROJECT_ROOT / "data" / "02_preannotation" / "lyrics")
        default_stems_dir = str(PROJECT_ROOT / "data" / "01_preprocess" / "demucs_stems")

    parser = argparse.ArgumentParser(description="Stage 5.2 歌词转写流水线（GPU 端运行，Mac 本地也可测试）")
    parser.add_argument("--input-dir", type=str, help="输入音频目录")
    parser.add_argument("--file-list", type=str, help="音频文件列表路径")
    parser.add_argument("--input-file", type=str, help="单个音频文件")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help=f"输出目录（默认 {default_output_dir}）")
    parser.add_argument("--stems-dir", type=str, default=default_stems_dir, help=f"Demucs stems 目录（默认 {default_stems_dir}）")
    parser.add_argument("--languages", type=str, default=None, help="只转写指定语言（逗号分隔）")
    parser.add_argument("--whisper-model", type=str, default="base", help="Whisper 语言检测模型大小")
    parser.add_argument("--faster-whisper-model", type=str, default="small", help="faster-whisper 转写模型大小")
    parser.add_argument("--rsync-dest", type=str, default=None, help="转写后回传 vocals.wav 到 Mac")
    parser.add_argument("--rsync-port", type=int, default=22, help="SSH 端口")
    parser.add_argument("--delete-after-rsync", action="store_true", help="回传成功后删除 GPU 上的 vocals.wav")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个音频")
    parser.add_argument("--dry-run", action="store_true", help="预览模式")
    parser.add_argument("--skip-demucs", action="store_true", help="跳过 Demucs 分离")
    parser.add_argument("--yamnet-results", type=str, default=None,
                        help="YAMNet 输出 CSV 路径，用于按 has_vocals 字段决定是否处理（上游开关）")
    parser.add_argument("--vocals-high-threshold", type=float, default=0.7,
                        help="YAMNet vocals_ratio 高阈值，高于此值明确有人声（默认0.7）")
    parser.add_argument("--vocals-low-threshold", type=float, default=0.3,
                        help="YAMNet vocals_ratio 低阈值，低于此值明确无人声（默认0.3）")
    parser.add_argument("--uncertain-strategy", type=str, default="run",
                        choices=["run", "librosa", "skip", "mark"],
                        help="不确定区（低阈值≤vocals_ratio≤高阈值）的处理策略："
                             "run=保守运行Demucs（默认，宁可多跑不可漏掉）；"
                             "librosa=用librosa粗筛二次确认；"
                             "skip=跳过不确定区（可能漏掉有人声）；"
                             "mark=标记为不确定，不跑Demucs")
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

    # 加载 YAMNet 结果（上游开关：按 has_vocals 字段决定是否处理）
    yamnet_df = None
    yamnet_vocals_map = {}
    if args.yamnet_results and os.path.exists(args.yamnet_results):
        logger.info(f"加载 YAMNet 结果: {args.yamnet_results}")
        yamnet_df = pd.read_csv(args.yamnet_results)
        # 构建 track_id -> (has_vocals, vocals_ratio) 映射
        for _, row in yamnet_df.iterrows():
            tid = str(row.get("track_id", ""))
            has_vocals = bool(row.get("has_vocals", False))
            vocals_ratio = float(row.get("vocals_ratio", 0.0))
            yamnet_vocals_map[tid] = (has_vocals, vocals_ratio)
        logger.info(f"YAMNet 结果加载完成: {len(yamnet_vocals_map)} 条记录")
        logger.info(f"  其中 has_vocals=True: {sum(1 for v in yamnet_vocals_map.values() if v[0])} 条")
        logger.info(f"  双阈值: 高={args.vocals_high_threshold:.0%}, 低={args.vocals_low_threshold:.0%}")
        logger.info(f"  不确定区策略: {args.uncertain_strategy}")
    else:
        logger.info("未提供 YAMNet 结果，将对所有音频执行 librosa 粗筛（应急降级）")

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
            # ============================================================
            # 关卡 A：YAMNet（唯一 Demucs 触发器）
            # ============================================================
            # 正常流程：YAMNet 的 has_vocals 是唯一决策依据
            # 应急降级：无 YAMNet 结果时，用 librosa 粗筛（准确率低）
            # ============================================================
            yamnet_info = yamnet_vocals_map.get(track_id) if yamnet_vocals_map else None

            if yamnet_info:
                # 有 YAMNet 结果：用 vocals_ratio 概率值双阈值决策（比二值化更稳健）
                has_vocals, vocals_ratio = yamnet_info
                result["yamnet_has_vocals"] = has_vocals
                result["yamnet_vocals_ratio"] = vocals_ratio

                if vocals_ratio > args.vocals_high_threshold:
                    # 高置信度有人声 → 直接继续
                    logger.info(f"  关卡A [YAMNet]: vocals_ratio={vocals_ratio:.1%} > {args.vocals_high_threshold:.0%} → 明确有人声，继续")
                    result["yamnet_decision"] = "high_confidence_vocals"
                elif vocals_ratio < args.vocals_low_threshold:
                    # 高置信度无人声 → 跳过
                    logger.info(f"  关卡A [YAMNet]: vocals_ratio={vocals_ratio:.1%} < {args.vocals_low_threshold:.0%} → 明确无人声，跳过 Demucs + ASR")
                    result["status"] = "skipped_no_vocals_high_confidence"
                    result["error"] = f"YAMNet vocals_ratio={vocals_ratio:.1%} < {args.vocals_low_threshold:.0%}"
                    result["yamnet_decision"] = "high_confidence_no_vocals"
                    results.append(result)
                    continue
                else:
                    # 不确定区（低阈值 ≤ vocals_ratio ≤ 高阈值）→ 按策略兜底
                    result["yamnet_decision"] = "uncertain"
                    logger.info(f"  关卡A [YAMNet]: vocals_ratio={vocals_ratio:.1%} 在不确定区 [{args.vocals_low_threshold:.0%}, {args.vocals_high_threshold:.0%}] → 兜底策略: {args.uncertain_strategy}")

                    if args.uncertain_strategy == "skip":
                        logger.info(f"    → 跳过不确定区（可能漏掉有人声）")
                        result["status"] = "skipped_uncertain_zone"
                        result["error"] = f"YAMNet vocals_ratio={vocals_ratio:.1%} 在不确定区，策略=skip"
                        results.append(result)
                        continue
                    elif args.uncertain_strategy == "mark":
                        logger.info(f"    → 标记为不确定，不跑 Demucs")
                        result["status"] = "marked_uncertain"
                        result["error"] = f"YAMNet vocals_ratio={vocals_ratio:.1%} 在不确定区，策略=mark"
                        results.append(result)
                        continue
                    elif args.uncertain_strategy == "librosa":
                        logger.info(f"    → 用 librosa 粗筛二次确认")
                        vocal_ratio = _fallback_vocal_ratio_librosa(audio_path)
                        result["fallback_vocal_ratio"] = vocal_ratio
                        if vocal_ratio <= 0.10:
                            logger.info(f"    → librosa 人声占比 {vocal_ratio:.1%} <= 10% → 跳过")
                            result["status"] = "skipped_uncertain_librosa_no_vocals"
                            result["error"] = f"不确定区 + librosa vocal_ratio={vocal_ratio:.1%} <= 10%"
                            results.append(result)
                            continue
                        logger.info(f"    → librosa 人声占比 {vocal_ratio:.1%} > 10% → 继续")
                    else:  # run（默认）：保守运行 Demucs
                        logger.info(f"    → 保守运行 Demucs（宁可多跑，不可漏掉）")
            else:
                # 无 YAMNet 结果：应急降级，用 librosa 粗筛
                logger.warning(f"  关卡A [应急]: 无 YAMNet 结果，使用 librosa 粗筛（准确率低）")
                vocal_ratio = _fallback_vocal_ratio_librosa(audio_path)
                result["fallback_vocal_ratio"] = vocal_ratio
                result["yamnet_decision"] = "fallback_librosa"

                if vocal_ratio <= 0.10:
                    logger.info(f"  关卡A [应急]: librosa人声占比 {vocal_ratio:.1%} <= 10% → 跳过")
                    result["status"] = "skipped_instrumental_fallback"
                    result["error"] = f"[应急] librosa vocal_ratio={vocal_ratio:.1%} <= 10%"
                    results.append(result)
                    continue
                logger.info(f"  关卡A [应急]: librosa人声占比 {vocal_ratio:.1%} > 10% → 继续")

            # ============================================================
            # 关卡 B：Whisper 语言检测（Demucs 之前，避免浪费分离算力）
            # ============================================================
            logger.info(f"  关卡B [Whisper]: 语言检测")
            lang, lang_conf = detect_language(whisper_model, audio_path)
            result["language"] = lang
            result["language_confidence"] = lang_conf

            if target_languages and lang not in target_languages:
                logger.info(f"  关卡B [Whisper]: 语言 {lang} 不在目标列表 → 跳过")
                result["status"] = "skipped_language"
                results.append(result)
                continue
            logger.info(f"  关卡B [Whisper]: 语言 {lang} (置信度 {lang_conf:.1%}) → 继续")

            # ============================================================
            # 关卡 C：Demucs 分离（只有通过关卡 A + B 才到这里）
            # ============================================================
            if args.skip_demucs:
                vocals_path = str(Path(args.stems_dir) / track_id / "vocals.wav")
                if not Path(vocals_path).exists():
                    logger.warning(f"  关卡C [Demucs]: vocals.wav 不存在，跳过")
                    result["status"] = "no_vocals"
                    results.append(result)
                    continue
            else:
                logger.info(f"  关卡C [Demucs]: 分离人声")
                vocals_path = separate_vocals_demucs(audio_path, args.stems_dir, track_id)
                if not vocals_path:
                    result["status"] = "demucs_failed"
                    result["error"] = "Demucs 分离失败"
                    results.append(result)
                    continue

            import soundfile as sf
            info = sf.info(vocals_path)
            result["duration"] = info.duration

            # ============================================================
            # 关卡 D：ASR 转写（语言决定路由）
            # ============================================================
            logger.info(f"  关卡D [ASR]: 转写（lang={lang}）")
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
