"""
【实验特征链暂停 · T3 终审计 2026-09-04 用户拍板】
KNN 传播因一致率 0% 已退役（见 archive/l4_knn_legacy/DEPRECATED.md），genre 标注改走
「文本LLM(P0)+Qwen听音频(P1)+人工裁决」多标签分层。本 L1 物理特征脚本作为历史资产
保留原地、不归档不删除，但不再进入当前 L4 生产流程；扩到 500 首或训练用途时可重启。

l1_physical_features.py
L1 物理标签层：提取音频的客观物理特征

功能：
- BPM（节拍速度）
- 调性（Key）
- 响度（LUFS）
- 时长（duration）
- 采样率（sample_rate）
- 声道数（channels）
- 信噪比估计（SNR）
- 频谱质心（spectral_centroid）
- 过零率（zero_crossing_rate）

用法：
    python l1_physical_features.py \
        --input-dir data/00_raw_collect/raw_audio \
        --output data/02_preannotation/l1_physical \
        --manifest data/00_raw_collect/audio_manifest.csv
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

# ===================== 配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def extract_physical_features(audio_path: str, audio_id_override: str = None) -> Dict:
    """
    提取音频的物理特征

    Args:
        audio_path: 音频文件路径

    Returns:
        特征字典
    """
    import librosa
    import soundfile as sf

    # 读取音频元信息
    try:
        info = sf.info(audio_path)
        sample_rate = info.samplerate
        channels = info.channels
        duration = info.duration
    except Exception:
        sample_rate = 0
        channels = 0
        duration = 0

    # 加载音频（重采样到 22050 用于特征提取）
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    features = {
        "audio_id": audio_id_override if audio_id_override else Path(audio_path).stem,
        "duration_sec": round(float(duration), 2) if duration else round(float(len(y) / sr), 2),
        "sample_rate": sample_rate,
        "channels": channels,
    }

    # BPM 检测
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["bpm"] = round(float(tempo), 2)
    except Exception as e:
        features["bpm"] = None
        logger.debug(f"BPM 检测失败: {e}")

    # 响度（LUFS）- 使用 pyloudnorm
    try:
        import pyloudnorm
        meter = pyloudnorm.Meter(sr)
        loudness = meter.integrated_loudness(y)
        features["lufs"] = round(float(loudness), 2)
    except Exception as e:
        features["lufs"] = None
        logger.debug(f"LUFS 检测失败: {e}")

    # 频谱质心
    try:
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid_mean"] = round(float(np.mean(spectral_centroids)), 2)
        features["spectral_centroid_std"] = round(float(np.std(spectral_centroids)), 2)
    except Exception:
        features["spectral_centroid_mean"] = None
        features["spectral_centroid_std"] = None

    # 过零率
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zero_crossing_rate_mean"] = round(float(np.mean(zcr)), 4)
    except Exception:
        features["zero_crossing_rate_mean"] = None

    # 频谱滚降点
    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        features["spectral_rolloff_mean"] = round(float(np.mean(rolloff)), 2)
    except Exception:
        features["spectral_rolloff_mean"] = None

    # RMS 能量
    try:
        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = round(float(np.mean(rms)), 6)
        features["rms_std"] = round(float(np.std(rms)), 6)
    except Exception:
        features["rms_mean"] = None
        features["rms_std"] = None

    # 调性估计（简单版，基于 chroma）
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        features["key"] = key_names[key_idx]
        features["key_strength"] = round(float(chroma_mean[key_idx]), 4)
    except Exception:
        features["key"] = None
        features["key_strength"] = None

    return features


def main():
    parser = argparse.ArgumentParser(
        description="L1 物理标签层：提取音频客观物理特征",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入音频目录（散列目录或扁平目录）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录（每个音频一个 JSON）")
    parser.add_argument("--manifest", type=str, default=None,
                        help="audio_manifest.csv 路径（用于解析散列目录）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找音频文件
    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    audio_files = []

    if args.manifest and Path(args.manifest).exists():
        # 从 manifest 解析路径
        df = pd.read_csv(args.manifest)
        if "audio_path" in df.columns:
            for _, row in df.iterrows():
                p = Path(row["audio_path"])
                if p.exists() and p.suffix.lower() in audio_extensions:
                    audio_files.append(p)
        else:
            # 散列目录递归查找
            for ext in audio_extensions:
                audio_files.extend(input_dir.rglob(f"*{ext}"))
    else:
        # 递归查找
        for ext in audio_extensions:
            audio_files.extend(input_dir.rglob(f"*{ext}"))

    audio_files = sorted(audio_files)

    # 建立 master_path → audio_id(ULID) 映射
    path_to_audio_id = {}
    if args.manifest and Path(args.manifest).exists():
        manifest_df = pd.read_csv(args.manifest)
        if "master_path" in manifest_df.columns and "audio_id" in manifest_df.columns:
            for _, row in manifest_df.iterrows():
                mp = row.get("master_path", "")
                if pd.notna(mp) and mp:
                    mp_abs = str(Path(mp).resolve()) if not Path(mp).is_absolute() else mp
                    path_to_audio_id[mp_abs] = row["audio_id"]
        logger.info(f"从 manifest 加载 {len(path_to_audio_id)} 个 master_path → audio_id 映射")

    if args.limit:
        audio_files = audio_files[:args.limit]

    logger.info(f"输入目录: {input_dir}")
    logger.info(f"音频文件数: {len(audio_files)}")
    logger.info(f"输出目录: {output_dir}")

    if not audio_files:
        logger.error("未找到音频文件")
        return

    # 处理每个音频
    results = []
    for idx, audio_path in enumerate(audio_files):
        try:
            audio_path_abs = str(audio_path.resolve())
            audio_id_override = path_to_audio_id.get(audio_path_abs)
            features = extract_physical_features(str(audio_path), audio_id_override)
            results.append(features)

            # 保存单个 JSON
            audio_id = features["audio_id"]
            with open(output_dir / f"{audio_id}_physical.json", "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False, indent=2)

            if (idx + 1) % 5 == 0 or idx == 0:
                logger.info(f"[{idx+1}/{len(audio_files)}] {audio_id}: "
                            f"BPM={features.get('bpm')}, LUFS={features.get('lufs')}, "
                            f"key={features.get('key')}")

        except Exception as e:
            logger.error(f"[{idx+1}/{len(audio_files)}] {audio_path.name}: {e}")

    # 保存汇总 CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / "_all_physical_features.csv", index=False, encoding="utf-8")
        logger.info(f"\n汇总 CSV: {output_dir / '_all_physical_features.csv'}")

    logger.info("\n" + "=" * 60)
    logger.info("L1 物理特征提取完成")
    logger.info("=" * 60)
    logger.info(f"  总数: {len(audio_files)}")
    logger.info(f"  成功: {len(results)}")
    logger.info(f"  失败: {len(audio_files) - len(results)}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
