"""
extract_features.py
音频特征提取框架

功能：
1. 从 audio_manifest.csv 读取音频列表（禁止 ls/find 扫描目录）
2. 调用 get_audio_physical_path 获取散列路径
3. 使用 librosa 提取多维度声学特征
4. 输出特征 CSV（每行一个音频，每列一个特征）
5. 支持断点续传（跳过已处理的文件）
6. 支持单文件测试和批量处理

提取的特征维度：
- 基础特征：时长、采样率、声道数
- 时域特征：RMS均值/标准差、过零率均值/标准差、能量
- 频域特征：频谱质心、频谱带宽、频谱滚降点、频谱对比度
- MFCC：1-20阶均值/标准差
- 色度特征：12维色度均值/标准差
- 节奏特征：BPM、节拍数
- 响度特征：LUFS（集成响度）

用法：
    # 批量处理所有音频
    python extract_features.py

    # 处理指定 audio_id
    python extract_features.py --audio-id 01M0E9X162CTB4D15WZQ5D8FVX

    # 只处理前 N 个（测试用）
    python extract_features.py --limit 5

    # 强制重新处理（不跳过已处理的）
    python extract_features.py --force

    # 指定输出文件
    python extract_features.py --output ./data/02_preannotation/features/audio_features.csv
"""
import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from get_audio_physical_path import get_audio_physical_path, validate_audio_id

# 时区
TZ = timezone(timedelta(hours=8))

# 输入输出路径
MANIFEST_CSV = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"
RAW_AUDIO_ROOT = PROJECT_ROOT / "data" / "00_raw_collect"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "02_preannotation" / "features"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "audio_features.csv"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"extract_features_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_audio_manifest() -> pd.DataFrame:
    """加载 audio_manifest.csv（禁止 ls/find 扫描目录）"""
    if not MANIFEST_CSV.exists():
        logger.error(f"audio_manifest.csv 不存在: {MANIFEST_CSV}")
        raise FileNotFoundError(f"audio_manifest.csv not found: {MANIFEST_CSV}")

    df = pd.read_csv(MANIFEST_CSV)
    logger.info(f"加载 audio_manifest.csv: {len(df)} 条记录")

    # 只处理 status=active 的音频
    if "status" in df.columns:
        active_df = df[df["status"] == "active"]
        logger.info(f"  其中 active 状态: {len(active_df)} 条")
        return active_df.reset_index(drop=True)

    return df


def extract_basic_features(y: np.ndarray, sr: int) -> Dict:
    """提取基础特征"""
    duration = len(y) / sr
    n_channels = 1 if y.ndim == 1 else y.shape[0]

    return {
        "duration_s": duration,
        "sample_rate": sr,
        "n_channels": n_channels,
        "n_samples": len(y),
    }


def extract_time_domain_features(y: np.ndarray, sr: int) -> Dict:
    """提取时域特征"""
    # RMS（均方根能量）
    rms = librosa.feature.rms(y=y)[0]

    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # 能量
    energy = np.sum(y ** 2)

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_max": float(np.max(rms)),
        "rms_min": float(np.min(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "zcr_max": float(np.max(zcr)),
        "energy": float(energy),
        "energy_db": float(10 * np.log10(energy + 1e-10)),
    }


def extract_spectral_features(y: np.ndarray, sr: int) -> Dict:
    """提取频域特征"""
    # 频谱质心
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 频谱带宽
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    # 频谱滚降点
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    # 频谱对比度
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # 频谱平坦度
    spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]

    features = {
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_rolloff_std": float(np.std(spectral_rolloff)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "spectral_flatness_std": float(np.std(spectral_flatness)),
    }

    # 频谱对比度（7个子带）
    for i in range(spectral_contrast.shape[0]):
        features[f"spectral_contrast_band{i+1}_mean"] = float(np.mean(spectral_contrast[i]))
        features[f"spectral_contrast_band{i+1}_std"] = float(np.std(spectral_contrast[i]))

    return features


def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 20) -> Dict:
    """提取 MFCC 特征"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    features = {}
    for i in range(n_mfcc):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_std"] = float(np.std(mfcc[i]))

    # MFCC 一阶差分（delta）
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(n_mfcc):
        features[f"mfcc_delta{i+1}_mean"] = float(np.mean(delta_mfcc[i]))

    return features


def extract_chroma_features(y: np.ndarray, sr: int) -> Dict:
    """提取色度特征（12维）"""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features = {}
    for i in range(12):
        note_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][i]
        features[f"chroma_{note_name}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{note_name}_std"] = float(np.std(chroma[i]))

    # 主音（能量最大的音）
    chroma_sum = np.sum(chroma, axis=1)
    dominant_note = np.argmax(chroma_sum)
    features["dominant_note"] = dominant_note
    features["dominant_note_name"] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][dominant_note]

    return features


def extract_rhythm_features(y: np.ndarray, sr: int) -> Dict:
    """提取节奏特征"""
    # 节拍跟踪
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # 节拍时间
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # 节拍间隔
    if len(beat_times) > 1:
        beat_intervals = np.diff(beat_times)
        beat_interval_mean = float(np.mean(beat_intervals))
        beat_interval_std = float(np.std(beat_intervals))
    else:
        beat_interval_mean = np.nan
        beat_interval_std = np.nan

    return {
        "bpm": float(tempo) if tempo is not None and not np.isnan(tempo) else np.nan,
        "n_beats": len(beat_frames),
        "beat_interval_mean": beat_interval_mean,
        "beat_interval_std": beat_interval_std,
        "first_beat_time": float(beat_times[0]) if len(beat_times) > 0 else np.nan,
        "last_beat_time": float(beat_times[-1]) if len(beat_times) > 0 else np.nan,
    }


def extract_loudness_features(y: np.ndarray, sr: int) -> Dict:
    """提取响度特征（LUFS）"""
    try:
        import pyloudnorm

        # 重采样到 48kHz（pyloudnorm 要求）
        if sr != 48000:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=48000)
            sr_loudness = 48000
        else:
            y_resampled = y
            sr_loudness = sr

        # 确保是二维（声道 x 样本）
        if y_resampled.ndim == 1:
            y_resampled = y_resampled[np.newaxis, :]

        # 创建响度表
        meter = pyloudnorm.Meter(sr_loudness)

        # 测量集成响度
        loudness_lufs = meter.integrated_loudness(y_resampled.T)

        return {
            "loudness_lufs": float(loudness_lufs) if not np.isnan(loudness_lufs) else np.nan,
        }
    except ImportError:
        logger.warning("pyloudnorm 未安装，跳过 LUFS 特征提取")
        return {"loudness_lufs": np.nan}
    except Exception as e:
        logger.warning(f"LUFS 提取失败: {e}")
        return {"loudness_lufs": np.nan}


def extract_all_features(audio_path: Path) -> Dict:
    """
    提取单个音频的所有特征

    返回：
        特征字典
    """
    # 加载音频
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    features = {}

    # 基础特征
    features.update(extract_basic_features(y, sr))

    # 时域特征
    features.update(extract_time_domain_features(y, sr))

    # 频域特征
    features.update(extract_spectral_features(y, sr))

    # MFCC 特征
    features.update(extract_mfcc_features(y, sr, n_mfcc=20))

    # 色度特征
    features.update(extract_chroma_features(y, sr))

    # 节奏特征
    features.update(extract_rhythm_features(y, sr))

    # 响度特征
    features.update(extract_loudness_features(y, sr))

    return features


def process_audio(audio_id: str, manifest_row: pd.Series) -> Optional[Dict]:
    """
    处理单个音频

    返回：
        包含 audio_id 和所有特征的字典，失败返回 None
    """
    try:
        # 验证 audio_id
        if not validate_audio_id(audio_id):
            logger.warning(f"无效的 audio_id: {audio_id}")
            return None

        # 获取物理路径（从 manifest 读取 file_relative_path，最可靠）
        if "file_relative_path" in manifest_row and pd.notna(manifest_row["file_relative_path"]):
            relative_path = manifest_row["file_relative_path"]
        else:
            # 回退：调用 get_audio_physical_path
            relative_path = get_audio_physical_path(audio_id)

        audio_path = RAW_AUDIO_ROOT / relative_path

        if not audio_path.exists():
            logger.warning(f"音频文件不存在: {audio_path}")
            return None

        # 如果是目录，尝试查找目录下的音频文件
        if audio_path.is_dir():
            audio_files = list(audio_path.glob("*"))
            if len(audio_files) > 0:
                audio_path = audio_files[0]
                logger.info(f"  目录路径，使用第一个文件: {audio_path.name}")
            else:
                logger.warning(f"目录下没有音频文件: {audio_path}")
                return None

        # 提取特征
        features = extract_all_features(audio_path)

        # 添加元数据
        result = {"audio_id": audio_id}
        result.update(features)

        # 从 manifest 添加原始信息
        if "original_filename" in manifest_row:
            result["original_filename"] = manifest_row["original_filename"]
        if "duration_seconds" in manifest_row:
            result["manifest_duration_s"] = manifest_row["duration_seconds"]

        logger.info(f"  ✅ {audio_id}: {len(features)} 个特征")
        return result

    except Exception as e:
        logger.error(f"  ❌ {audio_id}: 特征提取失败 - {e}")
        return None


def main():
    global librosa
    import librosa

    parser = argparse.ArgumentParser(
        description="音频特征提取框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio-id", type=str, default=None,
                        help="只处理指定的 audio_id（测试用）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 个音频（测试用）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新处理，不跳过已处理的音频")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件路径（默认 data/02_preannotation/features/audio_features.csv）")
    parser.add_argument("--no-loudness", action="store_true",
                        help="跳过 LUFS 响度特征（pyloudnorm 未安装时使用）")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("音频特征提取")
    logger.info("=" * 60)

    # 输出路径
    output_csv = Path(args.output) if args.output else DEFAULT_OUTPUT_CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出文件: {output_csv}")

    # 加载已处理的 audio_id（断点续传）
    processed_ids = set()
    if output_csv.exists() and not args.force:
        try:
            existing_df = pd.read_csv(output_csv)
            if "audio_id" in existing_df.columns:
                processed_ids = set(existing_df["audio_id"].values)
                logger.info(f"已处理 {len(processed_ids)} 个音频，将跳过（使用 --force 重新处理）")
        except Exception:
            pass

    # 加载音频清单
    manifest_df = load_audio_manifest()

    # 过滤
    if args.audio_id:
        manifest_df = manifest_df[manifest_df["audio_id"] == args.audio_id]
        if len(manifest_df) == 0:
            logger.error(f"未找到 audio_id: {args.audio_id}")
            sys.exit(1)
    if args.limit:
        manifest_df = manifest_df.head(args.limit)

    logger.info(f"待处理音频数: {len(manifest_df)}")

    # 批量处理
    all_features = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, row in manifest_df.iterrows():
        audio_id = row["audio_id"]

        # ADR-004: 排除不参与训练的样本（黄金集、challenge_set、val）
        # in_train_training=False 的样本只做KNN种子/真值/压力测试，不提取训练特征
        in_train = row.get("in_train_training", True)
        if in_train is False or (isinstance(in_train, str) and in_train.lower() == "false"):
            sample_type = row.get("sample_type", "unknown")
            logger.info(f"[{idx + 1}/{len(manifest_df)}] 跳过非训练样本: {audio_id} (sample_type={sample_type}, in_train_training=False)")
            skip_count += 1
            continue

        # 跳过已处理的
        if audio_id in processed_ids and not args.force:
            logger.info(f"[{idx + 1}/{len(manifest_df)}] 跳过（已处理）: {audio_id}")
            skip_count += 1
            continue

        logger.info(f"[{idx + 1}/{len(manifest_df)}] 处理: {audio_id}")

        result = process_audio(audio_id, row)

        if result is not None:
            all_features.append(result)
            success_count += 1
        else:
            fail_count += 1

        # 每处理 10 个保存一次（防止意外丢失）
        if len(all_features) % 10 == 0 and len(all_features) > 0:
            temp_df = pd.DataFrame(all_features)
            if output_csv.exists() and not args.force:
                existing_df = pd.read_csv(output_csv)
                temp_df = pd.concat([existing_df, temp_df], ignore_index=True)
            temp_df.to_csv(output_csv, index=False, encoding="utf-8")
            logger.info(f"  中间保存: {len(temp_df)} 条记录")

    # 最终保存
    if all_features:
        final_df = pd.DataFrame(all_features)

        if output_csv.exists() and not args.force:
            existing_df = pd.read_csv(output_csv)
            final_df = pd.concat([existing_df, final_df], ignore_index=True)
            # 去重（保留最后一次处理的结果）
            final_df = final_df.drop_duplicates(subset=["audio_id"], keep="last")

        final_df.to_csv(output_csv, index=False, encoding="utf-8")
        logger.info(f"最终保存: {len(final_df)} 条记录, {len(final_df.columns)} 个字段")

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("特征提取完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  跳过: {skip_count}")
    logger.info(f"  输出文件: {output_csv}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 打印特征字段统计
    if all_features:
        feature_count = len(all_features[0]) - 2  # 减去 audio_id 和 original_filename
        logger.info(f"")
        logger.info(f"每个音频提取 {feature_count} 个特征")
        logger.info(f"特征维度:")
        logger.info(f"  基础特征: 4")
        logger.info(f"  时域特征: 9")
        logger.info(f"  频域特征: 8 + 14(对比度) = 22")
        logger.info(f"  MFCC: 20x2(mean/std) + 20(delta) = 60")
        logger.info(f"  色度: 12x2(mean/std) + 2(主音) = 26")
        logger.info(f"  节奏: 6")
        logger.info(f"  响度: 1")
        logger.info(f"  总计: ~128 个特征")


if __name__ == "__main__":
    main()
