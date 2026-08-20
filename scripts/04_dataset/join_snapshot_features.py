"""
join_snapshot_features.py
GPU 快照特征 join 脚本

功能：
1. 读取快照中的 all_features.csv（汇总特征）
2. 遍历每个 track 目录，读取 beats.csv 和 f0.csv
3. 计算额外的统计特征（beats 间隔、f0 统计等）
4. 将所有特征 join 成完整数据集
5. 输出到 data/04_final_dataset/final_metadata/
6. 可选项：调用 freeze_version.py 生成版本

用法：
    # 从指定快照 join
    python join_snapshot_features.py --snapshot ./snapshots/gpu_backup_20260820_173500

    # join 后自动冻结版本
    python join_snapshot_features.py --snapshot ./snapshots/gpu_backup_20260820_173500 --freeze --note "mtg-jamendo 20首特征提取"

    # 只输出不冻结
    python join_snapshot_features.py --snapshot ./snapshots/gpu_backup_20260820_173500
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

# 时区
TZ = timezone(timedelta(hours=8))

# 输出目录
FINAL_METADATA_DIR = PROJECT_ROOT / "data" / "04_final_dataset" / "final_metadata"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"join_features_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_all_features(snapshot_path: Path) -> pd.DataFrame:
    """加载 all_features.csv"""
    csv_path = snapshot_path / "features" / "all_features.csv"

    if not csv_path.exists():
        # 尝试 csv 目录
        csv_path = snapshot_path / "csv" / "all_features.csv"

    if not csv_path.exists():
        logger.error(f"all_features.csv 不存在: {csv_path}")
        raise FileNotFoundError(f"all_features.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"加载 all_features.csv: {len(df)} 条记录, {len(df.columns)} 个字段")
    logger.info(f"  字段: {list(df.columns)}")
    return df


def analyze_beats(beats_csv: Path) -> Dict:
    """
    分析 beats.csv，计算统计特征

    返回：
        {
            "n_beats_detailed": 节拍数,
            "beat_interval_mean": 平均间隔(秒),
            "beat_interval_std": 间隔标准差,
            "beat_interval_min": 最小间隔,
            "beat_interval_max": 最大间隔,
            "first_beat_time": 第一个节拍时间,
            "last_beat_time": 最后一个节拍时间,
        }
    """
    result = {
        "n_beats_detailed": 0,
        "beat_interval_mean": np.nan,
        "beat_interval_std": np.nan,
        "beat_interval_min": np.nan,
        "beat_interval_max": np.nan,
        "first_beat_time": np.nan,
        "last_beat_time": np.nan,
    }

    if not beats_csv.exists():
        return result

    try:
        df = pd.read_csv(beats_csv)
        if "beat_time_s" not in df.columns:
            return result

        beat_times = df["beat_time_s"].values
        result["n_beats_detailed"] = len(beat_times)

        if len(beat_times) > 0:
            result["first_beat_time"] = float(beat_times[0])
            result["last_beat_time"] = float(beat_times[-1])

        if len(beat_times) > 1:
            intervals = np.diff(beat_times)
            result["beat_interval_mean"] = float(np.mean(intervals))
            result["beat_interval_std"] = float(np.std(intervals))
            result["beat_interval_min"] = float(np.min(intervals))
            result["beat_interval_max"] = float(np.max(intervals))

    except Exception as e:
        logger.warning(f"分析 beats.csv 失败 {beats_csv}: {e}")

    return result


def analyze_f0(f0_csv: Path) -> Dict:
    """
    分析 f0.csv，计算统计特征

    返回：
        {
            "n_f0_frames_detailed": F0 帧数,
            "f0_hz_mean": 平均基频(Hz),
            "f0_hz_std": 基频标准差,
            "f0_hz_min": 最小基频,
            "f0_hz_max": 最大基频,
            "f0_hz_median": 基频中位数,
            "f0_confidence_mean": 平均置信度,
            "f0_confidence_std": 置信度标准差,
            "f0_high_conf_ratio": 高置信度帧比例(conf>0.5),
            "f0_vocal_ratio": 有声帧比例(conf>0.3),
        }
    """
    result = {
        "n_f0_frames_detailed": 0,
        "f0_hz_mean": np.nan,
        "f0_hz_std": np.nan,
        "f0_hz_min": np.nan,
        "f0_hz_max": np.nan,
        "f0_hz_median": np.nan,
        "f0_confidence_mean": np.nan,
        "f0_confidence_std": np.nan,
        "f0_high_conf_ratio": np.nan,
        "f0_vocal_ratio": np.nan,
    }

    if not f0_csv.exists():
        return result

    try:
        df = pd.read_csv(f0_csv)
        if "f0_hz" not in df.columns or "confidence" not in df.columns:
            return result

        result["n_f0_frames_detailed"] = len(df)

        if len(df) > 0:
            f0_values = df["f0_hz"].values
            conf_values = df["confidence"].values

            result["f0_hz_mean"] = float(np.mean(f0_values))
            result["f0_hz_std"] = float(np.std(f0_values))
            result["f0_hz_min"] = float(np.min(f0_values))
            result["f0_hz_max"] = float(np.max(f0_values))
            result["f0_hz_median"] = float(np.median(f0_values))

            result["f0_confidence_mean"] = float(np.mean(conf_values))
            result["f0_confidence_std"] = float(np.std(conf_values))

            result["f0_high_conf_ratio"] = float(np.mean(conf_values > 0.5))
            result["f0_vocal_ratio"] = float(np.mean(conf_values > 0.3))

    except Exception as e:
        logger.warning(f"分析 f0.csv 失败 {f0_csv}: {e}")

    return result


def join_features(snapshot_path: Path) -> pd.DataFrame:
    """
    主 join 函数：将 all_features.csv 与每个 track 的 beats/f0 统计特征合并

    返回：
        合并后的 DataFrame
    """
    # 1. 加载汇总特征
    df = load_all_features(snapshot_path)

    # 2. 遍历每个 track，计算详细统计特征
    detailed_features = []
    features_dir = snapshot_path / "features"

    for idx, row in df.iterrows():
        track_id = row["track"]
        track_dir = features_dir / track_id

        logger.info(f"[{idx + 1}/{len(df)}] 分析 {track_id}")

        # 分析 beats
        beats_csv = track_dir / "beats.csv"
        beats_stats = analyze_beats(beats_csv)

        # 分析 f0
        f0_csv = track_dir / "f0.csv"
        f0_stats = analyze_f0(f0_csv)

        # 合并
        track_features = {"track": track_id}
        track_features.update(beats_stats)
        track_features.update(f0_stats)
        detailed_features.append(track_features)

    # 3. 创建详细特征 DataFrame
    detailed_df = pd.DataFrame(detailed_features)
    logger.info(f"详细特征计算完成: {len(detailed_df)} 条记录, {len(detailed_df.columns)} 个字段")

    # 4. 合并
    merged_df = pd.merge(df, detailed_df, on="track", how="left")
    logger.info(f"合并完成: {len(merged_df)} 条记录, {len(merged_df.columns)} 个字段")

    return merged_df


def generate_audio_manifest(merged_df: pd.DataFrame, snapshot_path: Path) -> pd.DataFrame:
    """
    生成 audio_manifest.csv（音频清单）

    注意：这是 mtg-jamendo 的数据，音频文件本身不在本地
    audio_manifest 记录元数据和特征摘要，audio_id 使用 track 名称
    """
    manifest = pd.DataFrame()

    # 使用 track 作为 audio_id（mtg-jamendo 格式）
    manifest["audio_id"] = merged_df["track"]
    manifest["original_filename"] = merged_df["track"].apply(lambda x: f"{x}.mp3")
    manifest["file_relative_path"] = merged_df["track"].apply(lambda x: f"raw_audio/mtg_jamendo/{x}.mp3")
    manifest["status"] = "active"
    manifest["quality_flags"] = ""
    manifest["duration_seconds"] = merged_df.get("dur_s", np.nan)
    manifest["sample_rate"] = 44100  # mtg-jamendo 默认
    manifest["bit_depth"] = 16
    manifest["channels"] = 2
    manifest["file_bytes"] = np.nan  # 未知
    manifest["sha256"] = ""  # 未知
    manifest["source"] = "mtg_jamendo"
    manifest["snapshot"] = snapshot_path.name
    manifest["imported_at"] = datetime.now(TZ).isoformat()
    manifest["updated_at"] = datetime.now(TZ).isoformat()

    return manifest


def save_outputs(merged_df: pd.DataFrame, manifest: pd.DataFrame,
                 snapshot_path: Path) -> Dict[str, Path]:
    """
    保存输出文件到 final_metadata/

    返回：
        输出文件路径字典
    """
    FINAL_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # 1. 保存完整 join 结果（CSV）
    dataset_csv = FINAL_METADATA_DIR / "dataset_joined.csv"
    merged_df.to_csv(dataset_csv, index=False, encoding="utf-8")
    outputs["dataset_csv"] = dataset_csv
    logger.info(f"已保存 dataset_joined.csv: {len(merged_df)} 行 x {len(merged_df.columns)} 列")

    # 2. 保存完整 join 结果（JSON）
    dataset_json = FINAL_METADATA_DIR / "dataset_joined.json"
    merged_df.to_json(dataset_json, orient="records", force_ascii=False, indent=2)
    outputs["dataset_json"] = dataset_json
    logger.info(f"已保存 dataset_joined.json")

    # 3. 保存 audio_manifest.csv
    manifest_csv = FINAL_METADATA_DIR / "audio_manifest.csv"
    manifest.to_csv(manifest_csv, index=False, encoding="utf-8")
    outputs["manifest_csv"] = manifest_csv
    logger.info(f"已保存 audio_manifest.csv: {len(manifest)} 条记录")

    # 4. 保存特征统计摘要
    summary = {
        "generated_at": datetime.now(TZ).isoformat(),
        "snapshot": snapshot_path.name,
        "total_tracks": len(merged_df),
        "total_features": len(merged_df.columns),
        "feature_columns": list(merged_df.columns),
        "status_distribution": merged_df["status"].value_counts().to_dict() if "status" in merged_df.columns else {},
        "duration_stats": {
            "mean": float(merged_df["dur_s"].mean()) if "dur_s" in merged_df.columns else None,
            "std": float(merged_df["dur_s"].std()) if "dur_s" in merged_df.columns else None,
            "min": float(merged_df["dur_s"].min()) if "dur_s" in merged_df.columns else None,
            "max": float(merged_df["dur_s"].max()) if "dur_s" in merged_df.columns else None,
        },
        "bpm_stats": {
            "mean": float(merged_df["bpm"].mean()) if "bpm" in merged_df.columns else None,
            "std": float(merged_df["bpm"].std()) if "bpm" in merged_df.columns else None,
            "min": float(merged_df["bpm"].min()) if "bpm" in merged_df.columns else None,
            "max": float(merged_df["bpm"].max()) if "bpm" in merged_df.columns else None,
        },
    }

    summary_json = FINAL_METADATA_DIR / "feature_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    outputs["summary_json"] = summary_json
    logger.info(f"已保存 feature_summary.json")

    # 5. 保存 readme.md
    readme = f"""# 数据集元数据

## 基本信息

| 项目 | 内容 |
|------|------|
| 生成时间 | {datetime.now(TZ).isoformat()} |
| 来源快照 | {snapshot_path.name} |
| 数据来源 | mtg-jamendo |
| 样本总数 | {len(merged_df)} |
| 特征总数 | {len(merged_df.columns)} |

## 文件清单

| 文件 | 说明 |
|------|------|
| dataset_joined.csv | 完整 join 结果（CSV） |
| dataset_joined.json | 完整 join 结果（JSON） |
| audio_manifest.csv | 音频清单 |
| feature_summary.json | 特征统计摘要 |

## 特征字段

### 基础特征（来自 all_features.csv）
{', '.join([c for c in merged_df.columns if c in ['track', 'status', 'lufs', 'bpm', 'n_beats', 'clap_dim', 'n_f0_frames', 'f0_conf_mean', 'rms_mean', 'centroid_mean', 'zcr_mean', 'mfcc1_mean', 'chroma_mean', 'dur_s', 'secs']])}

### 节拍详细特征（来自 beats.csv）
{', '.join([c for c in merged_df.columns if c.startswith('beat_') or c == 'n_beats_detailed' or c == 'first_beat_time' or c == 'last_beat_time'])}

### 基频详细特征（来自 f0.csv）
{', '.join([c for c in merged_df.columns if c.startswith('f0_')])}

---

*由 join_snapshot_features.py 自动生成*
"""
    readme_path = FINAL_METADATA_DIR / "readme.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    outputs["readme"] = readme_path
    logger.info(f"已保存 readme.md")

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="GPU 快照特征 join 脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snapshot", type=str, required=True,
                        help="快照目录路径（如 ./snapshots/gpu_backup_20260820_173500）")
    parser.add_argument("--freeze", action="store_true",
                        help="join 完成后自动调用 freeze_version.py 冻结版本")
    parser.add_argument("--note", type=str, default="",
                        help="版本备注（用于 --freeze）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认 data/04_final_dataset/final_metadata/）")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU 快照特征 Join")
    logger.info("=" * 60)

    snapshot_path = Path(args.snapshot).resolve()
    logger.info(f"快照目录: {snapshot_path}")

    if not snapshot_path.exists():
        logger.error(f"❌ 快照目录不存在: {snapshot_path}")
        sys.exit(1)

    # 1. Join 特征
    logger.info("-" * 40)
    logger.info("步骤 1/3: Join 特征")
    merged_df = join_features(snapshot_path)

    # 2. 生成 audio_manifest
    logger.info("-" * 40)
    logger.info("步骤 2/3: 生成 audio_manifest")
    manifest = generate_audio_manifest(merged_df, snapshot_path)

    # 3. 保存输出
    logger.info("-" * 40)
    logger.info("步骤 3/3: 保存输出")
    outputs = save_outputs(merged_df, manifest, snapshot_path)

    # 4. 可选：冻结版本
    if args.freeze:
        logger.info("-" * 40)
        logger.info("冻结版本")
        freeze_script = PROJECT_ROOT / "scripts" / "04_dataset" / "freeze_version.py"

        if freeze_script.exists():
            import subprocess
            cmd = [
                sys.executable,
                str(freeze_script),
                "--note", args.note or f"join from {snapshot_path.name}",
                "--src-snapshot", str(snapshot_path),
            ]
            logger.info(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            if result.returncode != 0:
                logger.error(f"freeze_version.py 执行失败，返回码: {result.returncode}")
        else:
            logger.warning(f"freeze_version.py 不存在: {freeze_script}")

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Join 完成")
    logger.info(f"   样本数: {len(merged_df)}")
    logger.info(f"   特征数: {len(merged_df.columns)}")
    logger.info(f"   输出目录: {FINAL_METADATA_DIR}")
    logger.info(f"   日志文件: {log_file}")
    logger.info("=" * 60)

    # 打印前几行预览
    logger.info("")
    logger.info("数据预览（前5行，关键列）:")
    key_columns = [c for c in ["track", "status", "bpm", "lufs", "dur_s", "f0_hz_mean", "beat_interval_mean"] if c in merged_df.columns]
    logger.info(merged_df[key_columns].head().to_string(index=False))


if __name__ == "__main__":
    main()
