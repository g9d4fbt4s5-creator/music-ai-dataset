"""
generate_recipe.py
生成训练配方（Stage 4 数据集版本）

功能：
- 读取数据集版本目录的划分清单
- 生成训练配方 JSON（数据混合比例、预处理参数、标签映射）
- 支持多数据集混合（如 mtg_jamendo + musiccaps + 自采）
- 生成 recipe.json 供训练脚本读取

用法：
    # 基于数据集版本生成配方
    python generate_recipe.py --dataset-version data/04_final_dataset/v20260821_143000/

    # 自定义训练参数
    python generate_recipe.py --dataset-version v20260821_143000 --batch-size 32 --lr 1e-4

    # 多数据集混合
    python generate_recipe.py --mix mtg_jamendo:0.6,musiccaps:0.3,custom:0.1
"""
import os
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"generate_recipe_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_dataset_version(version_dir: Path) -> Dict:
    """加载数据集版本信息"""
    if not version_dir.exists():
        logger.error(f"数据集版本目录不存在: {version_dir}")
        raise FileNotFoundError(f"Dataset version not found: {version_dir}")

    # 读取划分清单
    splits_dir = version_dir / "splits"
    splits = {}
    for split_name in ["train", "val", "test", "holdout_gold"]:
        split_file = splits_dir / f"{split_name}.csv"
        if split_file.exists():
            df = pd.read_csv(split_file)
            splits[split_name] = {
                "count": len(df),
                "audio_ids": df["audio_id"].tolist() if "audio_id" in df.columns else [],
            }
            logger.info(f"  {split_name}: {len(df)} 条")

    # 读取统计
    stats_file = version_dir / "stats" / "split_distribution.json"
    stats = {}
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            stats = json.load(f)

    # 读取血缘
    lineage_file = version_dir / "lineage.json"
    lineage = {}
    if lineage_file.exists():
        with open(lineage_file, "r", encoding="utf-8") as f:
            lineage = json.load(f)

    return {
        "version": version_dir.name,
        "path": str(version_dir),
        "splits": splits,
        "stats": stats,
        "lineage": lineage,
    }


def generate_recipe(
    dataset_info: Dict,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 100,
    audio_format: str = "flac",
    sample_rate: int = 44100,
    bit_depth: int = 24,
    channels: int = 2,
    chunk_size_sec: int = 30,
    overlap_ratio: float = 0.5,
    features: List[str] = None,
    label_mapping_path: Optional[Path] = None,
    mix_ratios: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    生成训练配方

    Args:
        dataset_info: 数据集信息
        batch_size: 批大小
        learning_rate: 学习率
        epochs: 训练轮数
        audio_format: 音频格式
        sample_rate: 采样率
        bit_depth: 位深
        channels: 声道数
        chunk_size_sec: 切片长度（秒）
        overlap_ratio: 重叠比例
        features: 特征列表
        label_mapping_path: 标签映射文件路径
        mix_ratios: 多数据集混合比例

    Returns:
        recipe dict
    """
    if features is None:
        features = ["mel", "cqt", "chroma", "mfcc"]

    recipe = {
        "recipe_version": "1.0",
        "generated_at": datetime.now(TZ).isoformat(),
        "dataset": {
            "version": dataset_info["version"],
            "path": dataset_info["path"],
            "splits": {
                k: {"count": v["count"]} for k, v in dataset_info["splits"].items()
            },
            "total_samples": sum(v["count"] for v in dataset_info["splits"].values()),
        },
        "audio": {
            "format": audio_format,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "channels": channels,
            "chunk_size_sec": chunk_size_sec,
            "overlap_ratio": overlap_ratio,
        },
        "features": {
            "types": features,
            "n_mels": 128,
            "n_cqt": 84,
            "n_chroma": 12,
            "n_mfcc": 20,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "warmup_steps": 1000,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
        },
        "labels": {
            "mapping_path": str(label_mapping_path) if label_mapping_path else None,
            "num_labels": None,  # 从映射文件读取
        },
        "data_mixing": {
            "enabled": mix_ratios is not None,
            "ratios": mix_ratios or {},
        },
        "augmentation": {
            "time_stretch": {"enabled": True, "min_rate": 0.8, "max_rate": 1.2},
            "pitch_shift": {"enabled": True, "min_semitones": -2, "max_semitones": 2},
            "add_noise": {"enabled": False, "snr_db": 20},
            "random_crop": {"enabled": True, "crop_sec": chunk_size_sec},
        },
        "preprocessing": {
            "normalize_loudness": {"enabled": True, "target_lufs": -14},
            "resample": {"enabled": True, "target_sr": sample_rate},
            "mono_convert": {"enabled": False},
        },
    }

    # 如果有标签映射，读取标签数量
    if label_mapping_path and label_mapping_path.exists():
        try:
            with open(label_mapping_path, "r", encoding="utf-8") as f:
                label_mapping = json.load(f)
            recipe["labels"]["num_labels"] = len(label_mapping)
            recipe["labels"]["label_names"] = list(label_mapping.keys())
            logger.info(f"标签映射: {len(label_mapping)} 个标签")
        except Exception as e:
            logger.warning(f"读取标签映射失败: {e}")

    return recipe


def save_recipe(recipe: Dict, output_path: Path):
    """保存训练配方"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)
    logger.info(f"训练配方已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="生成训练配方（recipe.json）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-version", type=str, required=True,
                        help="数据集版本目录（如 data/04_final_dataset/v20260821_143000/）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 recipe.json 路径（默认在数据集版本目录下）")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--sample-rate", type=int, default=44100, help="采样率")
    parser.add_argument("--chunk-size", type=int, default=30, help="切片长度（秒）")
    parser.add_argument("--label-mapping", type=str, default=None,
                        help="标签映射 JSON 路径")
    parser.add_argument("--mix", type=str, default=None,
                        help="多数据集混合比例，如 mtg:0.6,musiccaps:0.3,custom:0.1")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("生成训练配方")
    logger.info("=" * 60)

    # 加载数据集版本
    version_dir = Path(args.dataset_version)
    if not version_dir.is_absolute():
        version_dir = PROJECT_ROOT / version_dir
    dataset_info = load_dataset_version(version_dir)

    # 解析混合比例
    mix_ratios = None
    if args.mix:
        mix_ratios = {}
        for item in args.mix.split(","):
            name, ratio = item.split(":")
            mix_ratios[name.strip()] = float(ratio)
        logger.info(f"多数据集混合: {mix_ratios}")

    # 标签映射路径
    label_mapping_path = None
    if args.label_mapping:
        label_mapping_path = Path(args.label_mapping)
        if not label_mapping_path.is_absolute():
            label_mapping_path = PROJECT_ROOT / label_mapping_path

    # 生成配方
    recipe = generate_recipe(
        dataset_info,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        sample_rate=args.sample_rate,
        chunk_size_sec=args.chunk_size,
        label_mapping_path=label_mapping_path,
        mix_ratios=mix_ratios,
    )

    # 保存
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = version_dir / "recipe.json"
    save_recipe(recipe, output_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("配方生成完成")
    logger.info(f"  输出: {output_path}")
    logger.info(f"  数据集: {dataset_info['version']} ({recipe['dataset']['total_samples']} 条)")
    logger.info(f"  批大小: {args.batch_size} | 学习率: {args.lr} | 轮数: {args.epochs}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
