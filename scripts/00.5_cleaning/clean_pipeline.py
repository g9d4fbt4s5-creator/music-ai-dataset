"""
clean_pipeline.py
数据清洗主流程脚本

6阶段数据清洗流水线：
采集(00) → 数据清洗(00.5) → 预处理(01)

Stage 1: 元数据清洗（字段标准化、缺失补全、冲突消解、无效样本剔除）
Stage 2: 格式标准化（WAV/FLAC, 44.1kHz/48kHz, 16/24-bit）
Stage 3: 音频质量清洗（损坏检测、静音过滤、音质门槛、响度归一化）
Stage 4: 多级去重（精确去重、近似去重、片段级去重、跨集泄露防控）
Stage 5: 辅助清洗（风格聚类、语言过滤、PII移除）
Stage 6: 预处理输出（重采样、分块、特征提取）

用法：
    # 运行全部阶段
    python clean_pipeline.py

    # 只运行指定阶段
    python clean_pipeline.py --stages 1,2,3

    # 从指定阶段开始
    python clean_pipeline.py --from-stage 3

    # 使用自定义配置
    python clean_pipeline.py --config ./configs/cleaning_config.yaml

    # 预览模式（不实际处理）
    python clean_pipeline.py --dry-run

    # 只处理指定 audio_id
    python clean_pipeline.py --audio-id 01M0E9X162CTB4D15WZQ5D8FVX
"""
import os
import sys
import yaml
import logging
import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 时区
TZ = timezone(timedelta(hours=8))

# 默认配置文件
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cleaning_config.yaml"

# 添加 scripts 目录到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"clean_pipeline_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """加载配置文件"""
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    return config


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """加载音频清单"""
    if not manifest_path.exists():
        logger.error(f"音频清单不存在: {manifest_path}")
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info(f"加载音频清单: {len(df)} 条记录")
    return df


def save_report(report: Dict, report_path: Path):
    """保存清洗报告"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"清洗报告已保存: {report_path}")


def run_stage1_metadata_cleaning(df: pd.DataFrame, config: Dict, dry_run: bool = False) -> pd.DataFrame:
    """
    Stage 1: 元数据清洗

    - 字段标准化（GM128 / VAD / 三级流派）
    - 缺失补全 & 冲突消解
    - 无效样本剔除
    """
    logger.info("=" * 60)
    logger.info("Stage 1: 元数据清洗")
    logger.info("=" * 60)

    stage_config = config.get("stage1_metadata", {})
    if not stage_config.get("enabled", True):
        logger.info("Stage 1 已禁用，跳过")
        return df

    initial_count = len(df)
    report = {
        "stage": "metadata_cleaning",
        "initial_count": initial_count,
        "standardized": 0,
        "missing_filled": 0,
        "conflicts_resolved": 0,
        "invalid_removed": 0,
        "final_count": 0,
    }

    # 1. 字段标准化
    if stage_config.get("field_standardization", {}).get("instrument_standardization", True):
        logger.info("  [1/4] 乐器字段标准化 (GM128)")
        # TODO: 实现 GM128 标准化
        report["standardized"] += len(df)

    if stage_config.get("field_standardization", {}).get("emotion_standardization", True):
        logger.info("  [2/4] 情绪字段标准化 (VAD)")
        # TODO: 实现 VAD 标准化

    if stage_config.get("field_standardization", {}).get("genre_standardization", True):
        logger.info("  [3/4] 流派字段标准化 (三级流派)")
        # TODO: 实现三级流派标准化

    # 2. 缺失值处理
    missing_config = stage_config.get("missing_value", {})
    strategy = missing_config.get("strategy", "flag")
    logger.info(f"  [4/4] 缺失值处理 (策略: {strategy})")

    if strategy == "skip":
        # 跳过有缺失值的样本
        required_fields = missing_config.get("required_fields", [])
        if required_fields:
            before = len(df)
            df = df.dropna(subset=required_fields)
            report["missing_filled"] = before - len(df)
            logger.info(f"    剔除缺失样本: {before - len(df)}")
    elif strategy == "fill_default":
        # 用默认值填充
        defaults = missing_config.get("defaults", {})
        for col, default_val in defaults.items():
            if col in df.columns:
                fill_count = df[col].isna().sum()
                df[col] = df[col].fillna(default_val)
                report["missing_filled"] += fill_count
        logger.info(f"    填充缺失值: {report['missing_filled']}")
    elif strategy == "flag":
        # 标记缺失值，不剔除
        if "quality_flags" not in df.columns:
            df["quality_flags"] = ""
        missing_cols = df.columns[df.isna().any()].tolist()
        for col in missing_cols:
            mask = df[col].isna()
            df.loc[mask, "quality_flags"] += f"missing_{col};"
            report["missing_filled"] += mask.sum()
        logger.info(f"    标记缺失样本: {report['missing_filled']}")

    # 3. 无效样本剔除
    invalid_config = stage_config.get("invalid_filter", {})
    if invalid_config.get("remove_missing_required", False):
        required_fields = invalid_config.get("required_fields", ["audio_id", "file_relative_path"])
        before = len(df)
        df = df.dropna(subset=required_fields)
        report["invalid_removed"] = before - len(df)
        logger.info(f"  剔除无效样本: {report['invalid_removed']}")

    report["final_count"] = len(df)
    logger.info(f"  Stage 1 完成: {initial_count} → {len(df)} (剔除 {initial_count - len(df)})")

    return df


def run_stage2_format_normalization(df: pd.DataFrame, config: Dict, dry_run: bool = False) -> pd.DataFrame:
    """
    Stage 2: 格式标准化

    - WAV/FLAC, 44.1kHz/48kHz, 16/24-bit
    """
    logger.info("=" * 60)
    logger.info("Stage 2: 格式标准化")
    logger.info("=" * 60)

    stage_config = config.get("stage2_format", {})
    if not stage_config.get("enabled", True):
        logger.info("Stage 2 已禁用，跳过")
        return df

    target_format = stage_config.get("target_format", "wav")
    target_sr = stage_config.get("target_sample_rate", 44100)
    target_bit = stage_config.get("target_bit_depth", 16)
    target_ch = stage_config.get("target_channels", 2)

    logger.info(f"  目标格式: {target_format}")
    logger.info(f"  目标采样率: {target_sr} Hz")
    logger.info(f"  目标位深: {target_bit}-bit")
    logger.info(f"  目标声道: {target_ch}")

    # TODO: 实现格式标准化（使用 pydub 或 soundfile）
    # 1. 遍历音频文件
    # 2. 检查格式是否符合要求
    # 3. 不符合的转换格式
    # 4. 更新 manifest 中的路径

    logger.info("  [TODO] 格式标准化功能待实现")
    logger.info(f"  Stage 2 完成: {len(df)} 个样本")

    return df


def run_stage3_quality_cleaning(df: pd.DataFrame, config: Dict, dry_run: bool = False) -> pd.DataFrame:
    """
    Stage 3: 音频质量清洗

    - 损坏检测 & 静音过滤
    - 音质门槛（采样率/SNR/削波/动态范围）
    - 内容过滤（非音乐/人声/安全）
    - 响度归一化（-14 LUFS）
    """
    logger.info("=" * 60)
    logger.info("Stage 3: 音频质量清洗")
    logger.info("=" * 60)

    stage_config = config.get("stage3_quality", {})
    if not stage_config.get("enabled", True):
        logger.info("Stage 3 已禁用，跳过")
        return df

    initial_count = len(df)
    report = {
        "stage": "quality_cleaning",
        "initial_count": initial_count,
        "corrupted": 0,
        "silent": 0,
        "low_quality": 0,
        "loudness_normalized": 0,
        "final_count": 0,
    }

    # 1. 损坏检测
    if stage_config.get("corruption_detection", {}).get("enabled", True):
        logger.info("  [1/4] 损坏检测")
        # TODO: 实现损坏检测（尝试解码，失败则标记为损坏）
        # report["corrupted"] = ...

    # 2. 静音过滤
    if stage_config.get("silence_filter", {}).get("enabled", True):
        logger.info("  [2/4] 静音过滤")
        silence_threshold = stage_config.get("silence_filter", {}).get("silence_threshold", 0.001)
        max_silence_ratio = stage_config.get("silence_filter", {}).get("max_silence_ratio", 0.8)
        logger.info(f"    静音阈值: {silence_threshold}, 最大静音占比: {max_silence_ratio}")
        # TODO: 实现静音检测和过滤

    # 3. 音质门槛
    if stage_config.get("quality_threshold", {}).get("enabled", True):
        logger.info("  [3/4] 音质门槛检查")
        # TODO: 实现 SNR/削波/动态范围检测

    # 4. 响度归一化
    if stage_config.get("loudness_normalization", {}).get("enabled", True):
        logger.info("  [4/4] 响度归一化")
        target_lufs = stage_config.get("loudness_normalization", {}).get("target_lufs", -14)
        logger.info(f"    目标响度: {target_lufs} LUFS")
        # TODO: 实现响度归一化（使用 pyloudnorm）

    report["final_count"] = len(df)
    logger.info(f"  Stage 3 完成: {initial_count} → {len(df)}")

    return df


def run_stage4_deduplication(df: pd.DataFrame, config: Dict, dry_run: bool = False) -> pd.DataFrame:
    """
    Stage 4: 多级去重

    - 精确去重（MD5/SHA-256）
    - 近似去重（Chromaprint, 余弦相似度 > 0.9）
    - 片段级去重（滑动窗口切片）
    - 跨集泄露防控
    """
    logger.info("=" * 60)
    logger.info("Stage 4: 多级去重")
    logger.info("=" * 60)

    stage_config = config.get("stage4_dedup", {})
    if not stage_config.get("enabled", True):
        logger.info("Stage 4 已禁用，跳过")
        return df

    initial_count = len(df)
    report = {
        "stage": "deduplication",
        "initial_count": initial_count,
        "exact_duplicates": 0,
        "approximate_duplicates": 0,
        "segment_duplicates": 0,
        "final_count": 0,
    }

    # 1. 精确去重
    if stage_config.get("exact_dedup", {}).get("enabled", True):
        logger.info("  [1/4] 精确去重 (SHA-256)")
        checksum_csv = stage_config.get("exact_dedup", {}).get("checksum_csv", "")
        strategy = stage_config.get("exact_dedup", {}).get("strategy", "keep_first")
        logger.info(f"    策略: {strategy}")

        if checksum_csv:
            checksum_path = PROJECT_ROOT / checksum_csv
            if checksum_path.exists():
                checksums = pd.read_csv(checksum_path)
                before = len(df)
                # 根据 sha256 去重
                if "sha256" in checksums.columns and "audio_id" in checksums.columns:
                    df = df.merge(checksums[["audio_id", "sha256"]], on="audio_id", how="left")
                    if "sha256" in df.columns:
                        df = df.drop_duplicates(subset=["sha256"], keep="first")
                        df = df.drop(columns=["sha256"])
                    report["exact_duplicates"] = before - len(df)
                    logger.info(f"    精确去重: {report['exact_duplicates']} 个重复")
                else:
                    logger.info("    checksum 文件缺少 sha256 或 audio_id 列，跳过精确去重")
            else:
                logger.info(f"    checksum 文件不存在: {checksum_path}，跳过精确去重")
        else:
            logger.info("    未配置 checksum 文件，跳过精确去重")

    # 2. 近似去重
    if stage_config.get("approximate_dedup", {}).get("enabled", False):
        logger.info("  [2/4] 近似去重 (Chromaprint)")
        # TODO: 实现近似去重（需要 Chromaprint/fpcalc）

    # 3. 片段级去重
    if stage_config.get("segment_dedup", {}).get("enabled", False):
        logger.info("  [3/4] 片段级去重")
        # TODO: 实现片段级去重

    # 4. 跨集泄露防控
    if stage_config.get("cross_set_leakage", {}).get("enabled", False):
        logger.info("  [4/4] 跨集泄露防控")
        # TODO: 实现跨集泄露防控

    report["final_count"] = len(df)
    logger.info(f"  Stage 4 完成: {initial_count} → {len(df)} (去重 {initial_count - len(df)})")

    return df


def run_stage5_auxiliary_cleaning(df: pd.DataFrame, config: Dict, dry_run: bool = False) -> pd.DataFrame:
    """
    Stage 5: 辅助清洗

    - 风格一致性聚类
    - 语言过滤
    - PII 移除
    """
    logger.info("=" * 60)
    logger.info("Stage 5: 辅助清洗")
    logger.info("=" * 60)

    stage_config = config.get("stage5_auxiliary", {})
    if not stage_config.get("enabled", False):
        logger.info("Stage 5 已禁用，跳过")
        return df

    # TODO: 实现辅助清洗
    logger.info("  [TODO] 辅助清洗功能待实现")
    logger.info(f"  Stage 5 完成: {len(df)} 个样本")

    return df


def run_stage6_preprocess_output(df: pd.DataFrame, config: Dict, dry_run: bool = False) -> pd.DataFrame:
    """
    Stage 6: 预处理输出

    - 重采样 & 声道处理
    - 分块（5-30s, 50% 重叠）
    - 特征提取（Mel/CQT/Chroma）
    """
    logger.info("=" * 60)
    logger.info("Stage 6: 预处理输出")
    logger.info("=" * 60)

    stage_config = config.get("stage6_output", {})
    if not stage_config.get("enabled", True):
        logger.info("Stage 6 已禁用，跳过")
        return df

    # 1. 重采样
    if stage_config.get("resample", {}).get("enabled", True):
        logger.info("  [1/3] 重采样 & 声道处理")
        # TODO: 实现重采样（已在 extract_features.py 中部分实现）

    # 2. 分块
    if stage_config.get("chunking", {}).get("enabled", False):
        logger.info("  [2/3] 分块")
        # TODO: 实现分块

    # 3. 特征提取
    if stage_config.get("feature_extraction", {}).get("enabled", False):
        logger.info("  [3/3] 特征提取")
        # 使用独立的 extract_features.py
        logger.info("    建议使用独立的 scripts/01_preprocess/extract_features.py")

    logger.info(f"  Stage 6 完成: {len(df)} 个样本")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="数据清洗主流程脚本（6阶段流水线）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="配置文件路径")
    parser.add_argument("--stages", type=str, default=None,
                        help="只运行指定阶段，如 1,2,3")
    parser.add_argument("--from-stage", type=int, default=None,
                        help="从指定阶段开始运行")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不实际处理")
    parser.add_argument("--audio-id", type=str, default=None,
                        help="只处理指定的 audio_id")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("数据清洗流水线")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"预览模式: {'是' if args.dry_run else '否'}")

    # 加载配置
    config = load_config(Path(args.config))
    global_config = config.get("global", {})

    # 加载音频清单
    manifest_path = PROJECT_ROOT / global_config.get("manifest_csv", "data/00_raw_collect/audio_manifest.csv")
    df = load_manifest(manifest_path)

    # 过滤指定 audio_id
    if args.audio_id:
        df = df[df["audio_id"] == args.audio_id]
        logger.info(f"只处理 audio_id: {args.audio_id} ({len(df)} 个样本)")

    # 确定要运行的阶段
    all_stages = [1, 2, 3, 4, 5, 6]
    if args.stages:
        stages_to_run = [int(s) for s in args.stages.split(",")]
    elif args.from_stage:
        stages_to_run = [s for s in all_stages if s >= args.from_stage]
    else:
        stages_to_run = all_stages

    logger.info(f"运行阶段: {stages_to_run}")
    logger.info(f"初始样本数: {len(df)}")

    # 运行各个阶段
    stage_functions = {
        1: run_stage1_metadata_cleaning,
        2: run_stage2_format_normalization,
        3: run_stage3_quality_cleaning,
        4: run_stage4_deduplication,
        5: run_stage5_auxiliary_cleaning,
        6: run_stage6_preprocess_output,
    }

    for stage in stages_to_run:
        if stage in stage_functions:
            df = stage_functions[stage](df, config, dry_run=args.dry_run)

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("数据清洗完成")
    logger.info(f"  最终样本数: {len(df)}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 保存清洗后的 manifest
    if not args.dry_run:
        output_manifest = PROJECT_ROOT / "data" / "00.5_cleaned" / "cleaned_manifest.csv"
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_manifest, index=False, encoding="utf-8")
        logger.info(f"清洗后的清单已保存: {output_manifest}")


if __name__ == "__main__":
    import json
    main()
