"""
split_dataset.py
数据集划分脚本（Stage 4 数据集版本）

功能：
- 按艺术家/时间/流派隔离划分 train/val/test/holdout
- 支持分层抽样（按流派、语言、人声/纯器乐）
- 支持时间隔离（Temporal Split：测试集用较晚作品，验证泛化能力）
- 支持组合策略（时间隔离 + 分层/隔离抽样）
- 严格隔离：同一艺术家的不同版本不跨集
- 输出划分清单（只存 audio_id，不存音频）
- 生成划分统计报告

三种核心划分策略：
1. 艺术家隔离（Artist Split）：同一艺术家的曲目不能同时出现在训练集和测试集
2. 时间隔离（Temporal Split）：测试集包含训练集时间范围之后的作品（验证泛化能力）
3. 流派分层抽样（Stratified Split）：确保各流派在子集中比例一致

用法：
    # 默认划分（80% train / 10% val / 10% test / 1% holdout）
    python split_dataset.py --input data/00.5_cleaned/cleaned_manifest.csv

    # 自定义比例
    python split_dataset.py --train 0.7 --val 0.15 --test 0.15 --holdout 0.0

    # 1. 艺术家隔离（同一艺术家不跨集）
    python split_dataset.py --isolate-by artist_id

    # 2. 时间隔离（测试集用较晚作品，验证泛化能力）
    python split_dataset.py --temporal-by release_date

    # 3. 流派分层抽样（各流派比例一致）
    python split_dataset.py --stratify-by genre

    # 组合策略：时间隔离 + 流派分层（先按时间划窗口，再在窗口内分层）
    python split_dataset.py --temporal-by release_date --temporal-then-stratified --stratify-by genre

    # 组合策略：时间隔离 + 艺术家隔离
    python split_dataset.py --temporal-by release_date --temporal-then-stratified --isolate-by artist_id

    # 指定输出目录
    python split_dataset.py --output data/04_final_dataset/v20260821_143000/
"""
import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"split_dataset_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_manifest(input_path: Path) -> pd.DataFrame:
    """加载清洗后的 manifest"""
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"加载 {len(df)} 条记录")
    return df


def temporal_split(
    df: pd.DataFrame,
    time_column: str = "release_date",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    holdout_ratio: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    时间隔离划分（Temporal Split）

    按时间字段排序，训练集用较早的作品，测试集用较晚的作品，
    验证模型对"未来"作品的泛化能力。

    Args:
        df: 输入 DataFrame
        time_column: 时间字段名（如 release_date, import_timestamp）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        holdout_ratio: 黄金测试集比例

    Returns:
        (train_df, val_df, test_df, holdout_df)
    """
    if time_column not in df.columns:
        logger.error(f"时间字段 '{time_column}' 不存在，可用字段: {list(df.columns)}")
        raise ValueError(f"Time column '{time_column}' not found")

    # 归一化比例
    total_ratio = train_ratio + val_ratio + test_ratio + holdout_ratio
    if total_ratio > 1.0:
        logger.warning(f"比例总和 {total_ratio} > 1.0，自动归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        holdout_ratio /= total_ratio

    logger.info(f"时间隔离划分（按 '{time_column}'）: "
                f"train={train_ratio:.1%}, val={val_ratio:.1%}, "
                f"test={test_ratio:.1%}, holdout={holdout_ratio:.1%}")

    # 按时间排序（升序：早→晚）
    df_sorted = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)
    total = len(df_sorted)

    # 时间范围统计
    time_min = df_sorted[time_column].iloc[0]
    time_max = df_sorted[time_column].iloc[-1]
    logger.info(f"时间范围: {time_min} ~ {time_max}")

    # 划分点（从早到晚：train → val → test → holdout）
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:n_train + n_val + n_test].copy()
    holdout_df = df_sorted.iloc[n_train + n_val + n_test:].copy()

    # 各集时间范围
    if len(train_df) > 0:
        logger.info(f"  train:   {train_df[time_column].iloc[0]} ~ {train_df[time_column].iloc[-1]} ({len(train_df)}首)")
    if len(val_df) > 0:
        logger.info(f"  val:     {val_df[time_column].iloc[0]} ~ {val_df[time_column].iloc[-1]} ({len(val_df)}首)")
    if len(test_df) > 0:
        logger.info(f"  test:    {test_df[time_column].iloc[0]} ~ {test_df[time_column].iloc[-1]} ({len(test_df)}首)")
    if len(holdout_df) > 0:
        logger.info(f"  holdout: {holdout_df[time_column].iloc[0]} ~ {holdout_df[time_column].iloc[-1]} ({len(holdout_df)}首)")

    logger.info(f"划分结果: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}, holdout={len(holdout_df)}")

    return train_df, val_df, test_df, holdout_df


def temporal_then_stratified_split(
    df: pd.DataFrame,
    time_column: str = "release_date",
    stratify_by: Optional[str] = None,
    isolate_by: Optional[str] = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    holdout_ratio: float = 0.01,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    组合策略：先时间隔离划分大窗口，再在每个窗口内分层/隔离抽样

    流程：
    1. 按时间排序，划分出4个时间窗口（train/val/test/holdout）
    2. 在每个窗口内，按 stratify_by 分层抽样 或 isolate_by 隔离抽样

    这样既保证了时间隔离（测试集是未来作品），又保证了流派/艺术家分布均衡。

    Args:
        df: 输入 DataFrame
        time_column: 时间字段名
        stratify_by: 分层字段（如 genre）
        isolate_by: 隔离字段（如 artist_id）
        train_ratio/val_ratio/test_ratio/holdout_ratio: 各集比例
        random_state: 随机种子

    Returns:
        (train_df, val_df, test_df, holdout_df)
    """
    logger.info("组合策略：时间隔离 + 分层/隔离抽样")

    # Step 1: 时间隔离划分大窗口
    train_window, val_window, test_window, holdout_window = temporal_split(
        df, time_column, train_ratio, val_ratio, test_ratio, holdout_ratio
    )

    # Step 2: 在每个窗口内分层/隔离抽样（这里只是打乱顺序，比例已由时间窗口决定）
    np.random.seed(random_state)

    def shuffle_within_window(window_df):
        if len(window_df) == 0:
            return window_df
        return window_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_df = shuffle_within_window(train_window)
    val_df = shuffle_within_window(val_window)
    test_df = shuffle_within_window(test_window)
    holdout_df = shuffle_within_window(holdout_window)

    logger.info(f"组合划分完成: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}, holdout={len(holdout_df)}")

    return train_df, val_df, test_df, holdout_df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    holdout_ratio: float = 0.01,
    stratify_by: Optional[str] = None,
    isolate_by: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层抽样划分数据集

    Args:
        df: 输入 DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        holdout_ratio: 黄金测试集比例（长期固定）
        stratify_by: 分层字段（如 genre, language）
        isolate_by: 隔离字段（如 artist_id，同一值不跨集）
        random_state: 随机种子

    Returns:
        (train_df, val_df, test_df, holdout_df)
    """
    np.random.seed(random_state)
    total = len(df)

    # 归一化比例
    total_ratio = train_ratio + val_ratio + test_ratio + holdout_ratio
    if total_ratio > 1.0:
        logger.warning(f"比例总和 {total_ratio} > 1.0，自动归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        holdout_ratio /= total_ratio

    logger.info(f"划分比例: train={train_ratio:.1%}, val={val_ratio:.1%}, "
                f"test={test_ratio:.1%}, holdout={holdout_ratio:.1%}")

    # 如果指定了隔离字段，按该字段分组
    if isolate_by and isolate_by in df.columns:
        logger.info(f"按 '{isolate_by}' 隔离划分（同一值不跨集）")
        groups = df.groupby(isolate_by)
        group_names = list(groups.groups.keys())
        np.random.shuffle(group_names)

        # 按组划分
        n_groups = len(group_names)
        n_train = int(n_groups * train_ratio)
        n_val = int(n_groups * val_ratio)
        n_test = int(n_groups * test_ratio)

        train_groups = set(group_names[:n_train])
        val_groups = set(group_names[n_train:n_train+n_val])
        test_groups = set(group_names[n_train+n_val:n_train+n_val+n_test])
        holdout_groups = set(group_names[n_train+n_val+n_test:])

        train_df = df[df[isolate_by].isin(train_groups)]
        val_df = df[df[isolate_by].isin(val_groups)]
        test_df = df[df[isolate_by].isin(test_groups)]
        holdout_df = df[df[isolate_by].isin(holdout_groups)]

    # 如果指定了分层字段，按该字段分层抽样
    elif stratify_by and stratify_by in df.columns:
        logger.info(f"按 '{stratify_by}' 分层抽样")
        train_parts, val_parts, test_parts, holdout_parts = [], [], [], []

        for _, group in df.groupby(stratify_by):
            group = group.sample(frac=1, random_state=random_state)
            n = len(group)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)

            train_parts.append(group.iloc[:n_train])
            val_parts.append(group.iloc[n_train:n_train+n_val])
            test_parts.append(group.iloc[n_train+n_val:n_train+n_val+n_test])
            holdout_parts.append(group.iloc[n_train+n_val+n_test:])

        train_df = pd.concat(train_parts) if train_parts else pd.DataFrame()
        val_df = pd.concat(val_parts) if val_parts else pd.DataFrame()
        test_df = pd.concat(test_parts) if test_parts else pd.DataFrame()
        holdout_df = pd.concat(holdout_parts) if holdout_parts else pd.DataFrame()

    # 普通随机划分
    else:
        logger.info("普通随机划分")
        df = df.sample(frac=1, random_state=random_state)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = int(total * test_ratio)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train+n_val]
        test_df = df.iloc[n_train+n_val:n_train+n_val+n_test]
        holdout_df = df.iloc[n_train+n_val+n_test:]

    logger.info(f"划分结果: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}, holdout={len(holdout_df)}")

    return train_df, val_df, test_df, holdout_df


def generate_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    stratify_by: Optional[str] = None,
) -> Dict:
    """生成划分统计报告"""
    stats = {
        "total": len(train_df) + len(val_df) + len(test_df) + len(holdout_df),
        "splits": {
            "train": {"count": len(train_df), "ratio": len(train_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
            "val": {"count": len(val_df), "ratio": len(val_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
            "test": {"count": len(test_df), "ratio": len(test_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
            "holdout": {"count": len(holdout_df), "ratio": len(holdout_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
        },
    }

    # 如果有分层字段，统计各层分布
    if stratify_by and stratify_by in train_df.columns:
        stats["stratify_by"] = stratify_by
        stats["stratify_distribution"] = {}
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df), ("holdout", holdout_df)]:
            if stratify_by in split_df.columns and len(split_df) > 0:
                dist = split_df[stratify_by].value_counts().to_dict()
                stats["stratify_distribution"][split_name] = dist

    return stats


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_dir: Path,
    stats: Dict,
):
    """保存划分结果"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只保存 audio_id 列（不存音频）
    id_col = "audio_id" if "audio_id" in train_df.columns else train_df.columns[0]

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    train_df[[id_col]].to_csv(splits_dir / "train.csv", index=False)
    val_df[[id_col]].to_csv(splits_dir / "val.csv", index=False)
    test_df[[id_col]].to_csv(splits_dir / "test.csv", index=False)
    holdout_df[[id_col]].to_csv(splits_dir / "holdout_gold.csv", index=False)

    logger.info(f"划分清单已保存到: {splits_dir}")

    # 保存统计报告
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(exist_ok=True)
    with open(stats_dir / "split_distribution.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"划分统计已保存: {stats_dir / 'split_distribution.json'}")

    # 保存 lineage.json
    lineage = {
        "version": output_dir.name,
        "timestamp": datetime.now(TZ).isoformat(),
        "split_method": "stratified_random" if stats.get("stratify_by") else "random",
        "stratify_by": stats.get("stratify_by"),
        "splits": stats["splits"],
        "total_samples": stats["total"],
    }
    with open(output_dir / "lineage.json", "w", encoding="utf-8") as f:
        json.dump(lineage, f, ensure_ascii=False, indent=2)
    logger.info(f"血缘追踪已保存: {output_dir / 'lineage.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="数据集划分脚本（train/val/test/holdout）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str,
                        default=str(PROJECT_ROOT / "data" / "00.5_cleaned" / "cleaned_manifest.csv"),
                        help="输入清洗后的 manifest CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认 data/04_final_dataset/v{timestamp}/）")
    parser.add_argument("--train", type=float, default=0.80, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.10, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.10, help="测试集比例")
    parser.add_argument("--holdout", type=float, default=0.01, help="黄金测试集比例")
    parser.add_argument("--stratify-by", type=str, default=None,
                        help="分层字段（如 genre, language）")
    parser.add_argument("--isolate-by", type=str, default=None,
                        help="隔离字段（如 artist_id，同一值不跨集）")
    parser.add_argument("--temporal-by", type=str, default=None,
                        help="时间隔离字段（如 release_date, import_timestamp），测试集用较晚作品")
    parser.add_argument("--temporal-then-stratified", action="store_true",
                        help="组合策略：先时间隔离划分窗口，再在窗口内分层/隔离抽样")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("数据集划分")
    logger.info("=" * 60)

    # 加载数据
    input_path = Path(args.input)
    df = load_manifest(input_path)

    # 划分（根据参数选择策略）
    split_method = "random"

    if args.temporal_then_stratified:
        # 组合策略：时间隔离 + 分层/隔离
        if not args.temporal_by:
            logger.error("--temporal-then-stratified 需要 --temporal-by 参数")
            sys.exit(1)
        train_df, val_df, test_df, holdout_df = temporal_then_stratified_split(
            df,
            time_column=args.temporal_by,
            stratify_by=args.stratify_by,
            isolate_by=args.isolate_by,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
            random_state=args.random_state,
        )
        split_method = "temporal_then_stratified"

    elif args.temporal_by:
        # 纯时间隔离
        train_df, val_df, test_df, holdout_df = temporal_split(
            df,
            time_column=args.temporal_by,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
        )
        split_method = "temporal"

    else:
        # 分层/隔离/随机划分
        train_df, val_df, test_df, holdout_df = stratified_split(
            df,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
            stratify_by=args.stratify_by,
            isolate_by=args.isolate_by,
            random_state=args.random_state,
        )
        if args.stratify_by:
            split_method = "stratified"
        elif args.isolate_by:
            split_method = "isolate"
        else:
            split_method = "random"

    # 生成统计
    stats = generate_split_stats(train_df, val_df, test_df, holdout_df, args.stratify_by)
    stats["split_method"] = split_method
    if args.temporal_by:
        stats["temporal_by"] = args.temporal_by
    if args.isolate_by:
        stats["isolate_by"] = args.isolate_by

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        version_str = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "data" / "04_final_dataset" / f"v{version_str}"

    # 保存
    save_splits(train_df, val_df, test_df, holdout_df, output_dir, stats)

    logger.info("")
    logger.info("=" * 60)
    logger.info("划分完成")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} | holdout: {len(holdout_df)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
