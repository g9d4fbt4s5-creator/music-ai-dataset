#!/usr/bin/env python3
"""
分布对齐加权重采样 (Distribution-Aligned Weighted Sampling)

根据 holdout 的 P0 维度分布，给训练集样本计算采样权重，
使训练集的采样分布与 holdout 对齐，同时不删除任何样本。

核心原则:
- 只对齐 holdout 中实际存在的类别(避免双边0陷阱)
- Laplace 平滑处理低频类别
- 权重截断防止极端样本垄断
- 不删除任何样本，只调整采样概率

使用:
    from sample_weights import compute_sample_weights, build_weighted_sampler

    weights = compute_sample_weights(train_df, holdout_df, p0_columns)
    sampler = build_weighted_sampler(train_df, holdout_df, p0_columns)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
"""

import numpy as np
import pandas as pd
from typing import List, Optional

try:
    from torch.utils.data import WeightedRandomSampler
except ImportError:
    WeightedRandomSampler = None


def compute_sample_weights(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    p0_columns: List[str],
    smoothing: float = 1.0,
    clip_min: float = 0.05,
    clip_max: float = 20.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    根据 holdout 的 P0 维度分布，给训练集样本计算采样权重。

    Args:
        train_df: 训练集元数据
        holdout_df: holdout 测试集元数据(金标基准，只读)
        p0_columns: P0 维度列表(如 ['vocal_presence', 'bpm_bucket', 'genre_major'])
        smoothing: Laplace 平滑系数(默认1.0)
        clip_min/clip_max: 权重截断范围
        verbose: 是否打印分布对比日志

    Returns:
        与 train_df 等长的权重数组(已归一化，均值=1.0，总和=len(train_df))
    """
    n_train = len(train_df)
    weights = np.ones(n_train, dtype=np.float64)

    for col in p0_columns:
        if col not in train_df.columns or col not in holdout_df.columns:
            if verbose:
                print(f"[{col}] 跳过: 列不存在于 train_df 或 holdout_df")
            continue

        # 1. holdout 目标分布(只考虑 holdout 中存在的类别)
        holdout_counts = holdout_df[col].value_counts()
        n_holdout_cats = len(holdout_counts)
        holdout_total = len(holdout_df) + smoothing * n_holdout_cats
        holdout_dist = {
            k: (v + smoothing) / holdout_total
            for k, v in holdout_counts.items()
        }

        # 2. 训练集实际分布(同样 Laplace 平滑)
        train_counts = train_df[col].value_counts()
        # 只对 holdout 中出现的类别做对齐，训练集特有的边缘类别不惩罚
        aligned_categories = set(holdout_dist.keys())
        train_total = train_df[col].isin(aligned_categories).sum() + smoothing * len(aligned_categories)

        train_dist = {}
        for cat in aligned_categories:
            cnt = train_counts.get(cat, 0)
            train_dist[cat] = (cnt + smoothing) / train_total

        # 3. 逐样本计算权重(仅当样本属于 holdout 中存在的类别时)
        col_weights = np.ones(n_train, dtype=np.float64)
        for idx, val in train_df[col].items():
            if val in holdout_dist and val in train_dist:
                target = holdout_dist[val]
                actual = train_dist[val]
                # Laplace 已保证 >0，避免除零
                col_weights[idx] = target / actual

        weights *= col_weights

        if verbose:
            print(f"\n[{col}] 分布对齐:")
            print(f"  holdout类别: {list(holdout_dist.keys())}")
            print(f"  权重范围: {col_weights.min():.3f} ~ {col_weights.max():.3f}")
            # 打印 top3 高权重和低权重类别
            weight_by_cat = {}
            for cat in aligned_categories:
                mask = train_df[col] == cat
                if mask.any():
                    weight_by_cat[cat] = col_weights[mask].mean()
            sorted_cats = sorted(weight_by_cat.items(), key=lambda x: x[1], reverse=True)
            print(f"  高权重类别: {sorted_cats[:3]}")
            print(f"  低权重类别: {sorted_cats[-3:]}")

    # 4. 截断极端权重
    weights = np.clip(weights, clip_min, clip_max)

    # 5. 归一化: 均值 = 1.0，总和 = n_train
    weights = weights / weights.mean()

    if verbose:
        print(f"\n{'='*60}")
        print(f"最终权重统计:")
        print(f"  min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
        print(f"  有效样本覆盖率: {(weights > 0.1).sum()}/{n_train} ({(weights > 0.1).mean()*100:.1f}%)")
        print(f"  高权重(>2.0)样本: {(weights > 2.0).sum()} ({(weights > 2.0).mean()*100:.1f}%)")
        print(f"  低权重(<0.5)样本: {(weights < 0.5).sum()} ({(weights < 0.5).mean()*100:.1f}%)")

    return weights.astype(np.float32)


def build_weighted_sampler(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    p0_columns: List[str],
    num_samples_per_epoch: Optional[int] = None,
    **kwargs,
) -> "WeightedRandomSampler":
    """
    一键构建 WeightedRandomSampler，直接传给 DataLoader。

    Args:
        train_df: 训练集元数据
        holdout_df: holdout 测试集元数据
        p0_columns: P0 维度列表
        num_samples_per_epoch: 每epoch采样数(默认=训练集大小)
        **kwargs: 传给 compute_sample_weights 的额外参数

    Returns:
        WeightedRandomSampler 实例
    """
    if WeightedRandomSampler is None:
        raise ImportError(
            "torch 未安装，无法构建 WeightedRandomSampler。"
            "请先安装 torch: pip install torch"
        )

    weights = compute_sample_weights(train_df, holdout_df, p0_columns, **kwargs)

    if num_samples_per_epoch is None:
        num_samples_per_epoch = len(train_df)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples_per_epoch,
        replacement=True,  # 确保每 epoch 样本数稳定
    )


def analyze_distribution_gap(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    p0_columns: List[str],
) -> pd.DataFrame:
    """
    分析训练集与 holdout 的分布差距，用于诊断哪些维度需要对齐。

    Returns:
        DataFrame，每行一个(维度, 类别)，含 train_ratio/holdout_ratio/gap
    """
    rows = []
    for col in p0_columns:
        if col not in train_df.columns or col not in holdout_df.columns:
            continue

        train_dist = train_df[col].value_counts(normalize=True)
        holdout_dist = holdout_df[col].value_counts(normalize=True)
        all_cats = set(train_dist.index) | set(holdout_dist.index)

        for cat in all_cats:
            train_ratio = train_dist.get(cat, 0.0)
            holdout_ratio = holdout_dist.get(cat, 0.0)
            gap = abs(train_ratio - holdout_ratio)
            rows.append({
                "dimension": col,
                "category": cat,
                "train_ratio": round(train_ratio, 4),
                "holdout_ratio": round(holdout_ratio, 4),
                "gap": round(gap, 4),
                "needs_alignment": gap > 0.05,
            })

    df = pd.DataFrame(rows).sort_values("gap", ascending=False)
    return df


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 示例: 27首测试数据(实际用500首)
    print("=" * 60)
    print("分布对齐加权重采样 — 示例")
    print("=" * 60)

    # 构造示例数据
    np.random.seed(42)
    n_train = 500
    n_holdout = 50

    train_df = pd.DataFrame({
        "vocal_presence": np.random.choice(
            ["instrumental", "vocal", "mixed"], n_train, p=[0.7, 0.2, 0.1]
        ),
        "bpm_bucket": np.random.choice(
            ["slow(<80)", "medium(80-120)", "fast(120-160)", "very_fast(>160)"],
            n_train, p=[0.1, 0.3, 0.4, 0.2]
        ),
        "genre_major": np.random.choice(
            ["jazz", "blues", "classical", "electronic"], n_train, p=[0.6, 0.15, 0.1, 0.15]
        ),
    })

    holdout_df = pd.DataFrame({
        "vocal_presence": np.random.choice(
            ["instrumental", "vocal", "mixed"], n_holdout, p=[0.5, 0.35, 0.15]
        ),
        "bpm_bucket": np.random.choice(
            ["slow(<80)", "medium(80-120)", "fast(120-160)", "very_fast(>160)"],
            n_holdout, p=[0.15, 0.35, 0.35, 0.15]
        ),
        "genre_major": np.random.choice(
            ["jazz", "blues", "classical", "electronic"], n_holdout, p=[0.4, 0.25, 0.2, 0.15]
        ),
    })

    p0_columns = ["vocal_presence", "bpm_bucket", "genre_major"]

    # 1. 分析分布差距
    print("\n1. 训练集 vs holdout 分布差距分析:")
    gap_df = analyze_distribution_gap(train_df, holdout_df, p0_columns)
    print(gap_df[gap_df["needs_alignment"]].to_string(index=False))

    # 2. 计算权重
    print("\n2. 计算采样权重:")
    weights = compute_sample_weights(train_df, holdout_df, p0_columns)

    # 3. 构建 sampler(需要 torch)
    try:
        sampler = build_weighted_sampler(train_df, holdout_df, p0_columns)
        print(f"\n3. WeightedRandomSampler 构建成功: {len(sampler)} 样本/epoch")
    except ImportError as e:
        print(f"\n3. 跳过 sampler 构建: {e}")

    print("\n" + "=" * 60)
    print("完成。权重可直接传给 DataLoader(sampler=sampler)")
    print("=" * 60)
