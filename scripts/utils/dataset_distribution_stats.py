#!/usr/bin/env python3
"""
数据集质量分布统计脚本

统计 train / validation / holdout 三套集合的各项质量指标分布，
防止域偏移（如训练集SNR大多25dB以上，holdout大量marginal低信噪比样本，
导致模型评测指标异常下跌）。

使用:
    python dataset_distribution_stats.py \
        --qc-report data/00.5_cleaned/reports/vXXX/qc_gate_report.csv \
        --split-file data/01_splits/dataset_splits.csv \
        --output data/01_splits/distribution_report.md

如果没有 split-file，则统计全量数据分布。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_qc_report(qc_report_path: str) -> pd.DataFrame:
    """加载 QC Gate 报告"""
    df = pd.read_csv(qc_report_path)
    print(f"✅ 加载 QC 报告: {len(df)} 条")
    return df


def load_splits(split_file: Optional[str]) -> Optional[pd.DataFrame]:
    """加载数据集划分文件（audio_id, split）"""
    if not split_file or not os.path.exists(split_file):
        print("⚠️ 未提供划分文件，统计全量数据分布")
        return None
    df = pd.read_csv(split_file)
    print(f"✅ 加载划分文件: {len(df)} 条")
    return df


def compute_metric_stats(df: pd.DataFrame, metric: str) -> Dict:
    """计算单个指标的统计分布"""
    if metric not in df.columns:
        return {"available": False}

    values = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(values) == 0:
        return {"available": False, "count": 0}

    return {
        "available": True,
        "count": len(values),
        "mean": round(float(values.mean()), 2),
        "std": round(float(values.std()), 2),
        "min": round(float(values.min()), 2),
        "p25": round(float(values.quantile(0.25)), 2),
        "median": round(float(values.median()), 2),
        "p75": round(float(values.quantile(0.75)), 2),
        "max": round(float(values.max()), 2),
    }


def compute_branch_distribution(df: pd.DataFrame) -> Dict:
    """计算三分支分布"""
    if "final_branch" not in df.columns:
        return {}
    counts = df["final_branch"].value_counts().to_dict()
    total = len(df)
    return {
        branch: {
            "count": int(count),
            "ratio": round(count / total * 100, 1) if total > 0 else 0,
        }
        for branch, count in counts.items()
    }


def generate_report(
    all_stats: Dict[str, Dict],
    metrics: List[str],
    output_path: str,
) -> str:
    """生成 Markdown 格式的分布报告"""
    lines = []
    lines.append("# 数据集质量分布统计报告\n")
    lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 三分支分布
    lines.append("## 1. 三分支分布\n")
    lines.append("| 集合 | 总数 | pass | marginal | fail |")
    lines.append("|------|------|------|----------|------|")
    for split_name, stats in all_stats.items():
        branch_dist = stats.get("branch_distribution", {})
        total = stats.get("total", 0)
        pass_count = branch_dist.get("pass", {}).get("count", 0)
        marginal_count = branch_dist.get("marginal", {}).get("count", 0)
        fail_count = branch_dist.get("fail", {}).get("count", 0)
        lines.append(f"| {split_name} | {total} | {pass_count} | {marginal_count} | {fail_count} |")
    lines.append("")

    # 2. 各指标分布对比
    lines.append("## 2. 质量指标分布对比\n")
    for metric in metrics:
        metric_label = metric.replace("_", " ").title()
        lines.append(f"### {metric_label}\n")
        lines.append("| 集合 | count | mean | std | min | p25 | median | p75 | max |")
        lines.append("|------|-------|------|-----|-----|-----|--------|-----|-----|")
        for split_name, stats in all_stats.items():
            metric_stats = stats.get("metrics", {}).get(metric, {})
            if not metric_stats.get("available", False):
                lines.append(f"| {split_name} | - | - | - | - | - | - | - | - |")
                continue
            lines.append(
                f"| {split_name} | {metric_stats['count']} | {metric_stats['mean']} | "
                f"{metric_stats['std']} | {metric_stats['min']} | {metric_stats['p25']} | "
                f"{metric_stats['median']} | {metric_stats['p75']} | {metric_stats['max']} |"
            )
        lines.append("")

    # 3. 域偏移检测
    lines.append("## 3. 域偏移检测\n")
    lines.append("检测各集合之间质量指标分布是否存在显著差异（均值差异 > 1个标准差）。\n")

    split_names = list(all_stats.keys())
    if len(split_names) >= 2:
        baseline = split_names[0]
        for metric in metrics:
            baseline_stats = all_stats[baseline].get("metrics", {}).get(metric, {})
            if not baseline_stats.get("available", False):
                continue
            baseline_mean = baseline_stats["mean"]
            baseline_std = baseline_stats["std"] or 1  # 避免除零

            for compare_split in split_names[1:]:
                compare_stats = all_stats[compare_split].get("metrics", {}).get(metric, {})
                if not compare_stats.get("available", False):
                    continue
                compare_mean = compare_stats["mean"]
                diff = abs(compare_mean - baseline_mean)
                z_score = diff / baseline_std if baseline_std > 0 else 0

                if z_score > 1.0:
                    lines.append(
                        f"- ⚠️ **{metric}**: {baseline} mean={baseline_mean}, "
                        f"{compare_split} mean={compare_mean}, 差异={diff:.2f} "
                        f"({z_score:.2f}σ) — 可能存在域偏移"
                    )
        lines.append("")

    # 4. 建议
    lines.append("## 4. 建议\n")
    lines.append("- 如果检测到域偏移，建议重新划分数据集或对低质量集合做数据增强")
    lines.append("- marginal 样本可用于模型鲁棒性测试，但不应全部进入训练集")
    lines.append("- 定期运行此脚本，监控数据质量分布变化")
    lines.append("")

    report = "\n".join(lines)

    # 保存报告
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"✅ 分布报告已保存: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="数据集质量分布统计")
    parser.add_argument("--qc-report", required=True, help="QC Gate 报告 CSV 路径")
    parser.add_argument("--split-file", default=None, help="数据集划分文件（audio_id, split），可选")
    parser.add_argument("--output", default="data/01_splits/distribution_report.md", help="输出报告路径")
    args = parser.parse_args()

    # 加载数据
    qc_df = load_qc_report(args.qc_report)
    splits_df = load_splits(args.split_file)

    # 要统计的指标
    metrics = [
        "snr_db", "dynamic_range_db", "silence_ratio", "clipping_ratio",
        "loudness_lufs", "duration_sec",
    ]

    # 按集合统计
    all_stats = {}

    if splits_df is not None and "split" in splits_df.columns:
        # 合并 QC 报告和划分
        merged = qc_df.merge(splits_df[["audio_id", "split"]], on="audio_id", how="left")
        split_names = sorted(merged["split"].dropna().unique())

        for split_name in split_names:
            split_df = merged[merged["split"] == split_name]
            all_stats[split_name] = {
                "total": len(split_df),
                "branch_distribution": compute_branch_distribution(split_df),
                "metrics": {metric: compute_metric_stats(split_df, metric) for metric in metrics},
            }
    else:
        # 全量统计
        all_stats["all"] = {
            "total": len(qc_df),
            "branch_distribution": compute_branch_distribution(qc_df),
            "metrics": {metric: compute_metric_stats(qc_df, metric) for metric in metrics},
        }

    # 生成报告
    generate_report(all_stats, metrics, args.output)

    # 打印摘要
    print(f"\n{'='*60}")
    print("分布统计摘要")
    print(f"{'='*60}")
    for split_name, stats in all_stats.items():
        branch_dist = stats.get("branch_distribution", {})
        print(f"\n  {split_name}: {stats['total']} 条")
        for branch, info in branch_dist.items():
            print(f"    {branch}: {info['count']} ({info['ratio']}%)")


if __name__ == "__main__":
    main()
