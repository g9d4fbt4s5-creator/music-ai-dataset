#!/usr/bin/env python3
"""
数据集分布统计脚本 v2

统计所有核心子集的质量指标分布，防止域偏移。

子集优先级：
- P0 必统计（用于域偏移告警）：train, val, golden, holdout
- P1 按需统计（仅输出统计，不参与偏移告警）：ood, marginal
- fail：仅统计数量，不参与任何对比

输出产物：
- output/stats/dist_all_subsets.csv      # 全部子集各项统计
- output/stats/dist_core_compare.csv     # P0核心四组对比
- output/stats/distribution_warnings.csv # 只有P0组才会产生告警
- output/stats/fail_stats.csv            # fail样本数量统计

使用:
    python dataset_distribution_stats.py \
        --qc-report data/00.5_cleaned/reports/vXXX/qc_gate_report.csv \
        --split-file data/04_final_dataset/splits/dataset_splits.csv \
        --golden-dir data/03_human_annotation/golden_set \
        --holdout-dir data/00_raw_collect/holdout_pool \
        --ood-dir data/00_raw_collect/ood_pool \
        --output-dir data/04_final_dataset/stats
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 子集定义
SUBSET_DEFINITIONS = {
    # P0 核心对比组（用于域偏移告警）
    "train": {"priority": "P0", "description": "训练集"},
    "val": {"priority": "P0", "description": "验证集（早停依据）"},
    "golden": {"priority": "P0", "description": "内部回归冒烟集（few-shot/回归测试）"},
    "holdout": {"priority": "P0", "description": "冻结最终 benchmark"},
    # P1 附加输出组（仅输出统计，不参与偏移告警）
    "ood": {"priority": "P1", "description": "域外测试集（鲁棒性测试）"},
    "marginal": {"priority": "P1", "description": "待人工复核候选池"},
}

P0_SUBSETS = ["train", "val", "golden", "holdout"]
P1_SUBSETS = ["ood", "marginal"]

# 要统计的质量指标
QUALITY_METRICS = [
    "snr_db", "dynamic_range_db", "silence_ratio", "clipping_ratio",
    "loudness_lufs", "duration_sec",
]

# 域偏移告警阈值（均值差异 > 1个标准差）
DRIFT_THRESHOLD = 1.0


class DatasetDistributionStats:
    """数据集分布统计器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.warnings = []

    def load_subsets(
        self,
        qc_report_path: str,
        split_file: Optional[str] = None,
        golden_dir: Optional[str] = None,
        holdout_dir: Optional[str] = None,
        ood_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载所有子集的数据

        Returns:
            {subset_name: DataFrame}
        """
        subsets = {}

        # 加载 QC 报告
        qc_df = pd.read_csv(qc_report_path)
        print(f"✅ 加载 QC 报告: {len(qc_df)} 条")

        # 1. train / val：从 split_file 加载
        if split_file and os.path.exists(split_file):
            splits_df = pd.read_csv(split_file)
            for split_name in ["train", "val", "valid", "validation"]:
                if split_name in splits_df["split"].values:
                    actual_name = "val" if split_name in ["val", "valid", "validation"] else split_name
                    split_ids = splits_df[splits_df["split"] == split_name]["audio_id"].values
                    subsets[actual_name] = qc_df[qc_df["audio_id"].isin(split_ids)].copy()
                    print(f"✅ {actual_name}: {len(subsets[actual_name])} 条")

        # 2. golden：从目录加载（音频文件名 = audio_id）
        if golden_dir and os.path.exists(golden_dir):
            golden_ids = self._get_audio_ids_from_dir(golden_dir)
            subsets["golden"] = qc_df[qc_df["audio_id"].isin(golden_ids)].copy()
            print(f"✅ golden: {len(subsets['golden'])} 条")

        # 3. holdout：从目录加载
        if holdout_dir and os.path.exists(holdout_dir):
            holdout_ids = self._get_audio_ids_from_dir(holdout_dir)
            subsets["holdout"] = qc_df[qc_df["audio_id"].isin(holdout_ids)].copy()
            print(f"✅ holdout: {len(subsets['holdout'])} 条")

        # 4. ood：从目录加载
        if ood_dir and os.path.exists(ood_dir):
            ood_ids = self._get_audio_ids_from_dir(ood_dir)
            subsets["ood"] = qc_df[qc_df["audio_id"].isin(ood_ids)].copy()
            print(f"✅ ood: {len(subsets['ood'])} 条")

        # 5. marginal：从 QC 报告中筛选
        if "final_branch" in qc_df.columns:
            subsets["marginal"] = qc_df[qc_df["final_branch"] == "marginal"].copy()
            print(f"✅ marginal: {len(subsets['marginal'])} 条")

        # 6. fail：从 QC 报告中筛选（仅统计数量）
        if "final_branch" in qc_df.columns:
            fail_df = qc_df[qc_df["final_branch"] == "fail"].copy()
            self._save_fail_stats(fail_df)

        return subsets

    def _get_audio_ids_from_dir(self, directory: str) -> List[str]:
        """从目录中提取音频ID（文件名去掉扩展名）"""
        audio_ids = []
        for ext in ["*.mp3", "*.wav", "*.flac", "*.m4a"]:
            for f in Path(directory).rglob(ext):
                audio_ids.append(f.stem)
        return audio_ids

    def compute_metric_stats(self, df: pd.DataFrame, metric: str) -> Dict:
        """计算单个指标的统计分布"""
        if metric not in df.columns:
            # 尝试兼容列名
            if metric == "snr_db" and "snr" in df.columns:
                values = pd.to_numeric(df["snr"], errors="coerce").dropna()
            elif metric == "clipping_ratio" and "clip_ratio" in df.columns:
                values = pd.to_numeric(df["clip_ratio"], errors="coerce").dropna()
            elif metric == "dynamic_range_db" and "dynamic_range" in df.columns:
                values = pd.to_numeric(df["dynamic_range"], errors="coerce").dropna()
            else:
                return {"available": False, "count": 0}
        else:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()

        if len(values) == 0:
            return {"available": False, "count": 0}

        return {
            "available": True,
            "count": len(values),
            "mean": round(float(values.mean()), 2),
            "std": round(float(values.std()), 2) if len(values) > 1 else 0,
            "min": round(float(values.min()), 2),
            "p25": round(float(values.quantile(0.25)), 2),
            "median": round(float(values.median()), 2),
            "p75": round(float(values.quantile(0.75)), 2),
            "max": round(float(values.max()), 2),
        }

    def compute_branch_distribution(self, df: pd.DataFrame) -> Dict:
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

    def detect_drift(self, subsets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        检测 P0 子集之间的域偏移

        只对 P0 子集（train, val, golden, holdout）做两两对比。
        如果某特征均值差异 > 1个标准差，输出告警。
        """
        warnings = []
        p0_dfs = {name: subsets[name] for name in P0_SUBSETS if name in subsets}

        if len(p0_dfs) < 2:
            print("⚠️  P0 子集不足2个，跳过域偏移检测")
            return warnings

        # 以 train 为基准
        if "train" not in p0_dfs:
            print("⚠️  没有 train 子集，跳过域偏移检测")
            return warnings

        baseline_df = p0_dfs["train"]

        for metric in QUALITY_METRICS:
            baseline_stats = self.compute_metric_stats(baseline_df, metric)
            if not baseline_stats.get("available", False):
                continue

            baseline_mean = baseline_stats["mean"]
            baseline_std = baseline_stats["std"] or 1  # 避免除零

            for compare_name in ["val", "golden", "holdout"]:
                if compare_name not in p0_dfs:
                    continue
                compare_stats = self.compute_metric_stats(p0_dfs[compare_name], metric)
                if not compare_stats.get("available", False):
                    continue

                compare_mean = compare_stats["mean"]
                diff = abs(compare_mean - baseline_mean)
                z_score = diff / baseline_std if baseline_std > 0 else 0

                if z_score > DRIFT_THRESHOLD:
                    warning = {
                        "metric": metric,
                        "baseline": "train",
                        "baseline_mean": baseline_mean,
                        "compare": compare_name,
                        "compare_mean": compare_mean,
                        "diff": round(diff, 2),
                        "z_score": round(z_score, 2),
                        "severity": "high" if z_score > 2 else "medium",
                        "timestamp": datetime.now().isoformat(),
                    }
                    warnings.append(warning)
                    print(f"⚠️  域偏移告警: {metric} train={baseline_mean} vs {compare_name}={compare_mean} "
                          f"(diff={diff:.2f}, z={z_score:.2f})")

        self.warnings = warnings
        return warnings

    def save_all_subsets_stats(self, subsets: Dict[str, pd.DataFrame]) -> str:
        """保存全部子集的统计数据"""
        rows = []
        for subset_name, df in subsets.items():
            priority = SUBSET_DEFINITIONS.get(subset_name, {}).get("priority", "unknown")
            desc = SUBSET_DEFINITIONS.get(subset_name, {}).get("description", "")
            branch_dist = self.compute_branch_distribution(df)

            row = {
                "subset": subset_name,
                "priority": priority,
                "description": desc,
                "total_count": len(df),
                "pass_count": branch_dist.get("pass", {}).get("count", 0),
                "pass_ratio": branch_dist.get("pass", {}).get("ratio", 0),
                "marginal_count": branch_dist.get("marginal", {}).get("count", 0),
                "marginal_ratio": branch_dist.get("marginal", {}).get("ratio", 0),
                "fail_count": branch_dist.get("fail", {}).get("count", 0),
                "fail_ratio": branch_dist.get("fail", {}).get("ratio", 0),
            }

            for metric in QUALITY_METRICS:
                stats = self.compute_metric_stats(df, metric)
                row[f"{metric}_mean"] = stats.get("mean", "")
                row[f"{metric}_std"] = stats.get("std", "")
                row[f"{metric}_median"] = stats.get("median", "")

            rows.append(row)

        df_all = pd.DataFrame(rows)
        output_path = self.output_dir / "dist_all_subsets.csv"
        df_all.to_csv(output_path, index=False)
        print(f"✅ 全部子集统计: {output_path}")
        return str(output_path)

    def save_core_compare(self, subsets: Dict[str, pd.DataFrame]) -> str:
        """保存 P0 核心四组对比数据"""
        p0_dfs = {name: subsets[name] for name in P0_SUBSETS if name in subsets}

        rows = []
        for metric in QUALITY_METRICS:
            row = {"metric": metric}
            for subset_name in P0_SUBSETS:
                if subset_name in p0_dfs:
                    stats = self.compute_metric_stats(p0_dfs[subset_name], metric)
                    row[f"{subset_name}_mean"] = stats.get("mean", "")
                    row[f"{subset_name}_std"] = stats.get("std", "")
                    row[f"{subset_name}_median"] = stats.get("median", "")
                else:
                    row[f"{subset_name}_mean"] = ""
                    row[f"{subset_name}_std"] = ""
                    row[f"{subset_name}_median"] = ""
            rows.append(row)

        df_compare = pd.DataFrame(rows)
        output_path = self.output_dir / "dist_core_compare.csv"
        df_compare.to_csv(output_path, index=False)
        print(f"✅ P0核心四组对比: {output_path}")
        return str(output_path)

    def save_warnings(self) -> str:
        """保存域偏移告警"""
        if not self.warnings:
            print("✅ 无域偏移告警")
            # 仍然创建空文件
            output_path = self.output_dir / "distribution_warnings.csv"
            pd.DataFrame(columns=["metric", "baseline", "baseline_mean", "compare",
                                   "compare_mean", "diff", "z_score", "severity", "timestamp"]
                        ).to_csv(output_path, index=False)
            return str(output_path)

        df_warnings = pd.DataFrame(self.warnings)
        output_path = self.output_dir / "distribution_warnings.csv"
        df_warnings.to_csv(output_path, index=False)
        print(f"✅ 域偏移告警: {output_path} ({len(self.warnings)} 条)")
        return str(output_path)

    def _save_fail_stats(self, fail_df: pd.DataFrame) -> str:
        """保存 fail 样本统计（仅数量，不参与对比）"""
        output_path = self.output_dir / "fail_stats.csv"
        stats = {
            "total_fail": len(fail_df),
            "fail_ratio": 0,  # 需要全量数据计算
            "by_reason": {},
        }

        # 按 fail 原因统计
        if "reasons" in fail_df.columns:
            reasons = []
            for r in fail_df["reasons"].dropna():
                try:
                    r_dict = json.loads(r) if isinstance(r, str) else r
                    for key, val in r_dict.items():
                        if "fail" in str(val).lower() or "fail" in key.lower():
                            reasons.append(key)
                except (json.JSONDecodeError, TypeError):
                    pass
            if reasons:
                stats["by_reason"] = pd.Series(reasons).value_counts().to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✅ fail样本统计: {output_path} ({len(fail_df)} 条)")
        return str(output_path)

    def generate_report(self, subsets: Dict[str, pd.DataFrame]) -> str:
        """生成 Markdown 格式的分布报告"""
        lines = []
        lines.append("# 数据集分布统计报告\n")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 1. 子集概览
        lines.append("## 1. 子集概览\n")
        lines.append("| 子集 | 优先级 | 说明 | 总数 | pass | marginal | fail |")
        lines.append("|------|--------|------|------|------|----------|------|")
        for subset_name, df in subsets.items():
            priority = SUBSET_DEFINITIONS.get(subset_name, {}).get("priority", "?")
            desc = SUBSET_DEFINITIONS.get(subset_name, {}).get("description", "")
            branch_dist = self.compute_branch_distribution(df)
            lines.append(
                f"| {subset_name} | {priority} | {desc} | {len(df)} | "
                f"{branch_dist.get('pass', {}).get('count', 0)} | "
                f"{branch_dist.get('marginal', {}).get('count', 0)} | "
                f"{branch_dist.get('fail', {}).get('count', 0)} |"
            )
        lines.append("")

        # 2. P0 核心四组质量指标对比
        lines.append("## 2. P0 核心四组质量指标对比\n")
        p0_dfs = {name: subsets[name] for name in P0_SUBSETS if name in subsets}
        for metric in QUALITY_METRICS:
            metric_label = metric.replace("_", " ").title()
            lines.append(f"### {metric_label}\n")
            lines.append("| 子集 | count | mean | std | min | p25 | median | p75 | max |")
            lines.append("|------|-------|------|-----|-----|-----|--------|-----|-----|")
            for subset_name in P0_SUBSETS:
                if subset_name not in p0_dfs:
                    continue
                stats = self.compute_metric_stats(p0_dfs[subset_name], metric)
                if not stats.get("available", False):
                    lines.append(f"| {subset_name} | - | - | - | - | - | - | - | - |")
                    continue
                lines.append(
                    f"| {subset_name} | {stats['count']} | {stats['mean']} | "
                    f"{stats['std']} | {stats['min']} | {stats['p25']} | "
                    f"{stats['median']} | {stats['p75']} | {stats['max']} |"
                )
            lines.append("")

        # 3. 域偏移告警
        lines.append("## 3. 域偏移告警（仅 P0 子集）\n")
        if self.warnings:
            lines.append("| 指标 | 基准(train) | 对比子集 | 均值差异 | z-score | 严重程度 |")
            lines.append("|------|------------|----------|----------|---------|---------|")
            for w in self.warnings:
                lines.append(
                    f"| {w['metric']} | {w['baseline_mean']} | {w['compare']} | "
                    f"{w['diff']} | {w['z_score']} | {w['severity']} |"
                )
        else:
            lines.append("✅ 未检测到显著域偏移（P0 子集之间均值差异 < 1σ）")
        lines.append("")

        # 4. P1 子集独立统计
        lines.append("## 4. P1 子集独立统计（不参与偏移告警）\n")
        for subset_name in P1_SUBSETS:
            if subset_name not in subsets:
                continue
            df = subsets[subset_name]
            desc = SUBSET_DEFINITIONS.get(subset_name, {}).get("description", "")
            lines.append(f"### {subset_name} ({desc})\n")
            lines.append(f"总数: {len(df)}\n")
            branch_dist = self.compute_branch_distribution(df)
            lines.append(f"- pass: {branch_dist.get('pass', {}).get('count', 0)} "
                         f"({branch_dist.get('pass', {}).get('ratio', 0)}%)")
            lines.append(f"- marginal: {branch_dist.get('marginal', {}).get('count', 0)} "
                         f"({branch_dist.get('marginal', {}).get('ratio', 0)}%)")
            lines.append(f"- fail: {branch_dist.get('fail', {}).get('count', 0)} "
                         f"({branch_dist.get('fail', {}).get('ratio', 0)}%)")
            lines.append("")

        report = "\n".join(lines)
        output_path = self.output_dir / "distribution_report.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✅ 分布报告: {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="数据集分布统计脚本 v2")
    parser.add_argument("--qc-report", required=True, help="QC Gate 报告路径")
    parser.add_argument("--split-file", default=None, help="数据集划分文件（audio_id, split）")
    parser.add_argument("--golden-dir", default=None, help="golden 集目录")
    parser.add_argument("--holdout-dir", default=None, help="holdout 集目录")
    parser.add_argument("--ood-dir", default=None, help="OOD 集目录")
    parser.add_argument("--output-dir", default="data/04_final_dataset/stats", help="输出目录")
    args = parser.parse_args()

    stats = DatasetDistributionStats(args.output_dir)

    # 加载所有子集
    subsets = stats.load_subsets(
        qc_report_path=args.qc_report,
        split_file=args.split_file,
        golden_dir=args.golden_dir,
        holdout_dir=args.holdout_dir,
        ood_dir=args.ood_dir,
    )

    if not subsets:
        print("❌ 没有加载到任何子集数据")
        sys.exit(1)

    # 检测域偏移（仅 P0 子集）
    stats.detect_drift(subsets)

    # 保存所有输出
    stats.save_all_subsets_stats(subsets)
    stats.save_core_compare(subsets)
    stats.save_warnings()
    stats.generate_report(subsets)

    print(f"\n{'='*60}")
    print("分布统计完成")
    print(f"{'='*60}")
    print(f"输出目录: {args.output_dir}")
    print(f"  - dist_all_subsets.csv      (全部子集统计)")
    print(f"  - dist_core_compare.csv     (P0核心四组对比)")
    print(f"  - distribution_warnings.csv (域偏移告警)")
    print(f"  - fail_stats.csv            (fail样本统计)")
    print(f"  - distribution_report.md    (完整报告)")


if __name__ == "__main__":
    main()
