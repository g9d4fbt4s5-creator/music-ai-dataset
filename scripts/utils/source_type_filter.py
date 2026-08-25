#!/usr/bin/env python3
"""
source_type_filter.py
来源类型过滤工具（ADR-003 第7节定义）

功能：
- 定义被排除的 source_type 集合（AI生成、分轨人声等域外样本）
- 提供 filter_by_source_type() 函数，在 Stage 1/划分脚本入口处调用
- 生成过滤报告（排除数量、被排除的 audio_id 列表）

ADR-003 规则：
- ace_studio_generated / demucs_vocals / ace_studio_generated_demucs_vocals
  → 排除出训练/验证/测试/holdout，不进入下游 Stage 2-6
- normal → 正常放行
- 未知/空 → 保守放行（不排除，避免误杀）

用法：
    from scripts.utils.source_type_filter import filter_by_source_type, EXCLUDED_SOURCE_TYPES

    # 在 Stage 1 入口处
    df = filter_by_source_type(df, report_path="data/01_preprocess/source_type_filter_report.json")

    # 在划分脚本中（与 QC 过滤配合）
    df = filter_by_source_type(df)
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ADR-003 第7节定义的被排除 source_type
EXCLUDED_SOURCE_TYPES: Set[str] = {
    "ace_studio_generated",
    "demucs_vocals",
    "demucs_vocals_only",
    "ace_studio_generated_demucs_vocals",
    "ai_generated_vocals",
    "synthetic_dry_vocals",
}

# 已知的合法 source_type（用于校验）
KNOWN_SOURCE_TYPES: Set[str] = {
    "normal",
    "vinyl_rip",
    "live_recording",
    "studio_recording",
    *EXCLUDED_SOURCE_TYPES,
}


def filter_by_source_type(
    df: pd.DataFrame,
    excluded_types: Optional[Set[str]] = None,
    report_path: Optional[Path] = None,
    strict: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    按 source_type 过滤 DataFrame，排除域外样本（AI生成、分轨人声等）。

    Args:
        df: 输入 DataFrame，必须包含 audio_id 和 source_type 列
        excluded_types: 自定义被排除的 source_type 集合（默认使用 EXCLUDED_SOURCE_TYPES）
        report_path: 过滤报告输出路径（JSON格式，可选）
        strict: 严格模式，遇到未知 source_type 时警告（默认 False，保守放行）

    Returns:
        filtered_df: 过滤后的 DataFrame
        report: 过滤报告字典
    """
    if excluded_types is None:
        excluded_types = EXCLUDED_SOURCE_TYPES

    before_count = len(df)

    # 检查 source_type 列是否存在
    if "source_type" not in df.columns:
        logger.warning("DataFrame 缺少 source_type 列，跳过 source_type 过滤")
        report = {
            "filtered": False,
            "reason": "missing_source_type_column",
            "before_count": before_count,
            "after_count": before_count,
            "excluded_count": 0,
            "excluded_ids": [],
            "timestamp": datetime.now().isoformat(),
        }
        return df.copy(), report

    # 填充空值为 "normal"（保守策略：未知不排除）
    df = df.copy()
    df["source_type"] = df["source_type"].fillna("normal")

    # 严格模式：检查未知 source_type
    if strict:
        unknown_types = set(df["source_type"].unique()) - KNOWN_SOURCE_TYPES
        if unknown_types:
            logger.warning(f"发现未知 source_type: {unknown_types}，保守放行（不排除）")

    # 识别被排除的样本
    excluded_mask = df["source_type"].isin(excluded_types)
    excluded_df = df[excluded_mask]
    filtered_df = df[~excluded_mask].copy()

    excluded_count = len(excluded_df)
    after_count = len(filtered_df)

    # 生成报告
    excluded_ids = excluded_df["audio_id"].tolist() if "audio_id" in excluded_df.columns else []
    excluded_by_type = excluded_df["source_type"].value_counts().to_dict() if excluded_count > 0 else {}

    report = {
        "filtered": True,
        "before_count": before_count,
        "after_count": after_count,
        "excluded_count": excluded_count,
        "excluded_by_type": excluded_by_type,
        "excluded_ids": excluded_ids,
        "excluded_types_used": sorted(list(excluded_types)),
        "timestamp": datetime.now().isoformat(),
    }

    if excluded_count > 0:
        logger.info(f"source_type 过滤: {before_count} → {after_count} (排除 {excluded_count} 首)")
        for stype, count in excluded_by_type.items():
            logger.info(f"  {stype}: {count} 首")
    else:
        logger.info(f"source_type 过滤: {before_count} → {after_count} (无排除)")

    # 保存报告
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"source_type 过滤报告已保存: {report_path}")

    return filtered_df, report


def get_excluded_source_types() -> Set[str]:
    """获取当前被排除的 source_type 集合（只读副本）"""
    return set(EXCLUDED_SOURCE_TYPES)


def is_excluded(source_type: str) -> bool:
    """检查单个 source_type 是否被排除"""
    if pd.isna(source_type) or not source_type:
        return False
    return source_type in EXCLUDED_SOURCE_TYPES


if __name__ == "__main__":
    # 独立运行：对 audio_manifest.csv 执行过滤并输出报告
    import argparse

    parser = argparse.ArgumentParser(description="source_type 过滤工具")
    parser.add_argument("--input", default="data/00_raw_collect/audio_manifest.csv",
                        help="输入 manifest CSV")
    parser.add_argument("--output", default=None,
                        help="过滤后的输出 CSV（不指定则不保存，只输出报告）")
    parser.add_argument("--report", default="data/01_preprocess/source_type_filter_report.json",
                        help="过滤报告输出路径")
    parser.add_argument("--strict", action="store_true", help="严格模式，警告未知 source_type")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"加载 manifest: {len(df)} 条")

    filtered_df, report = filter_by_source_type(
        df,
        report_path=Path(args.report),
        strict=args.strict,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(output_path, index=False)
        logger.info(f"过滤后的 manifest 已保存: {output_path} ({len(filtered_df)} 条)")

    print("\n" + "=" * 60)
    print("source_type 过滤报告")
    print("=" * 60)
    print(f"  输入: {report['before_count']} 首")
    print(f"  输出: {report['after_count']} 首")
    print(f"  排除: {report['excluded_count']} 首")
    if report["excluded_by_type"]:
        print("  按类型:")
        for stype, count in report["excluded_by_type"].items():
            print(f"    {stype}: {count} 首")
    print("=" * 60)
