#!/usr/bin/env python3
"""
archive_badcase.py — Badcase 过程态→终态归档脚本

功能：
- 从 03_human_annotation/badcase/（过程态：审核中拒绝/返工的样本）
  筛选最终确认作为 DPO 负样本的 badcase
- 归档到 04_final_dataset/badcase_pool/（终态）
- 记录归档日志，避免过程态和终态混淆

归档条件（必须全部满足）：
1. review_decision = "reject" 或 "needs_revision"
2. 至少一名审核员标记为 badcase（review_flag 含 badcase 或 annotation_note 含 badcase）
3. 明确属于以下类别之一：
   - preannotation_error：预标注错误（模型标签错误）
   - audio_quality：音频质量问题漏网（QC 未过滤）
   - label_conflict：标签矛盾（不同标注员分歧大）
   - outlier：离群样本（风格/特征异常）

用法：
    # 预览（不实际归档）
    python archive_badcase.py --dry-run

    # 执行归档
    python archive_badcase.py --apply

    # 指定来源和目标
    python archive_badcase.py \
      --source data/03_human_annotation/badcase/ \
      --target data/04_final_dataset/badcase_pool/ \
      --apply
"""
import os
import sys
import json
import argparse
import shutil
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

DEFAULT_SOURCE = PROJECT_ROOT / "data" / "03_human_annotation" / "badcase"
DEFAULT_TARGET = PROJECT_ROOT / "data" / "04_final_dataset" / "badcase_pool"

# Badcase 类别
BADCASE_CATEGORIES = {
    "preannotation_error": "预标注错误",
    "audio_quality": "音频质量问题漏网",
    "label_conflict": "标签矛盾",
    "outlier": "离群样本",
}

# 归档条件
REQUIRED_REVIEW_DECISIONS = {"reject 拒绝", "needs_revision 需返工", "reject", "needs_revision"}
BADCASE_KEYWORDS = {"badcase", "bad case", "错误样本", "问题样本", "负样本"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_badcase_samples(source_dir: Path) -> List[Dict]:
    """加载过程态 badcase 样本（从 Label Studio 导出的 JSON）"""
    samples = []

    if not source_dir.exists():
        logger.warning(f"来源目录不存在: {source_dir}")
        return samples

    for json_file in sorted(source_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 支持单条或列表
            if isinstance(data, list):
                for item in data:
                    item["_source_file"] = str(json_file)
                    samples.append(item)
            else:
                data["_source_file"] = str(json_file)
                samples.append(data)
        except Exception as e:
            logger.warning(f"读取失败: {json_file}, {e}")

    logger.info(f"加载过程态 badcase 样本: {len(samples)} 条（来自 {source_dir}）")
    return samples


def is_badcase_candidate(sample: Dict) -> tuple:
    """
    检查样本是否符合 badcase 归档条件。

    Returns:
        (is_candidate: bool, reason: str, category: str)
    """
    # 1. 检查 review_decision
    review_decision = str(sample.get("review_decision", "")).strip()
    if review_decision not in REQUIRED_REVIEW_DECISIONS:
        return False, f"review_decision={review_decision} 不在拒绝/返工列表", None

    # 2. 检查 badcase 标记（review_flag 或 annotation_note）
    review_flag = str(sample.get("review_flag", "")).lower()
    annotation_note = str(sample.get("annotation_note", "")).lower()
    has_badcase_marker = any(kw in review_flag for kw in BADCASE_KEYWORDS) or \
                          any(kw in annotation_note for kw in BADCASE_KEYWORDS)

    # 也检查显式的 is_badcase 字段
    is_explicit_badcase = sample.get("is_badcase", False) is True

    if not has_badcase_marker and not is_explicit_badcase:
        return False, "无 badcase 标记（review_flag/annotation_note/is_badcase）", None

    # 3. 推断类别
    category = infer_badcase_category(sample)

    return True, f"符合条件: review_decision={review_decision}, category={category}", category


def infer_badcase_category(sample: Dict) -> str:
    """从 annotation_note 或 review_flag 推断 badcase 类别"""
    note = str(sample.get("annotation_note", "")).lower()
    flag = str(sample.get("review_flag", "")).lower()
    combined = note + " " + flag

    # 显式类别字段优先
    explicit_category = sample.get("badcase_category")
    if explicit_category in BADCASE_CATEGORIES:
        return explicit_category

    # 关键词推断
    if any(kw in combined for kw in ["预标注错误", "标签错误", "模型错误", "preannotation", "prediction error"]):
        return "preannotation_error"
    if any(kw in combined for kw in ["质量", "噪声", "爆音", "静音", "audio quality", "noise", "clipping"]):
        return "audio_quality"
    if any(kw in combined for kw in ["分歧", "矛盾", "不一致", "conflict", "disagreement", "iaa"]):
        return "label_conflict"
    if any(kw in combined for kw in ["离群", "异常", "outlier", "异常样本"]):
        return "outlier"

    return "preannotation_error"  # 默认


def archive_badcase(
    samples: List[Dict],
    target_dir: Path,
    dry_run: bool = True,
) -> Dict:
    """
    归档符合条件的 badcase 到终态目录。

    Args:
        samples: 过程态 badcase 样本列表
        target_dir: 终态目录
        dry_run: 预览模式（不实际写入）

    Returns:
        归档统计
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    archived = []
    skipped = []
    category_counts = defaultdict(int)

    for sample in samples:
        audio_id = sample.get("audio_id", sample.get("id", "unknown"))
        is_candidate, reason, category = is_badcase_candidate(sample)

        if not is_candidate:
            skipped.append({"audio_id": audio_id, "reason": reason})
            continue

        # 构建终态 badcase 记录
        badcase_record = {
            "audio_id": audio_id,
            "badcase_category": category,
            "badcase_category_label": BADCASE_CATEGORIES.get(category, "未知"),
            "review_decision": sample.get("review_decision"),
            "review_flag": sample.get("review_flag"),
            "annotation_note": sample.get("annotation_note", ""),
            "original_predictions": sample.get("predictions", sample.get("prediction", {})),
            "source_file": sample.get("_source_file", ""),
            "archived_at": datetime.now(TZ).isoformat(),
            "schema_version": "1.0",
        }

        # 保留原始标注结果（如果有）
        if "annotations" in sample:
            badcase_record["original_annotations"] = sample["annotations"]

        archived.append(badcase_record)
        category_counts[category] += 1

        if not dry_run:
            # 写入终态目录
            output_path = target_dir / f"{audio_id}_badcase.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(badcase_record, f, indent=2, ensure_ascii=False)

    # 生成归档清单
    manifest = {
        "archived_at": datetime.now(TZ).isoformat(),
        "total_candidates": len(samples),
        "archived_count": len(archived),
        "skipped_count": len(skipped),
        "category_distribution": dict(category_counts),
        "archived_ids": [b["audio_id"] for b in archived],
        "skipped": skipped,
        "schema_version": "1.0",
    }

    if not dry_run:
        manifest_path = target_dir / "badcase_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info(f"归档清单已保存: {manifest_path}")

    return manifest


def print_summary(manifest: Dict, dry_run: bool = True):
    """打印归档摘要"""
    mode = "🔍 预览模式（未实际写入）" if dry_run else "✅ 已执行归档"
    logger.info("=" * 60)
    logger.info(f"Badcase 归档 — {mode}")
    logger.info("=" * 60)
    logger.info(f"  候选总数: {manifest['total_candidates']}")
    logger.info(f"  已归档: {manifest['archived_count']}")
    logger.info(f"  已跳过: {manifest['skipped_count']}")
    logger.info(f"  类别分布:")
    for cat, count in manifest["category_distribution"].items():
        label = BADCASE_CATEGORIES.get(cat, cat)
        logger.info(f"    - {label} ({cat}): {count}")
    if manifest["skipped"]:
        logger.info(f"  跳过原因（前5条）:")
        for s in manifest["skipped"][:5]:
            logger.info(f"    - {s['audio_id']}: {s['reason']}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Badcase 过程态→终态归档脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE),
                        help="过程态 badcase 来源目录（默认 data/03_human_annotation/badcase/）")
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET),
                        help="终态 badcase 目标目录（默认 data/04_final_dataset/badcase_pool/）")
    parser.add_argument("--apply", action="store_true",
                        help="执行归档（默认 dry-run 预览）")
    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)
    dry_run = not args.apply

    logger.info(f"来源: {source_dir}")
    logger.info(f"目标: {target_dir}")
    logger.info(f"模式: {'预览' if dry_run else '执行'}")

    # 1. 加载过程态 badcase
    samples = load_badcase_samples(source_dir)

    if not samples:
        logger.warning("无 badcase 样本，退出")
        return

    # 2. 归档
    manifest = archive_badcase(samples, target_dir, dry_run=dry_run)

    # 3. 打印摘要
    print_summary(manifest, dry_run=dry_run)

    if dry_run and manifest["archived_count"] > 0:
        logger.info("")
        logger.info("💡 确认无误后，加 --apply 参数执行实际归档")


if __name__ == "__main__":
    main()
