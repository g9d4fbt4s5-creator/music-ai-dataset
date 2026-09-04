#!/usr/bin/env python3
"""
HITL 确认后标记 manifest 脚本（finalize_candidates.py）

功能：
1. 读取 Label Studio 导出的 HITL 确认结果（JSON 格式）
2. 将通过的候选标记为 sample_type（golden_seed / challenge_stress_test）
3. 更新 manifest，记录版本和血缘
4. 生成确认报告

设计原则（ADR-003/004/005）：
- 黄金集标记后必须在 train 中（校验）
- challenge 标记后必须不在 train/val 中（校验）
- 已有标记不被覆盖（--respect-existing 默认开启）
- 记录版本号和确认时间，支持回溯

用法：
    # 从 Label Studio 导出文件标记黄金集
    python scripts/03_human_annotation/finalize_candidates.py \
        --mode golden \
        --manifest data/00_raw_collect/audio_manifest.csv \
        --ls-export data/03_human_annotation/golden_set/ls_export.json \
        --respect-existing \
        --dry-run

    # 直接从候选池 JSON 标记（指定 approved_ids）
    python scripts/03_human_annotation/finalize_candidates.py \
        --mode challenge \
        --manifest data/00_raw_collect/audio_manifest.csv \
        --approved-ids audio_id1,audio_id2,audio_id3 \
        --respect-existing
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==================== 常量 ====================
GOLDEN_SAMPLE_TYPE = "golden_seed"
CHALLENGE_SAMPLE_TYPE = "challenge_stress_test"

# Label Studio 中确认字段的可能名称
APPROVE_VALUES = {"approve", "approved", "yes", "true", "1", "accept", "confirmed"}
REJECT_VALUES = {"reject", "rejected", "no", "false", "0", "decline", "denied"}
UNCERTAIN_VALUES = {"uncertain", "maybe", "unknown", "pending", "review"}


# ==================== 工具函数 ====================
def load_manifest(manifest_path: str) -> pd.DataFrame:
    """加载 manifest，校验必要字段"""
    df = pd.read_csv(manifest_path)
    required_cols = ["audio_id", "sample_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"manifest 缺少必要字段: {missing}")
        sys.exit(1)

    # 确保 sample_type 列存在且填充默认值
    if "sample_type" not in df.columns:
        df["sample_type"] = "normal"
    df["sample_type"] = df["sample_type"].fillna("normal")

    logger.info(f"加载 manifest: {len(df)} 首")
    return df


def parse_ls_export(ls_export_path: str, mode: str) -> Dict[str, str]:
    """
    解析 Label Studio 导出文件，返回 {audio_id: decision} 字典

    decision: approve / reject / uncertain

    支持两种格式：
    1. Label Studio 标准导出格式（annotations 数组）
    2. 简化格式（data + annotations）
    """
    with open(ls_export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Label Studio 导出文件应为 JSON 数组")
        sys.exit(1)

    results = {}
    for task in data:
        # 提取 audio_id
        audio_id = None
        if "data" in task:
            audio_id = task["data"].get("audio_id")
        if not audio_id and "audio_id" in task:
            audio_id = task["audio_id"]

        if not audio_id:
            logger.warning(f"跳过无 audio_id 的任务: {task.get('id', 'unknown')}")
            continue

        # 提取确认结果
        decision = "uncertain"
        if "annotations" in task and len(task["annotations"]) > 0:
            # Label Studio 标准格式
            annotation = task["annotations"][0]
            if "result" in annotation:
                for r in annotation["result"]:
                    if r.get("type") == "choices":
                        value = r.get("value", {}).get("choices", [])
                        if value:
                            choice = str(value[0]).lower().strip()
                            if choice in APPROVE_VALUES:
                                decision = "approve"
                            elif choice in REJECT_VALUES:
                                decision = "reject"
                            else:
                                decision = "uncertain"
        elif "final_decision" in task:
            # 简化格式
            choice = str(task["final_decision"]).lower().strip()
            if choice in APPROVE_VALUES:
                decision = "approve"
            elif choice in REJECT_VALUES:
                decision = "reject"

        results[audio_id] = decision

    logger.info(f"解析 Label Studio 导出: {len(results)} 个任务")
    approve_count = sum(1 for d in results.values() if d == "approve")
    reject_count = sum(1 for d in results.values() if d == "reject")
    uncertain_count = sum(1 for d in results.values() if d == "uncertain")
    logger.info(f"  通过: {approve_count}, 拒绝: {reject_count}, 不确定: {uncertain_count}")

    return results


def validate_golden_constraints(
    manifest_df: pd.DataFrame,
    approved_ids: Set[str],
    train_csv_path: Optional[str] = None,
) -> bool:
    """
    校验黄金集标记约束（ADR-003/004）：
    1. 黄金集必须在 train 中
    2. 黄金集不能与已有 challenge 重叠
    """
    valid = True

    # 校验 1：黄金集必须在 train 中
    if train_csv_path and Path(train_csv_path).exists():
        train_df = pd.read_csv(train_csv_path)
        train_ids = set(train_df["audio_id"].tolist())
        not_in_train = approved_ids - train_ids
        if not_in_train:
            logger.error(f"❌ 黄金集约束违反: {len(not_in_train)} 首不在 train 中")
            for aid in list(not_in_train)[:5]:
                logger.error(f"   - {aid}")
            valid = False
        else:
            logger.info(f"✅ 黄金集约束: 全部 {len(approved_ids)} 首在 train 中")
    else:
        logger.warning("未提供 --train-csv，跳过黄金集 train 约束校验")

    # 校验 2：不能与已有 challenge 重叠
    existing_challenge = set(manifest_df[
        manifest_df["sample_type"] == CHALLENGE_SAMPLE_TYPE
    ]["audio_id"].tolist())
    overlap = approved_ids & existing_challenge
    if overlap:
        logger.error(f"❌ 黄金集约束违反: {len(overlap)} 首已是 challenge_stress_test")
        valid = False

    return valid


def validate_challenge_constraints(
    manifest_df: pd.DataFrame,
    approved_ids: Set[str],
    splits_dir: Optional[str] = None,
) -> bool:
    """
    校验 challenge 标记约束（ADR-003/004）：
    1. challenge 不能在 train/val 中
    2. challenge 不能与已有 golden 重叠
    """
    valid = True

    # 校验 1：challenge 不能在 train/val 中
    if splits_dir and Path(splits_dir).exists():
        train_path = Path(splits_dir) / "train.csv"
        val_path = Path(splits_dir) / "val.csv"

        train_val_ids = set()
        if train_path.exists():
            train_val_ids.update(pd.read_csv(train_path)["audio_id"].tolist())
        if val_path.exists():
            train_val_ids.update(pd.read_csv(val_path)["audio_id"].tolist())

        in_train_val = approved_ids & train_val_ids
        if in_train_val:
            logger.error(f"❌ Challenge 约束违反: {len(in_train_val)} 首在 train/val 中")
            for aid in list(in_train_val)[:5]:
                logger.error(f"   - {aid}")
            valid = False
        else:
            logger.info(f"✅ Challenge 约束: 全部 {len(approved_ids)} 首不在 train/val 中")
    else:
        logger.warning("未提供 --splits-dir，跳过 challenge 位置约束校验")

    # 校验 2：不能与已有 golden 重叠
    existing_golden = set(manifest_df[
        manifest_df["sample_type"] == GOLDEN_SAMPLE_TYPE
    ]["audio_id"].tolist())
    overlap = approved_ids & existing_golden
    if overlap:
        logger.error(f"❌ Challenge 约束违反: {len(overlap)} 首已是 golden_seed")
        valid = False

    return valid


def update_manifest(
    manifest_df: pd.DataFrame,
    approved_ids: Set[str],
    mode: str,
    respect_existing: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    更新 manifest，标记 sample_type

    Returns:
        (updated_df, report)
    """
    sample_type = GOLDEN_SAMPLE_TYPE if mode == "golden" else CHALLENGE_SAMPLE_TYPE
    report = {
        "mode": mode,
        "sample_type": sample_type,
        "approved_count": len(approved_ids),
        "updated_count": 0,
        "skipped_existing": 0,
        "not_found": 0,
        "updated_ids": [],
        "skipped_ids": [],
        "not_found_ids": [],
    }

    manifest_ids = set(manifest_df["audio_id"].tolist())

    for audio_id in approved_ids:
        if audio_id not in manifest_ids:
            report["not_found"] += 1
            report["not_found_ids"].append(audio_id)
            logger.warning(f"  跳过（manifest中不存在）: {audio_id}")
            continue

        current_type = manifest_df.loc[
            manifest_df["audio_id"] == audio_id, "sample_type"
        ].iloc[0]

        # 已有标记不覆盖
        if respect_existing and current_type in [GOLDEN_SAMPLE_TYPE, CHALLENGE_SAMPLE_TYPE]:
            report["skipped_existing"] += 1
            report["skipped_ids"].append(audio_id)
            logger.info(f"  跳过（已有标记 {current_type}）: {audio_id}")
            continue

        # 更新标记
        manifest_df.loc[manifest_df["audio_id"] == audio_id, "sample_type"] = sample_type

        # 黄金集额外字段
        if mode == "golden":
            if "is_golden" in manifest_df.columns:
                manifest_df.loc[manifest_df["audio_id"] == audio_id, "is_golden"] = True
            if "in_train_training" in manifest_df.columns:
                manifest_df.loc[manifest_df["audio_id"] == audio_id, "in_train_training"] = False
            if "actual_split" in manifest_df.columns:
                manifest_df.loc[manifest_df["audio_id"] == audio_id, "actual_split"] = "golden"
            if "source_split_candidate" in manifest_df.columns:
                manifest_df.loc[manifest_df["audio_id"] == audio_id, "source_split_candidate"] = "train"

        # Challenge 额外字段
        if mode == "challenge":
            if "in_knn_pool" in manifest_df.columns:
                manifest_df.loc[manifest_df["audio_id"] == audio_id, "in_knn_pool"] = False
            if "actual_split" in manifest_df.columns:
                manifest_df.loc[manifest_df["audio_id"] == audio_id, "actual_split"] = "challenge"

        report["updated_count"] += 1
        report["updated_ids"].append(audio_id)
        logger.info(f"  标记为 {sample_type}: {audio_id}")

    return manifest_df, report


def save_report(report: Dict, output_path: str):
    """保存确认报告"""
    report["finalized_at"] = datetime.now().isoformat()
    report["version"] = datetime.now().strftime("v%Y%m%d_%H%M%S")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"确认报告已保存: {output_path}")


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description="HITL 确认后标记 manifest 脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模式
    parser.add_argument("--mode", type=str, required=True, choices=["golden", "challenge"],
                        help="标记模式：golden（黄金集）/ challenge（挑战集）")

    # 输入
    parser.add_argument("--manifest", type=str, required=True,
                        help="manifest CSV 路径")
    parser.add_argument("--ls-export", type=str, default=None,
                        help="Label Studio 导出文件路径（JSON 格式）")
    parser.add_argument("--approved-ids", type=str, default=None,
                        help="直接指定通过的 audio_id（逗号分隔，不使用 Label Studio 导出）")

    # 约束校验
    parser.add_argument("--train-csv", type=str, default=None,
                        help="train.csv 路径（golden 模式校验用）")
    parser.add_argument("--splits-dir", type=str, default=None,
                        help="划分结果目录（challenge 模式校验用）")

    # 通用参数
    parser.add_argument("--respect-existing", action="store_true", default=True,
                        help="不覆盖已有 golden_seed / challenge_stress_test 标记（默认开启）")
    parser.add_argument("--no-respect-existing", action="store_true",
                        help="覆盖已有标记（危险，不推荐）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览标记结果，不写文件")
    parser.add_argument("--output", type=str, default=None,
                        help="更新后的 manifest 输出路径（默认覆盖原文件）")
    parser.add_argument("--report", type=str, default=None,
                        help="确认报告输出路径")

    args = parser.parse_args()

    # 处理 --no-respect-existing
    if args.no_respect_existing:
        args.respect_existing = False

    logger.info("=" * 60)
    logger.info(f"finalize_candidates.py 启动")
    logger.info(f"模式: {args.mode}")
    logger.info(f"respect-existing: {args.respect_existing}")
    logger.info(f"dry-run: {args.dry_run}")
    logger.info("=" * 60)

    # 加载 manifest
    manifest_df = load_manifest(args.manifest)

    # 获取通过的 audio_id 列表
    approved_ids = set()
    if args.approved_ids:
        approved_ids = set(aid.strip() for aid in args.approved_ids.split(",") if aid.strip())
        logger.info(f"从 --approved-ids 获取: {len(approved_ids)} 首")
    elif args.ls_export:
        ls_results = parse_ls_export(args.ls_export, args.mode)
        approved_ids = set(aid for aid, d in ls_results.items() if d == "approve")
        logger.info(f"从 Label Studio 导出获取通过: {len(approved_ids)} 首")
    else:
        logger.error("必须指定 --ls-export 或 --approved-ids")
        sys.exit(1)

    if not approved_ids:
        logger.error("没有通过的候选，无需标记")
        sys.exit(0)

    # 约束校验
    logger.info("=" * 60)
    logger.info("约束校验")
    logger.info("=" * 60)

    if args.mode == "golden":
        valid = validate_golden_constraints(manifest_df, approved_ids, args.train_csv)
    else:
        valid = validate_challenge_constraints(manifest_df, approved_ids, args.splits_dir)

    if not valid:
        logger.error("❌ 约束校验失败，终止标记")
        logger.error("   请修正候选列表或划分结果后重试")
        sys.exit(1)

    logger.info("✅ 约束校验通过")

    # 更新 manifest
    logger.info("=" * 60)
    logger.info("更新 manifest")
    logger.info("=" * 60)

    manifest_df, report = update_manifest(
        manifest_df, approved_ids, args.mode, args.respect_existing
    )

    # 输出
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY-RUN 预览（不写文件）")
        logger.info("=" * 60)
        logger.info(f"通过候选: {report['approved_count']} 首")
        logger.info(f"将标记: {report['updated_count']} 首")
        logger.info(f"跳过（已有标记）: {report['skipped_existing']} 首")
        logger.info(f"跳过（不存在）: {report['not_found']} 首")
        if report["updated_ids"]:
            logger.info(f"将标记的 audio_id:")
            for aid in report["updated_ids"]:
                logger.info(f"  - {aid}")
    else:
        # 保存 manifest
        output_path = args.output or args.manifest
        manifest_df.to_csv(output_path, index=False)
        logger.info(f"manifest 已更新: {output_path}")

        # 保存报告
        report_path = args.report
        if not report_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.mode == "golden":
                report_path = f"data/03_human_annotation/golden_set/finalize_report_{timestamp}.json"
            else:
                report_path = f"data/03_human_annotation/challenge_set/finalize_report_{timestamp}.json"
        save_report(report, report_path)

    logger.info("=" * 60)
    logger.info("完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
