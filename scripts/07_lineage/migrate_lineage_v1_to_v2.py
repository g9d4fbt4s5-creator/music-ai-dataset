"""
migrate_lineage_v1_to_v2.py
血缘文件格式迁移脚本（v1.0 轻量血缘 → v2.0 算子级血缘）

功能：
- 扫描项目中所有 v1.0 格式的 lineage.json
- 自动转换为 v2.0 格式（lineage_v2.json）
- 保留原 v1.0 文件不变（不删除）
- 支持批量迁移和单个文件迁移

v1.0 格式（轻量血缘）：
{
  "version": "v20260821_115630",
  "timestamp": "2026-08-21T11:56:30",
  "split_method": "random",
  "splits": {"train": {"count": 1, "ratio": 0.5}, ...},
  "total_samples": 2
}

v2.0 格式（算子级血缘）：
{
  "lineage_version": "2.0",
  "dataset_version": "v20260821_115630",
  "created_at": "2026-08-21T11:56:30",
  "updated_at": "2026-08-24T...",
  "operators": [
    {"operator_name": "dataset_split", "operator_version": "1.0", ...}
  ],
  "splits": {"train": {"count": 1, "source_manifest": null}, ...},
  "upstream_lineage": null,
  "notes": "Migrated from v1.0 lineage format"
}

用法：
    # 迁移单个文件
    python migrate_lineage_v1_to_v2.py --input data/04_final_dataset/v20260821_115630/lineage.json

    # 批量迁移（扫描整个项目）
    python migrate_lineage_v1_to_v2.py --scan-all

    # 指定输出路径
    python migrate_lineage_v1_to_v2.py --input lineage.json --output lineage_v2.json

    # 覆盖原文件（不推荐）
    python migrate_lineage_v1_to_v2.py --input lineage.json --overwrite
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def is_v1_lineage(data: Dict) -> bool:
    """判断是否为 v1.0 格式的 lineage.json"""
    # v1.0 特征：有 "version" 字段，没有 "lineage_version" 字段
    return "version" in data and "lineage_version" not in data


def is_v2_lineage(data: Dict) -> bool:
    """判断是否为 v2.0 格式的 lineage.json"""
    return data.get("lineage_version") == "2.0"


def migrate_v1_to_v2(v1_data: Dict, source_path: Optional[str] = None) -> Dict:
    """
    将 v1.0 格式的 lineage.json 转换为 v2.0 格式

    Args:
        v1_data: v1.0 格式的字典
        source_path: 源文件路径（用于记录）

    Returns:
        v2.0 格式的字典
    """
    now = datetime.now(TZ).isoformat()

    # 提取 v1.0 字段
    dataset_version = v1_data.get("version", "unknown")
    timestamp = v1_data.get("timestamp", now)
    split_method = v1_data.get("split_method", "unknown")
    total_samples = v1_data.get("total_samples", 0)

    # 提取划分信息
    v1_splits = v1_data.get("splits", {})
    v2_splits = {}
    for split_name, split_info in v1_splits.items():
        v2_splits[split_name] = {
            "count": split_info.get("count", 0),
            "source_manifest": None,  # v1.0 没有记录来源
            "source_batch": None,
        }

    # 构建算子记录（v1.0 没有详细算子记录，只能推断）
    operators = []

    # 如果有划分信息，添加 dataset_split 算子
    if v1_splits:
        total_count = sum(s.get("count", 0) for s in v1_splits.values())
        operators.append({
            "operator_name": "dataset_split",
            "operator_version": "1.0",
            "model_version": None,
            "timestamp": timestamp,
            "input_manifest": None,  # v1.0 没有记录
            "input_filter": None,
            "input_count": total_count,
            "output_path": None,
            "output_count": total_count,
            "failed_count": 0,
            "failed_samples": [],
            "failure_reasons": {},
            "config": {
                "split_method": split_method,
                "stratify_by": v1_data.get("stratify_by"),
                "migrated_from": "v1.0",
            },
            "duration_sec": None,
            "status": "success",
            "error_message": None,
        })

    # 构建 v2.0 格式
    v2_data = {
        "lineage_version": "2.0",
        "dataset_version": dataset_version,
        "created_at": timestamp,
        "updated_at": now,
        "upstream_lineage": None,
        "notes": f"Migrated from v1.0 lineage format on {now}. Source: {source_path or 'unknown'}",
        "operators": operators,
        "splits": v2_splits,
        # 保留 v1.0 的原始字段（便于回溯）
        "v1_original": {
            "version": dataset_version,
            "timestamp": timestamp,
            "split_method": split_method,
            "total_samples": total_samples,
        },
    }

    return v2_data


def migrate_file(input_path: Path, output_path: Optional[Path] = None,
                 overwrite: bool = False) -> Optional[Path]:
    """
    迁移单个 lineage.json 文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（默认同目录下的 lineage_v2.json）
        overwrite: 是否覆盖原文件

    Returns:
        输出文件路径，或 None（如果跳过）
    """
    if not input_path.exists():
        logger.error(f"文件不存在: {input_path}")
        return None

    # 读取文件
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取文件失败: {input_path} - {e}")
        return None

    # 判断格式
    if is_v2_lineage(data):
        logger.info(f"已是 v2.0 格式，跳过: {input_path}")
        return None

    if not is_v1_lineage(data):
        logger.warning(f"未知格式，跳过: {input_path}")
        return None

    # 迁移
    logger.info(f"迁移 v1.0 → v2.0: {input_path}")
    v2_data = migrate_v1_to_v2(data, source_path=str(input_path))

    # 确定输出路径
    if overwrite:
        output_path = input_path
    elif output_path is None:
        output_path = input_path.parent / "lineage_v2.json"

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(v2_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 已保存: {output_path}")
    return output_path


def scan_and_migrate(root_dir: Path, dry_run: bool = False) -> List[Path]:
    """
    扫描目录并迁移所有 v1.0 格式的 lineage.json

    Args:
        root_dir: 根目录
        dry_run: 预览模式（只打印，不实际迁移）

    Returns:
        迁移的文件路径列表
    """
    migrated = []

    # 查找所有 lineage.json 文件
    lineage_files = list(root_dir.rglob("lineage.json"))
    # 排除已经是 v2 的文件（lineage_v2.json）
    lineage_files = [f for f in lineage_files if f.name == "lineage.json"]

    logger.info(f"找到 {len(lineage_files)} 个 lineage.json 文件")

    for input_path in lineage_files:
        # 读取并判断格式
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"读取失败: {input_path} - {e}")
            continue

        if is_v2_lineage(data):
            logger.info(f"  已是 v2.0，跳过: {input_path}")
            continue

        if not is_v1_lineage(data):
            logger.info(f"  未知格式，跳过: {input_path}")
            continue

        if dry_run:
            logger.info(f"  [预览] 将迁移: {input_path}")
            migrated.append(input_path)
        else:
            result = migrate_file(input_path)
            if result:
                migrated.append(result)

    return migrated


def main():
    parser = argparse.ArgumentParser(
        description="血缘文件格式迁移脚本（v1.0 → v2.0）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None,
                        help="输入 lineage.json 路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认同目录下的 lineage_v2.json）")
    parser.add_argument("--scan-all", action="store_true",
                        help="扫描整个项目并批量迁移")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖原文件（不推荐，默认保留原文件）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式（只打印，不实际迁移）")
    args = parser.parse_args()

    if args.scan_all:
        # 批量迁移
        logger.info("=" * 60)
        logger.info("批量迁移模式")
        logger.info("=" * 60)
        migrated = scan_and_migrate(PROJECT_ROOT, dry_run=args.dry_run)
        logger.info("")
        logger.info(f"迁移完成: {len(migrated)} 个文件")
        if args.dry_run:
            logger.info("（预览模式，未实际写入文件）")

    elif args.input:
        # 单个文件迁移
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path

        output_path = Path(args.output) if args.output else None

        if args.dry_run:
            logger.info(f"[预览] 将迁移: {input_path}")
        else:
            result = migrate_file(input_path, output_path, overwrite=args.overwrite)
            if result:
                logger.info(f"迁移完成: {result}")
            else:
                logger.info("未迁移（可能已是 v2.0 或格式错误）")

    else:
        parser.print_help()
        print("\n示例：")
        print("  # 迁移单个文件")
        print("  python migrate_lineage_v1_to_v2.py --input data/04_final_dataset/v20260821_115630/lineage.json")
        print("")
        print("  # 批量迁移")
        print("  python migrate_lineage_v1_to_v2.py --scan-all")
        print("")
        print("  # 预览模式")
        print("  python migrate_lineage_v1_to_v2.py --scan-all --dry-run")


if __name__ == "__main__":
    main()
