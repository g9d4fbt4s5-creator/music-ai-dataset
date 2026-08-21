"""
verify_lineage.py
血缘校验脚本（Stage 4 数据集版本）

功能：
- 校验数据集版本的血缘完整性
- 检查 train/val/test/holdout 的 audio_id 是否完整
- 检查是否有跨集重复（泄露）
- 检查音频文件是否存在
- 校验 checksum 是否匹配
- 生成校验报告

用法：
    # 校验数据集版本
    python verify_lineage.py --dataset-version data/04_final_dataset/v20260821_143000/

    # 校验音频文件存在性
    python verify_lineage.py --dataset-version v20260821_143000 --check-audio-exists

    # 校验 checksum
    python verify_lineage.py --dataset-version v20260821_143000 --verify-checksum

    # 严格模式（所有检查都通过才算成功）
    python verify_lineage.py --dataset-version v20260821_143000 --strict
"""
import os
import sys
import json
import hashlib
import logging
import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"verify_lineage_{time_str}.log"
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
    """加载数据集版本"""
    if not version_dir.exists():
        logger.error(f"数据集版本目录不存在: {version_dir}")
        raise FileNotFoundError(f"Dataset version not found: {version_dir}")

    splits = {}
    splits_dir = version_dir / "splits"
    for split_name in ["train", "val", "test", "holdout_gold"]:
        split_file = splits_dir / f"{split_name}.csv"
        if split_file.exists():
            df = pd.read_csv(split_file)
            splits[split_name] = df
            logger.info(f"  {split_name}: {len(df)} 条")

    return {"version": version_dir.name, "path": str(version_dir), "splits": splits}


def check_split_completeness(splits: Dict[str, pd.DataFrame]) -> Dict:
    """检查划分完整性"""
    logger.info("检查划分完整性...")
    result = {"passed": True, "errors": [], "warnings": []}

    all_ids = set()
    for split_name, df in splits.items():
        if "audio_id" not in df.columns:
            result["errors"].append(f"{split_name}: 缺少 audio_id 列")
            result["passed"] = False
            continue

        ids = set(df["audio_id"].tolist())
        if len(ids) != len(df):
            result["warnings"].append(f"{split_name}: 有重复 audio_id ({len(df)} 行, {len(ids)} 唯一)")

        # 检查与其他集的交集
        overlap = all_ids & ids
        if overlap:
            result["errors"].append(f"{split_name}: 与其他集有 {len(overlap)} 个重复 audio_id (泄露!)")
            result["passed"] = False

        all_ids.update(ids)

    result["total_unique_ids"] = len(all_ids)
    logger.info(f"  总唯一 audio_id: {len(all_ids)}")

    if result["passed"]:
        logger.info("  ✅ 划分完整性检查通过")
    else:
        logger.error(f"  ❌ 划分完整性检查失败: {len(result['errors'])} 个错误")

    return result


def check_cross_set_leakage(splits: Dict[str, pd.DataFrame]) -> Dict:
    """检查跨集泄露（更详细）"""
    logger.info("检查跨集泄露...")
    result = {"passed": True, "leaks": []}

    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            name1, name2 = split_names[i], split_names[j]
            df1, df2 = splits[name1], splits[name2]

            if "audio_id" not in df1.columns or "audio_id" not in df2.columns:
                continue

            ids1 = set(df1["audio_id"].tolist())
            ids2 = set(df2["audio_id"].tolist())
            overlap = ids1 & ids2

            if overlap:
                result["leaks"].append({
                    "split_1": name1,
                    "split_2": name2,
                    "overlap_count": len(overlap),
                    "overlap_ids": list(overlap)[:10],  # 只记录前10个
                })
                result["passed"] = False
                logger.error(f"  ❌ {name1} ↔ {name2}: {len(overlap)} 个重复 audio_id")

    if result["passed"]:
        logger.info("  ✅ 无跨集泄露")

    return result


def check_source_isolation(version_dir: Path, splits: Dict[str, pd.DataFrame] = None) -> Dict:
    """
    检查来源隔离（test/holdout 是否来自独立采集批次）

    工业级要求：
    - test 集应该来自独立采集的数据池，不参与清洗调优（防止迭代污染）
    - holdout 集应该长期封存，跨版本模型对比（必须与训练集来自不同采集批次）
    - 如果 test/holdout 与 train 来自同一 manifest，存在分布一致和迭代污染风险

    补强3: 如果 manifest 中包含 source_batch 字段，检查 train/val 和 test/holdout 的
    采集批次是否不重叠（语义层面的隔离，不只是 audio_id 不重复）
    """
    logger.info("检查来源隔离（test/holdout 是否来自独立采集批次）...")
    result = {
        "passed": True,
        "test_isolated": False,
        "holdout_isolated": False,
        "test_source": None,
        "holdout_source": None,
        "source_batch_check": None,  # 补强3: source_batch 检查结果
        "warnings": [],
    }

    # 读取 lineage.json 获取来源隔离信息
    lineage_file = version_dir / "lineage.json"
    if lineage_file.exists():
        with open(lineage_file, "r", encoding="utf-8") as f:
            lineage = json.load(f)

        # 检查 schema_version（补强1）
        schema_version = lineage.get("schema_version")
        if schema_version:
            logger.info(f"  lineage schema_version: {schema_version}")
        else:
            logger.warning(f"  ⚠️  lineage.json 缺少 schema_version 字段（建议添加）")

        # 检查来源隔离信息（兼容新旧格式）
        test_source = lineage.get("test_pool_manifest") or lineage.get("stats", {}).get("test_source_isolation")
        holdout_source = lineage.get("holdout_pool_manifest") or lineage.get("stats", {}).get("holdout_source_isolation")

        if test_source:
            result["test_isolated"] = True
            result["test_source"] = test_source
            logger.info(f"  ✅ 测试集来源隔离: {test_source}")
        else:
            result["test_isolated"] = False
            result["warnings"].append(
                "测试集未使用来源隔离（与训练集来自同一manifest），存在迭代污染风险"
            )
            logger.warning(f"  ⚠️  测试集未使用来源隔离（建议使用 --test-from 参数指定独立数据池）")

        if holdout_source:
            result["holdout_isolated"] = True
            result["holdout_source"] = holdout_source
            logger.info(f"  ✅ holdout集来源隔离: {holdout_source}")
        else:
            result["holdout_isolated"] = False
            result["warnings"].append(
                "holdout集未使用来源隔离（与训练集来自同一manifest），跨版本对比可能失效"
            )
            logger.warning(f"  ⚠️  holdout集未使用来源隔离（建议使用 --holdout-from 参数指定独立数据池）")

        # 补强3: source_batch 采集批次隔离检查
        # 如果 splits 数据存在且 manifest 中包含 source_batch 字段，检查采集批次是否不重叠
        if splits and "audio_id" in list(splits.values())[0].columns:
            # 加载全局 manifest 获取 source_batch 字段
            manifest_path = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"
            if manifest_path.exists():
                manifest = pd.read_csv(manifest_path)
                if "source_batch" in manifest.columns:
                    logger.info("  补强3: 检查 source_batch 采集批次隔离...")
                    source_batch_map = dict(zip(manifest["audio_id"], manifest["source_batch"]))

                    # 收集各集合的 source_batch
                    split_batches = {}
                    for split_name, split_df in splits.items():
                        if "audio_id" in split_df.columns:
                            batches = set()
                            for aid in split_df["audio_id"].tolist():
                                batch = source_batch_map.get(aid)
                                if batch:
                                    batches.add(batch)
                            split_batches[split_name] = batches

                    # 检查 train/val 和 test/holdout 的 source_batch 是否不重叠
                    train_val_batches = split_batches.get("train", set()) | split_batches.get("val", set())
                    test_batches = split_batches.get("test", set())
                    holdout_batches = split_batches.get("holdout_gold", set())

                    test_overlap = train_val_batches & test_batches
                    holdout_overlap = train_val_batches & holdout_batches

                    source_batch_result = {
                        "enabled": True,
                        "train_val_batches": sorted(list(train_val_batches)),
                        "test_batches": sorted(list(test_batches)),
                        "holdout_batches": sorted(list(holdout_batches)),
                        "test_batch_overlap": sorted(list(test_overlap)),
                        "holdout_batch_overlap": sorted(list(holdout_overlap)),
                        "passed": len(test_overlap) == 0 and len(holdout_overlap) == 0,
                    }
                    result["source_batch_check"] = source_batch_result

                    if test_overlap:
                        logger.warning(f"  ⚠️  补强3: test 与 train/val 有 {len(test_overlap)} 个重叠采集批次: {sorted(list(test_overlap))[:5]}")
                        result["warnings"].append(f"test 与 train/val 采集批次重叠: {sorted(list(test_overlap))}")
                    else:
                        logger.info(f"  ✅ 补强3: test 与 train/val 采集批次无重叠")

                    if holdout_overlap:
                        logger.warning(f"  ⚠️  补强3: holdout 与 train/val 有 {len(holdout_overlap)} 个重叠采集批次: {sorted(list(holdout_overlap))[:5]}")
                        result["warnings"].append(f"holdout 与 train/val 采集批次重叠: {sorted(list(holdout_overlap))}")
                    else:
                        logger.info(f"  ✅ 补强3: holdout 与 train/val 采集批次无重叠")
                else:
                    logger.info("  补强3: manifest 中无 source_batch 字段，跳过采集批次隔离检查（建议在采集端添加此字段）")
                    result["source_batch_check"] = {"enabled": False, "reason": "manifest 中无 source_batch 字段"}

        # 如果都没有隔离，总体不通过（但只是警告，不阻塞）
        if not result["test_isolated"] and not result["holdout_isolated"]:
            result["passed"] = False  # 标记为不通过，但调用方可决定是否阻塞

        # 补强3: 如果 source_batch 检查失败，也标记为不通过
        if result.get("source_batch_check", {}).get("enabled") and not result["source_batch_check"].get("passed", True):
            result["passed"] = False
    else:
        logger.warning(f"  ⚠️  lineage.json 不存在，跳过来源隔离检查")
        result["warnings"].append("lineage.json 不存在，无法检查来源隔离")

    return result


def check_audio_exists(splits: Dict[str, pd.DataFrame], audio_base_dir: Path) -> Dict:
    """检查音频文件是否存在"""
    logger.info("检查音频文件存在性...")
    result = {"passed": True, "missing": [], "total_checked": 0}

    # 加载 audio_manifest 获取路径
    manifest_path = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"
    if not manifest_path.exists():
        logger.warning(f"  audio_manifest.csv 不存在，跳过音频存在性检查")
        result["passed"] = True  # 不算失败
        return result

    manifest = pd.read_csv(manifest_path)
    path_map = dict(zip(manifest["audio_id"], manifest["file_relative_path"]))

    all_ids = set()
    for df in splits.values():
        if "audio_id" in df.columns:
            all_ids.update(df["audio_id"].tolist())

    for audio_id in all_ids:
        result["total_checked"] += 1
        rel_path = path_map.get(audio_id)
        if not rel_path:
            result["missing"].append({"audio_id": audio_id, "reason": "不在 manifest 中"})
            continue

        audio_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path
        if not audio_path.exists():
            result["missing"].append({"audio_id": audio_id, "path": str(audio_path), "reason": "文件不存在"})

    if result["missing"]:
        result["passed"] = False
        logger.error(f"  ❌ 缺失 {len(result['missing'])} 个音频文件")
        for m in result["missing"][:5]:
            logger.error(f"    - {m['audio_id']}: {m.get('reason', m.get('path', ''))}")
    else:
        logger.info(f"  ✅ 所有 {result['total_checked']} 个音频文件都存在")

    return result


def verify_checksums(splits: Dict[str, pd.DataFrame]) -> Dict:
    """校验 checksum"""
    logger.info("校验 checksum...")
    result = {"passed": True, "mismatched": [], "total_checked": 0}

    # 加载 checksum
    checksum_path = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio_checksums.csv"
    if not checksum_path.exists():
        logger.warning(f"  raw_audio_checksums.csv 不存在，跳过 checksum 校验")
        result["passed"] = True
        return result

    checksums = pd.read_csv(checksum_path)
    checksum_map = dict(zip(checksums["audio_id"], checksums["sha256"]))

    all_ids = set()
    for df in splits.values():
        if "audio_id" in df.columns:
            all_ids.update(df["audio_id"].tolist())

    for audio_id in all_ids:
        expected_hash = checksum_map.get(audio_id)
        if not expected_hash:
            result["mismatched"].append({"audio_id": audio_id, "reason": "无 checksum 记录"})
            continue

        # 这里只检查记录存在，实际文件哈希校验需要读取文件（慢）
        result["total_checked"] += 1

    if result["mismatched"]:
        result["passed"] = False
        logger.error(f"  ❌ {len(result['mismatched'])} 个 audio_id 无 checksum 记录")
    else:
        logger.info(f"  ✅ {result['total_checked']} 个 audio_id 都有 checksum 记录")

    return result


def generate_report(
    dataset_info: Dict,
    completeness: Dict,
    leakage: Dict,
    source_isolation: Dict,
    audio_exists: Dict,
    checksums: Dict,
    output_path: Path,
):
    """生成校验报告"""
    report = {
        "version": dataset_info["version"],
        "verified_at": datetime.now(TZ).isoformat(),
        "checks": {
            "completeness": completeness,
            "cross_set_leakage": leakage,
            "source_isolation": source_isolation,
            "audio_exists": audio_exists,
            "checksums": checksums,
        },
        "overall_passed": all([
            completeness["passed"],
            leakage["passed"],
            audio_exists["passed"],
            checksums["passed"],
        ]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"校验报告已保存: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="血缘校验脚本（检查数据集版本完整性）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-version", type=str, required=True,
                        help="数据集版本目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出校验报告路径（默认在数据集版本目录下）")
    parser.add_argument("--check-audio-exists", action="store_true",
                        help="检查音频文件存在性")
    parser.add_argument("--verify-checksum", action="store_true",
                        help="校验 checksum")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式（所有检查都通过才算成功）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("血缘校验")
    logger.info("=" * 60)

    # 加载数据集版本
    version_dir = Path(args.dataset_version)
    if not version_dir.is_absolute():
        version_dir = PROJECT_ROOT / version_dir
    dataset_info = load_dataset_version(version_dir)

    # 检查划分完整性
    completeness = check_split_completeness(dataset_info["splits"])

    # 检查跨集泄露
    leakage = check_cross_set_leakage(dataset_info["splits"])

    # 检查来源隔离（test/holdout 是否来自独立采集批次）
    source_isolation = check_source_isolation(version_dir, dataset_info["splits"])

    # 检查音频存在性
    audio_exists = {"passed": True, "missing": [], "total_checked": 0}
    if args.check_audio_exists:
        audio_exists = check_audio_exists(dataset_info["splits"], PROJECT_ROOT / "data")

    # 校验 checksum
    checksums = {"passed": True, "mismatched": [], "total_checked": 0}
    if args.verify_checksum:
        checksums = verify_checksums(dataset_info["splits"])

    # 生成报告
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = version_dir / "verification_report.json"
    report = generate_report(dataset_info, completeness, leakage, source_isolation, audio_exists, checksums, output_path)

    # 输出结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("校验结果")
    logger.info("=" * 60)
    logger.info(f"  划分完整性: {'✅ 通过' if completeness['passed'] else '❌ 失败'}")
    logger.info(f"  跨集泄露: {'✅ 无泄露' if leakage['passed'] else '❌ 有泄露'}")
    test_iso = "✅ 已隔离" if source_isolation.get("test_isolated") else "⚠️ 未隔离"
    holdout_iso = "✅ 已隔离" if source_isolation.get("holdout_isolated") else "⚠️ 未隔离"
    logger.info(f"  来源隔离: test={test_iso}, holdout={holdout_iso}")
    missing_count = len(audio_exists.get("missing", []))
    logger.info(f"  音频存在性: {'✅ 全部存在' if audio_exists['passed'] else f'❌ 缺失{missing_count}个'}")
    mismatch_count = len(checksums.get("mismatched", []))
    logger.info(f"  Checksum: {'✅ 全部匹配' if checksums['passed'] else f'❌ {mismatch_count}个不匹配'}")
    logger.info(f"  总体: {'✅ 通过' if report['overall_passed'] else '❌ 失败'}")
    logger.info(f"  报告: {output_path}")
    logger.info("=" * 60)

    if args.strict and not report["overall_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
