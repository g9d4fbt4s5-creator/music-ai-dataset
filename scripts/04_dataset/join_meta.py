"""
join_meta.py
MIR 特征统计表 x 官方 GT 元数据联合（track_id 主键）

⚠️ 核心约束（2026-08-20 用户规则）：
- 禁止读取 mp3 内部 ID3 标签（不可信，一律不碰）
- track_id 是唯一关联键：MIR 表的 track 列 = mp3 文件名去后缀 = TSV 的 TRACK_ID
- 所有 join/校验都以 track_id 做主键

功能：
1. 读取 GPU 拉回的 MIR 摘要表（all_features.csv / smoke_stats.csv）
2. 读取 MTG-Jamendo 官方元数据 TSV（手动解析多标签 TAGS）
3. 以 track_id 为键 inner join
4. 输出 dataset_joined.csv / dataset_joined.json 到指定目录

用法：
    # 基本用法
    python join_meta.py \\
        --snapshot snapshots/gpu_backup_20260820_173500 \\
        --output data/04_final_dataset/v20260820_173500

    # 指定元数据 TSV 路径
    python join_meta.py \\
        --snapshot snapshots/gpu_backup_20260820_173500 \\
        --output data/04_final_dataset/v20260820_173500 \\
        --metadata ~/Downloads/mtg_jamendo_meta/data/raw_30s_cleantags.tsv

    # 预览模式（只打印命中率，不写文件）
    python join_meta.py --snapshot ... --output ... --dry-run
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认元数据 TSV 路径（Mac 本地裁剪版，只含 GPU 实际下载的 track）
DEFAULT_METADATA_TSV = Path.home() / "Downloads" / "mtg_jamendo_meta" / "data" / "raw_30s_cleantags.tsv"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
from datetime import datetime, timezone, timedelta
TZ = timezone(timedelta(hours=8))
_time_str = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
_log_file = LOG_DIR / f"join_meta_{_time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def find_mir_csv(snapshot_dir: Path) -> Optional[Path]:
    """在快照目录中查找 MIR 摘要 CSV"""
    # 优先 all_features.csv（当前流水线产物名）
    candidates = [
        snapshot_dir / "csv" / "all_features.csv",
        snapshot_dir / "csv" / "smoke_stats.csv",       # 兼容旧名
        snapshot_dir / "all_features.csv",               # 顶层
        snapshot_dir / "smoke_stats.csv",               # 顶层旧名
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_metadata_tsv(tsv_path: Path) -> pd.DataFrame:
    """
    解析 MTG-Jamendo 元数据 TSV

    ⚠️ TAGS 列多标签用 tab 分隔，pd.read_csv(sep='\\t') 会列数报错，必须手动解析。
    策略：前 5 列固定，剩余合并为空格分隔的 TAGS 列。
    """
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows.append(p[:5] + [" ".join(p[5:])])

    df = pd.DataFrame(rows, columns=header[:5] + ["TAGS"])
    # DURATION 转浮点
    if "DURATION" in df.columns:
        df["DURATION"] = pd.to_numeric(df["DURATION"], errors="coerce")
    return df


def join_meta(
    snapshot_dir: Path,
    output_dir: Path,
    metadata_tsv: Path,
    dry_run: bool = False,
) -> dict:
    """
    执行 MIR x 元数据联合

    参数：
        snapshot_dir: 快照目录（含 csv/all_features.csv）
        output_dir: 输出目录（dataset_joined.csv/json）
        metadata_tsv: 元数据 TSV 路径
        dry_run: 预览模式（不写文件）

    返回：
        结果字典
    """
    result = {
        "mir_rows": 0,
        "meta_rows": 0,
        "joined_rows": 0,
        "hit_rate": 0.0,
        "output_csv": None,
        "output_json": None,
        "errors": [],
    }

    # ---- 1. 查找 MIR CSV ----
    mir_csv = find_mir_csv(snapshot_dir)
    if mir_csv is None:
        msg = f"快照目录中未找到 MIR 摘要 CSV: {snapshot_dir}"
        logger.error(msg)
        result["errors"].append(msg)
        return result

    logger.info(f"MIR 摘要表: {mir_csv}")
    df_mir = pd.read_csv(mir_csv)
    result["mir_rows"] = len(df_mir)
    logger.info(f"  MIR 行数: {len(df_mir)}")

    # ---- 2. 解析元数据 TSV ----
    if not metadata_tsv.exists():
        msg = f"元数据 TSV 不存在: {metadata_tsv}"
        logger.error(msg)
        result["errors"].append(msg)
        return result

    logger.info(f"元数据 TSV: {metadata_tsv}")
    df_meta = parse_metadata_tsv(metadata_tsv)
    result["meta_rows"] = len(df_meta)
    logger.info(f"  元数据行数: {len(df_meta)}")

    # ---- 3. track_id 关联 ----
    # MIR 表的 track 列 = TSV 的 TRACK_ID 列
    if "track" not in df_mir.columns:
        msg = f"MIR CSV 缺少 'track' 列，现有列: {list(df_mir.columns)}"
        logger.error(msg)
        result["errors"].append(msg)
        return result

    if "TRACK_ID" not in df_meta.columns:
        msg = f"元数据 TSV 缺少 'TRACK_ID' 列，现有列: {list(df_meta.columns)}"
        logger.error(msg)
        result["errors"].append(msg)
        return result

    df_join = pd.merge(
        df_mir,
        df_meta,
        left_on="track",
        right_on="TRACK_ID",
        how="inner",
    )

    result["joined_rows"] = len(df_join)
    hit_rate = len(df_join) / len(df_mir) * 100 if len(df_mir) > 0 else 0
    result["hit_rate"] = round(hit_rate, 2)

    logger.info(f"join 命中: {len(df_join)}/{len(df_mir)} ({hit_rate:.1f}%)")

    # 检查未命中的 track
    if len(df_join) < len(df_mir):
        unmatched = set(df_mir["track"]) - set(df_meta["TRACK_ID"])
        logger.warning(f"未命中元数据的 track ({len(unmatched)} 个):")
        for t in sorted(unmatched)[:10]:
            logger.warning(f"  {t}")
        if len(unmatched) > 10:
            logger.warning(f"  ... (共 {len(unmatched)} 个)")

    if dry_run:
        logger.info("[DRY-RUN] 预览模式，不写文件")
        return result

    if len(df_join) == 0:
        logger.error("join 结果为空，不写文件")
        result["errors"].append("join result is empty")
        return result

    # ---- 4. 输出 ----
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "dataset_joined.csv"
    out_json = output_dir / "dataset_joined.json"

    df_join.to_csv(out_csv, index=False)
    df_join.to_json(out_json, orient="records", force_ascii=False, indent=2)

    result["output_csv"] = str(out_csv)
    result["output_json"] = str(out_json)

    logger.info(f"已输出:")
    logger.info(f"  CSV : {out_csv}")
    logger.info(f"  JSON: {out_json}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="MIR 特征统计表 x 官方 GT 元数据联合（track_id 主键）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法
  python join_meta.py \\
      --snapshot snapshots/gpu_backup_20260820_173500 \\
      --output data/04_final_dataset/v20260820_173500

  # 指定元数据 TSV
  python join_meta.py \\
      --snapshot snapshots/gpu_backup_20260820_173500 \\
      --output data/04_final_dataset/v20260820_173500 \\
      --metadata ~/Downloads/mtg_jamendo_meta/data/raw_30s_cleantags.tsv

  # 预览模式
  python join_meta.py --snapshot ... --output ... --dry-run
        """,
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        required=True,
        help="快照目录路径（含 csv/all_features.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出目录路径（dataset_joined.csv/json）",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(DEFAULT_METADATA_TSV),
        help=f"元数据 TSV 路径（默认: {DEFAULT_METADATA_TSV}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式（只打印命中率，不写文件）",
    )

    args = parser.parse_args()

    # 解析路径（支持相对路径，相对于项目根目录）
    snapshot_dir = Path(args.snapshot)
    if not snapshot_dir.is_absolute():
        snapshot_dir = PROJECT_ROOT / snapshot_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    metadata_tsv = Path(args.metadata).expanduser()

    logger.info("=" * 60)
    logger.info("MIR x 元数据联合 (track_id 主键)")
    logger.info("=" * 60)
    logger.info(f"快照目录: {snapshot_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"元数据 TSV: {metadata_tsv}")
    logger.info(f"预览模式: {'是' if args.dry_run else '否'}")
    logger.info("")

    result = join_meta(
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
        metadata_tsv=metadata_tsv,
        dry_run=args.dry_run,
    )

    logger.info("")
    logger.info("=" * 60)
    if not result["errors"]:
        logger.info(f"join 完成: {result['joined_rows']}/{result['mir_rows']} "
                     f"({result['hit_rate']}%)")
        if result["output_csv"]:
            logger.info(f"  CSV : {result['output_csv']}")
            logger.info(f"  JSON: {result['output_json']}")
        logger.info(f"  日志: {_log_file}")
    else:
        logger.error("join 失败:")
        for err in result["errors"]:
            logger.error(f"  - {err}")
    logger.info("=" * 60)

    sys.exit(0 if not result["errors"] else 1)


if __name__ == "__main__":
    main()
