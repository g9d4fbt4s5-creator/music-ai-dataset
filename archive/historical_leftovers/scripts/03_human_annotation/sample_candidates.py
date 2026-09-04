#!/usr/bin/env python3
"""
黄金集 / Challenge 集候选采样脚本（自动候选 + HITL 确认）

设计原则（ADR-003/004/005）：
1. 黄金集候选必须在 Stage 4 划分后，从 train.csv 内筛选（禁止读取整个 manifest）
2. Challenge 候选 85 首阶段禁用 hdbscan_outlier 来源（HDBSCAN 在小样本上失效）
3. --respect-existing 默认开启，不覆盖已有 golden_seed / challenge_stress_test 标记
4. 输出候选池 JSON + Label Studio 导入文件，供 HITL 确认

用法：
    # 黄金集候选（划分后执行，强制读取 train.csv）
    python scripts/03_human_annotation/sample_candidates.py \
        --mode golden \
        --manifest data/00_raw_collect/audio_manifest.csv \
        --train-csv data/04_final_dataset/splits/splits/train.csv \
        --per-cluster 2 \
        --respect-existing \
        --output data/03_human_annotation/golden_set/candidates_v1.json

    # Challenge 候选（QC 后执行，85 首禁用 hdbscan_outlier）
    python scripts/03_human_annotation/sample_candidates.py \
        --mode challenge \
        --manifest data/00_raw_collect/audio_manifest.csv \
        --qc-report data/00.5_cleaned/reports/qc_gate_report.csv \
        --challenge-sources qc_marginal,short,long \
        --respect-existing \
        --output data/03_human_annotation/challenge_set/candidates_v1.json

    # 预览（不写文件）
    python scripts/03_human_annotation/sample_candidates.py --mode golden ... --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

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

# 85 首试点阶段禁用的 Challenge 来源
DISABLED_SOURCES_85 = {"hdbscan_outlier"}

# 黄金集硬性门槛
GOLDEN_MIN_DURATION = 120  # 秒
GOLDEN_MAX_DURATION = 600  # 秒（试点期放宽，Qwen-Omni 自动标注成本可控）

# Challenge 时长边界
CHALLENGE_SHORT_MIN = 5    # 秒（排除 fail 样本）
CHALLENGE_SHORT_MAX = 60   # 秒
CHALLENGE_LONG_MIN = 600   # 秒


# ==================== 工具函数 ====================
def load_manifest(manifest_path: str) -> pd.DataFrame:
    """加载 manifest，校验必要字段"""
    df = pd.read_csv(manifest_path)
    required_cols = ["audio_id", "sample_type", "duration_sec"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"manifest 缺少必要字段: {missing}")
        sys.exit(1)
    logger.info(f"加载 manifest: {len(df)} 首")
    return df


def load_train_ids(train_csv_path: str) -> Set[str]:
    """
    加载 train.csv，返回 train 中的 audio_id 集合
    黄金集模式强制调用此函数，禁止读取整个 manifest
    """
    if not Path(train_csv_path).exists():
        logger.error(f"train.csv 不存在: {train_csv_path}")
        logger.error("黄金集模式必须在 Stage 4 划分后执行，强制读取 train.csv")
        sys.exit(1)
    df = pd.read_csv(train_csv_path)
    if "audio_id" not in df.columns:
        logger.error(f"train.csv 缺少 audio_id 字段")
        sys.exit(1)
    train_ids = set(df["audio_id"].tolist())
    logger.info(f"加载 train.csv: {len(train_ids)} 首（黄金集候选仅在此范围内筛选）")
    return train_ids


def get_existing_ids(manifest_df: pd.DataFrame, respect_existing: bool = True) -> Set[str]:
    """
    获取已有标记的样本 ID（golden_seed + challenge_stress_test）
    --respect-existing 默认开启，这些样本不进入候选池
    """
    if not respect_existing:
        return set()
    existing = manifest_df[
        manifest_df["sample_type"].isin([GOLDEN_SAMPLE_TYPE, CHALLENGE_SAMPLE_TYPE])
    ]["audio_id"].tolist()
    logger.info(f"已有标记样本（排除出候选池）: {len(existing)} 首 "
                f"({GOLDEN_SAMPLE_TYPE} + {CHALLENGE_SAMPLE_TYPE})")
    return set(existing)


def build_candidate_record(
    row: pd.Series,
    candidate_type: str,
    reason: str,
    cluster_id: Optional[int] = None,
    center_dist: Optional[float] = None,
) -> Dict:
    """构建候选池记录"""
    record = {
        "audio_id": row["audio_id"],
        "candidate_type": candidate_type,  # golden / challenge
        "reason": reason,                    # 入选原因
        "duration_sec": row.get("duration_sec", None),
        "artist_id": row.get("artist_id", None),
        "source_type": row.get("source_type", None),
        "qc_branch": row.get("final_branch", row.get("qc_branch", None)),
        "cluster_id": cluster_id if cluster_id is not None else row.get("cluster_id", None),
        "cluster_center_dist": center_dist if center_dist is not None else row.get("cluster_center_dist", None),
        "file_relative_path": row.get("file_relative_path", None),
        "hitl_status": "pending",           # pending / approved / rejected / uncertain
        "hitl_note": "",
    }
    return record


def save_candidates(
    candidates: List[Dict],
    output_path: str,
    mode: str,
    metadata: Dict,
):
    """保存候选池 JSON"""
    output = {
        "candidate_version": datetime.now().strftime("v%Y%m%d_%H%M%S"),
        "mode": mode,
        "generated_at": datetime.now().isoformat(),
        "metadata": metadata,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"候选池已保存: {output_path} ({len(candidates)} 首)")


def generate_label_studio_import(candidates: List[Dict], output_path: str):
    """
    生成 Label Studio 导入文件（JSON 格式）
    供 HITL 确认环节使用
    """
    ls_tasks = []
    for c in candidates:
        task = {
            "data": {
                "audio_id": c["audio_id"],
                "audio": c.get("file_relative_path", ""),
                "candidate_type": c["candidate_type"],
                "reason": c["reason"],
                "duration_sec": c["duration_sec"],
                "artist_id": c["artist_id"],
                "cluster_id": c["cluster_id"],
                "cluster_center_dist": c["cluster_center_dist"],
            }
        }
        ls_tasks.append(task)

    ls_path = output_path.replace(".json", "_labelstudio.json")
    with open(ls_path, "w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, ensure_ascii=False, indent=2)
    logger.info(f"Label Studio 导入文件已生成: {ls_path} ({len(ls_tasks)} 个任务)")


# ==================== 黄金集候选采样 ====================
def sample_golden_candidates(
    manifest_df: pd.DataFrame,
    train_ids: Set[str],
    existing_ids: Set[str],
    per_cluster: int = 2,
    qc_report_df: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    """
    黄金集候选池筛选（仅在 train 候选池内）

    硬性门槛：
    1. 只保留 train_ids 中的样本（ADR-003/004 硬约束）
    2. QC pass（从 QC 报告读取，manifest 中不存 QC 结果）
    3. 时长 120-600s
    4. 非 AI 生成（source_type 不含 ace_studio）
    5. artist 已知（非 unknown_）
    6. 按 cluster 分层，每簇选距中心点最近的 per_cluster 首

    Args:
        manifest_df: 完整 manifest
        train_ids: train.csv 中的 audio_id 集合（强制约束）
        existing_ids: 已有标记样本（排除）
        per_cluster: 每簇候选数
        qc_report_df: QC 报告（可选，用于 QC pass 过滤）

    Returns:
        候选池记录列表
    """
    logger.info("=" * 60)
    logger.info("黄金集候选采样（仅在 train 候选池内）")
    logger.info("=" * 60)

    # 硬性门槛 1：只保留 train_ids 中的样本
    df = manifest_df[manifest_df["audio_id"].isin(train_ids)].copy()
    logger.info(f"  步骤1: 限制在 train 候选池内 → {len(df)} 首")

    # 排除已有标记
    df = df[~df["audio_id"].isin(existing_ids)]
    logger.info(f"  步骤2: 排除已有标记 → {len(df)} 首")

    # 硬性门槛 2：QC pass（从 QC 报告读取，manifest 中不存 QC 结果）
    if qc_report_df is not None:
        qc_col = "final_branch" if "final_branch" in qc_report_df.columns else "qc_branch"
        if qc_col in qc_report_df.columns:
            pass_ids = set(qc_report_df[qc_report_df[qc_col] == "pass"]["audio_id"].tolist())
            df = df[df["audio_id"].isin(pass_ids)]
            logger.info(f"  步骤3: QC pass（从QC报告）→ {len(df)} 首")
        else:
            logger.warning(f"  步骤3: QC 报告缺少 {qc_col} 字段，跳过 QC 过滤")
    else:
        # 尝试从 manifest 读取（兼容旧数据）
        qc_col = "final_branch" if "final_branch" in df.columns else "qc_branch"
        if qc_col in df.columns:
            df = df[df[qc_col] == "pass"]
            logger.info(f"  步骤3: QC pass（从manifest）→ {len(df)} 首")
        else:
            logger.warning(f"  步骤3: 未提供 --qc-report 且 manifest 无 QC 字段，跳过 QC 过滤")

    # 硬性门槛 3：时长 120-600s
    df = df[df["duration_sec"].between(GOLDEN_MIN_DURATION, GOLDEN_MAX_DURATION)]
    logger.info(f"  步骤4: 时长 {GOLDEN_MIN_DURATION}-{GOLDEN_MAX_DURATION}s → {len(df)} 首")

    # 硬性门槛 4：非 AI 生成
    if "source_type" in df.columns:
        df = df[~df["source_type"].astype(str).str.contains("ace_studio", na=False)]
        logger.info(f"  步骤5: 非 AI 生成 → {len(df)} 首")

    # 硬性门槛 5：artist 已知
    if "artist_id" in df.columns:
        df = df[~df["artist_id"].astype(str).str.startswith("unknown_", na=False)]
        logger.info(f"  步骤6: artist 已知 → {len(df)} 首")

    # 步骤 6：按 cluster 分层，每簇选距中心点最近的 per_cluster 首
    candidates = []
    if "cluster_id" not in df.columns:
        logger.warning("  manifest 缺少 cluster_id 字段，无法按簇分层，返回全部候选")
        for _, row in df.iterrows():
            candidates.append(build_candidate_record(row, "golden", "no_cluster_field"))
        return candidates

    for cluster_id, group in df.groupby("cluster_id"):
        # 按距中心点距离排序（越小越靠近中心）
        if "cluster_center_dist" in group.columns:
            group = group.sort_values("cluster_center_dist")
        else:
            logger.warning(f"  cluster {cluster_id} 缺少 cluster_center_dist，按 duration 排序")
            group = group.sort_values("duration_sec")

        top_n = group.head(per_cluster)
        for _, row in top_n.iterrows():
            candidates.append(build_candidate_record(
                row,
                candidate_type="golden",
                reason=f"cluster_{cluster_id}_center_top{per_cluster}",
                cluster_id=cluster_id,
                center_dist=row.get("cluster_center_dist", None),
            ))
        logger.info(f"  cluster {cluster_id}: {len(group)} 首候选 → 选 {len(top_n)} 首")

    logger.info(f"黄金集候选池总计: {len(candidates)} 首")
    return candidates


# ==================== Challenge 集候选采样 ====================
def sample_challenge_candidates(
    manifest_df: pd.DataFrame,
    qc_report_df: Optional[pd.DataFrame],
    existing_ids: Set[str],
    sources: List[str],
) -> List[Dict]:
    """
    Challenge 集候选池筛选

    来源（85 首阶段禁用 hdbscan_outlier）：
    - qc_marginal: QC marginal 样本（低信噪比、响度异常、静音偏多）
    - short: 超短（5-60s，排除 fail 样本）
    - long: 超长（>600s，非 dj_mix）
    - hdbscan_outlier: HDBSCAN outlier（85 首禁用，500 首启用）

    Args:
        manifest_df: 完整 manifest
        qc_report_df: QC 报告（qc_marginal 来源需要）
        existing_ids: 已有标记样本（排除）
        sources: 启用的来源列表

    Returns:
        候选池记录列表
    """
    logger.info("=" * 60)
    logger.info("Challenge 集候选采样")
    logger.info("=" * 60)

    # 校验：85 首阶段禁用 hdbscan_outlier
    effective_sources = []
    for s in sources:
        if s in DISABLED_SOURCES_85:
            logger.warning(f"  ⚠️ 来源 '{s}' 在 85 首试点阶段禁用（HDBSCAN 小样本失效），已跳过")
            logger.warning(f"     500 首全量阶段重新评估后可启用")
            continue
        effective_sources.append(s)

    if not effective_sources:
        logger.error("  所有来源均被禁用，无法生成候选池")
        sys.exit(1)

    logger.info(f"  启用来源: {effective_sources}")

    # 排除已有标记
    df = manifest_df[~manifest_df["audio_id"].isin(existing_ids)].copy()
    logger.info(f"  排除已有标记后: {len(df)} 首")

    candidates = []
    candidate_ids = set()  # 去重

    # 来源 1：qc_marginal
    if "qc_marginal" in effective_sources:
        if qc_report_df is None:
            logger.warning("  qc_marginal 来源需要 --qc-report，已跳过")
        else:
            qc_col = "final_branch" if "final_branch" in qc_report_df.columns else "qc_branch"
            if qc_col in qc_report_df.columns:
                qc_values = set(qc_report_df[qc_col].unique())
                if "marginal" not in qc_values:
                    logger.warning(f"  ⚠️ QC 报告中无 'marginal' 分类（当前值: {qc_values}）")
                    logger.warning(f"     当前 QC 流程可能只有 pass/fail，marginal 来源返回 0 首")
                    logger.warning(f"     如需 marginal 分类，请调整 QC Gate 阈值")
                marginal_ids = set(qc_report_df[qc_report_df[qc_col] == "marginal"]["audio_id"].tolist())
                marginal_df = df[df["audio_id"].isin(marginal_ids)]
                for _, row in marginal_df.iterrows():
                    if row["audio_id"] not in candidate_ids:
                        candidates.append(build_candidate_record(row, "challenge", "qc_marginal"))
                        candidate_ids.add(row["audio_id"])
                logger.info(f"  qc_marginal: {len(marginal_df)} 首")
            else:
                logger.warning(f"  QC 报告缺少 {qc_col} 字段，跳过 qc_marginal")

    # 来源 2：short（超短 5-60s）
    if "short" in effective_sources:
        short_df = df[df["duration_sec"].between(CHALLENGE_SHORT_MIN, CHALLENGE_SHORT_MAX)]
        for _, row in short_df.iterrows():
            if row["audio_id"] not in candidate_ids:
                candidates.append(build_candidate_record(row, "challenge", "too_short"))
                candidate_ids.add(row["audio_id"])
        logger.info(f"  too_short ({CHALLENGE_SHORT_MIN}-{CHALLENGE_SHORT_MAX}s): {len(short_df)} 首")

    # 来源 3：long（超长 >600s，非 dj_mix）
    if "long" in effective_sources:
        long_df = df[df["duration_sec"] > CHALLENGE_LONG_MIN]
        if "is_dj_mix" in long_df.columns:
            long_df = long_df[~long_df["is_dj_mix"].astype(bool)]
        for _, row in long_df.iterrows():
            if row["audio_id"] not in candidate_ids:
                candidates.append(build_candidate_record(row, "challenge", "too_long"))
                candidate_ids.add(row["audio_id"])
        logger.info(f"  too_long (>{CHALLENGE_LONG_MIN}s): {len(long_df)} 首")

    # 来源 4：hdbscan_outlier（85 首禁用，此处仅为 500 首预留）
    if "hdbscan_outlier" in effective_sources:
        if "cluster_id" in df.columns:
            outlier_df = df[df["cluster_id"] == -1]
            for _, row in outlier_df.iterrows():
                if row["audio_id"] not in candidate_ids:
                    candidates.append(build_candidate_record(row, "challenge", "hdbscan_outlier"))
                    candidate_ids.add(row["audio_id"])
            logger.info(f"  hdbscan_outlier: {len(outlier_df)} 首")
        else:
            logger.warning("  manifest 缺少 cluster_id 字段，跳过 hdbscan_outlier")

    logger.info(f"Challenge 候选池总计: {len(candidates)} 首（去重后）")
    return candidates


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description="黄金集 / Challenge 集候选采样脚本（自动候选 + HITL 确认）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 模式
    parser.add_argument("--mode", type=str, required=True, choices=["golden", "challenge"],
                        help="采样模式：golden（黄金集候选）/ challenge（挑战集候选）")

    # 输入
    parser.add_argument("--manifest", type=str, required=True,
                        help="manifest CSV 路径")
    parser.add_argument("--train-csv", type=str, default=None,
                        help="train.csv 路径（golden 模式必填，强制约束）")
    parser.add_argument("--qc-report", type=str, default=None,
                        help="QC 报告 CSV 路径（challenge 模式 qc_marginal 来源需要）")

    # 黄金集参数
    parser.add_argument("--per-cluster", type=int, default=2,
                        help="每簇候选数（golden 模式，默认 2）")

    # Challenge 参数
    parser.add_argument("--challenge-sources", type=str, default="qc_marginal,short,long",
                        help="Challenge 候选来源（逗号分隔），85 首禁用 hdbscan_outlier")

    # 通用参数
    parser.add_argument("--respect-existing", action="store_true", default=True,
                        help="排除已有 golden_seed / challenge_stress_test 标记（默认开启）")
    parser.add_argument("--no-respect-existing", action="store_true",
                        help="不排除已有标记（覆盖 --respect-existing）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览候选池，不写文件")
    parser.add_argument("--output", type=str, default=None,
                        help="候选池 JSON 输出路径")

    args = parser.parse_args()

    # 处理 --no-respect-existing
    if args.no_respect_existing:
        args.respect_existing = False

    logger.info("=" * 60)
    logger.info(f"sample_candidates.py 启动")
    logger.info(f"模式: {args.mode}")
    logger.info(f"respect-existing: {args.respect_existing}")
    logger.info(f"dry-run: {args.dry_run}")
    logger.info("=" * 60)

    # 加载 manifest
    manifest_df = load_manifest(args.manifest)

    # 获取已有标记
    existing_ids = get_existing_ids(manifest_df, args.respect_existing)

    # 根据模式执行
    if args.mode == "golden":
        # 黄金集模式：强制读取 train.csv
        if not args.train_csv:
            logger.error("❌ golden 模式必须指定 --train-csv")
            logger.error("   黄金集候选必须在 Stage 4 划分后，从 train.csv 内筛选（ADR-003/004）")
            sys.exit(1)

        train_ids = load_train_ids(args.train_csv)

        # 加载 QC 报告（黄金集 QC pass 过滤需要）
        qc_report_df = None
        if args.qc_report:
            qc_report_df = pd.read_csv(args.qc_report)
            logger.info(f"加载 QC 报告: {len(qc_report_df)} 首")
        else:
            logger.warning("golden 模式建议指定 --qc-report 以进行 QC pass 过滤")

        candidates = sample_golden_candidates(
            manifest_df=manifest_df,
            train_ids=train_ids,
            existing_ids=existing_ids,
            per_cluster=args.per_cluster,
            qc_report_df=qc_report_df,
        )

        metadata = {
            "mode": "golden",
            "per_cluster": args.per_cluster,
            "train_count": len(train_ids),
            "existing_excluded": len(existing_ids),
            "constraints": [
                "仅在 train 候选池内筛选（ADR-003/004）",
                f"QC pass",
                f"时长 {GOLDEN_MIN_DURATION}-{GOLDEN_MAX_DURATION}s",
                "非 AI 生成",
                "artist 已知",
                f"每簇选距中心最近的 {args.per_cluster} 首",
            ],
        }

    elif args.mode == "challenge":
        # Challenge 模式
        sources = [s.strip() for s in args.challenge_sources.split(",") if s.strip()]

        # 加载 QC 报告（如果需要）
        qc_report_df = None
        if "qc_marginal" in sources:
            if not args.qc_report:
                logger.warning("challenge-sources 包含 qc_marginal，但未指定 --qc-report，该来源将被跳过")
            else:
                qc_report_df = pd.read_csv(args.qc_report)
                logger.info(f"加载 QC 报告: {len(qc_report_df)} 首")

        candidates = sample_challenge_candidates(
            manifest_df=manifest_df,
            qc_report_df=qc_report_df,
            existing_ids=existing_ids,
            sources=sources,
        )

        metadata = {
            "mode": "challenge",
            "sources_requested": sources,
            "sources_disabled_85": list(DISABLED_SOURCES_85),
            "existing_excluded": len(existing_ids),
            "constraints": [
                "85 首试点阶段禁用 hdbscan_outlier（HDBSCAN 小样本失效）",
                "qc_marginal: QC marginal 样本",
                f"short: {CHALLENGE_SHORT_MIN}-{CHALLENGE_SHORT_MAX}s",
                f"long: >{CHALLENGE_LONG_MIN}s（非 dj_mix）",
            ],
        }

    # 输出
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY-RUN 预览（不写文件）")
        logger.info("=" * 60)
        logger.info(f"候选池大小: {len(candidates)} 首")
        for i, c in enumerate(candidates[:20]):  # 最多显示 20 首
            logger.info(f"  [{i+1}] {c['audio_id']} | {c['reason']} | "
                        f"duration={c['duration_sec']}s | cluster={c['cluster_id']}")
        if len(candidates) > 20:
            logger.info(f"  ... 还有 {len(candidates) - 20} 首")
    else:
        if not args.output:
            # 默认输出路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.mode == "golden":
                args.output = f"data/03_human_annotation/golden_set/candidates_{timestamp}.json"
            else:
                args.output = f"data/03_human_annotation/challenge_set/candidates_{timestamp}.json"

        save_candidates(candidates, args.output, args.mode, metadata)
        generate_label_studio_import(candidates, args.output)

    logger.info("=" * 60)
    logger.info("完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
