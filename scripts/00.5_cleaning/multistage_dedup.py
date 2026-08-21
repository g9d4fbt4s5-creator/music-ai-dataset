"""
multistage_dedup.py
多级去重流水线（Stage 4）

四个阶段：
1. 精确去重（Stage 4.1）：SHA-256哈希比对，完全相同的文件
2. 近似去重（Stage 4.2）：chroma特征+余弦相似度>0.9，同一首歌不同版本
3. 片段级去重（Stage 4.3）：滑动窗口切片+逐段指纹，同一首歌的不同节选
4. 跨集去重（Stage 4.4）：全局指纹库+子集标记，训练/验证/测试集泄露

用法：
    # 全部4阶段
    python multistage_dedup.py --all

    # 只跑精确去重
    python multistage_dedup.py --stage exact

    # 跑精确+近似
    python multistage_dedup.py --stage exact,approximate

    # 自定义相似度阈值
    python multistage_dedup.py --stage approximate --threshold 0.85

    # 片段级去重（10秒窗口，50%重叠）
    python multistage_dedup.py --stage segment --chunk-sec 10 --overlap 0.5

    # 跨集去重（指定子集列）
    python multistage_dedup.py --stage cross-set --split-col dataset_split

    # 预览模式
    python multistage_dedup.py --all --dry-run
"""
import os
import sys
import hashlib
import logging
import argparse
import pandas as pd
import numpy as np
import librosa
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from get_audio_physical_path import get_audio_absolute_path

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"multistage_dedup_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认路径
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"
DEFAULT_CHECKSUM = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio_checksums.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "00.5_cleaned" / "dedup_results"


def get_audio_path_from_row(row: pd.Series) -> Optional[str]:
    """从 manifest 行获取音频的绝对路径"""
    # 优先从 file_relative_path 读取完整路径
    rel_path = row.get("file_relative_path", "")
    if rel_path and isinstance(rel_path, str):
        abs_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path
        if abs_path.exists():
            return str(abs_path)

    # 回退：用 audio_id + format 构建路径
    audio_id = row.get("audio_id", "")
    fmt = row.get("format", "")
    if audio_id and fmt:
        try:
            abs_path = get_audio_absolute_path(audio_id, extension=fmt)
            if abs_path.exists():
                return str(abs_path)
        except Exception:
            pass

    return None
def exact_dedup(
    checksum_path: Path,
    manifest_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    精确去重：基于 SHA-256 哈希比对

    相同哈希的文件判定为完全重复，保留文件体积最大或元数据最完整的版本。

    Args:
        checksum_path: raw_audio_checksums.csv 路径
        manifest_path: audio_manifest.csv 路径（可选，用于元数据比较）
        dry_run: 预览模式，不输出结果

    Returns:
        (duplicates_df, keep_df): 重复对表 + 保留表
    """
    logger.info("=" * 60)
    logger.info("Stage 4.1: 精确去重（SHA-256哈希比对）")
    logger.info("=" * 60)

    if not checksum_path.exists():
        logger.error(f"checksum文件不存在: {checksum_path}")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(checksum_path)
    logger.info(f"加载 {len(df)} 条记录")

    # 按 sha256 分组
    hash_groups = df.groupby("sha256")
    duplicate_groups = {h: group for h, group in hash_groups if len(group) > 1}

    logger.info(f"发现 {len(duplicate_groups)} 组重复哈希")

    duplicates = []
    keep_records = []

    for sha256, group in duplicate_groups.items():
        audio_ids = group["audio_id"].tolist()
        logger.info(f"  哈希 {sha256[:16]}...: {len(audio_ids)} 个重复文件")

        # 决定保留哪个：优先文件体积最大，其次元数据最完整
        if "file_bytes" in group.columns:
            keep_idx = group["file_bytes"].idxmax()
        else:
            keep_idx = group.index[0]

        keep_id = group.loc[keep_idx, "audio_id"]
        keep_records.append({
            "sha256": sha256,
            "keep_audio_id": keep_id,
            "duplicate_count": len(audio_ids),
            "duplicate_audio_ids": ",".join(audio_ids),
        })

        # 记录所有重复对
        for i, aid1 in enumerate(audio_ids):
            for aid2 in audio_ids[i+1:]:
                duplicates.append({
                    "sha256": sha256,
                    "audio_id_1": aid1,
                    "audio_id_2": aid2,
                    "dedup_type": "exact",
                    "similarity": 1.0,
                    "keep": keep_id,
                    "remove": aid1 if aid1 != keep_id else aid2,
                })

    duplicates_df = pd.DataFrame(duplicates)
    keep_df = pd.DataFrame(keep_records)

    if not duplicates_df.empty:
        logger.info(f"精确去重结果: {len(duplicates_df)} 对重复，涉及 {len(keep_df)} 组")
    else:
        logger.info("精确去重结果: 未发现重复文件")

    if not dry_run:
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        duplicates_df.to_csv(output_dir / "exact_duplicates.csv", index=False)
        keep_df.to_csv(output_dir / "exact_keep_mapping.csv", index=False)
        logger.info(f"结果已保存到: {output_dir}")

    return duplicates_df, keep_df


# ===================== Stage 4.2: 近似去重 =====================
def extract_chroma_features(audio_path: str, sr: int = 22050, n_chroma: int = 12) -> Optional[np.ndarray]:
    """提取 chroma 特征（12维色度特征，取时间平均）"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        # 取前30秒，避免长音频计算过慢
        max_samples = 30 * sr
        if len(y) > max_samples:
            y = y[:max_samples]
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        # 时间平均，得到固定长度的特征向量
        chroma_avg = np.mean(chroma, axis=1)
        return chroma_avg
    except Exception as e:
        logger.warning(f"  提取特征失败 {audio_path}: {e}")
        return None


def approximate_dedup(
    manifest_path: Path,
    threshold: float = 0.9,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    近似去重：chroma特征 + 余弦相似度

    相似度 > threshold 判定为重复，保留音质评分最高的版本。

    Args:
        manifest_path: audio_manifest.csv 路径
        threshold: 相似度阈值，默认0.9
        limit: 只处理前N个
        dry_run: 预览模式

    Returns:
        (duplicates_df, keep_df)
    """
    logger.info("=" * 60)
    logger.info(f"Stage 4.2: 近似去重（chroma特征+余弦相似度，阈值={threshold}）")
    logger.info("=" * 60)

    if not manifest_path.exists():
        logger.error(f"manifest文件不存在: {manifest_path}")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit)
    logger.info(f"加载 {len(df)} 条记录")

    # 提取特征
    features = {}
    for idx, row in df.iterrows():
        audio_id = row["audio_id"]
        audio_path = get_audio_path_from_row(row)

        if not audio_path or not os.path.exists(audio_path):
            logger.warning(f"  文件不存在: {audio_id}")
            continue

        logger.info(f"  [{idx+1}/{len(df)}] 提取特征: {audio_id}")
        feat = extract_chroma_features(audio_path)
        if feat is not None:
            features[audio_id] = feat

    logger.info(f"成功提取 {len(features)} 个特征")

    if len(features) < 2:
        logger.info("特征数量不足，跳过近似去重")
        return pd.DataFrame(), pd.DataFrame()

    # 计算相似度矩阵
    audio_ids = list(features.keys())
    feat_matrix = np.array([features[aid] for aid in audio_ids])
    sim_matrix = cosine_similarity(feat_matrix)

    # 查找重复对
    duplicates = []
    dup_groups = defaultdict(list)

    for i in range(len(audio_ids)):
        for j in range(i+1, len(audio_ids)):
            sim = sim_matrix[i][j]
            if sim >= threshold:
                aid1, aid2 = audio_ids[i], audio_ids[j]
                duplicates.append({
                    "audio_id_1": aid1,
                    "audio_id_2": aid2,
                    "dedup_type": "approximate",
                    "similarity": round(float(sim), 4),
                })
                dup_groups[aid1].append(aid2)
                dup_groups[aid2].append(aid1)

    duplicates_df = pd.DataFrame(duplicates)

    # 决定保留哪个（简单策略：保留文件体积大的）
    keep_records = []
    processed = set()
    for aid in audio_ids:
        if aid in processed:
            continue
        if aid not in dup_groups:
            continue
        group = [aid] + dup_groups[aid]
        processed.update(group)

        # 找文件体积最大的
        group_df = df[df["audio_id"].isin(group)]
        if "file_bytes" in group_df.columns:
            keep_id = group_df.loc[group_df["file_bytes"].idxmax(), "audio_id"]
        else:
            keep_id = group[0]

        keep_records.append({
            "keep_audio_id": keep_id,
            "duplicate_count": len(group),
            "duplicate_audio_ids": ",".join(group),
            "max_similarity": round(float(max(
                sim_matrix[audio_ids.index(keep_id)][audio_ids.index(other)]
                for other in group if other != keep_id
            )), 4) if len(group) > 1 else 1.0,
        })

    keep_df = pd.DataFrame(keep_records)

    if not duplicates_df.empty:
        logger.info(f"近似去重结果: {len(duplicates_df)} 对重复，涉及 {len(keep_df)} 组")
    else:
        logger.info("近似去重结果: 未发现近似重复")

    if not dry_run:
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        duplicates_df.to_csv(output_dir / "approximate_duplicates.csv", index=False)
        keep_df.to_csv(output_dir / "approximate_keep_mapping.csv", index=False)
        logger.info(f"结果已保存到: {output_dir}")

    return duplicates_df, keep_df


# ===================== Stage 4.3: 片段级去重 =====================
def segment_dedup(
    manifest_path: Path,
    chunk_sec: int = 10,
    overlap: float = 0.5,
    threshold: float = 0.92,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    片段级去重：滑动窗口切片 + 逐段指纹比对

    将长音频按 chunk_sec 秒滑动窗口切片（overlap 重叠），
    逐段提取 chroma 指纹，比对片段间的相似度。

    目的：避免同一首歌的不同节选片段在数据集中重复出现。

    Args:
        manifest_path: audio_manifest.csv 路径
        chunk_sec: 切片长度（秒），默认10秒
        overlap: 重叠比例，默认0.5（50%）
        threshold: 相似度阈值，默认0.92
        limit: 只处理前N个
        dry_run: 预览模式

    Returns:
        duplicates_df: 片段重复对表
    """
    logger.info("=" * 60)
    logger.info(f"Stage 4.3: 片段级去重（{chunk_sec}s窗口，{overlap*100:.0f}%重叠，阈值={threshold}）")
    logger.info("=" * 60)

    if not manifest_path.exists():
        logger.error(f"manifest文件不存在: {manifest_path}")
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    if limit:
        df = df.head(limit)
    logger.info(f"加载 {len(df)} 条记录")

    # 提取所有片段的特征
    segment_features = {}  # key: (audio_id, segment_idx), value: chroma feature

    for idx, row in df.iterrows():
        audio_id = row["audio_id"]
        audio_path = get_audio_path_from_row(row)

        if not audio_path or not os.path.exists(audio_path):
            continue

        logger.info(f"  [{idx+1}/{len(df)}] 切片提取: {audio_id}")

        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            duration = len(y) / sr

            # 计算切片位置
            hop = int(chunk_sec * sr * (1 - overlap))
            chunk_samples = int(chunk_sec * sr)
            n_segments = max(1, (len(y) - chunk_samples) // hop + 1)

            for seg_idx in range(n_segments):
                start = seg_idx * hop
                end = min(start + chunk_samples, len(y))
                segment_y = y[start:end]

                if len(segment_y) < sr:  # 跳过太短的片段
                    continue

                chroma = librosa.feature.chroma_stft(y=segment_y, sr=sr, n_chroma=12)
                chroma_avg = np.mean(chroma, axis=1)
                segment_features[(audio_id, seg_idx)] = chroma_avg

        except Exception as e:
            logger.warning(f"  处理失败 {audio_id}: {e}")
            continue

    logger.info(f"成功提取 {len(segment_features)} 个片段特征")

    if len(segment_features) < 2:
        logger.info("片段特征数量不足，跳过片段级去重")
        return pd.DataFrame()

    # 计算相似度矩阵（只比对不同音频的片段）
    seg_keys = list(segment_features.keys())
    feat_matrix = np.array([segment_features[k] for k in seg_keys])
    sim_matrix = cosine_similarity(feat_matrix)

    duplicates = []
    for i in range(len(seg_keys)):
        for j in range(i+1, len(seg_keys)):
            aid1, seg1 = seg_keys[i]
            aid2, seg2 = seg_keys[j]

            # 只比对不同音频的片段（同一音频内部的片段重复是正常的）
            if aid1 == aid2:
                continue

            sim = sim_matrix[i][j]
            if sim >= threshold:
                duplicates.append({
                    "audio_id_1": aid1,
                    "segment_idx_1": seg1,
                    "audio_id_2": aid2,
                    "segment_idx_2": seg2,
                    "dedup_type": "segment",
                    "similarity": round(float(sim), 4),
                })

    duplicates_df = pd.DataFrame(duplicates)

    if not duplicates_df.empty:
        # 按音频对聚合
        audio_pairs = duplicates_df.groupby(["audio_id_1", "audio_id_2"]).size().reset_index(name="matching_segments")
        logger.info(f"片段级去重结果: {len(duplicates_df)} 对片段重复，涉及 {len(audio_pairs)} 对音频")
        logger.info(f"音频对详情:\n{audio_pairs.to_string(index=False)}")
    else:
        logger.info("片段级去重结果: 未发现片段重复")

    if not dry_run:
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        duplicates_df.to_csv(output_dir / "segment_duplicates.csv", index=False)
        logger.info(f"结果已保存到: {output_dir}")

    return duplicates_df


# ===================== Stage 4.4: 跨集去重 =====================
def cross_set_dedup(
    manifest_path: Path,
    split_col: str = "dataset_split",
    threshold: float = 0.9,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    跨集去重与泄露防控

    确保同一作品（含不同版本、remix、翻唱）只出现在单一数据子集中。
    严格执行训练/验证/测试集跨集去重。

    方法：
    1. 读取每个音频的子集标签（train/val/test/holdout）
    2. 计算所有音频间的 chroma 相似度
    3. 查找跨子集的高相似度对（泄露）
    4. 输出泄露报告，建议保留在优先级最高的子集中

    Args:
        manifest_path: audio_manifest.csv 路径
        split_col: 子集列名，默认 dataset_split
        threshold: 相似度阈值，默认0.9
        dry_run: 预览模式

    Returns:
        leakage_df: 跨集泄露对表
    """
    logger.info("=" * 60)
    logger.info(f"Stage 4.4: 跨集去重与泄露防控（阈值={threshold}）")
    logger.info("=" * 60)

    if not manifest_path.exists():
        logger.error(f"manifest文件不存在: {manifest_path}")
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)

    if split_col not in df.columns:
        logger.warning(f"未找到子集列 '{split_col}'，当前列: {list(df.columns)}")
        logger.info("跳过跨集去重（需要先划分数据集）")
        return pd.DataFrame()

    # 过滤出有子集标签的
    df = df[df[split_col].notna() & (df[split_col] != "")]
    logger.info(f"加载 {len(df)} 条有子集标签的记录")
    logger.info(f"子集分布:\n{df[split_col].value_counts()}")

    if len(df) < 2:
        logger.info("样本数量不足，跳过跨集去重")
        return pd.DataFrame()

    # 提取特征
    features = {}
    splits = {}
    for idx, row in df.iterrows():
        audio_id = row["audio_id"]
        audio_path = get_audio_path_from_row(row)

        if not audio_path or not os.path.exists(audio_path):
            continue

        feat = extract_chroma_features(audio_path)
        if feat is not None:
            features[audio_id] = feat
            splits[audio_id] = row[split_col]

    logger.info(f"成功提取 {len(features)} 个特征")

    if len(features) < 2:
        logger.info("特征数量不足，跳过跨集去重")
        return pd.DataFrame()

    # 计算相似度矩阵
    audio_ids = list(features.keys())
    feat_matrix = np.array([features[aid] for aid in audio_ids])
    sim_matrix = cosine_similarity(feat_matrix)

    # 子集优先级（泄露时保留在优先级高的子集中）
    split_priority = {"holdout": 4, "test": 3, "val": 2, "train": 1}

    # 查找跨子集泄露
    leakage = []
    for i in range(len(audio_ids)):
        for j in range(i+1, len(audio_ids)):
            aid1, aid2 = audio_ids[i], audio_ids[j]
            split1, split2 = splits[aid1], splits[aid2]

            # 只报告跨子集的
            if split1 == split2:
                continue

            sim = sim_matrix[i][j]
            if sim >= threshold:
                # 决定保留在哪个子集（优先级高的保留）
                keep_in_split = split1 if split_priority.get(split1, 0) >= split_priority.get(split2, 0) else split2
                remove_from_split = split2 if keep_in_split == split1 else split1

                leakage.append({
                    "audio_id_1": aid1,
                    "split_1": split1,
                    "audio_id_2": aid2,
                    "split_2": split2,
                    "dedup_type": "cross_set_leakage",
                    "similarity": round(float(sim), 4),
                    "recommend_keep_in": keep_in_split,
                    "recommend_remove_from": remove_from_split,
                })

    leakage_df = pd.DataFrame(leakage)

    if not leakage_df.empty:
        logger.info(f"跨集泄露结果: {len(leakage_df)} 对跨子集泄露")
        logger.info(f"泄露详情:\n{leakage_df[['split_1', 'split_2', 'similarity', 'recommend_keep_in']].to_string(index=False)}")
    else:
        logger.info("跨集泄露结果: 未发现跨子集泄露")

    if not dry_run:
        output_dir = DEFAULT_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        leakage_df.to_csv(output_dir / "cross_set_leakage.csv", index=False)
        logger.info(f"结果已保存到: {output_dir}")

    return leakage_df


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="多级去重流水线（Stage 4）")
    parser.add_argument("--all", action="store_true", help="运行全部4阶段")
    parser.add_argument("--stage", type=str, default=None,
                        help="指定阶段（逗号分隔）：exact,approximate,segment,cross-set")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                        help="audio_manifest.csv 路径")
    parser.add_argument("--checksum", type=str, default=str(DEFAULT_CHECKSUM),
                        help="raw_audio_checksums.csv 路径")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="近似去重相似度阈值，默认0.9")
    parser.add_argument("--chunk-sec", type=int, default=10,
                        help="片段级去重切片长度（秒），默认10")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="片段级去重重叠比例，默认0.5")
    parser.add_argument("--split-col", type=str, default="dataset_split",
                        help="跨集去重的子集列名，默认dataset_split")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前N个（用于测试）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不输出结果文件")
    args = parser.parse_args()

    # 确定要运行的阶段
    stages = set()
    if args.all:
        stages = {"exact", "approximate", "segment", "cross-set"}
    elif args.stage:
        stages = set(s.strip() for s in args.stage.split(","))
    else:
        logger.error("请指定 --all 或 --stage")
        return

    manifest_path = Path(args.manifest)
    checksum_path = Path(args.checksum)

    logger.info("=" * 60)
    logger.info("多级去重流水线（Stage 4）")
    logger.info(f"运行阶段: {', '.join(sorted(stages))}")
    logger.info(f"manifest: {manifest_path}")
    logger.info(f"checksum: {checksum_path}")
    logger.info("=" * 60)

    all_results = {}

    # Stage 4.1: 精确去重
    if "exact" in stages:
        exact_dup, exact_keep = exact_dedup(checksum_path, manifest_path, args.dry_run)
        all_results["exact"] = {"duplicates": exact_dup, "keep": exact_keep}

    # Stage 4.2: 近似去重
    if "approximate" in stages:
        approx_dup, approx_keep = approximate_dedup(
            manifest_path, args.threshold, args.limit, args.dry_run
        )
        all_results["approximate"] = {"duplicates": approx_dup, "keep": approx_keep}

    # Stage 4.3: 片段级去重
    if "segment" in stages:
        seg_dup = segment_dedup(
            manifest_path, args.chunk_sec, args.overlap,
            args.threshold + 0.02, args.limit, args.dry_run
        )
        all_results["segment"] = {"duplicates": seg_dup}

    # Stage 4.4: 跨集去重
    if "cross-set" in stages:
        cross_dup = cross_set_dedup(
            manifest_path, args.split_col, args.threshold, args.dry_run
        )
        all_results["cross-set"] = {"leakage": cross_dup}

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("多级去重汇总")
    logger.info("=" * 60)
    for stage, results in all_results.items():
        if "duplicates" in results:
            n = len(results["duplicates"])
            logger.info(f"  {stage}: {n} 对重复")
        if "leakage" in results:
            n = len(results["leakage"])
            logger.info(f"  {stage}: {n} 对跨集泄露")

    logger.info("")
    logger.info(f"结果目录: {DEFAULT_OUTPUT_DIR}")
    logger.info("完成！")


if __name__ == "__main__":
    main()
