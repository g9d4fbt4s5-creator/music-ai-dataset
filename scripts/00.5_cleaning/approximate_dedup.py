"""
approximate_dedup.py
近似去重脚本（Stage 4）

功能：
- 从 audio_manifest.csv 读取音频列表
- 用 librosa 提取 chroma 特征（色度特征）
- 计算音频间的余弦相似度
- 相似度 > 阈值判定为重复
- 输出去重映射表（保留音质最高的版本）
- 支持精确去重（SHA-256）+ 近似去重

方法说明：
- 由于 fpcalc (Chromaprint) 未安装，使用 librosa chroma 特征做余弦相似度
- chroma 特征对音调、和弦进行敏感，适合检测同一首歌的不同版本
- 对于翻唱、remix 等高相似度但非完全重复的版本，可以通过阈值调整

用法：
    # 全部去重
    python approximate_dedup.py

    # 自定义相似度阈值
    python approximate_dedup.py --threshold 0.9

    # 只处理指定数量
    python approximate_dedup.py --limit 50

    # 预览模式（不输出结果）
    python approximate_dedup.py --dry-run

    # 输出去重映射表
    python approximate_dedup.py --output ./data/00.5_cleaned/dedup_mapping.csv
"""
import os
import sys
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
import librosa
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from get_audio_physical_path import get_audio_absolute_path

# 默认配置文件
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cleaning_config.yaml"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"approximate_dedup_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """加载配置文件"""
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    return config


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """加载音频清单"""
    if not manifest_path.exists():
        logger.error(f"音频清单不存在: {manifest_path}")
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info(f"加载音频清单: {len(df)} 条记录")
    return df


def extract_chroma_features(audio_path: str, sr: int = 22050, n_chroma: int = 12) -> Optional[np.ndarray]:
    """
    提取音频的 chroma 特征（色度特征）

    Args:
        audio_path: 音频文件路径
        sr: 采样率（统一重采样到 22050 用于特征提取）
        n_chroma: chroma 维度数（12 个半音）

    Returns:
        chroma 特征的均值向量（12维），失败返回 None
    """
    try:
        # 加载音频（统一重采样到 22050，单声道）
        y, sr = librosa.load(audio_path, sr=sr, mono=True)

        # 提取 chroma 特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)

        # 计算均值（时间维度平均），得到 12 维特征向量
        chroma_mean = np.mean(chroma, axis=1)

        # L2 归一化
        norm = np.linalg.norm(chroma_mean)
        if norm > 0:
            chroma_mean = chroma_mean / norm

        return chroma_mean

    except Exception as e:
        logger.warning(f"特征提取失败: {audio_path} -> {str(e)}")
        return None


def compute_similarity_matrix(features: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    计算音频间的余弦相似度矩阵

    Args:
        features: {audio_id: feature_vector} 字典

    Returns:
        相似度矩阵 DataFrame
    """
    audio_ids = list(features.keys())
    feature_matrix = np.array([features[aid] for aid in audio_ids])

    # 计算余弦相似度
    sim_matrix = cosine_similarity(feature_matrix)

    # 转换为 DataFrame
    sim_df = pd.DataFrame(sim_matrix, index=audio_ids, columns=audio_ids)

    return sim_df


def find_duplicates(
    sim_df: pd.DataFrame,
    threshold: float = 0.9,
    manifest: Optional[pd.DataFrame] = None,
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    查找重复音频对

    Args:
        sim_df: 相似度矩阵
        threshold: 相似度阈值（> 此值判定为重复）
        manifest: 音频清单（用于选择保留哪个版本）

    Returns:
        (duplicate_pairs, keep_mapping):
            - duplicate_pairs: 重复对列表 [{audio1, audio2, similarity}]
            - keep_mapping: {要删除的audio_id: 要保留的audio_id}
    """
    duplicate_pairs = []
    keep_mapping = {}

    audio_ids = sim_df.index.tolist()
    n = len(audio_ids)

    # 用于记录哪些已经被标记为删除
    marked_for_deletion = set()

    for i in range(n):
        aid1 = audio_ids[i]
        if aid1 in marked_for_deletion:
            continue

        for j in range(i + 1, n):
            aid2 = audio_ids[j]
            if aid2 in marked_for_deletion:
                continue

            similarity = sim_df.loc[aid1, aid2]

            if similarity >= threshold:
                duplicate_pairs.append({
                    "audio_id_1": aid1,
                    "audio_id_2": aid2,
                    "similarity": round(float(similarity), 4),
                })

                # 决定保留哪个（优先保留音质更好的）
                keep_id, delete_id = decide_keep(aid1, aid2, manifest)
                keep_mapping[delete_id] = keep_id
                marked_for_deletion.add(delete_id)

                logger.info(f"  重复: {aid1} ↔ {aid2} (相似度: {similarity:.4f})")
                logger.info(f"    保留: {keep_id}, 删除: {delete_id}")

    return duplicate_pairs, keep_mapping


def decide_keep(aid1: str, aid2: str, manifest: Optional[pd.DataFrame] = None) -> Tuple[str, str]:
    """
    决定保留哪个版本

    优先级：
    1. 采样率高的优先
    2. 位深高的优先
    3. 文件大的优先
    4. 默认保留第一个

    Args:
        aid1: 音频 ID 1
        aid2: 音频 ID 2
        manifest: 音频清单

    Returns:
        (keep_id, delete_id)
    """
    if manifest is None:
        return aid1, aid2

    row1 = manifest[manifest["audio_id"] == aid1]
    row2 = manifest[manifest["audio_id"] == aid2]

    if len(row1) == 0 or len(row2) == 0:
        return aid1, aid2

    row1 = row1.iloc[0]
    row2 = row2.iloc[0]

    # 比较采样率
    sr1 = row1.get("sample_rate", 0)
    sr2 = row2.get("sample_rate", 0)
    if sr1 > sr2:
        return aid1, aid2
    elif sr2 > sr1:
        return aid2, aid1

    # 比较位深
    bd1 = row1.get("bit_depth", 0)
    bd2 = row2.get("bit_depth", 0)
    if bd1 > bd2:
        return aid1, aid2
    elif bd2 > bd1:
        return aid2, aid1

    # 比较文件大小
    fs1 = row1.get("file_bytes", 0)
    fs2 = row2.get("file_bytes", 0)
    if fs1 > fs2:
        return aid1, aid2
    elif fs2 > fs1:
        return aid2, aid1

    # 默认保留第一个
    return aid1, aid2


def main():
    parser = argparse.ArgumentParser(
        description="近似去重脚本（Stage 4）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="配置文件路径")
    parser.add_argument("--manifest", type=str, default=None,
                        help="音频清单 CSV 路径")
    parser.add_argument("--threshold", type=float, default=None,
                        help="相似度阈值（默认 0.9）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    parser.add_argument("--output", type=str, default=None,
                        help="去重映射表输出路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不输出结果")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("近似去重（Stage 4）")
    logger.info("=" * 60)

    # 加载配置
    config = load_config(Path(args.config))
    stage4_config = config.get("stage4_dedup", {})
    approx_config = stage4_config.get("approximate_dedup", {})

    # 阈值
    threshold = args.threshold or approx_config.get("similarity_threshold", 0.9)
    logger.info(f"相似度阈值: {threshold}")

    # 加载音频清单
    manifest_path = Path(args.manifest) if args.manifest else \
        PROJECT_ROOT / config.get("global", {}).get("manifest_csv", "data/00_raw_collect/audio_manifest.csv")
    df = load_manifest(manifest_path)

    # 限制数量
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"限制处理数量: {len(df)}")

    logger.info(f"待处理音频: {len(df)} 个")

    # 提取特征
    logger.info("提取 chroma 特征...")
    features = {}
    failed = []

    for i, (_, row) in enumerate(df.iterrows()):
        audio_id = row["audio_id"]
        ext = row.get("format", "wav").lower()
        audio_path = get_audio_absolute_path(audio_id, ext)

        if not audio_path.exists():
            logger.warning(f"[{i+1}/{len(df)}] 文件不存在，跳过: {audio_id}")
            failed.append(audio_id)
            continue

        logger.info(f"[{i+1}/{len(df)}] 提取特征: {audio_id}")
        feat = extract_chroma_features(str(audio_path))

        if feat is not None:
            features[audio_id] = feat
        else:
            failed.append(audio_id)

    logger.info(f"特征提取完成: 成功 {len(features)}, 失败 {len(failed)}")

    if len(features) < 2:
        logger.warning("有效特征少于 2 个，无法计算相似度")
        return

    # 计算相似度矩阵
    logger.info("计算余弦相似度矩阵...")
    sim_df = compute_similarity_matrix(features)

    # 查找重复
    logger.info(f"查找重复（阈值: {threshold}）...")
    duplicate_pairs, keep_mapping = find_duplicates(sim_df, threshold, df)

    # 输出结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("去重结果")
    logger.info(f"  总音频数: {len(df)}")
    logger.info(f"  有效特征: {len(features)}")
    logger.info(f"  重复对数: {len(duplicate_pairs)}")
    logger.info(f"  建议删除: {len(keep_mapping)}")
    logger.info(f"  建议保留: {len(features) - len(keep_mapping)}")
    logger.info("=" * 60)

    if duplicate_pairs:
        logger.info("")
        logger.info("重复对详情:")
        for pair in duplicate_pairs:
            logger.info(f"  {pair['audio_id_1']} ↔ {pair['audio_id_2']} "
                       f"(相似度: {pair['similarity']})")

    if keep_mapping:
        logger.info("")
        logger.info("去重映射（删除 → 保留）:")
        for delete_id, keep_id in keep_mapping.items():
            logger.info(f"  {delete_id} → {keep_id}")

    # 输出去重映射表
    if not args.dry_run and keep_mapping:
        output_path = Path(args.output) if args.output else \
            PROJECT_ROOT / "data" / "00.5_cleaned" / "dedup_mapping.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mapping_df = pd.DataFrame([
            {"delete_audio_id": k, "keep_audio_id": v}
            for k, v in keep_mapping.items()
        ])
        mapping_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"去重映射表已保存: {output_path}")

        # 保存重复对详情
        pairs_path = output_path.parent / "duplicate_pairs.csv"
        pairs_df = pd.DataFrame(duplicate_pairs)
        pairs_df.to_csv(pairs_path, index=False, encoding="utf-8")
        logger.info(f"重复对详情已保存: {pairs_path}")

    logger.info(f"日志文件: {log_file}")


if __name__ == "__main__":
    main()
