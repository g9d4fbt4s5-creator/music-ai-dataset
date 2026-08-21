#!/usr/bin/env python3
"""
style_consistency_clustering.py
Stage 5.3 风格一致性聚类（MERT/CLAP 嵌入 + DBSCAN）

两阶段设计：
  阶段1（GPU）：extract 模式 — 用 MERT/CLAP 提取音频嵌入，输出 Parquet
  阶段2（Mac本地）：cluster 模式 — 读取嵌入，DBSCAN 聚类，识别异常离群点

为什么两阶段？
  - MERT/CLAP 在 Mac 本地 16GB 内存容易 OOM（CLAP 2.35GB 模型+音频加载）
  - DBSCAN 是 CPU 任务，Mac 本地完全可以跑，且结果可视化方便
  - 嵌入向量只有 512/768 维，文件很小（500首约几MB），传输成本低

用法：
  # 阶段1：GPU 上提取嵌入
  python3 style_consistency_clustering.py --mode extract \\
      --input-dir /root/autodl-tmp/jazz_500_audio-low \\
      --output /root/autodl-tmp/embeddings/mert_embeddings.parquet \\
      --model mert

  # 阶段2：Mac 本地聚类
  python3 style_consistency_clustering.py --mode cluster \\
      --input mert_embeddings.parquet \\
      --output-dir data/05_auxiliary/style_clustering \\
      --eps 0.5 --min-samples 5

  # 用 CLAP 嵌入
  python3 style_consistency_clustering.py --mode extract --model clap ...

输出：
  extract 模式：
    - {output}.parquet：track_id + 嵌入向量（512/768维）
  cluster 模式：
    - clustering_results.csv：track_id + cluster_label + is_outlier + distance_to_centroid
    - outliers.csv：异常离群点列表（cluster_label=-1）
    - cluster_summary.csv：每个聚类的统计信息
    - tsne_visualization.png：t-SNE 降维可视化（可选）
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ===================== 配置 =====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"style_clustering_{time_str}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===================== 阶段1：嵌入提取（GPU） =====================
def extract_mert_embeddings(audio_dir: str, output_path: str, model_name: str = "mert-v1-95M"):
    """
    用 MERT 提取音频嵌入

    MERT-v1-95M 输出 768 维嵌入
    """
    import torch
    import librosa
    from transformers import Wav2Vec2FeatureExtractor, AutoModel

    logger.info(f"加载 MERT 模型: {model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        logger.info("使用 GPU 加速")

    # 扫描音频文件
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, f))
    audio_files = sorted(audio_files)
    logger.info(f"找到 {len(audio_files)} 个音频文件")

    embeddings = []
    track_ids = []

    for i, audio_path in enumerate(audio_files):
        track_id = Path(audio_path).stem
        logger.info(f"[{i+1}/{len(audio_files)}] {track_id}")

        try:
            # 加载音频（MERT 需要 24kHz 单声道）
            y, sr = librosa.load(audio_path, sr=24000, mono=True)

            # 特征提取
            inputs = feature_extractor(y, sampling_rate=24000, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = model(**inputs)

            # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>  token 的嵌入（768维）
            # MERT 输出 last_hidden_state: [batch, seq_len, 768]
            # 取全局平均池化作为曲目级嵌入
            hidden_states = outputs.last_hidden_state.cpu().numpy()[0]  # [seq_len, 768]
            embedding = np.mean(hidden_states, axis=0)  # [768]

            embeddings.append(embedding)
            track_ids.append(track_id)
            logger.info(f"  嵌入维度: {embedding.shape}")

        except Exception as e:
            logger.error(f"  提取失败: {e}")
            continue

    # 保存
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings[0].shape[0])])
    df.insert(0, "track_id", track_ids)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"嵌入已保存: {output_path} ({len(track_ids)} 首, {embeddings[0].shape[0]} 维)")

    return df


def extract_clap_embeddings(audio_dir: str, output_path: str):
    """
    用 LAION CLAP 提取音频嵌入

    CLAP HTSAT-base 输出 512 维嵌入
    """
    import torch
    import librosa
    import laion_clap

    logger.info("加载 LAION CLAP 模型 (HTSAT-base)")
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # 默认下载 HTSAT-base
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        logger.info("使用 GPU 加速")

    # 扫描音频文件
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(root, f))
    audio_files = sorted(audio_files)
    logger.info(f"找到 {len(audio_files)} 个音频文件")

    embeddings = []
    track_ids = []

    for i, audio_path in enumerate(audio_files):
        track_id = Path(audio_path).stem
        logger.info(f"[{i+1}/{len(audio_files)}] {track_id}")

        try:
            # CLAP 需要 48kHz 单声道，int16 格式
            y, sr = librosa.load(audio_path, sr=48000, mono=True)
            # 转为 int16
            y_int16 = (y * 32767).astype(np.int16)

            # 推理
            with torch.no_grad():
                audio_embed = model.get_audio_embedding_from_data(
                    [y_int16], use_tensor=True
                )
            embedding = audio_embed.cpu().numpy()[0]  # [512]

            embeddings.append(embedding)
            track_ids.append(track_id)
            logger.info(f"  嵌入维度: {embedding.shape}")

        except Exception as e:
            logger.error(f"  提取失败: {e}")
            continue

    # 保存
    df = pd.DataFrame(embeddings, columns=[f"dim_{i}" for i in range(embeddings[0].shape[0])])
    df.insert(0, "track_id", track_ids)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"嵌入已保存: {output_path} ({len(track_ids)} 首, {embeddings[0].shape[0]} 维)")

    return df


# ===================== 阶段2：DBSCAN 聚类（Mac本地） =====================
def load_embeddings(input_path: str) -> Tuple[np.ndarray, List[str]]:
    """加载嵌入向量（支持多种列名格式）"""
    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"不支持的文件格式: {input_path}")

    # 自动识别 ID 列（track_id 或 audio_id）
    id_col = None
    for col in ["track_id", "audio_id", "id", "filename"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        id_col = df.columns[0]  # 默认第一列

    track_ids = df[id_col].tolist()

    # 自动识别嵌入列（dim_ 开头，或所有数值列）
    embedding_cols = [c for c in df.columns if c.startswith("dim_")]
    if not embedding_cols:
        # 如果没有 dim_ 列，使用所有数值列（排除 ID 列和非数值列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除常见的元数据列
        meta_cols = {"duration_s", "sample_rate", "n_channels", "n_samples", "bpm", "n_beats",
                     "beat_interval_mean", "beat_interval_std", "first_beat_time", "last_beat_time",
                     "loudness_lufs", "dominant_note"}
        embedding_cols = [c for c in numeric_cols if c not in meta_cols]
        logger.info(f"自动识别 {len(embedding_cols)} 个数值特征列作为嵌入")

    if not embedding_cols:
        raise ValueError("未找到嵌入列（需要 dim_ 开头的列或数值列）")

    embeddings = df[embedding_cols].values.astype(np.float32)

    # 处理 NaN
    if np.isnan(embeddings).any():
        logger.warning("嵌入中存在 NaN，用 0 填充")
        embeddings = np.nan_to_num(embeddings, nan=0.0)

    logger.info(f"加载嵌入: {len(track_ids)} 首, {embeddings.shape[1]} 维 (ID列: {id_col})")
    return embeddings, track_ids


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 归一化嵌入向量"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # 避免除零
    normalized = embeddings / norms
    logger.info("L2 归一化完成")
    return normalized


def dbscan_clustering(
    embeddings: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "cosine",
) -> np.ndarray:
    """
    DBSCAN 聚类

    Args:
        embeddings: 嵌入向量（已归一化）
        eps: 邻域半径（cosine 距离下，0.5 是合理起点）
        min_samples: 核心点的最小邻居数
        metric: 距离度量（cosine/euclidean）

    Returns:
        np.ndarray: 聚类标签（-1 表示异常离群点）
    """
    from sklearn.cluster import DBSCAN

    logger.info(f"DBSCAN 聚类: eps={eps}, min_samples={min_samples}, metric={metric}")
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = clustering.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = np.sum(labels == -1)
    logger.info(f"聚类完成: {n_clusters} 个聚类, {n_outliers} 个异常离群点 ({n_outliers/len(labels)*100:.1f}%)")

    return labels


def compute_cluster_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    track_ids: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    计算聚类统计信息

    Returns:
        tuple: (results_df, outliers_df, summary_df)
    """
    n_samples = len(track_ids)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # 计算每个聚类的质心
    centroids = {}
    for label in unique_labels:
        if label == -1:
            continue
        mask = labels == label
        centroids[label] = np.mean(embeddings[mask], axis=0)

    # 计算每个样本到所属聚类质心的距离
    distances = np.zeros(n_samples)
    for i, label in enumerate(labels):
        if label == -1:
            distances[i] = -1  # 异常点没有质心
        else:
            distances[i] = np.linalg.norm(embeddings[i] - centroids[label])

    # 结果表
    results_df = pd.DataFrame({
        "track_id": track_ids,
        "cluster_label": labels,
        "is_outlier": labels == -1,
        "distance_to_centroid": distances,
    })

    # 异常点表
    outliers_df = results_df[results_df["is_outlier"]].copy()
    logger.info(f"异常离群点: {len(outliers_df)} 个")

    # 聚类汇总表
    summary_data = []
    for label in sorted(unique_labels):
        if label == -1:
            name = "Outliers"
        else:
            name = f"Cluster_{label}"
        mask = labels == label
        size = np.sum(mask)
        if label != -1 and size > 1:
            avg_dist = np.mean(distances[mask])
            max_dist = np.max(distances[mask])
        else:
            avg_dist = -1
            max_dist = -1
        summary_data.append({
            "cluster": name,
            "label": label,
            "size": size,
            "percentage": f"{size/n_samples*100:.1f}%",
            "avg_distance_to_centroid": round(avg_dist, 4) if avg_dist >= 0 else "N/A",
            "max_distance_to_centroid": round(max_dist, 4) if max_dist >= 0 else "N/A",
        })
    summary_df = pd.DataFrame(summary_data)

    return results_df, outliers_df, summary_df


def visualize_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    perplexity: int = 30,
):
    """
    t-SNE 降维可视化

    注意：t-SNE 在样本数 <  perplexity 时会报错，需要调整 perplexity
    """
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")  # 非交互式后端
        import matplotlib.pyplot as plt

        n_samples = len(embeddings)
        actual_perplexity = min(perplexity, n_samples - 1)
        if actual_perplexity < 2:
            logger.warning("样本太少，跳过 t-SNE 可视化")
            return

        logger.info(f"t-SNE 降维: perplexity={actual_perplexity}")
        tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # 绘图
        plt.figure(figsize=(12, 8))
        unique_labels = set(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(sorted(unique_labels), colors):
            mask = labels == label
            if label == -1:
                marker = "x"
                label_name = "Outlier"
                alpha = 0.6
            else:
                marker = "o"
                label_name = f"Cluster {label}"
                alpha = 0.7
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                marker=marker,
                label=label_name,
                alpha=alpha,
                s=50,
            )

        plt.title("Style Consistency Clustering (t-SNE)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"t-SNE 可视化已保存: {output_path}")

    except ImportError as e:
        logger.warning(f"跳过可视化（缺少依赖）: {e}")
    except Exception as e:
        logger.warning(f"可视化失败: {e}")


def auto_tune_eps(embeddings: np.ndarray, min_samples: int = 5, metric: str = "cosine") -> float:
    """
    自动调优 eps 参数

    方法：计算每个点到第 min_samples 个最近邻的距离，取拐点（knee point）
    """
    from sklearn.neighbors import NearestNeighbors

    logger.info("自动调优 eps...")
    nn = NearestNeighbors(n_neighbors=min_samples, metric=metric)
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # 取第 min_samples 个最近邻的距离（索引 min_samples-1）
    k_distances = np.sort(distances[:, min_samples - 1])

    # 简单方法：取 75 分位数作为 eps
    eps = np.percentile(k_distances, 75)
    logger.info(f"自动调优 eps = {eps:.4f}（75分位数）")
    return eps


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(description="Stage 5.3 风格一致性聚类（MERT/CLAP 嵌入 + DBSCAN）")
    parser.add_argument("--mode", type=str, required=True, choices=["extract", "cluster"],
                        help="extract: GPU提取嵌入; cluster: Mac本地聚类")
    # extract 模式参数
    parser.add_argument("--input-dir", type=str, help="输入音频目录（extract模式）")
    parser.add_argument("--model", type=str, default="mert", choices=["mert", "clap"],
                        help="嵌入模型（mert/clap）")
    parser.add_argument("--mert-model", type=str, default="mert-v1-95M", help="MERT 模型名称")
    # cluster 模式参数
    parser.add_argument("--input", type=str, help="输入嵌入文件（cluster模式，.parquet/.csv）")
    parser.add_argument("--output-dir", type=str, default="data/05_auxiliary/style_clustering", help="输出目录")
    parser.add_argument("--eps", type=float, default=None, help="DBSCAN eps（None=自动调优）")
    parser.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="距离度量")
    parser.add_argument("--no-visualize", action="store_true", help="跳过 t-SNE 可视化")
    # 通用参数
    parser.add_argument("--output", type=str, help="输出文件路径（extract模式）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个音频")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stage 5.3 风格一致性聚类")
    logger.info(f"  模式: {args.mode}")
    if args.mode == "extract":
        logger.info(f"  输入目录: {args.input_dir}")
        logger.info(f"  模型: {args.model}")
        logger.info(f"  输出: {args.output}")
    else:
        logger.info(f"  输入嵌入: {args.input}")
        logger.info(f"  输出目录: {args.output_dir}")
        logger.info(f"  eps: {args.eps or '自动调优'}")
        logger.info(f"  min_samples: {args.min_samples}")
        logger.info(f"  metric: {args.metric}")
    logger.info("=" * 60)

    if args.mode == "extract":
        # 阶段1：提取嵌入（GPU）
        if not args.input_dir:
            logger.error("extract 模式需要 --input-dir")
            return
        if not args.output:
            logger.error("extract 模式需要 --output")
            return

        if args.model == "mert":
            extract_mert_embeddings(args.input_dir, args.output, args.mert_model)
        elif args.model == "clap":
            extract_clap_embeddings(args.input_dir, args.output)

    elif args.mode == "cluster":
        # 阶段2：DBSCAN 聚类（Mac本地）
        if not args.input:
            logger.error("cluster 模式需要 --input")
            return

        # 加载嵌入
        embeddings, track_ids = load_embeddings(args.input)

        if args.limit:
            embeddings = embeddings[:args.limit]
            track_ids = track_ids[:args.limit]
            logger.info(f"限制前 {args.limit} 个样本")

        # L2 归一化
        embeddings_normalized = normalize_embeddings(embeddings)

        # 自动调优 eps
        eps = args.eps
        if eps is None:
            eps = auto_tune_eps(embeddings_normalized, args.min_samples, args.metric)

        # DBSCAN 聚类
        labels = dbscan_clustering(embeddings_normalized, eps, args.min_samples, args.metric)

        # 计算统计信息
        results_df, outliers_df, summary_df = compute_cluster_stats(
            embeddings_normalized, labels, track_ids
        )

        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "clustering_results.csv"
        outliers_path = output_dir / "outliers.csv"
        summary_path = output_dir / "cluster_summary.csv"

        results_df.to_csv(results_path, index=False, encoding="utf-8")
        outliers_df.to_csv(outliers_path, index=False, encoding="utf-8")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8")

        logger.info(f"聚类结果已保存: {results_path}")
        logger.info(f"异常点列表已保存: {outliers_path}")
        logger.info(f"聚类汇总已保存: {summary_path}")

        # 打印汇总
        logger.info("")
        logger.info("=== 聚类汇总 ===")
        logger.info(summary_df.to_string(index=False))

        # t-SNE 可视化
        if not args.no_visualize:
            viz_path = output_dir / "tsne_visualization.png"
            visualize_tsne(embeddings_normalized, labels, str(viz_path))

        logger.info("")
        logger.info("=" * 60)
        logger.info("聚类完成")
        logger.info(f"  总样本: {len(track_ids)}")
        logger.info(f"  聚类数: {len(set(labels)) - (1 if -1 in labels else 0)}")
        logger.info(f"  异常点: {np.sum(labels == -1)} ({np.sum(labels == -1)/len(labels)*100:.1f}%)")
        logger.info(f"  eps: {eps:.4f}")
        logger.info(f"  输出目录: {output_dir}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
