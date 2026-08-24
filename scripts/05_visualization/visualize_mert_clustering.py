"""
visualize_mert_clustering.py
MERT 嵌入向量聚类可视化：t-SNE + UMAP + DBSCAN

功能：
- 加载 MERT 768维嵌入向量
- 加载 L3 真实标签（用于着色）
- t-SNE 降维到2D/3D
- UMAP 降维到2D
- DBSCAN 聚类
- 用 plotly 生成交互式 HTML 可视化

用法：
    python visualize_mert_clustering.py \
        --mert-dir data/02_preannotation/l2_mert_embedding \
        --l3-dir data/02_preannotation/l3_structural/text_labels \
        --output reports/mert_clustering_visualization.html
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

COLOR_PALETTE = [
    '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
]


def load_mert_embeddings(mert_dir: Path) -> Tuple[List[str], np.ndarray]:
    audio_ids = []
    embeddings = []
    for f in sorted(mert_dir.glob("*_mert_embedding.npy")):
        audio_id = f.stem.replace("_mert_embedding", "")
        emb = np.load(f)
        audio_ids.append(audio_id)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    logger.info(f"MERT 嵌入: {len(audio_ids)} 首, 维度: {embeddings.shape[1]}")
    return audio_ids, embeddings


def load_l3_labels(l3_dir: Path) -> Dict[str, Dict]:
    labels = {}
    for f in l3_dir.glob("*_text_labels.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        audio_id = data.get("audio_id", f.stem.replace("_text_labels", ""))
        labels[audio_id] = data
    logger.info(f"L3 标签: {len(labels)} 首")
    return labels


def run_tsne(embeddings, n_components=2, perplexity=5, random_state=42):
    logger.info(f"运行 t-SNE (n_components={n_components}, perplexity={perplexity})...")
    tsne = TSNE(n_components=n_components, perplexity=min(perplexity, len(embeddings)-1),
                random_state=random_state, init='pca', learning_rate='auto')
    result = tsne.fit_transform(embeddings)
    logger.info(f"  t-SNE 完成: {result.shape}")
    return result


def run_umap(embeddings, n_components=2, n_neighbors=5, min_dist=0.1, random_state=42):
    if not HAS_UMAP:
        logger.warning("umap-learn 未安装，跳过 UMAP")
        return None
    logger.info(f"运行 UMAP (n_components={n_components}, n_neighbors={n_neighbors})...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=min(n_neighbors, len(embeddings)-1),
                        min_dist=min_dist, random_state=random_state)
    result = reducer.fit_transform(embeddings)
    logger.info(f"  UMAP 完成: {result.shape}")
    return result


def run_dbscan(embeddings, eps=2.0, min_samples=2):
    logger.info(f"运行 DBSCAN (eps={eps}, min_samples={min_samples})...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(embeddings_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"  DBSCAN 完成: {n_clusters} 个簇, {n_noise} 个噪声点")
    return labels, n_clusters


def create_scatter_2d(coords, audio_ids, labels, color_by="genre", title="t-SNE"):
    df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'audio_id': audio_ids})
    df['genre'] = df['audio_id'].map(lambda aid: labels.get(aid, {}).get('genre', 'unknown'))
    df['mood'] = df['audio_id'].map(lambda aid: ', '.join(labels.get(aid, {}).get('mood', [])[:2]))
    df['instrumentation'] = df['audio_id'].map(lambda aid: ', '.join(labels.get(aid, {}).get('instrumentation', [])[:3]))
    df['caption'] = df['audio_id'].map(lambda aid: labels.get(aid, {}).get('caption', '')[:60])

    fig = px.scatter(df, x='x', y='y', color=color_by,
                     hover_data=['audio_id', 'genre', 'mood', 'instrumentation', 'caption'],
                     title=title, color_discrete_sequence=COLOR_PALETTE)
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', title_x=0.5)
    return fig


def create_scatter_3d(coords, audio_ids, labels, color_by="genre", title="t-SNE 3D"):
    df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'z': coords[:, 2], 'audio_id': audio_ids})
    df['genre'] = df['audio_id'].map(lambda aid: labels.get(aid, {}).get('genre', 'unknown'))
    df['mood'] = df['audio_id'].map(lambda aid: ', '.join(labels.get(aid, {}).get('mood', [])[:2]))
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=color_by,
                         hover_data=['audio_id', 'genre', 'mood'], title=title,
                         color_discrete_sequence=COLOR_PALETTE)
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(title_x=0.5)
    return fig


def create_cluster_viz(coords, audio_ids, cluster_labels, title="DBSCAN 聚类"):
    df = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1], 'audio_id': audio_ids,
                       'cluster': [f"Cluster {l}" if l != -1 else "Noise" for l in cluster_labels]})
    fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['audio_id', 'cluster'],
                     title=title, color_discrete_sequence=COLOR_PALETTE)
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', title_x=0.5)
    return fig


def create_combined_html(figures, output_path):
    html_parts = [fig.to_html(full_html=False, include_plotlyjs='cdn') for fig in figures]
    combined = f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>MERT 聚类可视化</title>
    <style>body{{font-family:-apple-system,sans-serif;margin:20px;background:#f5f5f5}}
    .chart{{background:white;border-radius:8px;padding:20px;margin-bottom:20px;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
    h1{{color:#333;text-align:center}}.summary{{background:#e8f4f8;padding:15px;border-radius:8px;margin-bottom:20px}}</style>
    </head><body><h1>MERT 嵌入聚类可视化</h1>
    <div class="summary"><p><strong>说明：</strong>基于 MERT-v1-95M 768维音乐理解嵌入，使用 t-SNE 和 UMAP 降维到2D/3D，DBSCAN 自动聚类。鼠标悬停查看详细标签。</p></div>
    {''.join(f'<div class="chart">{h}</div>' for h in html_parts)}
    </body></html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    logger.info(f"✅ 可视化已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MERT 嵌入聚类可视化：t-SNE + UMAP + DBSCAN")
    parser.add_argument("--mert-dir", default="data/02_preannotation/l2_mert_embedding")
    parser.add_argument("--l3-dir", default="data/02_preannotation/l3_structural/text_labels")
    parser.add_argument("--output", default="reports/mert_clustering_visualization.html")
    parser.add_argument("--perplexity", type=int, default=5)
    parser.add_argument("--eps", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()

    mert_dir = Path(args.mert_dir)
    l3_dir = Path(args.l3_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MERT 聚类可视化")
    logger.info("=" * 60)

    logger.info("\n[Step 1] 加载数据...")
    audio_ids, embeddings = load_mert_embeddings(mert_dir)
    l3_labels = load_l3_labels(l3_dir)

    logger.info("\n[Step 2] t-SNE 2D...")
    tsne_2d = run_tsne(embeddings, 2, args.perplexity)

    logger.info("\n[Step 3] t-SNE 3D...")
    tsne_3d = run_tsne(embeddings, 3, args.perplexity)

    logger.info("\n[Step 4] UMAP 2D...")
    umap_2d = run_umap(embeddings, 2, args.perplexity)

    logger.info("\n[Step 5] DBSCAN 聚类...")
    cluster_labels, n_clusters = run_dbscan(tsne_2d, args.eps, args.min_samples)

    logger.info("\n[Step 6] 创建可视化...")
    figures = []
    figures.append(create_scatter_2d(tsne_2d, audio_ids, l3_labels, "genre",
                                       f"t-SNE 2D - 按流派着色 (MERT 768d, n={len(audio_ids)})"))
    figures.append(create_scatter_2d(tsne_2d, audio_ids, l3_labels, "mood", "t-SNE 2D - 按情绪着色"))
    figures.append(create_scatter_3d(tsne_3d, audio_ids, l3_labels, "genre", "t-SNE 3D - 按流派着色"))
    if umap_2d is not None:
        figures.append(create_scatter_2d(umap_2d, audio_ids, l3_labels, "genre", "UMAP 2D - 按流派着色"))
    figures.append(create_cluster_viz(tsne_2d, audio_ids, cluster_labels,
                                        f"DBSCAN 聚类结果 (eps={args.eps}, {n_clusters} 簇)"))

    create_combined_html(figures, output_path)

    logger.info("\n" + "=" * 60)
    logger.info("可视化完成")
    logger.info("=" * 60)
    logger.info(f"  样本数: {len(audio_ids)}, 嵌入维度: {embeddings.shape[1]}")
    logger.info(f"  DBSCAN 簇数: {n_clusters}")
    logger.info(f"  输出文件: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
