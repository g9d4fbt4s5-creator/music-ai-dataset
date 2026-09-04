#!/usr/bin/env python3
"""
【实验特征链暂停 · T3 终审计 2026-09-04 用户拍板】
MERT 嵌入聚类可视化属 KNN/聚类实验链（聚类轮廓系数 0.049、KNN 一致率 0% 已证伪）。
作为诊断资产保留原地、不归档不删除，但不再服务当前 L4 生产；写报告/复盘时仍可运行。

MERT 聚类可视化 — t-SNE/UMAP 降维 + plotly 交互散点图

读取 MERT 768维嵌入，降维到2D，按流派/情绪/人声着色，生成交互式HTML。

使用:
    python visualize_mert_clustering.py \
        --embeddings-dir data/00.5_cleaned/reports/vXXX/l2_mert_embedding \
        --labels-dir data/02_preannotation/l4_propagated \
        --output data/00.5_cleaned/reports/vXXX/stats/mert_clustering.html
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ plotly 未安装，将生成静态数据JSON而非HTML")


def load_embeddings(embeddings_dir: str) -> tuple:
    """加载 MERT 嵌入"""
    embeddings_dir = Path(embeddings_dir)
    audio_ids = []
    vectors = []

    for f in sorted(embeddings_dir.glob("*.npy")):
        aid = f.stem.replace("_mert_embedding", "")
        vec = np.load(f)
        if vec.ndim > 1:
            vec = vec.flatten()[:768]
        audio_ids.append(aid)
        vectors.append(vec)

    return audio_ids, np.array(vectors)


def load_labels(labels_dir: str, audio_ids: list) -> dict:
    """加载 L4 融合标签用于着色"""
    labels_dir = Path(labels_dir)
    labels = {aid: {} for aid in audio_ids}

    for f in labels_dir.glob("*_full_tags.json"):
        with open(f) as fp:
            data = json.load(fp)
        aid = data.get("audio_id", "")
        if aid in labels:
            labels[aid] = {
                "genre": data.get("genre", "unknown"),
                "mood": (data.get("mood", ["unknown"]) or ["unknown"])[0],
                "vocal_presence": data.get("vocal_presence", "unknown"),
                "quality": data.get("quality_assessment", "unknown"),
                "propagated_from": data.get("propagated_from", "deepseek"),
                "bpm": data.get("bpm", 0),
                "caption": data.get("caption", "")[:80],
            }

    return labels


def reduce_dimensions(vectors: np.ndarray, method: str = "tsne",
                       perplexity: int = 30, n_neighbors: int = 15) -> np.ndarray:
    """降维到2D"""
    n_samples = len(vectors)

    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(perplexity, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                    init="pca", learning_rate="auto")
        return tsne.fit_transform(vectors)

    elif method == "umap":
        import umap
        n_n = min(n_neighbors, n_samples - 1)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_n)
        return reducer.fit_transform(vectors)

    else:
        raise ValueError(f"未知降维方法: {method}")


def build_interactive_html(audio_ids: list, coords_tsne: np.ndarray,
                           coords_umap: np.ndarray, labels: dict,
                           output_path: str):
    """构建 plotly 交互式HTML"""
    if not PLOTLY_AVAILABLE:
        # 保存静态数据
        data = {
            "audio_ids": audio_ids,
            "tsne": coords_tsne.tolist(),
            "umap": coords_umap.tolist(),
            "labels": labels,
        }
        json_path = output_path.replace(".html", "_data.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"plotly不可用，已保存数据到: {json_path}")
        return

    # 准备数据
    genres = [labels[aid].get("genre", "unknown") for aid in audio_ids]
    moods = [labels[aid].get("mood", "unknown") for aid in audio_ids]
    vocals = [labels[aid].get("vocal_presence", "unknown") for aid in audio_ids]
    qualities = [labels[aid].get("quality", "unknown") for aid in audio_ids]
    bpms = [labels[aid].get("bpm", 0) for aid in audio_ids]
    captions = [labels[aid].get("caption", "") for aid in audio_ids]
    propagated = [labels[aid].get("propagated_from", "deepseek") for aid in audio_ids]

    hover_text = [
        f"<b>{aid[:16]}</b><br>"
        f"流派: {g}<br>情绪: {m}<br>人声: {v}<br>"
        f"BPM: {b}<br>质量: {q}<br>来源: {p}<br>"
        f"<i>{c}</i>"
        for aid, g, m, v, b, q, p, c in
        zip(audio_ids, genres, moods, vocals, bpms, qualities, propagated, captions)
    ]

    # 颜色映射
    genre_colors = {
        "jazz": "#1f77b4", "blues": "#ff7f0e", "classical": "#2ca02c",
        "pop": "#d62728", "rock": "#9467bd", "electronic": "#8c564b",
        "folk": "#e377c2", "world": "#7f7f7f", "other": "#bcbd22",
        "unknown": "#17becf",
    }
    mood_colors = {
        "relaxed": "#2ca02c", "energetic": "#ff7f0e", "melancholic": "#1f77b4",
        "happy": "#ffbb78", "mysterious": "#9467bd", "intense": "#d62728",
        "calm": "#98df8a", "nostalgic": "#c5b0d5", "unknown": "#7f7f7f",
    }
    vocal_colors = {
        "instrumental": "#1f77b4", "vocal": "#ff7f0e",
        "mixed": "#2ca02c", "unknown": "#7f7f7f",
    }

    def make_scatter(coords, color_key, color_map, title):
        colors = [color_map.get(v, "#7f7f7f") for v in color_key]
        return go.Scatter(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(size=10, color=colors, line=dict(width=1, color="white")),
            text=hover_text, hoverinfo="text",
            name=title,
        )

    # 创建2x2子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "t-SNE — 按流派着色",
            "t-SNE — 按情绪着色",
            "UMAP — 按流派着色",
            "UMAP — 按人声着色",
        ),
        horizontal_spacing=0.08, vertical_spacing=0.12,
    )

    fig.add_trace(make_scatter(coords_tsne, genres, genre_colors, "genre"), row=1, col=1)
    fig.add_trace(make_scatter(coords_tsne, moods, mood_colors, "mood"), row=1, col=2)
    fig.add_trace(make_scatter(coords_umap, genres, genre_colors, "genre"), row=2, col=1)
    fig.add_trace(make_scatter(coords_umap, vocals, vocal_colors, "vocal"), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"MERT 768d 嵌入聚类可视化 (n={len(audio_ids)})",
            font=dict(size=18),
        ),
        height=900, width=1200,
        showlegend=False,
        template="plotly_white",
    )

    # 更新坐标轴
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Dim 1", row=i, col=j)
            fig.update_yaxes(title_text="Dim 2", row=i, col=j)

    fig.write_html(output_path, include_plotlyjs="cdn")
    print(f"交互式HTML已保存: {output_path}")


def run_visualization(embeddings_dir: str, labels_dir: str, output_path: str):
    """主流程"""
    print("加载 MERT 嵌入...")
    audio_ids, vectors = load_embeddings(embeddings_dir)
    print(f"  {len(audio_ids)} 个样本, {vectors.shape[1]} 维")

    print("加载标签...")
    labels = load_labels(labels_dir, audio_ids)
    labeled = sum(1 for aid in audio_ids if labels[aid])
    print(f"  {labeled}/{len(audio_ids)} 个有标签")

    print("t-SNE 降维...")
    coords_tsne = reduce_dimensions(vectors, method="tsne")

    print("UMAP 降维...")
    coords_umap = reduce_dimensions(vectors, method="umap")

    print("生成交互式HTML...")
    build_interactive_html(audio_ids, coords_tsne, coords_umap, labels, output_path)

    # 统计
    genre_dist = {}
    for aid in audio_ids:
        g = labels[aid].get("genre", "unknown")
        genre_dist[g] = genre_dist.get(g, 0) + 1

    print(f"\n{'='*60}")
    print(f"MERT 聚类可视化完成")
    print(f"{'='*60}")
    print(f"  样本数: {len(audio_ids)}")
    print(f"  流派分布: {genre_dist}")
    print(f"  输出: {output_path}")
    print(f"\n  在浏览器中打开HTML可交互查看:")
    print(f"  - 悬停查看样本详情(流派/情绪/BPM/Caption)")
    print(f"  - 4个子图: t-SNE(流派/情绪) + UMAP(流派/人声)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERT 聚类可视化")
    parser.add_argument("--embeddings-dir", required=True, help="MERT嵌入目录")
    parser.add_argument("--labels-dir", required=True, help="L4标签目录")
    parser.add_argument("--output", required=True, help="输出HTML路径")
    args = parser.parse_args()

    run_visualization(args.embeddings_dir, args.labels_dir, args.output)
