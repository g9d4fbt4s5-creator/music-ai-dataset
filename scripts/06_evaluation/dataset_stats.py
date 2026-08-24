#!/usr/bin/env python3
"""
数据集统计评测脚本

自动生成数据集分布统计报告，包括:
- 流派/BPM/调性/响度/时长/人声/来源分布
- 坏样本/边际样本统计
- 标注置信度/来源分布
- OOD 样本统计与分布距离
- MERT 嵌入 t-SNE/UMAP 可视化

使用:
    python dataset_stats.py \
        --manifest data/00_raw_collect/audio_manifest.csv \
        --qc-report data/00.5_cleaned/reports/vXXX/qc_gate_report.csv \
        --l4-dir data/02_preannotation/l4_propagated \
        --mert-dir data/00.5_cleaned/reports/vXXX/l2_mert_embedding \
        --output-dir data/00.5_cleaned/reports/vXXX/stats/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(manifest_path, qc_report_path, l4_dir, mert_dir):
    """加载所有数据源"""
    manifest = pd.read_csv(manifest_path)
    qc = pd.read_csv(qc_report_path) if qc_report_path and os.path.exists(qc_report_path) else pd.DataFrame()

    # L4 标签
    l4_data = []
    if l4_dir and os.path.exists(l4_dir):
        for f in Path(l4_dir).glob("*_full_tags.json"):
            with open(f) as fp:
                l4_data.append(json.load(fp))
    l4_df = pd.DataFrame(l4_data) if l4_data else pd.DataFrame()

    # MERT 嵌入
    embeddings = {}
    if mert_dir and os.path.exists(mert_dir):
        for f in Path(mert_dir).glob("*.npy"):
            aid = f.stem.replace("_mert_embedding", "")
            embeddings[aid] = np.load(f)

    return manifest, qc, l4_df, embeddings


# ========== 分布统计 ==========

def genre_distribution(l4_df):
    """流派分布"""
    if l4_df.empty or "genre" not in l4_df.columns:
        return {}
    dist = l4_df["genre"].value_counts().to_dict()
    return {k: int(v) for k, v in dist.items()}


def bpm_distribution(manifest, l4_df):
    """BPM 分布统计"""
    bpms = []
    if "bpm" in manifest.columns:
        bpms.extend(manifest["bpm"].dropna().tolist())
    if not l4_df.empty and "bpm" in l4_df.columns:
        bpms.extend(l4_df["bpm"].dropna().tolist())

    if not bpms:
        return {}

    bpms = np.array([b for b in bpms if b > 0])
    bins = [0, 60, 80, 100, 120, 140, 160, 200, 999]
    labels = ["<60", "60-80", "80-100", "100-120", "120-140", "140-160", "160-200", ">200"]
    hist = {labels[i]: int(((bpms >= bins[i]) & (bpms < bins[i+1])).sum()) for i in range(len(labels))}

    return {
        "count": int(len(bpms)),
        "mean": round(float(np.mean(bpms)), 1),
        "median": round(float(np.median(bpms)), 1),
        "std": round(float(np.std(bpms)), 1),
        "min": round(float(np.min(bpms)), 1),
        "max": round(float(np.max(bpms)), 1),
        "p25": round(float(np.percentile(bpms, 25)), 1),
        "p75": round(float(np.percentile(bpms, 75)), 1),
        "histogram": hist,
    }


def key_distribution(l4_df):
    """调性分布"""
    if l4_df.empty or "key" not in l4_df.columns:
        return {}
    keys = l4_df["key"].dropna()
    dist = keys.value_counts().to_dict()
    # 分离大小调
    major = sum(1 for k in keys if "major" in str(k).lower() or not str(k).endswith("m"))
    minor = sum(1 for k in keys if "minor" in str(k).lower() or str(k).endswith("m"))
    return {
        "distribution": {k: int(v) for k, v in dist.items()},
        "major_count": int(major),
        "minor_count": int(minor),
    }


def loudness_distribution(l4_df):
    """响度分布"""
    if l4_df.empty or "loudness_db" not in l4_df.columns:
        return {}
    loudness = l4_df["loudness_db"].dropna()
    if loudness.empty:
        return {}
    return {
        "count": int(len(loudness)),
        "mean": round(float(np.mean(loudness)), 1),
        "median": round(float(np.median(loudness)), 1),
        "std": round(float(np.std(loudness)), 1),
        "min": round(float(np.min(loudness)), 1),
        "max": round(float(np.max(loudness)), 1),
    }


def duration_distribution(manifest):
    """时长分布"""
    if "duration_sec" not in manifest.columns:
        return {}
    durations = manifest["duration_sec"].dropna()
    if durations.empty:
        return {}

    bins = [0, 30, 60, 120, 180, 300, 600, 900, 1800, 99999]
    labels = ["<30s", "30-60s", "1-2min", "2-3min", "3-5min", "5-10min", "10-15min", "15-30min", ">30min"]
    hist = {labels[i]: int(((durations >= bins[i]) & (durations < bins[i+1])).sum()) for i in range(len(labels))}

    return {
        "count": int(len(durations)),
        "mean_sec": round(float(np.mean(durations)), 1),
        "median_sec": round(float(np.median(durations)), 1),
        "total_minutes": round(float(np.sum(durations) / 60), 1),
        "long_form_count": int((durations > 900).sum()),
        "dj_mix_count": int((durations > 1800).sum()),
        "histogram": hist,
    }


def vocal_distribution(l4_df, qc):
    """人声分布"""
    result = {}
    if not l4_df.empty and "vocal_presence" in l4_df.columns:
        result["l4_vocal_presence"] = l4_df["vocal_presence"].value_counts().to_dict()
    if not qc.empty and "has_vocals" in qc.columns:
        result["yamnet_has_vocals"] = {
            "true": int(qc["has_vocals"].sum()),
            "false": int((~qc["has_vocals"]).sum()),
        }
    return result


def source_distribution(manifest):
    """来源分布"""
    result = {}
    if "source" in manifest.columns:
        result["by_source"] = manifest["source"].value_counts().to_dict()
    if "collection_batch" in manifest.columns:
        result["by_batch"] = manifest["collection_batch"].value_counts().to_dict()
    return result


# ========== 质量统计 ==========

def quality_statistics(qc):
    """坏样本/边际样本统计"""
    if qc.empty:
        return {}

    result = {
        "total": int(len(qc)),
        "pass": int((qc["final_branch"] == "pass").sum()),
        "marginal": int((qc["final_branch"] == "marginal").sum()),
        "fail": int((qc["final_branch"] == "fail").sum()),
        "flag_for_review": int(qc["flag_for_review"].sum()) if "flag_for_review" in qc.columns else 0,
    }

    # 各分支占比
    result["pass_rate"] = round(result["pass"] / result["total"], 3) if result["total"] else 0
    result["marginal_rate"] = round(result["marginal"] / result["total"], 3) if result["total"] else 0
    result["fail_rate"] = round(result["fail"] / result["total"], 3) if result["total"] else 0

    # 告警阈值检查
    result["alerts"] = []
    if result["fail_rate"] > 0.05:
        result["alerts"].append(f"fail_rate={result['fail_rate']:.1%}>5%阈值")
    if result["marginal_rate"] > 0.15:
        result["alerts"].append(f"marginal_rate={result['marginal_rate']:.1%}>15%阈值")

    # flags 统计
    if "flags" in qc.columns:
        all_flags = []
        for flags_str in qc["flags"].dropna():
            try:
                all_flags.extend(json.loads(flags_str))
            except:
                pass
        from collections import Counter
        result["flag_distribution"] = dict(Counter(all_flags).most_common())

    return result


# ========== 标注统计 ==========

def annotation_statistics(l4_df):
    """标注置信度/来源分布"""
    if l4_df.empty:
        return {}

    result = {
        "total_labeled": int(len(l4_df)),
        "coverage": 1.0,  # L4 全量覆盖
    }

    # 标签来源分布
    if "propagated_from" in l4_df.columns:
        golden = int((l4_df["propagated_from"] == "golden_set").sum())
        propagated = int(l4_df["propagated_from"].notna().sum() - golden)
        deepseek_only = int(len(l4_df) - golden - propagated)
        result["source_distribution"] = {
            "golden_set_l3": golden,
            "knn_propagated": propagated,
            "deepseek_only": deepseek_only,
        }

    # fusion 来源统计
    if "fusion" in l4_df.columns:
        mood_sources = []
        inst_sources = []
        for fusion_str in l4_df["fusion"].dropna():
            try:
                f = json.loads(fusion_str) if isinstance(fusion_str, str) else fusion_str
                mood_sources.append(f.get("mood_source", "unknown"))
                inst_sources.append(f.get("instrumentation_source", "unknown"))
            except:
                pass
        from collections import Counter
        result["mood_source_dist"] = dict(Counter(mood_sources).most_common())
        result["instrument_source_dist"] = dict(Counter(inst_sources).most_common())

    # KNN 传播相似度分布
    if "propagation_similarity" in l4_df.columns:
        sims = l4_df["propagation_similarity"].dropna()
        sims = sims[sims > 0]  # 排除黄金集(1.0)和无传播(0)
        sims = sims[sims < 1.0]
        if not sims.empty:
            result["knn_similarity"] = {
                "count": int(len(sims)),
                "mean": round(float(np.mean(sims)), 4),
                "median": round(float(np.median(sims)), 4),
                "min": round(float(np.min(sims)), 4),
                "max": round(float(np.max(sims)), 4),
                "high_confidence(>0.7)": int((sims > 0.7).sum()),
                "medium_confidence(0.5-0.7)": int(((sims >= 0.5) & (sims <= 0.7)).sum()),
                "low_confidence(<0.5)": int((sims < 0.5).sum()),
            }

    return result


# ========== OOD 统计 ==========

def ood_statistics(manifest, embeddings, ood_tag="ood"):
    """OOD 样本统计与分布距离"""
    result = {"ood_count": 0, "ood_ids": []}

    # 检查是否有 OOD 标记
    if "pool" in manifest.columns:
        ood_mask = manifest["pool"].str.contains(ood_tag, case=False, na=False)
        result["ood_count"] = int(ood_mask.sum())
        result["ood_ids"] = manifest.loc[ood_mask, "audio_id"].tolist()

    # 如果有嵌入，计算 train vs ood 分布距离
    if embeddings and result["ood_count"] > 0:
        train_embs = []
        ood_embs = []
        for aid, emb in embeddings.items():
            if aid in result["ood_ids"]:
                ood_embs.append(emb)
            else:
                train_embs.append(emb)

        if train_embs and ood_embs:
            train_mean = np.mean(train_embs, axis=0)
            ood_mean = np.mean(ood_embs, axis=0)
            # 余弦距离
            cos_dist = 1 - np.dot(train_mean, ood_mean) / (np.linalg.norm(train_mean) * np.linalg.norm(ood_mean))
            result["distribution_distance"] = {
                "method": "cosine_distance_between_centroids",
                "cosine_distance": round(float(cos_dist), 4),
                "interpretation": "距离越大表示OOD与训练集分布差异越大",
            }

    return result


# ========== 嵌入可视化 ==========

def embedding_visualization(embeddings, l4_df, output_dir):
    """MERT 嵌入 t-SNE/UMAP 降维可视化数据"""
    if not embeddings:
        return {}

    ids = sorted(embeddings.keys())
    X = np.array([embeddings[i] for i in ids])

    result = {"n_samples": len(ids), "embedding_dim": X.shape[1]}

    # t-SNE
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(ids)-1))
        X_tsne = tsne.fit_transform(X)
        result["tsne"] = {
            ids[i]: [round(float(X_tsne[i, 0]), 4), round(float(X_tsne[i, 1]), 4)]
            for i in range(len(ids))
        }
    except Exception as e:
        result["tsne_error"] = str(e)

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(ids)-1))
        X_umap = reducer.fit_transform(X)
        result["umap"] = {
            ids[i]: [round(float(X_umap[i, 0]), 4), round(float(X_umap[i, 1]), 4)]
            for i in range(len(ids))
        }
    except Exception as e:
        result["umap_error"] = str(e)

    # 流派标签（用于着色）
    if not l4_df.empty and "genre" in l4_df.columns:
        genre_map = dict(zip(l4_df["audio_id"], l4_df["genre"]))
        result["genre_labels"] = {aid: genre_map.get(aid, "unknown") for aid in ids}

    # 保存可视化数据
    vis_path = os.path.join(output_dir, "embedding_visualization.json")
    with open(vis_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ========== 主流程 ==========

def run_stats(manifest_path, qc_report_path, l4_dir, mert_dir, output_dir):
    """主流程: 生成完整统计报告"""
    os.makedirs(output_dir, exist_ok=True)

    print("加载数据...")
    manifest, qc, l4_df, embeddings = load_data(manifest_path, qc_report_path, l4_dir, mert_dir)
    print(f"  manifest: {len(manifest)} 条")
    print(f"  qc: {len(qc)} 条")
    print(f"  l4: {len(l4_df)} 条")
    print(f"  embeddings: {len(embeddings)} 个")

    report = {
        "version": "v1.0.0",
        "generated_at": pd.Timestamp.now().isoformat(),
        "total_samples": int(len(manifest)),
    }

    # 分布统计
    print("\n计算分布统计...")
    report["genre_distribution"] = genre_distribution(l4_df)
    report["bpm_distribution"] = bpm_distribution(manifest, l4_df)
    report["key_distribution"] = key_distribution(l4_df)
    report["loudness_distribution"] = loudness_distribution(l4_df)
    report["duration_distribution"] = duration_distribution(manifest)
    report["vocal_distribution"] = vocal_distribution(l4_df, qc)
    report["source_distribution"] = source_distribution(manifest)

    # 质量统计
    print("计算质量统计...")
    report["quality_statistics"] = quality_statistics(qc)

    # 标注统计
    print("计算标注统计...")
    report["annotation_statistics"] = annotation_statistics(l4_df)

    # OOD 统计
    print("计算OOD统计...")
    report["ood_statistics"] = ood_statistics(manifest, embeddings)

    # 嵌入可视化
    print("计算嵌入可视化...")
    report["embedding_visualization"] = embedding_visualization(embeddings, l4_df, output_dir)

    # 保存完整报告
    report_path = os.path.join(output_dir, "dataset_stats_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"数据集统计报告生成完成: {report_path}")
    print(f"{'='*60}")
    print(f"  总样本数: {report['total_samples']}")
    print(f"  流派分布: {report['genre_distribution']}")
    if report["bpm_distribution"]:
        print(f"  BPM: mean={report['bpm_distribution'].get('mean')}, median={report['bpm_distribution'].get('median')}")
    if report["duration_distribution"]:
        print(f"  总时长: {report['duration_distribution'].get('total_minutes')} 分钟")
        print(f"  长曲(>15min): {report['duration_distribution'].get('long_form_count')}")
    if report["quality_statistics"]:
        q = report["quality_statistics"]
        print(f"  质量: pass={q.get('pass')}, marginal={q.get('marginal')}, fail={q.get('fail')}")
    if report["annotation_statistics"]:
        a = report["annotation_statistics"]
        print(f"  标注来源: {a.get('source_distribution')}")
    if report["ood_statistics"]:
        print(f"  OOD样本: {report['ood_statistics'].get('ood_count')}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集统计评测")
    parser.add_argument("--manifest", required=True, help="audio_manifest.csv 路径")
    parser.add_argument("--qc-report", default="", help="qc_gate_report.csv 路径")
    parser.add_argument("--l4-dir", default="", help="L4 融合标签目录")
    parser.add_argument("--mert-dir", default="", help="MERT 嵌入目录")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()

    run_stats(args.manifest, args.qc_report, args.l4_dir, args.mert_dir, args.output_dir)
