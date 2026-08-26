#!/usr/bin/env python3
"""
L4 KNN 传播（防泄漏版）
- 种子池：黄金集（高置信）+ CLAP zero-shot top-1（中置信）
- 距离度量：MERT 嵌入（768维）余弦相似度
- 防泄漏：只在 train 拟合，val 预测（标记伪标签），test/holdout/ood 不传播
- 多标签阈值：genre 0.40, mood 0.25, instruments 0.25
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# 多标签传播阈值
THRESHOLDS = {
    "genre": 0.40,
    "mood": 0.25,
    "instruments": 0.25,
}


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """加载 manifest，建立 hash_audioid -> ULID audio_id 映射"""
    df = pd.read_csv(manifest_path)
    # 从 master_path 提取文件名（不含扩展名）= hash_audioid
    hash_to_ulid = {}
    for _, row in df.iterrows():
        mp = row.get("master_path", "")
        if pd.notna(mp) and mp:
            hash_id = Path(mp).stem
            hash_to_ulid[hash_id] = row["audio_id"]
    print(f"[Manifest] 加载 {len(df)} 条记录，建立 {len(hash_to_ulid)} 个 hash→ULID 映射")
    return df, hash_to_ulid


def load_mert_embeddings(embedding_dir: Path, hash_to_ulid: dict) -> dict:
    """加载 MERT 嵌入，映射到 ULID audio_id"""
    embeddings = {}
    for f in embedding_dir.glob("*.npy"):
        hash_id = f.stem.replace("_mert_embedding", "")
        ulid = hash_to_ulid.get(hash_id)
        if ulid:
            emb = np.load(f)
            if emb.ndim > 1:
                emb = emb.flatten()[:768]
            embeddings[ulid] = emb
    print(f"[MERT] 加载 {len(embeddings)} 个嵌入（768维）")
    return embeddings


def load_clap_semantic(semantic_dir: Path, hash_to_ulid: dict) -> dict:
    """加载 CLAP zero-shot 结果，提取 genre/mood top-1，映射到 ULID"""
    labels = {}
    for f in semantic_dir.glob("*.json"):
        hash_id = f.stem.replace("_semantic", "")
        ulid = hash_to_ulid.get(hash_id)
        if ulid:
            with open(f) as fp:
                data = json.load(fp)
            label_dict = {}
            if data.get("genre"):
                label_dict["genre"] = data["genre"][0]["label"]
                label_dict["genre_score"] = data["genre"][0]["score"]
            if data.get("mood"):
                label_dict["mood"] = data["mood"][0]["label"]
                label_dict["mood_score"] = data["mood"][0]["score"]
            if label_dict:
                labels[ulid] = label_dict
    print(f"[CLAP] 加载 {len(labels)} 个语义标签（genre/mood top-1）")
    return labels


def load_golden_labels(golden_dir: Path) -> dict:
    """加载黄金集人工标注（高置信种子）"""
    labels = {}
    ann_dir = golden_dir / "annotations"
    if ann_dir.exists():
        for f in ann_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
            ulid = data.get("audio_id")
            if ulid:
                ann = data.get("annotation", {})
                label_dict = {}
                if ann.get("genre_level1"):
                    label_dict["genre"] = ann["genre_level1"]
                    label_dict["genre_confidence"] = "golden"
                if ann.get("mood_tags") and len(ann["mood_tags"]) > 0:
                    label_dict["mood"] = ann["mood_tags"][0]
                    label_dict["mood_confidence"] = "golden"
                if ann.get("instruments") and len(ann["instruments"]) > 0:
                    label_dict["instruments"] = ann["instruments"][0]
                    label_dict["instruments_confidence"] = "golden"
                if label_dict:
                    labels[ulid] = label_dict
    print(f"[Golden] 加载 {len(labels)} 个黄金集标注（高置信种子）")
    return labels


def load_splits(split_dir: Path) -> dict:
    """加载数据划分结果"""
    splits = {}
    for name in ["train", "val", "test", "holdout_gold"]:
        f = split_dir / f"{name}.csv"
        if f.exists():
            df = pd.read_csv(f)
            splits[name] = set(df["audio_id"].tolist())
            print(f"[Split] {name}: {len(splits[name])} 样本")
    return splits


def build_seed_pool(golden_labels: dict, clap_labels: dict, train_ids: set) -> dict:
    """构建种子池：黄金集（高置信）+ CLAP top-1（中置信），只包含 train 中的样本"""
    seeds = {}
    # 高置信：黄金集
    for ulid, labels in golden_labels.items():
        if ulid in train_ids:
            seeds[ulid] = {**labels, "source": "golden"}
    # 中置信：CLAP top-1（补充黄金集没有的标签类型）
    for ulid, labels in clap_labels.items():
        if ulid in train_ids and ulid not in seeds:
            seeds[ulid] = {**labels, "source": "clap_zero_shot"}
        elif ulid in train_ids and ulid in seeds:
            # 合并：黄金集优先，CLAP补充缺失的标签类型
            for k, v in labels.items():
                if k not in seeds[ulid]:
                    seeds[ulid][k] = v
    print(f"[Seed] 种子池: {len(seeds)} 个样本（train中）")
    return seeds


def propagate_labels(target_ids: set, seed_pool: dict, embeddings: dict,
                      nn_model: NearestNeighbors, seed_id_list: list,
                      source_prefix: str = "train") -> dict:
    """对目标样本传播标签"""
    results = {}
    target_with_emb = [aid for aid in target_ids if aid in embeddings and aid not in seed_pool]
    if not target_with_emb:
        return results

    target_embs = np.vstack([embeddings[aid] for aid in target_with_emb])
    distances, indices = nn_model.kneighbors(target_embs, n_neighbors=min(3, len(seed_id_list)))

    for i, aid in enumerate(target_with_emb):
        propagated = {}
        for j in range(len(seed_id_list)):
            if j >= indices.shape[1]:
                break
            nearest_seed = seed_id_list[indices[i][j]]
            dist = distances[i][j]
            seed_labels = seed_pool[nearest_seed]

            for label_type in ["genre", "mood", "instruments"]:
                if label_type in seed_labels and label_type not in propagated:
                    threshold = THRESHOLDS.get(label_type, 0.40)
                    if dist < threshold:
                        propagated[label_type] = {
                            "value": seed_labels[label_type],
                            "source": f"{source_prefix}_knn",
                            "seed": nearest_seed,
                            "distance": float(dist),
                            "threshold": threshold,
                        }
        if propagated:
            results[aid] = propagated
    return results


def main():
    parser = argparse.ArgumentParser(description="L4 KNN 传播（防泄漏版）")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--mert-dir", type=str, required=True, help="MERT 嵌入目录")
    parser.add_argument("--clap-dir", type=str, required=True, help="CLAP semantic 目录")
    parser.add_argument("--golden-dir", type=str, required=True, help="黄金集目录")
    parser.add_argument("--split-dir", type=str, required=True, help="数据划分目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    manifest_df, hash_to_ulid = load_manifest(Path(args.manifest))
    embeddings = load_mert_embeddings(Path(args.mert_dir), hash_to_ulid)
    clap_labels = load_clap_semantic(Path(args.clap_dir), hash_to_ulid)
    golden_labels = load_golden_labels(Path(args.golden_dir))
    splits = load_splits(Path(args.split_dir))

    train_ids = splits.get("train", set())
    val_ids = splits.get("val", set())
    test_ids = splits.get("test", set())
    holdout_ids = splits.get("holdout_gold", set())

    # 2. 构建种子池（只在 train 中）
    seed_pool = build_seed_pool(golden_labels, clap_labels, train_ids)

    # 3. 拟合 KNN（只在 train 种子上）
    seed_with_emb = [aid for aid in seed_pool if aid in embeddings]
    if len(seed_with_emb) < 2:
        print(f"[ERROR] train 中种子样本不足（{len(seed_with_emb)} 个），无法传播")
        return

    seed_embs = np.vstack([embeddings[aid] for aid in seed_with_emb])
    # 归一化（余弦相似度）
    norms = np.linalg.norm(seed_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    seed_embs_norm = seed_embs / norms

    n_neighbors = min(3, len(seed_with_emb))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn_model.fit(seed_embs_norm)
    print(f"[KNN] 拟合完成：{len(seed_with_emb)} 个种子，n_neighbors={n_neighbors}")

    # 4. 传播标签
    # train：传播未标注样本
    train_results = propagate_labels(train_ids, seed_pool, embeddings, nn_model, seed_with_emb, "train")
    print(f"[Train] 传播 {len(train_results)}/{len(train_ids)} 个样本")

    # val：传播（标记伪标签）
    val_results = propagate_labels(val_ids, seed_pool, embeddings, nn_model, seed_with_emb, "val_pseudo")
    print(f"[Val] 传播 {len(val_results)}/{len(val_ids)} 个样本（伪标签）")

    # test/holdout：不传播
    print(f"[Test] 不传播（{len(test_ids)} 个样本，保持标签纯净）")
    print(f"[Holdout] 不传播（{len(holdout_ids)} 个样本，保持标签纯净）")

    # 5. 保存结果
    with open(output_dir / "train_propagated.json", "w") as f:
        json.dump(train_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "val_propagated.json", "w") as f:
        json.dump(val_results, f, indent=2, ensure_ascii=False)

    # 6. 生成 manifest_with_l4.csv
    manifest_with_l4 = manifest_df.copy()
    manifest_with_l4["l4_genre"] = ""
    manifest_with_l4["l4_mood"] = ""
    manifest_with_l4["l4_instruments"] = ""
    manifest_with_l4["l4_source"] = ""

    for aid, labels in {**train_results, **val_results}.items():
        for label_type, info in labels.items():
            col = f"l4_{label_type}"
            if col in manifest_with_l4.columns:
                manifest_with_l4.loc[manifest_with_l4["audio_id"] == aid, col] = info["value"]
                manifest_with_l4.loc[manifest_with_l4["audio_id"] == aid, "l4_source"] = info["source"]

    # 种子样本也标记
    for aid, labels in seed_pool.items():
        for label_type in ["genre", "mood", "instruments"]:
            if label_type in labels:
                col = f"l4_{label_type}"
                if col in manifest_with_l4.columns:
                    manifest_with_l4.loc[manifest_with_l4["audio_id"] == aid, col] = labels[label_type]
                    manifest_with_l4.loc[manifest_with_l4["audio_id"] == aid, "l4_source"] = labels.get("source", "seed")

    manifest_with_l4.to_csv(output_dir / "manifest_with_l4.csv", index=False)
    print(f"\n[Output] manifest_with_l4.csv 已保存：{len(manifest_with_l4)} 行")
    print(f"  有 genre 标签: {(manifest_with_l4['l4_genre'] != '').sum()}")
    print(f"  有 mood 标签: {(manifest_with_l4['l4_mood'] != '').sum()}")
    print(f"  有 instruments 标签: {(manifest_with_l4['l4_instruments'] != '').sum()}")
    print(f"  test/holdout 不传播（保持纯净）: {len(test_ids) + len(holdout_ids)} 个样本")


if __name__ == "__main__":
    main()
