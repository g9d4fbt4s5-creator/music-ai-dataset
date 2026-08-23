"""
run_end_to_end_preannotation.py
端到端预标注流水线：L1 → L2 → L3(mock) → L4 → Label Studio

功能：
- 加载 L1 物理特征（BPM/调性/LUFS等）
- 加载 L2 嵌入 + 语义候选（CLAP zero-shot）
- L3 mock：选 5% 黄金集，用 L2 top-3 标签生成结构化标签
- L4 KNN 传播：黄金种子标签传播到全量
- 输出 ls_preannotations.jsonl（Label Studio 格式）

用法：
    python run_end_to_end_preannotation.py \
        --l1-dir data/02_preannotation/l1_physical \
        --l2-embedding-dir data/02_preannotation/l2_embedding \
        --l2-semantic-dir data/02_preannotation/l2_semantic \
        --output-dir data/02_preannotation/l4_propagated \
        --ls-output data/02_preannotation/ls_preannotations.jsonl \
        --golden-ratio 0.1 \
        --k 5
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
from collections import defaultdict

# ===================== 配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ===================== 数据加载 =====================

def load_l1_features(l1_dir: Path) -> Dict[str, Dict]:
    """加载 L1 物理特征"""
    features = {}
    for f in l1_dir.glob("*_physical.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        raw_id = data.get("audio_id", f.stem.replace("_physical", ""))

        # 提取纯 audio_id：如果格式是 {hash}_{ulid}，取后半部分
        # ULID 是 26 字符的 base32 字符串
        if "_" in raw_id:
            parts = raw_id.split("_")
            # 找最后一个看起来像 ULID 的部分（26字符，base32）
            for part in reversed(parts):
                if len(part) == 26 and all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in part.upper()):
                    raw_id = part
                    break

        data["audio_id"] = raw_id
        features[raw_id] = data
    logger.info(f"L1 物理特征: {len(features)} 首")
    return features


def load_l2_embeddings(embedding_dir: Path) -> Dict[str, np.ndarray]:
    """加载 L2 嵌入向量"""
    embeddings = {}
    for f in embedding_dir.glob("*_embedding.npy"):
        audio_id = f.stem.replace("_embedding", "")
        embeddings[audio_id] = np.load(f)
    logger.info(f"L2 嵌入向量: {len(embeddings)} 首")
    return embeddings


def load_l2_semantic(semantic_dir: Path) -> Dict[str, Dict]:
    """加载 L2 语义候选"""
    semantics = {}
    for f in semantic_dir.glob("*_semantic.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        audio_id = data.get("audio_id", f.stem.replace("_semantic", ""))
        semantics[audio_id] = data
    logger.info(f"L2 语义候选: {len(semantics)} 首")
    return semantics


# ===================== L3 Mock：黄金种子生成 =====================

def generate_golden_seeds(l1_features: Dict, l2_semantics: Dict,
                            golden_ratio: float = 0.1) -> Tuple[List[str], Dict[str, Dict]]:
    """
    生成黄金种子集（mock L3）

    选置信度最高的 N% 作为黄金种子，用 L2 top-3 标签生成结构化标签。

    Args:
        l1_features: L1 物理特征
        l2_semantics: L2 语义候选
        golden_ratio: 黄金集比例（默认 10%）

    Returns:
        (golden_ids, golden_labels)
    """
    # 计算每首的置信度（genre top-1 score）
    confidence = {}
    for audio_id, sem in l2_semantics.items():
        genre_top = sem.get("genre", [{}])[0]
        confidence[audio_id] = genre_top.get("score", 0)

    # 按置信度排序，选 top N%
    sorted_ids = sorted(confidence, key=confidence.get, reverse=True)
    n_golden = max(1, int(len(sorted_ids) * golden_ratio))
    golden_ids = sorted_ids[:n_golden]

    logger.info(f"黄金种子集: {n_golden} 首 (top {golden_ratio*100:.0f}% 置信度)")

    # 生成黄金标签
    golden_labels = {}
    for audio_id in golden_ids:
        sem = l2_semantics.get(audio_id, {})
        l1 = l1_features.get(audio_id, {})

        # 用 L2 top-3 标签 + L1 特征生成结构化标签
        genre_top3 = [g["label"] for g in sem.get("genre", [])[:3]]
        mood_top3 = [m["label"] for m in sem.get("mood", [])[:3]]
        instrument_top3 = [i["label"] for i in sem.get("instrumentation", [])[:3]]
        scene_top3 = [s["label"] for s in sem.get("scene", [])[:3]]

        label = {
            "audio_id": audio_id,
            "is_golden_seed": True,
            "genre": genre_top3[0] if genre_top3 else "unknown",
            "genre_candidates": genre_top3,
            "mood": mood_top3[:2],
            "instrumentation": instrument_top3[:3],
            "scene": scene_top3[0] if scene_top3 else "unknown",
            "bpm": l1.get("bpm"),
            "key": l1.get("key"),
            "lufs": l1.get("lufs"),
            "duration_sec": l1.get("duration_sec"),
            "vocal_presence": "instrumental",  # Jazz 默认纯器乐
            "caption": f"A {genre_top3[0] if genre_top3 else 'jazz'} piece with "
                       f"{', '.join(instrument_top3[:2]) if instrument_top3 else 'various instruments'}, "
                       f"{' and '.join(mood_top3[:2]) if mood_top3 else 'mixed mood'} mood, "
                       f"BPM {l1.get('bpm', 'unknown')}, key {l1.get('key', 'unknown')}.",
            "source": "l2_mock_golden_seed",
        }
        golden_labels[audio_id] = label

    return golden_ids, golden_labels


# ===================== L4 KNN 传播 =====================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def knn_propagate(embeddings: Dict[str, np.ndarray],
                   golden_labels: Dict[str, Dict],
                   golden_ids: List[str],
                   k: int = 5) -> Dict[str, Dict]:
    """
    KNN 标签传播

    对每个非黄金样本，找 K 个最近邻黄金样本，按距离加权投票。

    Args:
        embeddings: 所有样本的嵌入向量
        golden_labels: 黄金样本的标签
        golden_ids: 黄金样本 ID 列表
        k: KNN 邻居数

    Returns:
        所有样本的传播后标签
    """
    all_ids = list(embeddings.keys())
    propagated = {}

    for audio_id in all_ids:
        if audio_id in golden_labels:
            # 黄金样本直接用其标签
            propagated[audio_id] = golden_labels[audio_id].copy()
            propagated[audio_id]["propagation_method"] = "golden_seed"
            continue

        # 计算与所有黄金样本的相似度
        similarities = []
        for gid in golden_ids:
            sim = cosine_similarity(embeddings[audio_id], embeddings[gid])
            similarities.append((gid, sim))

        # 取 top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 加权投票
        genre_votes = defaultdict(float)
        mood_votes = defaultdict(float)
        instrument_votes = defaultdict(float)
        total_weight = 0

        for gid, sim in top_k:
            weight = max(sim, 0.01)  # 避免负权重
            label = golden_labels[gid]

            genre_votes[label.get("genre", "unknown")] += weight
            for m in label.get("mood", []):
                mood_votes[m] += weight
            for inst in label.get("instrumentation", []):
                instrument_votes[inst] += weight
            total_weight += weight

        # 归一化，取 top
        propagated_label = {
            "audio_id": audio_id,
            "is_golden_seed": False,
            "propagation_method": "knn_weighted",
            "knn_k": k,
            "knn_neighbors": [{"id": gid, "similarity": round(sim, 4)} for gid, sim in top_k],
        }

        if genre_votes:
            top_genre = max(genre_votes, key=genre_votes.get)
            propagated_label["genre"] = top_genre
            propagated_label["genre_confidence"] = round(genre_votes[top_genre] / total_weight, 4)
            propagated_label["genre_candidates"] = sorted(genre_votes, key=genre_votes.get, reverse=True)[:3]

        if mood_votes:
            propagated_label["mood"] = sorted(mood_votes, key=mood_votes.get, reverse=True)[:2]

        if instrument_votes:
            propagated_label["instrumentation"] = sorted(instrument_votes, key=instrument_votes.get, reverse=True)[:3]

        propagated[audio_id] = propagated_label

    logger.info(f"L4 KNN 传播完成: {len(propagated)} 首 "
                f"({len(golden_ids)} 黄金种子 + {len(propagated) - len(golden_ids)} 传播)")

    return propagated


# ===================== Label Studio 输出 =====================

def convert_to_label_studio(propagated: Dict[str, Dict],
                              l1_features: Dict[str, Dict],
                              audio_dir: Optional[Path] = None) -> List[Dict]:
    """
    转换为 Label Studio 格式

    Args:
        propagated: 传播后标签
        l1_features: L1 物理特征
        audio_dir: 音频目录（用于生成音频路径）

    Returns:
        Label Studio tasks 列表
    """
    tasks = []

    for audio_id, label in propagated.items():
        l1 = l1_features.get(audio_id, {})

        # 构建预标注结果
        predictions = [{
            "model_version": "preannotation_v1.0",
            "result": [
                # 流派
                {
                    "from_name": "genre",
                    "to_name": "audio",
                    "type": "choices",
                    "value": {"choices": [label.get("genre", "unknown")]}
                },
                # 情绪
                {
                    "from_name": "mood",
                    "to_name": "audio",
                    "type": "choices",
                    "value": {"choices": label.get("mood", [])}
                },
                # 乐器
                {
                    "from_name": "instrumentation",
                    "to_name": "audio",
                    "type": "taxonomy",
                    "value": {"taxonomy": [[inst] for inst in label.get("instrumentation", [])]}
                },
                # 描述
                {
                    "from_name": "caption",
                    "to_name": "audio",
                    "type": "textarea",
                    "value": {"text": [label.get("caption", "")]}
                },
            ]
        }]

        task = {
            "id": audio_id,
            "data": {
                "audio": f"/data/local-files/?d={audio_id}.mp3" if audio_dir else audio_id,
                "audio_id": audio_id,
                "bpm": l1.get("bpm"),
                "key": l1.get("key"),
                "lufs": l1.get("lufs"),
                "duration_sec": l1.get("duration_sec"),
                "is_golden_seed": label.get("is_golden_seed", False),
                "propagation_method": label.get("propagation_method", "unknown"),
            },
            "predictions": predictions,
        }
        tasks.append(task)

    logger.info(f"Label Studio tasks: {len(tasks)} 个")
    return tasks


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="端到端预标注流水线：L1 → L2 → L3(mock) → L4 → Label Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--l1-dir", type=str, default="data/02_preannotation/l1_physical",
                        help="L1 物理特征目录")
    parser.add_argument("--l2-embedding-dir", type=str, default="data/02_preannotation/l2_embedding",
                        help="L2 嵌入向量目录")
    parser.add_argument("--l2-semantic-dir", type=str, default="data/02_preannotation/l2_semantic",
                        help="L2 语义候选目录")
    parser.add_argument("--output-dir", type=str, default="data/02_preannotation/l4_propagated",
                        help="L4 传播结果输出目录")
    parser.add_argument("--ls-output", type=str, default="data/02_preannotation/ls_preannotations.jsonl",
                        help="Label Studio 输出文件")
    parser.add_argument("--golden-ratio", type=float, default=0.1,
                        help="黄金集比例（默认 10%%）")
    parser.add_argument("--k", type=int, default=5,
                        help="KNN 邻居数（默认 5）")
    args = parser.parse_args()

    l1_dir = Path(args.l1_dir)
    l2_embedding_dir = Path(args.l2_embedding_dir)
    l2_semantic_dir = Path(args.l2_semantic_dir)
    output_dir = Path(args.output_dir)
    ls_output = Path(args.ls_output)

    output_dir.mkdir(parents=True, exist_ok=True)
    ls_output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("端到端预标注流水线")
    logger.info("=" * 60)

    # Step 1: 加载 L1
    logger.info("\n[Step 1] 加载 L1 物理特征...")
    l1_features = load_l1_features(l1_dir)

    # Step 2: 加载 L2
    logger.info("\n[Step 2] 加载 L2 嵌入 + 语义候选...")
    embeddings = load_l2_embeddings(l2_embedding_dir)
    l2_semantics = load_l2_semantic(l2_semantic_dir)

    # 取交集（同时有 L1 和 L2 的样本）
    common_ids = set(l1_features.keys()) & set(embeddings.keys()) & set(l2_semantics.keys())
    logger.info(f"共同样本数: {len(common_ids)}")

    l1_features = {k: v for k, v in l1_features.items() if k in common_ids}
    embeddings = {k: v for k, v in embeddings.items() if k in common_ids}
    l2_semantics = {k: v for k, v in l2_semantics.items() if k in common_ids}

    # Step 3: L3 mock - 生成黄金种子
    logger.info("\n[Step 3] L3 mock - 生成黄金种子集...")
    golden_ids, golden_labels = generate_golden_seeds(l1_features, l2_semantics, args.golden_ratio)

    # 保存黄金种子
    golden_dir = output_dir / "golden_seeds"
    golden_dir.mkdir(exist_ok=True)
    for gid, glabel in golden_labels.items():
        with open(golden_dir / f"{gid}_golden.json", "w", encoding="utf-8") as f:
            json.dump(glabel, f, ensure_ascii=False, indent=2)

    # Step 4: L4 KNN 传播
    logger.info("\n[Step 4] L4 KNN 标签传播...")
    propagated = knn_propagate(embeddings, golden_labels, golden_ids, args.k)

    # 保存传播结果
    for audio_id, label in propagated.items():
        with open(output_dir / f"{audio_id}_full_tags.json", "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

    # 保存汇总 CSV
    df = pd.DataFrame([
        {
            "audio_id": aid,
            "genre": label.get("genre"),
            "genre_confidence": label.get("genre_confidence"),
            "mood": ", ".join(label.get("mood", [])),
            "instrumentation": ", ".join(label.get("instrumentation", [])),
            "is_golden_seed": label.get("is_golden_seed"),
            "propagation_method": label.get("propagation_method"),
        }
        for aid, label in propagated.items()
    ])
    df.to_csv(output_dir / "_all_propagated_tags.csv", index=False, encoding="utf-8")
    logger.info(f"汇总 CSV: {output_dir / '_all_propagated_tags.csv'}")

    # Step 5: 转换为 Label Studio 格式
    logger.info("\n[Step 5] 转换为 Label Studio 格式...")
    tasks = convert_to_label_studio(propagated, l1_features)

    with open(ls_output, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    logger.info(f"Label Studio 输出: {ls_output}")

    # 统计
    logger.info("\n" + "=" * 60)
    logger.info("流水线完成")
    logger.info("=" * 60)
    logger.info(f"  总样本数: {len(propagated)}")
    logger.info(f"  黄金种子: {len(golden_ids)}")
    logger.info(f"  KNN 传播: {len(propagated) - len(golden_ids)}")
    logger.info(f"  K 值: {args.k}")
    logger.info(f"  流派分布:")
    genre_dist = defaultdict(int)
    for label in propagated.values():
        genre_dist[label.get("genre", "unknown")] += 1
    for genre, count in sorted(genre_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    {genre}: {count}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  LS 文件: {ls_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
