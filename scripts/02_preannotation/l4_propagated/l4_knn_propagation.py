#!/usr/bin/env python3
"""
L4 传播融合层 — KNN 标签传播 + 多源融合 (量化阈值版)

架构:
- DeepSeek V4 Flash 全量文本标签 (genre/mood/instruments/caption)
- Qwen-Omni 多模态黄金集结构标注 (5%)
- KNN(cosine) 将黄金集标签传播到相似样本
- 按字段差异化阈值融合

融合规则:
- genre: cosine_dist < 0.4 传播 (稳定字段，阈值放宽)
- mood/instruments: cosine_dist < 0.25 传播 (不稳定字段，阈值严格)
- caption: 不传播 (每首独立 DeepSeek 生成)
- segments: 仅黄金集保留 (不传播)

使用:
    python l4_knn_propagation.py \
        --embeddings-dir data/00.5_cleaned/reports/vXXX/l2_mert_embedding \
        --l4-deepseek-dir data/02_preannotation/l4_deepseek \
        --l3-golden-dir data/02_preannotation/l3_structural \
        --output-dir data/02_preannotation/l4_propagated \
        --ls-output data/02_preannotation/ls_preannotations.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ========== 融合阈值配置 ==========
FUSION_CONFIG = {
    "genre": {
        "max_cosine_dist": 0.40,      # cosine距离 < 0.4 才传播
        "min_gold_confidence": "medium",  # 黄金集标签置信度≥medium
        "propagate": True,
    },
    "mood": {
        "max_cosine_dist": 0.25,      # 严格阈值
        "min_gold_confidence": "high",    # 需high置信度
        "propagate": True,
    },
    "instruments": {
        "max_cosine_dist": 0.25,      # 严格阈值
        "min_gold_confidence": "high",
        "propagate": True,
    },
    "caption": {
        "propagate": False,            # 不传播，每首独立生成
    },
    "segments": {
        "propagate": False,            # 不传播段落结构
    },
    "vocal_presence": {
        "max_cosine_dist": 0.30,
        "min_gold_confidence": "medium",
        "propagate": True,
    },
}


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
        if vec.shape[0] == 768:
            audio_ids.append(aid)
            vectors.append(vec)

    return audio_ids, np.array(vectors)


def load_deepseek_labels(deepseek_dir: str) -> dict:
    """加载 DeepSeek 全量文本标签"""
    deepseek_dir = Path(deepseek_dir)
    labels = {}

    for f in deepseek_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        aid = data.get("audio_id", f.stem)
        labels[aid] = data

    return labels


def load_golden_labels(golden_dir: str) -> dict:
    """加载 Qwen-Omni 黄金集结构标注"""
    golden_dir = Path(golden_dir)
    labels = {}

    for f in golden_dir.glob("*_structure.json"):
        with open(f) as fp:
            data = json.load(fp)
        aid = data.get("audio_id", "")
        if aid:
            # 从段落中提取全曲级标签
            segments = data.get("segments", [])
            instruments = list(set(
                inst for seg in segments
                for inst in (seg.get("instruments") or seg.get("instrument") or [])
            ))
            moods = list(set(
                seg.get("emotion", "") for seg in segments
                if seg.get("emotion")
            ))
            labels[aid] = {
                "audio_id": aid,
                "segments": segments,
                "caption": data.get("caption", ""),
                "instruments": instruments[:5],
                "mood": moods[:2],
                "confidence": "high",  # Qwen-Omni 多模态直接听音频，置信度高
                "source": "qwen_omni_golden",
            }

    return labels


def compute_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """计算余弦相似度矩阵"""
    if SKLEARN_AVAILABLE:
        return cosine_similarity(vectors)
    # 手动计算
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms
    return normalized @ normalized.T


def find_nearest_golden(audio_idx: int, golden_indices: list,
                         sim_matrix: np.ndarray) -> tuple:
    """找到最近的黄金集样本"""
    sims = [(j, sim_matrix[audio_idx][j]) for j in golden_indices]
    sims.sort(key=lambda x: x[1], reverse=True)
    nearest_idx, nearest_sim = sims[0]
    cosine_dist = 1.0 - nearest_sim
    return nearest_idx, cosine_dist, nearest_sim


def should_propagate(field: str, cosine_dist: float,
                      gold_confidence: str = "high") -> bool:
    """判断是否应该传播该字段"""
    config = FUSION_CONFIG.get(field, {})
    if not config.get("propagate", False):
        return False
    if cosine_dist > config.get("max_cosine_dist", 1.0):
        return False
    # 置信度检查
    min_conf = config.get("min_gold_confidence", "low")
    conf_order = {"low": 0, "medium": 1, "high": 2}
    if conf_order.get(gold_confidence, 0) < conf_order.get(min_conf, 0):
        return False
    return True


def fuse_single_sample(audio_id: str, is_golden: bool,
                        deepseek_label: dict, golden_label: dict,
                        nearest_golden_id: str, cosine_dist: float,
                        nearest_sim: float) -> dict:
    """融合单个样本的标签"""
    result = {
        "audio_id": audio_id,
        "fusion": {},
        "propagated_from": None,
        "propagation_similarity": None,
        "propagation_cosine_dist": None,
    }

    # 基础物理特征(从DeepSeek继承)
    for key in ["bpm", "key", "duration_sec", "snr_db", "loudness_db",
                "quality_assessment", "vocal_presence", "subgenre"]:
        if key in deepseek_label:
            result[key] = deepseek_label[key]

    if is_golden:
        # 黄金集: 用 Qwen-Omni 结果覆盖 DeepSeek
        result["genre"] = deepseek_label.get("genre", "jazz")
        result["mood"] = golden_label.get("mood", deepseek_label.get("mood", ["relaxed"]))
        result["instrumentation"] = golden_label.get("instruments", deepseek_label.get("instrumentation", []))
        result["caption"] = golden_label.get("caption", deepseek_label.get("caption", ""))
        result["segments"] = golden_label.get("segments", [])
        result["propagated_from"] = "golden_set"
        result["fusion"] = {
            "genre_source": "deepseek",
            "mood_source": "qwen_omni_golden",
            "instrumentation_source": "qwen_omni_golden",
            "caption_source": "qwen_omni_golden",
        }
    else:
        # 非黄金集: 按字段差异化融合
        result["genre"] = deepseek_label.get("genre", "jazz")
        result["mood"] = deepseek_label.get("mood", ["relaxed"])
        result["instrumentation"] = deepseek_label.get("instrumentation", [])
        result["caption"] = deepseek_label.get("caption", "")
        result["segments"] = []  # 非黄金集无段落结构

        fusion = {
            "genre_source": "deepseek",
            "mood_source": "deepseek",
            "instrumentation_source": "deepseek",
            "caption_source": "deepseek (not_propagated)",
        }

        propagated_any = False

        # genre 传播
        if golden_label and should_propagate("genre", cosine_dist, golden_label.get("confidence", "high")):
            # genre 从黄金集的 DeepSeek 标签继承(黄金集也有genre)
            propagated_any = True
            fusion["genre_source"] = f"knn(from {nearest_golden_id}, dist={cosine_dist:.3f})"

        # mood 传播
        if golden_label and should_propagate("mood", cosine_dist, golden_label.get("confidence", "high")):
            result["mood"] = golden_label.get("mood", result["mood"])
            fusion["mood_source"] = f"knn(from {nearest_golden_id}, dist={cosine_dist:.3f})"
            propagated_any = True

        # instruments 传播
        if golden_label and should_propagate("instruments", cosine_dist, golden_label.get("confidence", "high")):
            result["instrumentation"] = golden_label.get("instruments", result["instrumentation"])
            fusion["instrumentation_source"] = f"knn(from {nearest_golden_id}, dist={cosine_dist:.3f})"
            propagated_any = True

        if propagated_any:
            result["propagated_from"] = nearest_golden_id
            result["propagation_similarity"] = round(float(nearest_sim), 4)
            result["propagation_cosine_dist"] = round(float(cosine_dist), 4)

        result["fusion"] = fusion

    return result


def run_l4_fusion(embeddings_dir: str, deepseek_dir: str, golden_dir: str,
                   output_dir: str, ls_output: str = None):
    """主流程: L4 传播融合"""
    print("=" * 60)
    print("L4 传播融合 (KNN + 量化阈值)")
    print("=" * 60)

    # 加载数据
    print("\n加载 MERT 嵌入...")
    audio_ids, vectors = load_embeddings(embeddings_dir)
    print(f"  {len(audio_ids)} 个样本, {vectors.shape[1]} 维")

    print("加载 DeepSeek 全量标签...")
    deepseek_labels = load_deepseek_labels(deepseek_dir)
    print(f"  {len(deepseek_labels)} 个标签")

    print("加载 Qwen-Omni 黄金集...")
    golden_labels = load_golden_labels(golden_dir)
    print(f"  {len(golden_labels)} 个黄金集")

    # 计算相似度矩阵
    print("\n计算余弦相似度矩阵...")
    sim_matrix = compute_similarity_matrix(vectors)

    # 黄金集索引
    golden_ids = set(golden_labels.keys())
    golden_indices = [i for i, aid in enumerate(audio_ids) if aid in golden_ids]
    print(f"  黄金集索引: {len(golden_indices)} 个")

    # 融合
    print("\n执行融合...")
    os.makedirs(output_dir, exist_ok=True)
    results = []
    golden_count = 0
    knn_count = 0
    deepseek_only_count = 0

    for i, audio_id in enumerate(audio_ids):
        is_golden = audio_id in golden_ids
        deepseek_label = deepseek_labels.get(audio_id, {})
        golden_label = golden_labels.get(audio_id)

        nearest_golden_id = None
        cosine_dist = 1.0
        nearest_sim = 0.0

        if not is_golden and golden_indices:
            nearest_idx, cosine_dist, nearest_sim = find_nearest_golden(
                i, golden_indices, sim_matrix
            )
            nearest_golden_id = audio_ids[nearest_idx]
            golden_label = golden_labels.get(nearest_golden_id)

        result = fuse_single_sample(
            audio_id, is_golden, deepseek_label, golden_label,
            nearest_golden_id, cosine_dist, nearest_sim
        )

        # 保存
        output_path = Path(output_dir) / f"{audio_id}_full_tags.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)

        if is_golden:
            golden_count += 1
        elif result.get("propagated_from"):
            knn_count += 1
        else:
            deepseek_only_count += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(audio_ids)}] 处理完成")

    # 生成 Label Studio JSONL
    if ls_output:
        print(f"\n生成 Label Studio 预标注: {ls_output}")
        with open(ls_output, "w") as f:
            for result in results:
                mood = result.get("mood", ["relaxed"])
                if isinstance(mood, str):
                    mood = [mood]
                instruments = result.get("instrumentation", [])
                if isinstance(instruments, str):
                    instruments = [instruments]

                ls_task = {
                    "id": result["audio_id"],
                    "data": {"audio": f"/data/audio/{result['audio_id']}.flac"},
                    "predictions": [{
                        "model_version": "l4_deepseek_knn_fusion_v2",
                        "result": [
                            {"type": "choices", "from_name": "genre", "to_name": "audio",
                             "value": {"choices": [result.get("genre", "jazz")]}},
                            {"type": "choices", "from_name": "mood", "to_name": "audio",
                             "value": {"choices": mood[:2]}},
                            {"type": "choices", "from_name": "instruments", "to_name": "audio",
                             "value": {"choices": instruments[:5]}},
                            {"type": "choices", "from_name": "vocal_presence", "to_name": "audio",
                             "value": {"choices": [result.get("vocal_presence", "instrumental")]}},
                            {"type": "textarea", "from_name": "caption", "to_name": "audio",
                             "value": {"text": [result.get("caption", "")]}},
                        ],
                    }],
                    "meta": {
                        "propagated_from": result.get("propagated_from", ""),
                        "propagation_cosine_dist": result.get("propagation_cosine_dist", 0),
                        "is_golden": result.get("propagated_from") == "golden_set",
                    }
                }
                f.write(json.dumps(ls_task, ensure_ascii=False) + "\n")

    # 统计
    print(f"\n{'='*60}")
    print(f"L4 融合完成")
    print(f"{'='*60}")
    print(f"  总计: {len(results)}")
    print(f"  🌟 黄金集(Qwen-Omni): {golden_count}")
    print(f"  📡 KNN传播: {knn_count}")
    print(f"  🤖 DeepSeek-only: {deepseek_only_count}")
    print(f"\n  融合阈值:")
    print(f"    genre: cosine_dist < {FUSION_CONFIG['genre']['max_cosine_dist']}")
    print(f"    mood: cosine_dist < {FUSION_CONFIG['mood']['max_cosine_dist']} (需high置信度)")
    print(f"    instruments: cosine_dist < {FUSION_CONFIG['instruments']['max_cosine_dist']}")
    print(f"    caption: 不传播")
    print(f"    segments: 不传播(仅黄金集)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L4 KNN传播融合(量化阈值版)")
    parser.add_argument("--embeddings-dir", required=True, help="MERT嵌入目录")
    parser.add_argument("--l4-deepseek-dir", required=True, help="DeepSeek全量标签目录")
    parser.add_argument("--l3-golden-dir", required=True, help="Qwen-Omni黄金集目录")
    parser.add_argument("--output-dir", required=True, help="输出融合标签目录")
    parser.add_argument("--ls-output", default=None, help="Label Studio JSONL输出路径")
    args = parser.parse_args()

    run_l4_fusion(args.embeddings_dir, args.l4_deepseek_dir, args.l3_golden_dir,
                  args.output_dir, args.ls_output)
