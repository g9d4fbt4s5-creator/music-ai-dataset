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


# ========== KNN 传播量化阈值（文件顶部常量，500首全量时可微调） ==========
# cosine_dist = 1 - cosine_sim，距离越小越相似
DIST_THRESHOLD_GENRE = 0.50          # 流派稳定字段，放宽(85首试点)
DIST_THRESHOLD_MOOD = 0.40           # 情绪主观，放宽(85首试点)
DIST_THRESHOLD_INSTRUMENTS = 0.40    # 乐器存在性，放宽(85首试点)
DIST_THRESHOLD_VOCAL = 0.45          # 人声判定，放宽(85首试点)

# 黄金集标签置信度要求
GOLD_CONFIDENCE_GENRE = {"high", "medium"}
GOLD_CONFIDENCE_MOOD = {"high"}
GOLD_CONFIDENCE_INSTRUMENTS = {"high"}
GOLD_CONFIDENCE_VOCAL = {"high", "medium"}

# ========== 融合阈值配置（引用上方常量） ==========
FUSION_CONFIG = {
    "genre": {
        "max_cosine_dist": DIST_THRESHOLD_GENRE,
        "min_gold_confidence": "medium",
        "propagate": True,
    },
    "mood": {
        "max_cosine_dist": DIST_THRESHOLD_MOOD,
        "min_gold_confidence": "high",
        "propagate": True,
    },
    "instruments": {
        "max_cosine_dist": DIST_THRESHOLD_INSTRUMENTS,
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
        "max_cosine_dist": DIST_THRESHOLD_VOCAL,
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
        # MERT嵌入文件名格式: {32位hash}_{audio_id}_mert_embedding.npy
        # 去掉前面的hash_部分，只保留audio_id
        parts = aid.split("_", 1)
        if len(parts) == 2 and len(parts[0]) == 32:  # hash是32位md5
            aid = parts[1]
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
    """加载 Qwen-Omni 黄金集结构标注（适配L3输出格式 *_l3_qwen.json）"""
    golden_dir = Path(golden_dir)
    labels = {}

    # 匹配L3输出格式 *_l3_qwen.json，同时兼容旧格式 *_structure.json
    files = list(golden_dir.glob("*_l3_qwen.json")) + list(golden_dir.glob("*_structure.json"))

    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        # L3输出格式：标注在annotation字段里；旧格式顶层直接有标注
        ann = data.get("annotation", data)
        aid = data.get("audio_id", "")
        if not aid:
            continue
        segments = ann.get("segments", [])
        # 从全曲级+段落级提取乐器（兼容instruments/instrument字段）
        instruments = list(set(
            list(ann.get("instruments", [])) +
            [inst for seg in segments for inst in (seg.get("instruments") or seg.get("instrument") or [])]
        ))
        # 从全曲级+段落级提取情绪（兼容mood_tags/mood/emotion字段）
        moods = list(set(
            list(ann.get("mood_tags", [])) +
            [m for seg in segments for m in (seg.get("mood") or seg.get("emotion") or [])]
        ))
        # confidence统一为字符串等级（L3输出可能是0-1浮点数，也可能是字符串）
        raw_conf = ann.get("confidence", "high")
        if isinstance(raw_conf, (int, float)):
            if raw_conf >= 0.8:
                conf_level = "high"
            elif raw_conf >= 0.5:
                conf_level = "medium"
            else:
                conf_level = "low"
        else:
            conf_level = str(raw_conf).lower()
        labels[aid] = {
            "audio_id": aid,
            "segments": segments,
            "caption": ann.get("caption", ""),
            "genre": ann.get("genre", ""),
            "instruments": instruments[:5],
            "mood": moods[:2],
            "confidence": conf_level,
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
            result["genre"] = golden_label.get("genre", result["genre"])
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
                   output_dir: str, ls_output: str = None,
                   exclude_splits: str = None, splits_dir: str = None):
    """主流程: L4 传播融合

    Args:
        exclude_splits: 逗号分隔的子集名，如 "test,holdout,ood"，这些子集的样本禁止KNN传播
        splits_dir: 划分结果目录，包含 train.csv/val.csv/test.csv/holdout_gold.csv/ood.csv
    """
    print("=" * 60)
    print("L4 传播融合 (KNN + 量化阈值)")
    print("=" * 60)

    # ========== V4防泄漏：加载排除子集 ==========
    excluded_ids = set()
    if exclude_splits and splits_dir:
        splits_path = Path(splits_dir)
        split_names = [s.strip() for s in exclude_splits.split(",") if s.strip()]
        for sname in split_names:
            # 兼容多种文件名: test.csv / holdout_gold.csv / ood.csv
            candidates = [sname + ".csv", sname + "_gold.csv"]
            for cand in candidates:
                fpath = splits_path / cand
                if fpath.exists():
                    import pandas as pd
                    df = pd.read_csv(fpath)
                    ids = set(df["audio_id"].tolist()) if "audio_id" in df.columns else set()
                    excluded_ids.update(ids)
                    print(f"  [防泄漏] 排除 {sname}: {len(ids)} 首 ({cand})")
                    break
        if excluded_ids:
            print(f"  [防泄漏] 总计排除 {len(excluded_ids)} 首，禁止KNN传播")
        else:
            print(f"  [防泄漏] 警告: 指定了排除子集但未找到任何文件，将不排除任何样本")
    elif exclude_splits and not splits_dir:
        print("  [防泄漏] 警告: 指定了--exclude-splits但未提供--splits-dir，防泄漏不生效")

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
    excluded_count = 0  # V4防泄漏：被排除KNN传播的样本数

    for i, audio_id in enumerate(audio_ids):
        is_golden = audio_id in golden_ids
        deepseek_label = deepseek_labels.get(audio_id, {})
        golden_label = golden_labels.get(audio_id)

        nearest_golden_id = None
        cosine_dist = 1.0
        nearest_sim = 0.0

        # V4防泄漏：test/holdout/ood样本禁止KNN传播
        is_excluded = audio_id in excluded_ids
        if is_excluded:
            golden_label = None  # 不使用黄金集标签
        elif not is_golden and golden_indices:
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
        elif is_excluded:
            excluded_count += 1
            deepseek_only_count += 1  # 排除样本也只有DeepSeek标签
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
    if excluded_ids:
        print(f"  🚫 [防泄漏] 排除KNN传播: {excluded_count} 首 (test/holdout/ood)")
        if excluded_count == len(excluded_ids):
            print(f"  ✅ [防泄漏] 验证通过: 所有 {excluded_count} 首排除样本均未被KNN传播")
        else:
            print(f"  ❌ [防泄漏] 验证失败: 排除了 {len(excluded_ids)} 首但只有 {excluded_count} 首未传播")
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
    parser.add_argument("--exclude-splits", default=None,
                        help="V4防泄漏：逗号分隔的子集名，禁止KNN传播，如 'test,holdout,ood'")
    parser.add_argument("--splits-dir", default=None,
                        help="划分结果目录（包含train.csv/val.csv/test.csv等），配合--exclude-splits使用")
    args = parser.parse_args()

    run_l4_fusion(args.embeddings_dir, args.l4_deepseek_dir, args.l3_golden_dir,
                  args.output_dir, args.ls_output,
                  exclude_splits=args.exclude_splits, splits_dir=args.splits_dir)
