"""
l4_knn_propagation.py
L4 传播层：KNN 标签传播 + 多源融合

功能：
- 读取 L2 CLAP 嵌入（GPU端产出）
- 读取 L3 文本标签（DeepSeek V4 Flash，全量）
- 读取 L3 纠错标签（DeepSeek V4 Pro，10%抽样）
- 读取 L3 音频结构（Qwen-Audio，5%黄金集）
- KNN 传播：从黄金集标签传播到全量
- 多源融合：加权投票整合各层标签
- 输出最终预标注 JSONL（Label Studio 可导入）

用法：
    # 完整 KNN 传播 + 融合
    python l4_knn_propagation.py \
        --embeddings-dir data/02_preannotation/model_output_cache/clap_embeddings \
        --l3-text-dir data/02_preannotation/l3_structural/text_labels \
        --l3-corrected-dir data/02_preannotation/l3_structural/corrected_labels \
        --l3-audio-dir data/02_preannotation/l3_structural/audio_structure \
        --output data/02_preannotation/l4_propagated \
        --config configs/preannotation/preannotation_config.yaml

    # 只做融合（不做KNN传播）
    python l4_knn_propagation.py \
        --l3-text-dir data/02_preannotation/l3_structural/text_labels \
        --output data/02_preannotation/l4_propagated \
        --no-knn

    # 输出 Label Studio 预标注格式
    python l4_knn_propagation.py \
        --embeddings-dir data/02_preannotation/model_output_cache/clap_embeddings \
        --l3-text-dir data/02_preannotation/l3_structural/text_labels \
        --output data/02_preannotation/l4_propagated \
        --ls-output data/02_preannotation/ls_preannotations.jsonl
"""
import os
import sys
import json
import yaml
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ===================== 工具函数 =====================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(file_path: Path) -> Optional[Dict]:
    """加载 JSON 文件"""
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"JSON 加载失败 {file_path}: {e}")
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ===================== 数据加载 =====================

def load_embeddings(embeddings_dir: Path) -> Dict[str, np.ndarray]:
    """加载 CLAP 嵌入"""
    embeddings = {}
    if not embeddings_dir.exists():
        logger.warning(f"嵌入目录不存在: {embeddings_dir}")
        return embeddings

    for file_path in embeddings_dir.glob("*.npy"):
        sample_id = file_path.stem
        try:
            embeddings[sample_id] = np.load(file_path)
        except Exception as e:
            logger.warning(f"嵌入加载失败 {file_path}: {e}")

    # 也支持 JSON 格式嵌入
    for file_path in embeddings_dir.glob("*.json"):
        sample_id = file_path.stem
        if sample_id not in embeddings:
            data = load_json(file_path)
            if data and "embedding" in data:
                embeddings[sample_id] = np.array(data["embedding"])

    logger.info(f"加载嵌入: {len(embeddings)} 个")
    return embeddings


def load_labels_from_dir(labels_dir: Path, label_key: str = "labels") -> Dict[str, Dict]:
    """从目录加载标签"""
    labels = {}
    if not labels_dir.exists():
        logger.warning(f"标签目录不存在: {labels_dir}")
        return labels

    for file_path in labels_dir.glob("*.json"):
        if file_path.name.startswith("_"):
            continue
        sample_id = file_path.stem
        data = load_json(file_path)
        if data:
            labels[sample_id] = data.get(label_key, data)

    logger.info(f"加载标签 ({label_key}): {len(labels)} 个")
    return labels


# ===================== KNN 传播 =====================

def knn_propagate(target_id: str, target_embedding: np.ndarray,
                  seed_embeddings: Dict[str, np.ndarray],
                  seed_labels: Dict[str, Dict],
                  k: int = 5, distance_metric: str = "cosine") -> Optional[Dict]:
    """
    KNN 标签传播

    Args:
        target_id: 目标样本 ID
        target_embedding: 目标样本嵌入
        seed_embeddings: 种子样本嵌入
        seed_labels: 种子样本标签
        k: K 近邻数
        distance_metric: 距离度量

    Returns:
        传播后的标签，失败返回 None
    """
    if not seed_embeddings or not seed_labels:
        return None

    # 计算相似度
    similarities = []
    for seed_id, seed_emb in seed_embeddings.items():
        if seed_id == target_id:
            continue
        if distance_metric == "cosine":
            sim = cosine_similarity(target_embedding, seed_emb)
        else:
            # 欧氏距离转相似度
            dist = np.linalg.norm(target_embedding - seed_emb)
            sim = 1.0 / (1.0 + dist)
        similarities.append((seed_id, sim))

    # 取 top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]

    if not top_k:
        return None

    # 加权投票
    label_weights = defaultdict(lambda: defaultdict(float))
    total_weight = 0.0

    for seed_id, sim in top_k:
        if seed_id not in seed_labels:
            continue
        weight = sim  # 距离加权
        total_weight += weight

        seed_label = seed_labels[seed_id]
        for label_type, label_value in seed_label.items():
            if isinstance(label_value, list):
                for item in label_value:
                    label_weights[label_type][item] += weight
            elif isinstance(label_value, (str, int, float, bool)):
                label_weights[label_type][str(label_value)] += weight

    if total_weight == 0:
        return None

    # 归一化，取 top-k 标签
    propagated = {}
    for label_type, value_weights in label_weights.items():
        sorted_values = sorted(value_weights.items(), key=lambda x: x[1], reverse=True)
        if label_type in ["genre", "mood", "instrumentation"]:
            # 列表类型：取 top-3
            propagated[label_type] = [v for v, w in sorted_values[:3]]
        else:
            # 单值类型：取最高权重
            propagated[label_type] = sorted_values[0][0] if sorted_values else None

    # 添加传播元数据
    propagated["_propagation_info"] = {
        "method": "knn",
        "k": k,
        "n_seeds": len(top_k),
        "avg_similarity": float(np.mean([sim for _, sim in top_k])),
        "neighbors": [{"id": sid, "similarity": sim} for sid, sim in top_k],
    }

    return propagated


# ===================== 多源融合 =====================

def fuse_labels(l2_labels: Optional[Dict],
                l3_text_labels: Optional[Dict],
                l3_corrected_labels: Optional[Dict],
                l3_audio_labels: Optional[Dict],
                knn_labels: Optional[Dict],
                weights: Dict) -> Dict:
    """
    多源标签融合（加权投票）

    Args:
        l2_labels: L2 CLAP zero-shot 标签
        l3_text_labels: L3 DeepSeek 文本标签
        l3_corrected_labels: L3 DeepSeek Pro 纠错标签
        l3_audio_labels: L3 Qwen-Audio 音频结构标签
        knn_labels: KNN 传播标签
        weights: 各源权重

    Returns:
        融合后的标签
    """
    # 收集所有可用标签源
    sources = []
    if l2_labels:
        sources.append(("l2_semantic", l2_labels, weights.get("l2_semantic", 0.3)))
    if l3_text_labels:
        sources.append(("l3_text", l3_text_labels, weights.get("l3_text", 0.4)))
    if l3_corrected_labels:
        sources.append(("l3_corrected", l3_corrected_labels, weights.get("l3_corrected", 0.2)))
    if l3_audio_labels:
        sources.append(("l3_audio", l3_audio_labels, weights.get("l3_audio", 0.1)))
    if knn_labels:
        sources.append(("knn", knn_labels, 0.2))  # KNN 传播额外权重

    if not sources:
        return {}

    # 加权投票
    label_weights = defaultdict(lambda: defaultdict(float))
    total_weights = defaultdict(float)

    for source_name, labels, weight in sources:
        # 跳过元数据字段
        clean_labels = {k: v for k, v in labels.items() if not k.startswith("_")}

        for label_type, label_value in clean_labels.items():
            if isinstance(label_value, list):
                for item in label_value:
                    label_weights[label_type][str(item)] += weight
                    total_weights[label_type] += weight
            elif isinstance(label_value, (str, int, float, bool)):
                label_weights[label_type][str(label_value)] += weight
                total_weights[label_type] += weight

    # 生成融合结果
    fused = {}
    for label_type, value_weights in label_weights.items():
        sorted_values = sorted(value_weights.items(), key=lambda x: x[1], reverse=True)

        if label_type in ["genre", "mood", "instrumentation"]:
            # 列表类型：取 top-3
            fused[label_type] = [v for v, w in sorted_values[:3]]
        else:
            # 单值类型：取最高权重
            fused[label_type] = sorted_values[0][0] if sorted_values else None

        # 置信度
        if total_weights[label_type] > 0:
            fused[f"{label_type}_confidence"] = sorted_values[0][1] / total_weights[label_type]

    # 添加融合元数据
    fused["_fusion_info"] = {
        "method": "weighted_voting",
        "n_sources": len(sources),
        "sources": [{"name": name, "weight": w} for name, _, w in sources],
        "timestamp": datetime.now().isoformat(),
    }

    return fused


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="L4 传播层：KNN 标签传播 + 多源融合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embeddings-dir", type=str, default=None,
                        help="CLAP 嵌入目录（用于KNN传播）")
    parser.add_argument("--l2-dir", type=str, default=None,
                        help="L2 语义标签目录（CLAP zero-shot）")
    parser.add_argument("--l3-text-dir", type=str, default=None,
                        help="L3 文本标签目录（DeepSeek V4 Flash）")
    parser.add_argument("--l3-corrected-dir", type=str, default=None,
                        help="L3 纠错标签目录（DeepSeek V4 Pro）")
    parser.add_argument("--l3-audio-dir", type=str, default=None,
                        help="L3 音频结构目录（Qwen-Audio）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--ls-output", type=str, default=None,
                        help="Label Studio 预标注 JSONL 输出路径")
    parser.add_argument("--config", type=str,
                        default="configs/preannotation/preannotation_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--no-knn", action="store_true",
                        help="不做KNN传播，只做多源融合")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    l4_config = config["l4_propagated"]
    knn_config = l4_config["knn"]
    fusion_config = l4_config["fusion"]

    # 目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载嵌入
    embeddings = {}
    if args.embeddings_dir and not args.no_knn:
        embeddings = load_embeddings(Path(args.embeddings_dir))

    # 加载各层标签
    l2_labels = load_labels_from_dir(Path(args.l2_dir), "labels") if args.l2_dir else {}
    l3_text_labels = load_labels_from_dir(Path(args.l3_text_dir), "labels") if args.l3_text_dir else {}
    l3_corrected_labels = load_labels_from_dir(Path(args.l3_corrected_dir), "labels") if args.l3_corrected_dir else {}
    l3_audio_labels = load_labels_from_dir(Path(args.l3_audio_dir), "labels") if args.l3_audio_dir else {}

    # 确定所有样本 ID
    all_sample_ids = set()
    all_sample_ids.update(embeddings.keys())
    all_sample_ids.update(l2_labels.keys())
    all_sample_ids.update(l3_text_labels.keys())
    all_sample_ids.update(l3_corrected_labels.keys())
    all_sample_ids.update(l3_audio_labels.keys())

    sample_ids = sorted(all_sample_ids)
    if args.limit:
        sample_ids = sample_ids[:args.limit]

    logger.info(f"总样本数: {len(sample_ids)}")
    logger.info(f"KNN传播: {'禁用' if args.no_knn else '启用'}")
    logger.info(f"嵌入数: {len(embeddings)}")
    logger.info(f"L2标签数: {len(l2_labels)}")
    logger.info(f"L3文本标签数: {len(l3_text_labels)}")
    logger.info(f"L3纠错标签数: {len(l3_corrected_labels)}")
    logger.info(f"L3音频结构数: {len(l3_audio_labels)}")

    # 确定 KNN 种子集（有 L3 标签的样本）
    seed_labels = {}
    seed_embeddings = {}
    if not args.no_knn and embeddings:
        for sid in sample_ids:
            if sid in l3_text_labels and sid in embeddings:
                seed_labels[sid] = l3_text_labels[sid]
                seed_embeddings[sid] = embeddings[sid]
        logger.info(f"KNN种子集: {len(seed_labels)} 个")

    # 处理每个样本
    results = {}
    ls_tasks = []

    for idx, sample_id in enumerate(sample_ids):
        if (idx + 1) % 100 == 0:
            logger.info(f"处理进度: {idx+1}/{len(sample_ids)}")

        # KNN 传播
        knn_labels = None
        if not args.no_knn and sample_id in embeddings and seed_labels:
            knn_labels = knn_propagate(
                sample_id, embeddings[sample_id],
                seed_embeddings, seed_labels,
                k=knn_config.get("k", 5),
                distance_metric=knn_config.get("distance_metric", "cosine"),
            )

        # 多源融合
        fused = fuse_labels(
            l2_labels.get(sample_id),
            l3_text_labels.get(sample_id),
            l3_corrected_labels.get(sample_id),
            l3_audio_labels.get(sample_id),
            knn_labels,
            fusion_config.get("weights", {}),
        )

        if fused:
            results[sample_id] = fused

            # 保存单个结果
            output_file = output_dir / f"{sample_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({"sample_id": sample_id, "labels": fused}, f, ensure_ascii=False, indent=2)

            # 构建 Label Studio task
            if args.ls_output:
                ls_task = {
                    "id": sample_id,
                    "data": {
                        "audio": f"/data/audio/{sample_id}.wav",
                        "sample_id": sample_id,
                    },
                    "predictions": [{
                        "result": [],
                        "model_version": "l4_knn_fusion_v1",
                    }],
                    "preannotations": {k: v for k, v in fused.items() if not k.startswith("_")},
                }
                ls_tasks.append(ls_task)

    # 保存 Label Studio JSONL
    if args.ls_output and ls_tasks:
        ls_output_path = Path(args.ls_output)
        ls_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ls_output_path, "w", encoding="utf-8") as f:
            for task in ls_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        logger.info(f"Label Studio 预标注已保存: {ls_output_path} ({len(ls_tasks)} 条)")

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("KNN传播 + 多源融合完成")
    logger.info("=" * 60)
    logger.info(f"  总样本数: {len(sample_ids)}")
    logger.info(f"  成功融合: {len(results)}")
    logger.info(f"  KNN种子集: {len(seed_labels)}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)

    # 保存汇总
    summary = {
        "total": len(sample_ids),
        "success": len(results),
        "knn_seeds": len(seed_labels),
        "knn_enabled": not args.no_knn,
        "timestamp": datetime.now().isoformat(),
        "sample_ids": sample_ids,
    }
    with open(output_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
