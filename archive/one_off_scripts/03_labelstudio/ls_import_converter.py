#!/usr/bin/env python3
"""
L3 → Label Studio 预标注转换脚本

将 L3 Qwen-Omni 输出的结构标注 JSON 转换为 Label Studio 可导入的格式，
包括 TimeSeries 段落标注和分类标签。

支持:
- 段落结构 (Intro/Theme/Improv/Outro) → LabelsOnAudio 时间区间
- 乐器/情绪/流派 → Choices
- Caption → TextArea

使用:
    python ls_import_converter.py \
        --l3-dir data/02_preannotation/l3_structural \
        --l4-dir data/02_preannotation/l4_propagated \
        --audio-base-url /data/audio \
        --output data/02_preannotation/ls_preannotations.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def convert_structure_to_ls(segments, audio_id, duration):
    """将 L3 段落结构转换为 Label Studio LabelsOnAudio 格式"""
    ls_results = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", duration))
        label = seg.get("label", "未知段落")
        instruments = seg.get("instruments", [])
        emotion = seg.get("emotion", "")
        confidence = float(seg.get("confidence", 0.5))

        # 时间区间标注
        ls_results.append({
            "id": f"seg_{audio_id}_{i}",
            "type": "labels",
            "from_name": "structure",
            "to_name": "audio",
            "value": {
                "start": start,
                "end": end,
                "labels": [label]
            },
            "score": confidence,
        })

        # 乐器标注（附加在该段落上）
        if instruments:
            ls_results.append({
                "id": f"inst_{audio_id}_{i}",
                "type": "choices",
                "from_name": "segment_instruments",
                "to_name": "audio",
                "value": {"choices": instruments[:5]},
                "score": confidence,
            })

        # 情绪标注
        if emotion:
            ls_results.append({
                "id": f"emo_{audio_id}_{i}",
                "type": "choices",
                "from_name": "segment_emotion",
                "to_name": "audio",
                "value": {"choices": [emotion]},
                "score": confidence,
            })

    return ls_results


def convert_l4_to_ls(l4_data, audio_id):
    """将 L4 融合标签转换为 Label Studio Choices/Text 格式"""
    ls_results = []

    # 流派
    genre = l4_data.get("genre", "")
    if genre:
        ls_results.append({
            "id": f"genre_{audio_id}",
            "type": "choices",
            "from_name": "genre",
            "to_name": "audio",
            "value": {"choices": [genre]},
        })

    # 子流派
    subgenre = l4_data.get("subgenre", "")
    if subgenre:
        ls_results.append({
            "id": f"subgenre_{audio_id}",
            "type": "choices",
            "from_name": "subgenre",
            "to_name": "audio",
            "value": {"choices": [subgenre] if isinstance(subgenre, str) else subgenre[:3]},
        })

    # 情绪（全曲级）
    mood = l4_data.get("mood", [])
    if isinstance(mood, str):
        mood = [mood]
    if mood:
        ls_results.append({
            "id": f"mood_{audio_id}",
            "type": "choices",
            "from_name": "mood",
            "to_name": "audio",
            "value": {"choices": mood[:2]},
        })

    # 乐器（全曲级）
    instruments = l4_data.get("instrumentation", [])
    if isinstance(instruments, str):
        instruments = [instruments]
    if instruments:
        ls_results.append({
            "id": f"instruments_{audio_id}",
            "type": "choices",
            "from_name": "instruments",
            "to_name": "audio",
            "value": {"choices": instruments[:5]},
        })

    # 人声
    vocal = l4_data.get("vocal_presence", "")
    if vocal:
        ls_results.append({
            "id": f"vocal_{audio_id}",
            "type": "choices",
            "from_name": "vocal_presence",
            "to_name": "audio",
            "value": {"choices": [vocal]},
        })

    # 质量评估
    quality = l4_data.get("quality_assessment", "")
    if quality:
        ls_results.append({
            "id": f"quality_{audio_id}",
            "type": "choices",
            "from_name": "quality_assessment",
            "to_name": "audio",
            "value": {"choices": [quality]},
        })

    # Caption
    caption = l4_data.get("caption", "")
    if caption:
        ls_results.append({
            "id": f"caption_{audio_id}",
            "type": "textarea",
            "from_name": "caption",
            "to_name": "audio",
            "value": {"text": [caption]},
        })

    return ls_results


def build_ls_task(audio_id, audio_url, l3_data, l4_data, model_version):
    """构建单个 Label Studio 任务"""
    results = []

    # L3 结构标注（黄金集才有）
    if l3_data and "segments" in l3_data:
        duration = float(l3_data.get("duration", 0))
        results.extend(convert_structure_to_ls(l3_data["segments"], audio_id, duration))

    # L4 融合标签（全量都有）
    if l4_data:
        results.extend(convert_l4_to_ls(l4_data, audio_id))

    # 来源追溯（隐藏字段）
    fusion = l4_data.get("fusion", {}) if l4_data else {}
    propagated_from = l4_data.get("propagated_from", "") if l4_data else ""
    propagation_similarity = l4_data.get("propagation_similarity", 0) if l4_data else 0

    task = {
        "id": audio_id,
        "data": {"audio": audio_url},
        "predictions": [{
            "model_version": model_version,
            "result": results,
        }],
        # 元数据（Label Studio 会存储但不显示）
        "meta": {
            "audio_id": audio_id,
            "propagated_from": propagated_from,
            "propagation_similarity": propagation_similarity,
            "fusion_sources": json.dumps(fusion, ensure_ascii=False),
            "is_golden": propagated_from == "golden_set",
        }
    }
    return task


def run_converter(l3_dir, l4_dir, audio_base_url, output_path, model_version):
    """主流程: 读取 L3/L4 结果，转换为 Label Studio JSONL"""
    l3_dir = Path(l3_dir)
    l4_dir = Path(l4_dir)

    # 加载 L3 结果
    l3_data = {}
    if l3_dir.exists():
        for f in l3_dir.glob("*_structure.json"):
            with open(f) as fp:
                data = json.load(fp)
                l3_data[data["audio_id"]] = data

    # 加载 L4 结果
    l4_data = {}
    if l4_dir.exists():
        for f in l4_dir.glob("*_full_tags.json"):
            with open(f) as fp:
                data = json.load(fp)
                l4_data[data["audio_id"]] = data

    # 合并所有 audio_id
    all_ids = set(l3_data.keys()) | set(l4_data.keys())
    print(f"L3 黄金集: {len(l3_data)} 首")
    print(f"L4 全量标签: {len(l4_data)} 首")
    print(f"总计: {len(all_ids)} 首")

    # 转换
    tasks = []
    golden_count = 0
    propagated_count = 0
    for audio_id in sorted(all_ids):
        audio_url = f"{audio_base_url.rstrip('/')}/{audio_id}.flac"
        l3 = l3_data.get(audio_id)
        l4 = l4_data.get(audio_id)
        task = build_ls_task(audio_id, audio_url, l3, l4, model_version)
        tasks.append(task)

        if task["meta"]["is_golden"]:
            golden_count += 1
        elif task["meta"]["propagated_from"]:
            propagated_count += 1

    # 写入 JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Label Studio 预标注转换完成: {output_path}")
    print(f"{'='*60}")
    print(f"  总计: {len(tasks)} 条")
    print(f"  🌟 黄金集(L3结构标注): {golden_count} 条")
    print(f"  📡 KNN传播: {propagated_count} 条")
    print(f"  🤖 DeepSeek-only: {len(tasks) - golden_count - propagated_count} 条")
    print(f"\n  导入 Label Studio:")
    print(f"  Settings → Import → 选择 {output_path}")

    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L3/L4 → Label Studio 预标注转换")
    parser.add_argument("--l3-dir", default="data/02_preannotation/l3_structural", help="L3 结构标注目录")
    parser.add_argument("--l4-dir", default="data/02_preannotation/l4_propagated", help="L4 融合标签目录")
    parser.add_argument("--audio-base-url", default="/data/audio", help="音频文件基础 URL")
    parser.add_argument("--output", default="data/02_preannotation/ls_preannotations.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--model-version", default="l4_deepseek_knn_fusion_v1", help="模型版本标识")
    args = parser.parse_args()

    run_converter(args.l3_dir, args.l4_dir, args.audio_base_url, args.output, args.model_version)
