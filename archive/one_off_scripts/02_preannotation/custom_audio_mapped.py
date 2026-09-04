#!/usr/bin/env python3
"""
custom_audio_mapped.py — 自有新采集音频标签映射（收敛到 TagMapper）

将自有音频的原始粗标签（从文本描述/模型输出提取）通过统一的 TagMapper 映射为：
- GM128 乐器编码
- VAD 情绪三元组
- 三级流派（primary + secondary）
- 软/硬黑名单分级

与 tag_mapping_musiccaps.py 的区别：
- MusicCaps: 从 CSV 的 aspect_list 字段解析标签
- 自有音频: 从 JSON 的 raw_tags 字段读取标签（已由 L2 语义候选/大模型提取）

使用:
    python custom_audio_mapped.py \
        --input data/02_preannotation/l2_semantic/*.json \
        --mapping configs/label_mapping_dict.json \
        --output data/02_preannotation/custom_audio_mapped_v4.json
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "02_preannotation"))

from tag_mapper import TagMapper


def extract_raw_tags_from_json(data: Dict) -> List[str]:
    """
    从自有音频 JSON 中提取原始粗标签。

    支持多种字段格式：
    - raw_tags: ["piano", "sad", "jazz"]
    - tags: ["piano", "sad"]
    - aspect_list: ["piano", "sad"]
    - instruments + moods + genres 分开存储
    """
    raw_tags = []

    # 直接的标签列表字段
    for field in ["raw_tags", "tags", "aspect_list", "labels"]:
        if field in data and isinstance(data[field], list):
            raw_tags.extend([str(t).strip().lower() for t in data[field] if t])

    # 分开存储的字段
    for field in ["instruments", "moods", "genres", "emotions"]:
        if field in data:
            val = data[field]
            if isinstance(val, list):
                raw_tags.extend([str(t).strip().lower() for t in val if t])
            elif isinstance(val, str):
                raw_tags.append(val.strip().lower())

    # 从文本描述提取（简单关键词匹配）
    if not raw_tags and "caption" in data:
        caption = str(data["caption"]).lower()
        # 简单的关键词提取（实际项目中应由大模型/CLAP完成）
        keywords = ["piano", "guitar", "drum", "bass", "violin", "cello",
                    "jazz", "rock", "pop", "classical", "electronic",
                    "happy", "sad", "calm", "energetic", "dark",
                    "noise", "speech", "silence"]
        for kw in keywords:
            if kw in caption:
                raw_tags.append(kw)

    # 去重
    return list(dict.fromkeys(raw_tags))


def process_custom_audio(input_pattern: str, mapping_path: str, output_path: str,
                         include_soft_blacklist: bool = True):
    """
    处理自有音频 JSON，输出映射后的 JSON。

    Args:
        input_pattern: 输入 JSON glob 模式（如 "data/02_preannotation/l2_semantic/*.json"）
        mapping_path: label_mapping_dict.json 路径
        output_path: 输出 JSON 路径
        include_soft_blacklist: 是否包含软黑名单样本
    """
    print("=" * 60)
    print("自有音频标签映射（收敛到 TagMapper）")
    print("=" * 60)

    # 1. 加载映射器
    mapper = TagMapper(mapping_path)
    print(f"映射字典版本: {mapper.version}")
    print(f"乐器: {len(mapper.inst_map)}, 情绪: {len(mapper.emotion_map)}, "
          f"流派: {len(mapper.genre_map)}")
    print(f"硬黑名单: {len(mapper.hard_blacklist)}, 软黑名单: {len(mapper.soft_blacklist)}")

    # 2. 查找输入文件
    input_files = sorted(glob.glob(input_pattern))
    if not input_files:
        print(f"❌ 未找到匹配文件: {input_pattern}")
        return []
    print(f"\n找到 {len(input_files)} 个输入 JSON 文件")

    # 3. 逐条映射
    results = []
    hard_filtered = 0
    soft_flagged = 0
    unmapped_total = 0

    for fpath in input_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️ 跳过无效文件 {fpath}: {e}")
            continue

        audio_id = data.get("audio_id", Path(fpath).stem)
        raw_tags = extract_raw_tags_from_json(data)

        if not raw_tags:
            # 无原始标签，仍保留记录（标记为无标签）
            mapped = {
                "raw_tags": [],
                "gm128_instruments": [],
                "genre_primary": None,
                "genre_secondary": [],
                "vad_emotions": [],
                "unmapped_original_tags": [],
                "blacklist_hit": [],
                "blacklist_severity": "none",
                "sample_tier": "normal",
                "mapping_version": mapper.version,
            }
        else:
            mapped = mapper.map_all(raw_tags)

        # 硬黑名单：直接过滤
        if mapped["sample_tier"] == "fail":
            hard_filtered += 1
            continue

        # 软黑名单：标记但保留
        if mapped["sample_tier"] == "marginal":
            soft_flagged += 1
            if not include_soft_blacklist:
                continue

        # 统计未映射
        unmapped_total += len(mapped["unmapped_original_tags"])

        # 构建输出记录
        record = {
            "audio_id": audio_id,
            "file_path": data.get("file_path", data.get("audio_path", "")),
            "duration_sec": data.get("duration_sec", data.get("duration", 0)),
            "source": data.get("source", "custom"),
            "raw_tags": raw_tags,
            **mapped,
            "preannotation_version": "custom_v1",
        }
        results.append(record)

    # 4. 输出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 5. 统计
    print(f"\n{'='*60}")
    print(f"映射完成")
    print(f"{'='*60}")
    print(f"  输入文件: {len(input_files)}")
    print(f"  硬黑名单过滤: {hard_filtered}")
    print(f"  软黑名单标记: {soft_flagged} (已保留)")
    print(f"  输出记录: {len(results)}")
    print(f"  未映射标签总数: {unmapped_total}")
    print(f"  输出: {output_path}")

    if unmapped_total > 0:
        print(f"\n  ⚠️ 存在未映射标签，建议通过 Label Studio 【映射建议】闭环更新字典")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自有音频标签映射（收敛到 TagMapper）")
    parser.add_argument("--input", required=True,
                        help="输入 JSON glob 模式（如 'data/02_preannotation/l2_semantic/*.json'）")
    parser.add_argument("--mapping", default=str(PROJECT_ROOT / "configs" / "label_mapping_dict.json"),
                        help="映射字典路径")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "02_preannotation" / "custom_audio_mapped_v4.json"),
                        help="输出 JSON 路径")
    parser.add_argument("--exclude-soft-blacklist", action="store_true",
                        help="排除软黑名单样本（默认保留并标记marginal）")
    args = parser.parse_args()

    process_custom_audio(args.input, args.mapping, args.output,
                         include_soft_blacklist=not args.exclude_soft_blacklist)
