#!/usr/bin/env python3
"""
tag_mapping_musiccaps.py — MusicCaps 标签映射（收敛到 TagMapper）

将 MusicCaps 的 aspect_list 原始标签通过统一的 TagMapper 映射为：
- GM128 乐器编码
- VAD 情绪三元组
- 三级流派（primary + secondary）
- 软/硬黑名单分级

所有映射规则收敛到 TagMapper，本脚本只负责：
1. 读取 MusicCaps CSV
2. 解析 aspect_list
3. 调用 TagMapper.map_all()
4. 过滤硬黑名单样本
5. 输出 V4 可直接读取的 JSON

使用:
    python tag_mapping_musiccaps.py \
        --input data/00_raw_collect/raw_metadata/musiccaps_full.csv \
        --mapping configs/label_mapping_dict.json \
        --output data/02_preannotation/musiccaps_mapped_v4.json
"""

import argparse
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "02_preannotation"))

from tag_mapper import TagMapper


def parse_aspect_list(raw_str: str) -> List[str]:
    """解析 MusicCaps aspect list 字符串为标签列表"""
    if not raw_str or raw_str == "":
        return []
    try:
        result = ast.literal_eval(raw_str)
        if isinstance(result, list):
            return [str(t).strip().lower() for t in result if t]
        return []
    except (ValueError, SyntaxError):
        # 兜底：逗号分隔
        return [t.strip().lower() for t in raw_str.split(",") if t.strip()]


def process_musiccaps(input_csv: str, mapping_path: str, output_path: str,
                      include_soft_blacklist: bool = True):
    """
    处理 MusicCaps CSV，输出映射后的 JSON。

    Args:
        input_csv: MusicCaps CSV 路径
        mapping_path: label_mapping_dict.json 路径
        output_path: 输出 JSON 路径
        include_soft_blacklist: 是否包含软黑名单样本（标记marginal但保留）
    """
    import pandas as pd

    print("=" * 60)
    print("MusicCaps 标签映射（收敛到 TagMapper）")
    print("=" * 60)

    # 1. 加载映射器
    mapper = TagMapper(mapping_path)
    print(f"映射字典版本: {mapper.version}")
    print(f"乐器映射: {len(mapper.inst_map)}, 情绪: {len(mapper.emotion_map)}, "
          f"流派: {len(mapper.genre_map)}")
    print(f"硬黑名单: {len(mapper.hard_blacklist)}, 软黑名单: {len(mapper.soft_blacklist)}")

    # 2. 读取 CSV
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"\n读取 {total} 条 MusicCaps 记录")

    # 3. 逐条映射
    results = []
    hard_filtered = 0
    soft_flagged = 0
    unmapped_total = 0

    for idx, row in df.iterrows():
        # 解析 aspect_list
        aspect_str = row.get("aspect_list", row.get("aspect", ""))
        raw_tags = parse_aspect_list(str(aspect_str))

        if not raw_tags:
            continue

        # 调用统一映射器
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
            "audio_id": row.get("ytid", row.get("audio_id", f"mc_{idx}")),
            "file_path": row.get("file_path", ""),
            "caption": row.get("caption", row.get("text", "")),
            "start_s": float(row.get("start_s", row.get("start", 0))),
            "end_s": float(row.get("end_s", row.get("end", 0))),
            "raw_tags": raw_tags,
            **mapped,  # genre_primary/secondary, gm128_instruments, vad_emotions, unmapped, blacklist, sample_tier
            "preannotation_version": "musiccaps_v1",
        }
        results.append(record)

        if (idx + 1) % 1000 == 0:
            print(f"  处理进度: {idx+1}/{total}")

    # 4. 输出
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 5. 统计
    print(f"\n{'='*60}")
    print(f"映射完成")
    print(f"{'='*60}")
    print(f"  原始记录: {total}")
    print(f"  硬黑名单过滤: {hard_filtered} (speech/silence等)")
    print(f"  软黑名单标记: {soft_flagged} (noise/distorted等, 已保留)")
    print(f"  输出记录: {len(results)}")
    print(f"  未映射标签总数: {unmapped_total}")
    print(f"  输出: {output_path}")

    if unmapped_total > 0:
        print(f"\n  ⚠️ 存在 {unmapped_total} 个未映射标签，建议:")
        print(f"     1. 运行 stat_unmapped.py 统计高频未映射标签")
        print(f"     2. 在 Label Studio 中用【映射建议】格式标注")
        print(f"     3. export_annotations.py 提取 → merge_mapping.py 合并")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MusicCaps 标签映射（收敛到 TagMapper）")
    parser.add_argument("--input", required=True, help="MusicCaps CSV 路径")
    parser.add_argument("--mapping", default=str(PROJECT_ROOT / "configs" / "label_mapping_dict.json"),
                        help="映射字典路径")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "02_preannotation" / "musiccaps_mapped_v4.json"),
                        help="输出 JSON 路径")
    parser.add_argument("--exclude-soft-blacklist", action="store_true",
                        help="排除软黑名单样本（默认保留并标记marginal）")
    args = parser.parse_args()

    process_musiccaps(args.input, args.mapping, args.output,
                      include_soft_blacklist=not args.exclude_soft_blacklist)
