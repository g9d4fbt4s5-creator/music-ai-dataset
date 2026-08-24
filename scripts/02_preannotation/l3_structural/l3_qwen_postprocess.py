#!/usr/bin/env python3
"""
l3_qwen_postprocess.py — Qwen-Omni L3 输出后处理映射

将 Qwen3.5-Omni-Flash 的中文输出（段落/乐器/情绪）映射为 V4 标准格式：
- 段落: 前奏→Intro, 主题呈示→Verse, 即兴独奏→Solo, 未知→Unknown
- 乐器: 钢琴→GM001, 萨克斯→GM065 (通过映射字典中文key)
- 情绪: 平静→温柔舒缓 + VAD(0.5,0.2,0.5)
- Caption: 直接填入

输出符合 V4 Label Studio 预标注 JSON 格式。

使用:
    python l3_qwen_postprocess.py \
        --input data/02_preannotation/l3_structural/01M0XXX_structure.json \
        --mapping configs/label_mapping_dict.json \
        --output data/02_preannotation/l3_structural/01M0XXX_v4.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "02_preannotation"))

from tag_mapper import TagMapper


# ========== 段落结构映射（Qwen 中文 → V4 英文） ==========
STRUCTURE_MAP = {
    "前奏": "Intro",
    "引子": "Intro",
    "开场": "Intro",
    "主题呈示": "Verse",
    "主题": "Verse",
    "主旋律": "Chorus",
    "副歌": "Chorus",
    "即兴独奏": "Solo",
    "独奏": "Solo",
    "华彩": "Solo",
    "间奏": "Bridge",
    "桥段": "Bridge",
    "过渡": "Bridge",
    "尾奏": "Outro",
    "尾声": "Outro",
    "结束": "Outro",
    "器乐段": "Instrumental",
    "纯器乐": "Instrumental",
    "分解段": "Breakdown",
    "静音": "Silence",
    "静默": "Silence",
    "未知段落": "Unknown",
    "未知": "Unknown",
    "未标注": "Unknown",
}

# ========== 情绪映射（Qwen 中文 → V4 mood + VAD） ==========
MOOD_MAP = {
    "平静": {"mood": "温柔舒缓 Calm", "vad": [0.5, 0.2, 0.5]},
    "舒缓": {"mood": "温柔舒缓 Calm", "vad": [0.6, 0.2, 0.3]},
    "温柔": {"mood": "温柔舒缓 Calm", "vad": [0.6, 0.25, 0.4]},
    "欢快": {"mood": "欢快活泼 Joyful", "vad": [0.8, 0.7, 0.6]},
    "活泼": {"mood": "欢快活泼 Joyful", "vad": [0.85, 0.75, 0.65]},
    "愉悦": {"mood": "欢快活泼 Joyful", "vad": [0.8, 0.65, 0.6]},
    "激昂": {"mood": "激昂热血 Intense", "vad": [0.7, 0.9, 0.8]},
    "热血": {"mood": "激昂热血 Intense", "vad": [0.75, 0.95, 0.85]},
    "紧张": {"mood": "紧张悬疑 Tense", "vad": [0.3, 0.8, 0.7]},
    "悬疑": {"mood": "紧张悬疑 Tense", "vad": [0.25, 0.75, 0.65]},
    "神秘": {"mood": "神秘 Mysterious", "vad": [0.5, 0.5, 0.4]},
    "空灵": {"mood": "空灵治愈 Ethereal", "vad": [0.7, 0.25, 0.4]},
    "治愈": {"mood": "空灵治愈 Ethereal", "vad": [0.7, 0.3, 0.45]},
    "忧郁": {"mood": "忧郁伤感 Melancholic", "vad": [0.2, 0.3, 0.3]},
    "伤感": {"mood": "忧郁伤感 Melancholic", "vad": [0.15, 0.35, 0.25]},
    "浪漫": {"mood": "浪漫甜蜜 Romantic", "vad": [0.75, 0.4, 0.45]},
    "甜蜜": {"mood": "浪漫甜蜜 Romantic", "vad": [0.8, 0.45, 0.5]},
    "大气": {"mood": "大气史诗 Epic", "vad": [0.6, 0.85, 0.85]},
    "史诗": {"mood": "大气史诗 Epic", "vad": [0.65, 0.9, 0.9]},
    "怀旧": {"mood": "怀旧 Nostalgic", "vad": [0.45, 0.3, 0.4]},
}


def map_structure(paragraph: str) -> str:
    """映射段落标签：中文→V4英文，未知用Unknown兜底"""
    if not paragraph:
        return "Unknown"
    # 精确匹配
    if paragraph in STRUCTURE_MAP:
        return STRUCTURE_MAP[paragraph]
    # 模糊匹配（包含关键词）
    for cn, en in STRUCTURE_MAP.items():
        if cn in paragraph or paragraph in cn:
            return en
    return "Unknown"


def map_mood(emotion: str) -> Dict:
    """映射情绪：中文→V4 mood + VAD"""
    if not emotion:
        return {"mood": "神秘 Mysterious", "vad": [0.5, 0.5, 0.4]}
    if emotion in MOOD_MAP:
        return MOOD_MAP[emotion]
    # 模糊匹配
    for cn, mapping in MOOD_MAP.items():
        if cn in emotion or emotion in cn:
            return mapping
    return {"mood": "神秘 Mysterious", "vad": [0.5, 0.5, 0.4]}


def map_instruments(instruments: List[str], mapper: TagMapper) -> List[str]:
    """
    映射乐器：中文→GM128编码。

    优先用映射字典的中文key，其次用TagMapper.map_instrument()，
    都失败则保留原文（标记为未映射）。
    """
    if not instruments:
        return []

    gm128_codes = []
    for inst in instruments:
        inst_clean = str(inst).strip()
        # 1. 映射字典直接查（支持中文key）
        if inst_clean.lower() in mapper.inst_map:
            code = mapper._gm_number_to_code(mapper.inst_map[inst_clean.lower()])
            gm128_codes.append(code)
            continue
        # 2. TagMapper 映射（英文key）
        code = mapper.map_instrument(inst_clean)
        if code:
            gm128_codes.append(code)
            continue
        # 3. 保留原文（未映射）
        gm128_codes.append(inst_clean)

    # 去重保序
    seen = set()
    result = []
    for code in gm128_codes:
        if code not in seen:
            seen.add(code)
            result.append(code)
    return result


def convert_qwen_to_v4(qwen_output: Dict, audio_id: str,
                        mapper: TagMapper, audio_path: str = "") -> Dict:
    """
    Qwen-Omni 输出 → V4 预标注 JSON。

    Args:
        qwen_output: Qwen-Omni 原始输出（含 segments/caption/confidence）
        audio_id: 音频ID
        mapper: TagMapper 实例
        audio_path: 音频文件路径

    Returns:
        V4 格式的预标注 JSON
    """
    predictions = []
    all_moods = []
    all_instruments = set()

    segments = qwen_output.get("segments", [])

    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        paragraph = seg.get("paragraph", seg.get("label", seg.get("type", "")))
        instruments = seg.get("instruments", seg.get("instrument", []))
        emotion = seg.get("emotion", seg.get("mood", ""))

        if isinstance(instruments, str):
            instruments = [instruments]

        # 1. Structure 区间
        structure_label = map_structure(paragraph)
        predictions.append({
            "id": f"struct_{audio_id}_{i}",
            "type": "labels",
            "from_name": "structure",
            "to_name": "audio_source",
            "value": {
                "start": start,
                "end": end,
                "labels": [structure_label]
            },
            "score": qwen_output.get("confidence", 0.7),
        })

        # 2. Instruments 区间（同一段时间）
        gm128_codes = map_instruments(instruments, mapper)
        if gm128_codes:
            predictions.append({
                "id": f"inst_{audio_id}_{i}",
                "type": "labels",
                "from_name": "instruments",
                "to_name": "audio_source",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": gm128_codes[:5]
                },
                "score": 0.6,
            })
            all_instruments.update(gm128_codes)

        # 3. 收集情绪
        if emotion:
            mood_info = map_mood(emotion)
            all_moods.append(mood_info["mood"])

    # 4. 整首情绪（取出现最多的）
    if all_moods:
        from collections import Counter
        dominant_mood = Counter(all_moods).most_common(1)[0][0]
        predictions.append({
            "id": f"mood_{audio_id}",
            "type": "choices",
            "from_name": "mood",
            "to_name": "audio_source",
            "value": {"choices": [dominant_mood]},
            "score": 0.7,
        })

        # mood_vad 备注
        dominant_vad = next((m["vad"] for m in [map_mood(e) for e in all_moods]
                             if m["mood"] == dominant_mood), [0.5, 0.5, 0.4])
        predictions.append({
            "id": f"mood_vad_{audio_id}",
            "type": "textarea",
            "from_name": "mood_vad",
            "to_name": "audio_source",
            "value": {"text": [f"Valence={dominant_vad[0]}, Arousal={dominant_vad[1]}, Dominance={dominant_vad[2]}"]},
        })

    # 5. Caption
    caption = qwen_output.get("caption", "")
    if caption:
        predictions.append({
            "id": f"caption_{audio_id}",
            "type": "textarea",
            "from_name": "caption",
            "to_name": "audio_source",
            "value": {"text": [caption]},
        })

    # 6. 人声（从乐器推断）
    has_vocal = any("GM091" in i or "GM092" in i or "vocal" in i.lower()
                     for i in all_instruments)
    predictions.append({
        "id": f"vocal_{audio_id}",
        "type": "choices",
        "from_name": "vocal_presence",
        "to_name": "audio_source",
        "value": {"choices": ["有人声 Vocal" if has_vocal else "纯器乐 Instrumental"]},
        "score": 0.8,
    })

    # 构建 V4 任务
    task = {
        "audio_id": audio_id,
        "data": {
            "audio": audio_path or f"/data/audio/{audio_id}.flac",
            "is_golden": True,
            "golden_display": "block",
            "marginal_display": "none",
            "qc_flags": "golden_set_l3",
        },
        "predictions": [{
            "model_version": "l3_qwen_omni_v1",
            "result": predictions,
        }],
        "meta": {
            "audio_id": audio_id,
            "is_golden": True,
            "qwen_confidence": qwen_output.get("confidence", 0.7),
            "mapping_version": mapper.version,
            "preannotation_version": "l3_qwen_v1",
            "segment_count": len(segments),
            "instruments": sorted(list(all_instruments)),
        }
    }

    return task


def process_file(input_path: str, mapping_path: str, output_path: str) -> Dict:
    """处理单个 Qwen 输出文件"""
    with open(input_path, "r", encoding="utf-8") as f:
        qwen_output = json.load(f)

    audio_id = qwen_output.get("audio_id", Path(input_path).stem.replace("_structure", ""))
    mapper = TagMapper(mapping_path)

    task = convert_qwen_to_v4(qwen_output, audio_id, mapper)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    print(f"✅ 转换完成: {input_path} → {output_path}")
    print(f"   段落数: {len(qwen_output.get('segments', []))}")
    print(f"   乐器: {task['meta']['instruments']}")
    print(f"   映射字典版本: {mapper.version}")

    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen-Omni L3 输出后处理映射")
    parser.add_argument("--input", required=True, help="Qwen-Omni 输出 JSON 路径")
    parser.add_argument("--mapping", default=str(PROJECT_ROOT / "configs" / "label_mapping_dict.json"),
                        help="映射字典路径")
    parser.add_argument("--output", required=True, help="输出 V4 JSON 路径")
    args = parser.parse_args()

    process_file(args.input, args.mapping, args.output)
