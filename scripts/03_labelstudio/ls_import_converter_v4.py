#!/usr/bin/env python3
"""
L4 → Label Studio V4 预标注导入转换脚本

将 L4 传播融合结果 + L3 黄金集结构标注 + L1 物理特征
转换为 Label Studio V4 模板可导入的 JSONL 格式。

V4 模板字段映射:
- structure: 10类时间区间 (L3 Qwen-Omni)
- instruments: GM128乐器区间 (L2/L3)
- genre_primary / genre_secondary: 流派主/次 (L4融合)
- mood / mood_vad: 情绪多选+VAD (L4 DeepSeek)
- key_tonic / key_mode / key_modulation: 调性 (L1预填)
- vocal_presence: 人声 (L1 YAMNet)
- quality_grade: A-E质量级 (Stage3质检)
- caption: 自然语言描述 (L4 DeepSeek)
- review_decision / golden_set / review_flag: 审核 (人工填写, 预填默认值)
- annotation_note: 修正说明 (人工填写)

元数据只读区 (data字段):
- bpm, key, lufs, snr, duration_sec
- source_batch, propagation_source, knn_sim
- fusion_strategy, l2_confidence, annotation_source
- qc_flags, marginal_display, golden_display

使用:
    python ls_import_converter_v4.py \
        --l4-dir data/02_preannotation/l4_propagated \
        --l3-dir data/02_preannotation/l3_structural \
        --l1-dir data/02_preannotation/l1_physical \
        --qc-report data/00.5_cleaned/reports/vXXX/qc_gate_report.csv \
        --audio-base-url /data/audio \
        --output data/02_preannotation/ls_preannotations_v4.jsonl
"""

import argparse
import csv
import json
import os
from pathlib import Path


# ========== V4 标签映射表 ==========

# L3 段落标签 → V4 10类结构标签
STRUCTURE_LABEL_MAP = {
    "intro": "Intro",
    "verse": "Verse",
    "pre_chorus": "Pre-Chorus",
    "pre-chorus": "Pre-Chorus",
    "chorus": "Chorus",
    "bridge": "Bridge",
    "instrumental": "Instrumental",
    "solo": "Solo",
    "breakdown": "Breakdown",
    "outro": "Outro",
    "silence": "Silence",
    "theme": "Verse",       # Jazz theme → Verse
    "improvisation": "Solo",  # Jazz improv → Solo
    "head": "Chorus",        # Jazz head → Chorus
    "尾奏": "Outro",
    "即兴独奏": "Solo",
    "主题": "Verse",
}

# 乐器 → V4 GM128标签
INSTRUMENT_LABEL_MAP = {
    "piano": "钢琴 Piano",
    "acoustic guitar": "木吉他 Acoustic Guitar",
    "electric guitar": "电吉他 Electric Guitar",
    "guitar": "电吉他 Electric Guitar",
    "strings": "弦乐 Strings",
    "violin": "弦乐 Strings",
    "cello": "弦乐 Strings",
    "drums": "鼓组 Drum Kit",
    "drum": "鼓组 Drum Kit",
    "percussion": "鼓组 Drum Kit",
    "bass": "贝斯 Bass",
    "double bass": "贝斯 Bass",
    "vocal": "人声演唱 Lead Vocal",
    "vocals": "人声演唱 Lead Vocal",
    "lead vocal": "人声演唱 Lead Vocal",
    "backup vocal": "和声 Backup Vocal",
    "backing vocal": "和声 Backup Vocal",
    "harmony": "和声 Backup Vocal",
    "synth": "合成器 Synth",
    "synthesizer": "合成器 Synth",
    "keyboard": "合成器 Synth",
    "brass": "管乐 Brass",
    "trumpet": "管乐 Brass",
    "saxophone": "管乐 Brass",
    "trombone": "管乐 Brass",
    "萨克斯": "管乐 Brass",
    "钢琴": "钢琴 Piano",
    "吉他": "电吉他 Electric Guitar",
    "贝斯": "贝斯 Bass",
    "鼓": "鼓组 Drum Kit",
    "人声": "人声演唱 Lead Vocal",
    "合成器": "合成器 Synth",
    "弦乐": "弦乐 Strings",
}

# 流派 → V4 主标签
GENRE_PRIMARY_MAP = {
    "jazz": "爵士 Jazz",
    "pop": "流行 Pop",
    "rock": "摇滚 Rock",
    "electronic": "电子 Electronic",
    "hiphop": "嘻哈 HipHop",
    "hip-hop": "嘻哈 HipHop",
    "classical": "古典 Classical",
    "folk": "民谣 Folk",
    "world": "世界音乐 World",
    "soundtrack": "原声配乐 Soundtrack",
    "experimental": "实验先锋 Experimental",
    "other": "爵士 Jazz",  # 默认归Jazz(本项目以Jazz为主)
}

# 流派子标签 → V4 次标签
GENRE_SECONDARY_MAP = {
    "swing": "摇摆 Swing",
    "bebop": "比波普 Bebop",
    "fusion": "融合爵士 Fusion",
    "cool jazz": "融合爵士 Fusion",
    "free jazz": "实验先锋 Experimental",
    "alt-rock": "另类摇滚 Alt-Rock",
    "indie-rock": "独立摇滚 Indie-Rock",
    "metal": "重金属 Metal",
    "house": "House",
    "techno": "Techno",
    "dubstep": "Dubstep",
    "trap": "Trap",
    "baroque": "巴洛克 Baroque",
    "romantic": "浪漫派 Romantic",
    "contemporary": "现代派 Contemporary",
}

# 情绪 → V4 情绪标签
MOOD_MAP = {
    "happy": "欢快活泼 Joyful",
    "joyful": "欢快活泼 Joyful",
    "calm": "温柔舒缓 Calm",
    "relaxed": "温柔舒缓 Calm",
    "peaceful": "温柔舒缓 Calm",
    "intense": "激昂热血 Intense",
    "energetic": "激昂热血 Intense",
    "excited": "激昂热血 Intense",
    "sad": "忧郁伤感 Melancholic",
    "melancholic": "忧郁伤感 Melancholic",
    "melancholy": "忧郁伤感 Melancholic",
    "romantic": "浪漫甜蜜 Romantic",
    "ethereal": "空灵治愈 Ethereal",
    "dreamy": "空灵治愈 Ethereal",
    "tense": "紧张悬疑 Tense",
    "suspenseful": "紧张悬疑 Tense",
    "epic": "大气史诗 Epic",
    "mysterious": "神秘 Mysterious",
    "nostalgic": "怀旧 Nostalgic",
    "平静": "温柔舒缓 Calm",
    "舒缓": "温柔舒缓 Calm",
    "怀旧": "怀旧 Nostalgic",
    "激昂": "激昂热血 Intense",
}

# 质量等级 → V4 A-E级
QUALITY_GRADE_MAP = {
    "good": "B级 4分 良好",
    "warning": "C级 3分 可接受",
    "bad": "D级 2分 较差",
    "excellent": "A级 5分 母带级",
    "marginal": "C级 3分 可接受",
    "fail": "E级 1分 极差/废弃",
    "pass": "B级 4分 良好",
}


def map_label(value, mapping, default=None):
    """通用标签映射"""
    if not value:
        return default
    key = str(value).lower().strip()
    return mapping.get(key, default)


def load_l3_structure(l3_dir, audio_id):
    """加载 L3 黄金集结构标注"""
    l3_dir = Path(l3_dir)
    for pattern in [f"{audio_id}_structure.json", f"{audio_id}.json"]:
        f = l3_dir / pattern
        if f.exists():
            with open(f) as fp:
                return json.load(fp)
    return None


def load_l1_features(l1_dir, audio_id):
    """加载 L1 物理特征"""
    l1_dir = Path(l1_dir)
    for pattern in [f"{audio_id}_physical.json", f"{audio_id}.json"]:
        f = l1_dir / pattern
        if f.exists():
            with open(f) as fp:
                return json.load(fp)
    return None


def load_qc_report(qc_report_path):
    """加载 QC Gate 报告"""
    if not qc_report_path or not os.path.exists(qc_report_path):
        return {}
    qc_data = {}
    with open(qc_report_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("audio_id", "")
            if aid:
                qc_data[aid] = row
    return qc_data


def convert_structure_to_v4(segments, audio_id, duration):
    """将 L3 段落结构转换为 V4 LabelsOnAudio 格式"""
    ls_results = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", duration))
        raw_label = seg.get("label", seg.get("type", "未知段落"))
        label = map_label(raw_label, STRUCTURE_LABEL_MAP, "Instrumental")
        confidence = float(seg.get("confidence", 0.7))

        ls_results.append({
            "id": f"struct_{audio_id}_{i}",
            "type": "labels",
            "from_name": "structure",
            "to_name": "audio_source",
            "value": {
                "start": start,
                "end": end,
                "labels": [label]
            },
            "score": confidence,
        })
    return ls_results


def convert_instruments_to_v4(segments, audio_id):
    """将 L3 段落中的乐器转换为 V4 乐器区间"""
    ls_results = []
    for i, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        raw_instruments = seg.get("instruments", seg.get("instrument", []))
        if isinstance(raw_instruments, str):
            raw_instruments = [raw_instruments]

        mapped_instruments = []
        for inst in raw_instruments:
            mapped = map_label(inst, INSTRUMENT_LABEL_MAP)
            if mapped and mapped not in mapped_instruments:
                mapped_instruments.append(mapped)

        if mapped_instruments:
            ls_results.append({
                "id": f"inst_{audio_id}_{i}",
                "type": "labels",
                "from_name": "instruments",
                "to_name": "audio_source",
                "value": {
                    "start": start,
                    "end": end,
                    "labels": mapped_instruments[:3]
                },
                "score": 0.6,
            })
    return ls_results


def convert_l4_to_v4(l4_data, audio_id, l1_data, qc_data):
    """将 L4 融合标签转换为 V4 Choices/Text 格式"""
    ls_results = []

    # 流派主标签
    genre_raw = l4_data.get("genre", "jazz")
    genre_primary = map_label(genre_raw, GENRE_PRIMARY_MAP, "爵士 Jazz")
    ls_results.append({
        "id": f"genre_p_{audio_id}",
        "type": "choices",
        "from_name": "genre_primary",
        "to_name": "audio_source",
        "value": {"choices": [genre_primary]},
        "score": l4_data.get("confidence", 0.7),
    })

    # 流派次标签 (从 subgenre 或 genre 推断)
    subgenre = l4_data.get("subgenre", "")
    genre_secondary = []
    if subgenre:
        mapped = map_label(subgenre, GENRE_SECONDARY_MAP)
        if mapped:
            genre_secondary.append(mapped)
    if genre_secondary:
        ls_results.append({
            "id": f"genre_s_{audio_id}",
            "type": "choices",
            "from_name": "genre_secondary",
            "to_name": "audio_source",
            "value": {"choices": genre_secondary[:2]},
        })

    # 情绪
    mood_raw = l4_data.get("mood", ["calm"])
    if isinstance(mood_raw, str):
        mood_raw = [mood_raw]
    mood_labels = []
    for m in mood_raw:
        mapped = map_label(m, MOOD_MAP)
        if mapped and mapped not in mood_labels:
            mood_labels.append(mapped)
    if not mood_labels:
        mood_labels = ["温柔舒缓 Calm"]
    ls_results.append({
        "id": f"mood_{audio_id}",
        "type": "choices",
        "from_name": "mood",
        "to_name": "audio_source",
        "value": {"choices": mood_labels[:3]},
        "score": l4_data.get("confidence", 0.6),
    })

    # 调性 (从 L1 预填)
    if l1_data:
        key_raw = l1_data.get("key", l1_data.get("tonic", ""))
        if key_raw:
            # 解析主音和调式
            key_str = str(key_raw).lower()
            tonic = key_str.split()[0].upper() if key_str else "C"
            mode = "Major 大调" if "major" in key_str else "Minor 小调" if "minor" in key_str else "Major 大调"
            ls_results.append({
                "id": f"key_tonic_{audio_id}",
                "type": "choices",
                "from_name": "key_tonic",
                "to_name": "audio_source",
                "value": {"choices": [tonic]},
            })
            ls_results.append({
                "id": f"key_mode_{audio_id}",
                "type": "choices",
                "from_name": "key_mode",
                "to_name": "audio_source",
                "value": {"choices": [mode]},
            })

    # 人声
    vocal_raw = l4_data.get("vocal_presence", l1_data.get("has_vocals", False) if l1_data else False)
    if isinstance(vocal_raw, bool):
        vocal_label = "有人声 Vocal" if vocal_raw else "纯器乐 Instrumental"
    else:
        vocal_label = str(vocal_raw)
        if "instrumental" in vocal_label.lower():
            vocal_label = "纯器乐 Instrumental"
        elif "vocal" in vocal_label.lower():
            vocal_label = "有人声 Vocal"
        else:
            vocal_label = "纯器乐 Instrumental"
    ls_results.append({
        "id": f"vocal_{audio_id}",
        "type": "choices",
        "from_name": "vocal_presence",
        "to_name": "audio_source",
        "value": {"choices": [vocal_label]},
        "score": 0.8,
    })

    # 质量等级 (从 QC 报告)
    quality_raw = "good"
    if qc_data:
        quality_raw = qc_data.get("quality_grade", qc_data.get("qc_result", "good"))
    elif l1_data:
        quality_raw = l1_data.get("quality", "good")
    quality_grade = map_label(quality_raw, QUALITY_GRADE_MAP, "C级 3分 可接受")
    ls_results.append({
        "id": f"quality_{audio_id}",
        "type": "choices",
        "from_name": "quality_grade",
        "to_name": "audio_source",
        "value": {"choices": [quality_grade]},
    })

    # Caption
    caption = l4_data.get("caption", "")
    if caption:
        ls_results.append({
            "id": f"caption_{audio_id}",
            "type": "textarea",
            "from_name": "caption",
            "to_name": "audio_source",
            "value": {"text": [caption]},
        })

    # 审核决策预填 (默认值，人工修改)
    is_golden = l4_data.get("propagated_from") == "golden_set"
    ls_results.append({
        "id": f"review_{audio_id}",
        "type": "choices",
        "from_name": "review_decision",
        "to_name": "audio_source",
        "value": {"choices": ["approve_with_edits 通过并修正"]},
    })
    ls_results.append({
        "id": f"golden_{audio_id}",
        "type": "choices",
        "from_name": "golden_set",
        "to_name": "audio_source",
        "value": {"choices": ["yes 加入黄金集" if is_golden else "no 不加入"]},
    })
    ls_results.append({
        "id": f"flag_{audio_id}",
        "type": "choices",
        "from_name": "review_flag",
        "to_name": "audio_source",
        "value": {"choices": ["golden_standard 黄金标准" if is_golden else "no_review_needed 无需复核"]},
    })

    return ls_results


def build_metadata(l4_data, l1_data, qc_data, audio_id):
    """构建 V4 元数据只读区 (data 字段)"""
    meta = {
        "bpm": l1_data.get("bpm", 0) if l1_data else l4_data.get("bpm", 0),
        "key": l1_data.get("key", "未知") if l1_data else l4_data.get("key", "未知"),
        "lufs": l1_data.get("loudness", l1_data.get("lufs", 0)) if l1_data else 0,
        "snr": l1_data.get("snr", l1_data.get("snr_db", 0)) if l1_data else 0,
        "duration_sec": l4_data.get("duration_sec", l1_data.get("duration", 0) if l1_data else 0),
        "source_batch": l4_data.get("source_batch", "unknown"),
        "propagation_source": l4_data.get("propagated_from", "deepseek"),
        "knn_sim": l4_data.get("propagation_similarity", 0),
        "fusion_strategy": "deepseek+knn" if l4_data.get("propagated_from") else "deepseek_only",
        "l2_confidence": l4_data.get("l2_confidence", "medium"),
        "annotation_source": l4_data.get("propagated_from", "deepseek"),
        "qc_flags": "",
        "marginal_display": "none",
        "golden_display": "none",
    }

    # QC 标记
    if qc_data:
        qc_result = qc_data.get("qc_result", qc_data.get("decision", ""))
        if qc_result == "marginal":
            meta["qc_flags"] = "marginal(需人工复核)"
            meta["marginal_display"] = "block"
        elif qc_result == "fail":
            meta["qc_flags"] = "fail(质量不合格)"
            meta["marginal_display"] = "block"

    # 黄金集标记
    if l4_data.get("propagated_from") == "golden_set":
        meta["golden_display"] = "block"

    return meta


def convert_single(audio_id, l4_data, l3_data, l1_data, qc_data, audio_base_url):
    """转换单条样本为 V4 Label Studio 格式"""
    duration = l4_data.get("duration_sec", l1_data.get("duration", 180) if l1_data else 180)

    # 预标注结果
    predictions = []

    # L3 结构段落 (仅黄金集有)
    if l3_data and l3_data.get("segments"):
        predictions.extend(convert_structure_to_v4(l3_data["segments"], audio_id, duration))
        predictions.extend(convert_instruments_to_v4(l3_data["segments"], audio_id))

    # L4 标签
    predictions.extend(convert_l4_to_v4(l4_data, audio_id, l1_data, qc_data))

    # 元数据
    meta = build_metadata(l4_data, l1_data, qc_data, audio_id)

    task = {
        "id": audio_id,
        "data": {
            "audio": f"{audio_base_url}/{audio_id}.flac",
            **meta,
        },
        "predictions": [{
            "model_version": "l4_deepseek_knn_fusion_v4",
            "result": predictions,
        }],
        "meta": {
            "audio_id": audio_id,
            "is_golden": l4_data.get("propagated_from") == "golden_set",
            "propagated_from": l4_data.get("propagated_from", ""),
            "knn_cosine_dist": l4_data.get("propagation_cosine_dist", 0),
        }
    }

    return task


def run_conversion(l4_dir, l3_dir, l1_dir, qc_report, audio_base_url, output_path):
    """主流程: 批量转换"""
    print("=" * 60)
    print("L4 → Label Studio V4 预标注转换")
    print("=" * 60)

    l4_dir = Path(l4_dir)
    l4_files = sorted(l4_dir.glob("*_full_tags.json"))
    print(f"\n找到 {len(l4_files)} 个 L4 融合标签")

    qc_data = load_qc_report(qc_report)
    print(f"QC 报告: {len(qc_data)} 条记录" if qc_data else "QC 报告: 未提供")

    tasks = []
    golden_count = 0
    knn_count = 0
    deepseek_count = 0

    for f in l4_files:
        with open(f) as fp:
            l4_data = json.load(fp)
        audio_id = l4_data.get("audio_id", f.stem.replace("_full_tags", ""))

        l3_data = load_l3_structure(l3_dir, audio_id)
        l1_data = load_l1_features(l1_dir, audio_id)
        qc_row = qc_data.get(audio_id, {})

        task = convert_single(audio_id, l4_data, l3_data, l1_data, qc_row, audio_base_url)
        tasks.append(task)

        source = l4_data.get("propagated_from", "deepseek")
        if source == "golden_set":
            golden_count += 1
        elif source and source != "deepseek":
            knn_count += 1
        else:
            deepseek_count += 1

    # 写入 JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"转换完成")
    print(f"{'='*60}")
    print(f"  总计: {len(tasks)} 条")
    print(f"  🌟 黄金集: {golden_count}")
    print(f"  📡 KNN传播: {knn_count}")
    print(f"  🤖 DeepSeek-only: {deepseek_count}")
    print(f"  输出: {output_path}")
    print(f"\n  导入 Label Studio:")
    print(f"  1. 创建项目 → 模板选择 labeling_interface_v4.xml")
    print(f"  2. 导入 → 选择 {output_path}")
    print(f"  3. 音频路径需配置本地存储或 S3 存储")

    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L4 → Label Studio V4 预标注转换")
    parser.add_argument("--l4-dir", required=True, help="L4 融合标签目录")
    parser.add_argument("--l3-dir", default="", help="L3 黄金集结构标注目录")
    parser.add_argument("--l1-dir", default="", help="L1 物理特征目录")
    parser.add_argument("--qc-report", default="", help="QC Gate 报告 CSV 路径")
    parser.add_argument("--audio-base-url", default="/data/audio", help="音频基础URL")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    args = parser.parse_args()

    run_conversion(args.l4_dir, args.l3_dir, args.l1_dir, args.qc_report,
                   args.audio_base_url, args.output)
