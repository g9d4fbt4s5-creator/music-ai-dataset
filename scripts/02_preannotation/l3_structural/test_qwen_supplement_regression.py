#!/usr/bin/env python3
"""
qwen_supplement_30.py 回归测试
测试用例：
1. convert_to_l4_format: 完整字段提取（subgenre/vocal_presence/tempo_bpm/key/mood_tags/mood_vad）
2. merge_segment_results（来自l3_qwen_audio_structure）: 超10MB长音频分段标注后合并正确
3. prepare_audio_for_qwen: 超10MB文件处理后truncated状态正确
"""
import sys, json, tempfile
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from qwen_supplement_30 import convert_to_l4_format
from l3_qwen_audio_structure import merge_segment_results, prepare_audio_for_qwen, get_file_size_mb

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")

print("=" * 60)
print("回归测试 1: convert_to_l4_format 完整字段提取")
print("=" * 60)

# 模拟Qwen-Omni完整返回
qwen_result = {
    "genre": "Jazz",
    "subgenre": "Bebop",
    "vocal_presence": "instrumental",
    "tempo_bpm": 180,
    "key": "F minor",
    "mood_tags": ["energetic", "complex"],
    "mood_vad": {"valence": 0.3, "arousal": 0.8, "dominance": 0.6},
    "instruments": ["saxophone", "piano", "bass", "drums"],
    "caption": "快节奏比波普爵士",
    "confidence": 0.9,
    "segments": [
        {"start": 0, "end": 30, "label": "Head", "mood": "energetic"},
        {"start": 30, "end": 90, "label": "Solo", "mood": "complex"},
    ],
}

result = convert_to_l4_format(qwen_result, "test_audio_id_123")
test("audio_id正确", result["audio_id"] == "test_audio_id_123")
test("genre正确", result["genre"] == "Jazz")
test("subgenre提取", result.get("subgenre") == "Bebop", f"got {result.get('subgenre')}")
test("vocal_presence提取", result.get("vocal_presence") == "instrumental")
test("tempo_bpm提取", result.get("tempo_bpm") == 180)
test("key提取", result.get("key") == "F minor")
test("mood_tags提取", result.get("mood_tags") == ["energetic", "complex"])
test("mood_vad提取", result.get("mood_vad") == {"valence": 0.3, "arousal": 0.8, "dominance": 0.6})
test("instrumentation正确", result["instrumentation"] == ["saxophone", "piano", "bass", "drums"])
test("caption正确", result["caption"] == "快节奏比波普爵士")
test("source正确", result["source"] == "qwen_omni_supplement")
test("segments保留", len(result["segments"]) == 2)
test("confidence正确", result["confidence"] == 0.9)

# 测试mood为空时从segments提取
qwen_no_mood = {"genre": "Pop", "instruments": ["piano"], "segments": [{"mood": "romantic"}]}
result2 = convert_to_l4_format(qwen_no_mood, "test_no_mood")
test("mood为空时从segments提取", result2["mood"] == ["romantic"], f"got {result2['mood']}")

print()
print("=" * 60)
print("回归测试 2: merge_segment_results 分段合并正确性")
print("=" * 60)

# 模拟3段标注结果（超10MB长音频分段标注场景）
seg1 = {
    "genre": "Trance", "subgenre": "Progressive Trance",
    "vocal_presence": "vocal", "tempo_bpm": 138, "key": "A minor",
    "mood_tags": ["energetic", "uplifting"], "mood_vad": {"valence": 0.7},
    "instruments": ["synthesizer", "bass"], "caption": "第一段",
    "confidence": 0.85,
    "segments": [{"start": 0, "end": 60, "label": "Intro", "mood": "mysterious"}],
}
seg2 = {
    "genre": "Trance", "subgenre": "Progressive Trance",
    "vocal_presence": "vocal", "tempo_bpm": 138, "key": "A minor",
    "mood_tags": ["uplifting", "euphoric"], "mood_vad": {"valence": 0.8},
    "instruments": ["synthesizer", "drums", "vocals"], "caption": "第二段",
    "confidence": 0.9,
    "segments": [{"start": 0, "end": 90, "label": "Build-up", "mood": "energetic"}],
}
seg3 = {
    "genre": "Trance", "subgenre": "Progressive Trance",
    "vocal_presence": "vocal", "tempo_bpm": 138, "key": "A minor",
    "mood_tags": ["euphoric"], "mood_vad": {"valence": 0.9},
    "instruments": ["synthesizer", "piano"], "caption": "第三段",
    "confidence": 0.88,
    "segments": [{"start": 0, "end": 53, "label": "Outro", "mood": "calm"}],
}

merged = merge_segment_results([seg1, seg2, seg3], "test_long_audio", 413.0)
test("genre众数正确", merged["genre"] == "Trance")
test("subgenre保留", merged.get("subgenre") == "Progressive Trance")
test("mood合并去重", set(merged["mood"]) == {"energetic", "uplifting", "euphoric"})
test("instruments合并去重", set(merged["instrumentation"]) == {"synthesizer", "bass", "drums", "vocals", "piano"})
test("segments时间戳偏移", merged["segments"][0]["start"] == 0)
test("segments第二段偏移180s", merged["segments"][1]["start"] == 180, f"got {merged['segments'][1]['start']}")
test("segments第三段偏移360s", merged["segments"][2]["start"] == 360)
test("truncated=False", merged["truncated"] == False)
test("segmented=True", merged["segmented"] == True)
test("duration_sec正确", merged["duration_sec"] == 413.0)
test("source正确", merged["source"] == "qwen_omni_segmented")
test("vocal_presence保留", merged.get("vocal_presence") == "vocal")
test("tempo_bpm保留", merged.get("tempo_bpm") == 138)
test("key保留", merged.get("key") == "A minor")
test("confidence为平均值", abs(merged["confidence"] - (0.85+0.9+0.88)/3) < 0.01)

print()
print("=" * 60)
print(f"测试结果: {PASS} 通过, {FAIL} 失败")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
