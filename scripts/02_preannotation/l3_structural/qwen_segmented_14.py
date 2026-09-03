#!/usr/bin/env python3
"""
对14首被截断的Qwen补标样本做分段标注（类似Don Carlos做法）：
切3段(0-180/180-360/360-end)，分别调Qwen-Omni API，合并标注结果。
"""
import sys, os, json, csv, time, shutil
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from l3_qwen_audio_structure import call_qwen_omni, get_audio_duration, get_file_size_mb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data/00_raw_collect/audio_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/02_preannotation/l4_deepseek"
SEG_DIR = Path("/tmp/qwen_segmented_14")
SEG_DIR.mkdir(parents=True, exist_ok=True)

# 14首被截断的样本ID（从之前的输出中提取）
TRUNCATED_IDS = [
    "01M0ZV75E1JTESWENDMCDN4G3Y",
    "01M0ZV75GA4AQAY1KNJ0PD68GQ",
    "01M0ZV75GQPN21PZZESQBZA47M",
    "01M0ZV75HFQNAS9VNCF717E4MC",
    "01M0ZV75HXBGQX8HM75G0WXBPV",
    "01M0ZV75MCZDA2T7FAXRDPVDA9",
    "01M0ZV75PJG0F27RP7YANG0ZGW",
    "2996876392774F67887DA90C3E",
    "3DD429C6458C404A9891416635",
    "78250AD45D2842A3AA18C9071D",
    "818C48C49AE547DCBC6D3D9B43",
    "8E0D1A1B284746B3A1FCE15B41",
    "924CF2535CAB4C48A485EF4845",
    "AA089CC166EF4DE8AA78014AD9",
    "A0F9B5560A194DE5A25A22BD62",
    "B7C8FC27F3C349B0B39EB7D55A",
    "E6A98A1C67AC4D9FA1E7437524",
]

def load_manifest():
    manifest = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row.get("audio_id", "")
            if aid:
                manifest[aid] = row
    return manifest

def extract_segment(input_path, output_path, start_sec, end_sec):
    """用ffmpeg提取音频片段"""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start_sec), "-to", str(end_sec),
        "-c:a", "libmp3lame", "-b:a", "192k",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return Path(output_path).exists()

def merge_segments(segment_results, audio_id, duration):
    """合并3段标注结果"""
    # genre: 众数
    genres = [r.get("genre", "") for r in segment_results if r.get("genre")]
    genre = Counter(genres).most_common(1)[0][0] if genres else ""

    # mood: 合并去重
    moods = []
    for r in segment_results:
        for m in r.get("mood", []):
            if m not in moods:
                moods.append(m)

    # instruments: 合并去重
    instruments = []
    for r in segment_results:
        for inst in r.get("instruments", []) or r.get("instrumentation", []):
            if inst not in instruments:
                instruments.append(inst)

    # segments: 合并并偏移时间戳
    segments = []
    for i, r in enumerate(segment_results):
        offset = i * 180
        for seg in r.get("segments", []):
            seg_copy = dict(seg)
            seg_copy["start"] = seg.get("start", 0) + offset
            seg_copy["end"] = seg.get("end", 0) + offset
            segments.append(seg_copy)

    caption = segment_results[0].get("caption", "") if segment_results else ""

    return {
        "audio_id": audio_id,
        "genre": genre,
        "mood": moods,
        "instrumentation": instruments,
        "caption": caption,
        "source": "qwen_omni_segmented",
        "segments": segments,
        "confidence": 0.85,
        "truncated": False,
        "segmented": True,
        "duration_sec": duration,
    }

def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("DASHSCOPE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not found")
        sys.exit(1)

    manifest = load_manifest()
    success = 0
    failed = []

    for idx, aid in enumerate(TRUNCATED_IDS, 1):
        print(f"\n[{idx}/{len(TRUNCATED_IDS)}] {aid}")
        row = manifest.get(aid)
        if not row:
            print(f"  ❌ manifest中找不到")
            failed.append(aid)
            continue

        master_path = row.get("master_path", "")
        if not master_path or not Path(master_path).exists():
            possible = list((PROJECT_ROOT / "data/01_preprocess/processed_master").glob(f"*{aid}*"))
            if possible:
                master_path = str(possible[0])
            else:
                print(f"  ❌ 找不到母版")
                failed.append(aid)
                continue

        duration = get_audio_duration(master_path)
        print(f"  母版: {Path(master_path).name}, {duration:.0f}s")

        # 切3段
        segment_results = []
        seg_count = min(3, int(duration // 180) + 1)
        for seg_i in range(seg_count):
            start = seg_i * 180
            end = min((seg_i + 1) * 180, duration)
            if end - start < 10:
                continue
            seg_path = SEG_DIR / f"{aid}_seg{seg_i}_{start}_{int(end)}.mp3"
            if not seg_path.exists():
                ok = extract_segment(master_path, str(seg_path), start, end)
                if not ok:
                    print(f"  ⚠️ 段{seg_i}提取失败，跳过")
                    continue
            print(f"  段{seg_i}: {start}-{int(end)}s, {get_file_size_mb(str(seg_path)):.1f}MB")

            # 调API
            try:
                result = call_qwen_omni(str(seg_path), api_key)
                if result and not result.get("parse_error"):
                    segment_results.append(result)
                    print(f"    ✅ genre={result.get('genre','?')}")
                else:
                    print(f"    ❌ 段{seg_i}标注失败")
            except Exception as e:
                print(f"    ❌ 段{seg_i}API错误: {e}")
            time.sleep(1)

        if not segment_results:
            print(f"  ❌ 所有段都失败")
            failed.append(aid)
            continue

        # 合并
        merged = merge_segments(segment_results, aid, duration)
        out_file = OUTPUT_DIR / f"{aid}_text_labels.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"  ✅ 合并完成: genre={merged['genre']}, mood={len(merged['mood'])}个, instr={len(merged['instrumentation'])}个, segments={len(merged['segments'])}个, truncated=False")
        success += 1

    print(f"\n{'='*60}")
    print(f"分段标注完成: {success}/{len(TRUNCATED_IDS)} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败列表: {failed}")

if __name__ == "__main__":
    main()
