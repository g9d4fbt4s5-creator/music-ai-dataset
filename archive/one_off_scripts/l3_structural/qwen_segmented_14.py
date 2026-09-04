#!/usr/bin/env python3
"""
对被截断的Qwen补标样本做分段标注（复用l3_qwen_audio_structure的唯一实现）。
不重复实现分段+合并逻辑，直接调用annotate_long_audio。
"""
import sys, os, json, csv, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from l3_qwen_audio_structure import annotate_long_audio, get_audio_duration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data/00_raw_collect/audio_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/02_preannotation/l4_deepseek"
SEG_DIR = Path("/tmp/qwen_segmented_14")
SEG_DIR.mkdir(parents=True, exist_ok=True)

# 被截断的样本ID（qwen_supplement时truncated=True的）
TRUNCATED_IDS = [
    "01M0ZV75E1JTESWENDMCDN4G3Y", "01M0ZV75GA4AQAY1KNJ0PD68GQ",
    "01M0ZV75GQPN21PZZESQBZA47M", "01M0ZV75HFQNAS9VNCF717E4MC",
    "01M0ZV75HXBGQX8HM75G0WXBPV", "01M0ZV75MCZDA2T7FAXRDPVDA9",
    "01M0ZV75PJG0F27RP7YANG0ZGW", "2996876392774F67887DA90C3E",
    "3DD429C6458C404A9891416635", "78250AD45D2842A3AA18C9071D",
    "818C48C49AE547DCBC6D3D9B43", "8E0D1A1B284746B3A1FCE15B41",
    "924CF2535CAB4C48A485EF4845", "AA089CC166EF4DE8AA78014AD9",
    "A0F9B5560A194DE5A25A22BD62", "B7C8FC27F3C349B0B39EB7D55A",
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

        # 调用唯一实现：分段标注+合并
        result = annotate_long_audio(master_path, api_key, SEG_DIR, aid)
        if result:
            out_file = OUTPUT_DIR / f"{aid}_text_labels.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  ✅ 完成: genre={result['genre']}, truncated=False")
            success += 1
        else:
            print(f"  ❌ 分段标注失败")
            failed.append(aid)

    print(f"\n{'='*60}")
    print(f"分段标注完成: {success}/{len(TRUNCATED_IDS)} 成功, {len(failed)} 失败")
    if failed:
        print(f"失败列表: {failed}")

if __name__ == "__main__":
    main()
