#!/usr/bin/env python3
"""
对4首文件过大导致Base64超限的样本做分段标注+合并。
直接调用l3_qwen_audio_structure.annotate_long_audio唯一实现。
"""
import sys, os, json, time, shutil
from pathlib import Path
from glob import glob

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from l3_qwen_audio_structure import annotate_long_audio, get_audio_duration

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/02_preannotation/l4_deepseek"
SEG_DIR = Path("/tmp/qwen_segmented_retry_4")
SEG_DIR.mkdir(parents=True, exist_ok=True)
MASTER_DIR = PROJECT_ROOT / "data/01_preprocess/processed_master"

# 4首因文件过大失败的样本
FAILED_IDS = [
    "01M0ZV75FJ5XT8F2BWAWABTQ0P",  # 526s, 20MB
    "01M0ZV75EF73YWR1X5V3MHVA3A",  # 440s, 16MB
    "01M0ZV75CA0XZA10YGA99D89XQ",  # 688s, 26MB
    "01M0ZV75N3CVB5WH17V4GCGDQ3",  # 482s, 16MB
]


def find_master(audio_id: str) -> str:
    matches = glob(str(MASTER_DIR / "**" / f"*_{audio_id}.flac"), recursive=True)
    return matches[0] if matches else ""


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("DASHSCOPE_API_KEY="):
                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not found")
        sys.exit(1)
    print(f"API key 已加载: {api_key[:8]}...")

    # 备份旧标签
    backup_dir = PROJECT_ROOT / "archive/l4_deepseek_v4_flash_old"
    backup_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for idx, aid in enumerate(FAILED_IDS, 1):
        print(f"\n[{idx}/4] {aid}")

        master_path = find_master(aid)
        if not master_path:
            print(f"  ❌ 找不到母版")
            failed.append(aid)
            continue

        duration = get_audio_duration(master_path)
        print(f"  母版: {Path(master_path).name}, {duration:.0f}s")

        # 备份旧标签
        old_file = OUTPUT_DIR / f"{aid}_text_labels.json"
        if old_file.exists():
            shutil.copy2(old_file, backup_dir / old_file.name)

        # 分段标注+合并
        try:
            result = annotate_long_audio(master_path, api_key, SEG_DIR, aid)
        except Exception as e:
            print(f"  ❌ 分段标注异常: {e}")
            failed.append(aid)
            continue

        if result:
            # 确保source字段正确
            result["source"] = "qwen_omni_supplement"
            result["reannotated_from"] = "deepseek_v4_flash"
            result["truncated"] = False

            out_file = OUTPUT_DIR / f"{aid}_text_labels.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            genre = result.get("genre", "?")
            subgenre = result.get("subgenre", "")
            print(f"  ✅ 完成: genre={genre}, subgenre={subgenre}, truncated=False")
            success += 1
        else:
            print(f"  ❌ 分段标注失败")
            failed.append(aid)

        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"4首分段重标完成: {success}/4 成功, {len(failed)} 失败")
    if failed:
        print(f"失败列表: {failed}")


if __name__ == "__main__":
    main()
