#!/usr/bin/env python3
"""
对9首deepseek_v4_flash残留的unknown样本用Qwen-Omni重标。
直接从processed_master搜索音频文件，不依赖manifest的master_path。
输出: data/02_preannotation/l4_deepseek/{audio_id}_text_labels.json (source=qwen_omni_supplement)
"""
import sys, os, json, logging, time, shutil
from pathlib import Path
from glob import glob

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from l3_qwen_audio_structure import (
    prepare_audio_for_qwen,
    call_qwen_omni,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data/02_preannotation/l4_deepseek"
TMP_DIR = Path("/tmp/qwen_reannotate_9")
MASTER_DIR = PROJECT_ROOT / "data/01_preprocess/processed_master"

# 9首deepseek_v4_flash残留
TARGET_IDS = [
    "01M0ZV75FJ5XT8F2BWAWABTQ0P",
    "01M0ZV75KZ1JMS1XKVRZ326PCW",
    "01M0ZV75PYAVPDFDGZD69DE3YF",
    "01M0ZV75EF73YWR1X5V3MHVA3A",
    "01M0ZV75CA0XZA10YGA99D89XQ",
    "01M0ZV75MN54813KGGWVFJA864",
    "01M0ZV75CTPEDPQSJSDWQP40E4",
    "01M0ZV75N3CVB5WH17V4GCGDQ3",
    "01M0ZV75H12W7ATW11QWSKXC5N",
]


def find_master(audio_id: str) -> str:
    """从processed_master搜索音频文件"""
    matches = glob(str(MASTER_DIR / "**" / f"*_{audio_id}.flac"), recursive=True)
    if matches:
        return matches[0]
    # 尝试mp3
    matches = glob(str(MASTER_DIR / "**" / f"*_{audio_id}.mp3"), recursive=True)
    if matches:
        return matches[0]
    return ""


def convert_to_l4_format(qwen_result: dict, audio_id: str, truncated: bool) -> dict:
    """转换为L4期望格式，完整提取所有字段"""
    instr = qwen_result.get("instruments") or qwen_result.get("instrumentation") or []
    if isinstance(instr, str):
        instr = [instr]

    mood = qwen_result.get("mood_tags") or qwen_result.get("mood") or qwen_result.get("moods") or []
    if isinstance(mood, str):
        mood = [mood]
    if not mood:
        for seg in qwen_result.get("segments", []):
            m = seg.get("mood", "")
            if m and m not in mood:
                mood.append(m)

    genre = qwen_result.get("genre", "")
    if isinstance(genre, list):
        genre = genre[0] if genre else ""
    subgenre = qwen_result.get("subgenre", "")
    caption = qwen_result.get("caption", "") or qwen_result.get("description", "")

    return {
        "audio_id": audio_id,
        "genre": genre,
        "subgenre": subgenre,
        "mood": mood,
        "mood_tags": qwen_result.get("mood_tags", []),
        "mood_vad": qwen_result.get("mood_vad", {}),
        "instrumentation": instr,
        "vocal_presence": qwen_result.get("vocal_presence", ""),
        "tempo_bpm": qwen_result.get("tempo_bpm", 0),
        "key": qwen_result.get("key", ""),
        "caption": caption,
        "source": "qwen_omni_supplement",
        "segments": qwen_result.get("segments", []),
        "confidence": qwen_result.get("confidence", 0.8),
        "truncated": truncated,
        "reannotated_from": "deepseek_v4_flash",
    }


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载API key
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
        logger.error("未找到 DASHSCOPE_API_KEY")
        sys.exit(1)
    logger.info(f"API key 已加载: {api_key[:8]}...")

    # 备份旧的deepseek_v4_flash标签
    backup_dir = PROJECT_ROOT / "archive/l4_deepseek_v4_flash_old"
    backup_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []
    results = []

    for i, aid in enumerate(TARGET_IDS, 1):
        logger.info(f"[{i}/9] 处理 {aid}")

        # 找母版
        master_path = find_master(aid)
        if not master_path:
            logger.error(f"  ❌ 找不到母版文件")
            failed.append(aid)
            continue
        logger.info(f"  母版: {Path(master_path).name}")

        # 备份旧标签
        old_file = OUTPUT_DIR / f"{aid}_text_labels.json"
        if old_file.exists():
            shutil.copy2(old_file, backup_dir / old_file.name)

        # 预处理音频
        try:
            work_path, truncated = prepare_audio_for_qwen(master_path, TMP_DIR)
        except Exception as e:
            logger.error(f"  ❌ 音频预处理失败: {e}")
            failed.append(aid)
            continue

        # 调用Qwen-Omni
        try:
            result = call_qwen_omni(work_path, api_key)
        except Exception as e:
            logger.error(f"  ❌ API调用失败: {e}")
            failed.append(aid)
            continue

        if not result or result.get("parse_error"):
            logger.error(f"  ❌ 标注结果解析失败")
            failed.append(aid)
            continue

        # 转换格式并保存
        l4_format = convert_to_l4_format(result, aid, truncated)
        out_file = OUTPUT_DIR / f"{aid}_text_labels.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(l4_format, f, ensure_ascii=False, indent=2)

        genre = l4_format.get("genre", "?")
        subgenre = l4_format.get("subgenre", "")
        mood = l4_format.get("mood", [])[:2]
        instr_count = len(l4_format.get("instrumentation", []))
        logger.info(f"  ✅ genre={genre}, subgenre={subgenre}, mood={mood}, instr={instr_count}个, truncated={truncated}")
        success += 1
        results.append((aid, genre))

        time.sleep(1)

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("9首Qwen重标完成")
    logger.info("=" * 60)
    logger.info(f"  总数: 9")
    logger.info(f"  成功: {success}")
    logger.info(f"  失败: {len(failed)}")
    if failed:
        logger.info(f"  失败列表: {failed}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  旧标签备份: {backup_dir}")
    logger.info("")
    logger.info("  流派分布:")
    from collections import Counter
    genre_dist = Counter(g for _, g in results)
    for g, c in genre_dist.most_common():
        logger.info(f"    {g}: {c}")


if __name__ == "__main__":
    main()
