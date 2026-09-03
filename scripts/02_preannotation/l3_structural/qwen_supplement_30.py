#!/usr/bin/env python3
"""
对原 deepseek-only 的 30 首样本用 Qwen-Omni 补标，直接输出 L4 期望的格式。
输出字段: genre(字符串), mood(列表), instrumentation(列表), caption(字符串), audio_id(字符串), source
"""
import sys
import os
import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 把 l3_qwen_audio_structure.py 的目录加入 path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from l3_qwen_audio_structure import (
    prepare_audio_for_qwen,
    call_qwen_omni,
    get_audio_duration,
    get_file_size_mb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data/00_raw_collect/audio_manifest.csv"
IDS_FILE = "/tmp/deepseek_only_30_ids.txt"
OUTPUT_DIR = PROJECT_ROOT / "data/02_preannotation/l4_deepseek"
TMP_DIR = Path("/tmp/qwen_supplement_30")


def load_manifest() -> Dict[str, Dict]:
    """加载 manifest，返回 audio_id -> 行数据"""
    manifest = {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("audio_id", "")
            if aid:
                manifest[aid] = row
    return manifest


def load_ids() -> List[str]:
    """加载 30 首 audio_id 列表"""
    with open(IDS_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


def convert_to_l4_format(qwen_result: Dict, audio_id: str) -> Dict:
    """
    将 Qwen-Omni 输出转换为 L4 load_deepseek_labels 期望的格式。
    L4 期望: genre(字符串), mood(列表), instrumentation(列表), caption(字符串), audio_id
    """
    # Qwen 输出的字段名可能是 instruments 或 instrumentation
    instr = qwen_result.get("instruments") or qwen_result.get("instrumentation") or []
    if isinstance(instr, str):
        instr = [instr]

    mood = qwen_result.get("mood") or qwen_result.get("moods") or []
    if isinstance(mood, str):
        mood = [mood]

    genre = qwen_result.get("genre", "")
    if isinstance(genre, list):
        genre = genre[0] if genre else ""

    caption = qwen_result.get("caption", "") or qwen_result.get("description", "")

    return {
        "audio_id": audio_id,
        "genre": genre,
        "mood": mood,
        "instrumentation": instr,
        "caption": caption,
        "source": "qwen_omni_supplement",
        "segments": qwen_result.get("segments", []),
        "confidence": qwen_result.get("confidence", 0.8),
    }


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载 API key
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        # 尝试从 .env 加载
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

    manifest = load_manifest()
    ids = load_ids()
    logger.info(f"manifest: {len(manifest)}首, 待标注: {len(ids)}首")

    # 先备份原来的 DeepSeek 标签（只备份这30首的）
    backup_dir = PROJECT_ROOT / "archive/l4_deepseek_old"
    backup_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []
    results = []

    for i, aid in enumerate(ids, 1):
        logger.info(f"[{i}/{len(ids)}] 处理 {aid}")

        # 找到母版路径
        row = manifest.get(aid)
        if not row:
            logger.error(f"  ❌ manifest中找不到 {aid}")
            failed.append(aid)
            continue

        master_path = row.get("master_path", "")
        if not master_path or not Path(master_path).exists():
            # 尝试从 master_dir 拼接
            logger.warning(f"  master_path不存在: {master_path}, 尝试其他路径")
            # 搜索母版文件
            possible = list((PROJECT_ROOT / "data/01_preprocess/processed_master").glob(f"*{aid}*"))
            if possible:
                master_path = str(possible[0])
            else:
                logger.error(f"  ❌ 找不到母版文件")
                failed.append(aid)
                continue

        logger.info(f"  母版: {Path(master_path).name}")

        # 备份原来的 DeepSeek 标签
        old_file = OUTPUT_DIR / f"{aid}_text_labels.json"
        if old_file.exists():
            import shutil
            shutil.copy2(old_file, backup_dir / old_file.name)

        # 预处理音频
        try:
            work_path, truncated = prepare_audio_for_qwen(master_path, TMP_DIR)
        except Exception as e:
            logger.error(f"  ❌ 音频预处理失败: {e}")
            failed.append(aid)
            continue

        # 调用 Qwen-Omni
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
        l4_format = convert_to_l4_format(result, aid)
        l4_format["truncated"] = truncated
        out_file = OUTPUT_DIR / f"{aid}_text_labels.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(l4_format, f, ensure_ascii=False, indent=2)

        genre = l4_format.get("genre", "?")
        mood = l4_format.get("mood", [])[:2]
        instr_count = len(l4_format.get("instrumentation", []))
        logger.info(f"  ✅ genre={genre}, mood={mood}, instr={instr_count}个, truncated={truncated}")
        success += 1
        results.append((aid, genre))

        # 避免速率限制
        time.sleep(1)

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("Qwen-Omni 补标完成")
    logger.info("=" * 60)
    logger.info(f"  总数: {len(ids)}")
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
