"""
tag_mapping_musiccaps.py
MusicCaps 标签映射：将 aspect_list 原始标签映射为 GM128 乐器编码、VAD 情绪数值、三级流派
"""
import pandas as pd
import ast
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ===================== 路径配置（相对于项目根目录） =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"tag_mapping_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 标签映射字典
MAPPING_JSON_PATH = PROJECT_ROOT / "data/02_preannotation/label_mapping/label_mapping_dict.json"
with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
    mapping_cfg = json.load(f)

INSTRUMENT_GM128_MAP: Dict[str, int] = mapping_cfg["instrument_gm128_map"]
EMOTION_VAD_MAP: Dict[str, Dict[str, float]] = mapping_cfg["emotion_vad_map"]
GENRE_3LEVEL_MAP: Dict[str, List[str]] = mapping_cfg["genre_3level_map"]
BLACKLIST = mapping_cfg["blacklist_tags"]

# 输入输出路径
INPUT_CSV = PROJECT_ROOT / "data/00_raw_collect/raw_metadata/musiccaps_full.csv"
OUTPUT_JSON = PROJECT_ROOT / "data/02_preannotation/preann_csv/musiccaps_mapped_label.json"


def parse_aspect_list(raw_str: str) -> List[str]:
    """解析MusicCaps aspect list字符串为标签列表"""
    try:
        return ast.literal_eval(raw_str)
    except Exception:
        return []


def map_tags(aspect_tags: List[str]):
    """
    原始标签列表 -> 映射为GM128乐器、VAD情绪、三级流派
    返回：gm_insts, vad_list, genre_list, unmapped_tags
    """
    gm_insts = []
    vad_list = []
    genre_list = []
    unmapped_tags = []

    for tag in aspect_tags:
        tag = tag.strip().lower()
        if tag in BLACKLIST:
            continue

        if tag in INSTRUMENT_GM128_MAP:
            gm_insts.append(INSTRUMENT_GM128_MAP[tag])
        elif tag in EMOTION_VAD_MAP:
            vad_list.append(EMOTION_VAD_MAP[tag])
        elif tag in GENRE_3LEVEL_MAP:
            genre_list.append(GENRE_3LEVEL_MAP[tag])
        else:
            unmapped_tags.append(tag)

    # 去重
    gm_insts = list(sorted(set(gm_insts)))
    unmapped_tags = list(sorted(set(unmapped_tags)))
    return gm_insts, vad_list, genre_list, unmapped_tags


def main():
    logger.info("=" * 60)
    logger.info("MusicCaps 标签映射开始")
    logger.info("=" * 60)
    
    logger.info(f"输入文件: {INPUT_CSV}")
    logger.info(f"映射字典: {MAPPING_JSON_PATH}")
    
    if not INPUT_CSV.exists():
        logger.error(f"找不到输入文件: {INPUT_CSV}")
        return
    
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"读取完成，总样本数: {len(df)}")
    
    output_rows = []
    unmapped_total = 0

    for idx, row in df.iterrows():
        ytid = str(row["ytid"])
        start_s = int(row["start_s"])
        end_s = int(row["end_s"])
        caption_text = str(row["caption"])
        aspect_raw = str(row["aspect_list"])

        raw_tags = parse_aspect_list(aspect_raw)
        gm_insts, vad_list, genre_list, unmapped_tags = map_tags(raw_tags)
        
        unmapped_total += len(unmapped_tags)

        item = {
            "ytid": ytid,
            "start_s": start_s,
            "end_s": end_s,
            "caption_text": caption_text,
            "raw_aspect_tags": raw_tags,
            "gm128_instrument_codes": gm_insts,
            "vad_emotion": vad_list,
            "genre_3level": genre_list,
            "unmapped_original_tags": unmapped_tags
        }
        output_rows.append(item)
        
        # 每1000条打个日志
        if (idx + 1) % 1000 == 0:
            logger.info(f"已处理 {idx + 1}/{len(df)} 条")

    # 输出json
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fw:
        json.dump(output_rows, fw, ensure_ascii=False, indent=2)

    logger.info(f"✅ 映射完成，输出: {OUTPUT_JSON}")
    logger.info(f"总样本数: {len(output_rows)}")
    logger.info(f"未映射标签总次数: {unmapped_total}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
