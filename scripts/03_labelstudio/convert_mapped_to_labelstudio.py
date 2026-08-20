"""
convert_mapped_to_labelstudio.py
输入：musiccaps_mapped_label.json / custom_audio_mapped.json（自有音频输出）
输出：labelstudio_musiccaps_tasks.json，可直接导入Label-Studio

功能：
1. 增加GM128编码反向查表，输出可读乐器名称
2. 增加VAD反向查表，输出可读情绪文本
3. 字段完全兼容原有流水线，支持MusicCaps & 自有音频（local_audio_path）
4. 输出data字段完全匹配前面XML模板的$变量
"""
import json
import os
import logging
from datetime import datetime
from pathlib import Path

# ======================== 路径配置（相对于项目根目录） ========================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"convert_ls_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 映射字典
MAPPING_DICT_PATH = PROJECT_ROOT / "data/02_preannotation/label_mapping/label_mapping_dict.json"

# 输入：经过映射后的数据集json（MusicCaps / 自有音频二选一修改）
INPUT_MAPPED_JSON = PROJECT_ROOT / "data/02_preannotation/preann_csv/musiccaps_mapped_label.json"
# INPUT_MAPPED_JSON = PROJECT_ROOT / "data/02_preannotation/preann_csv/custom_audio_mapped.json"

# 输出LS导入任务包
OUTPUT_LS_TASK = PROJECT_ROOT / "data/03_human_review/labelstudio_musiccaps_tasks.json"

# =========================================================


def main():
    logger.info("=" * 60)
    logger.info("映射数据集 → Label Studio 任务包转换")
    logger.info("=" * 60)
    
    logger.info(f"输入文件: {INPUT_MAPPED_JSON}")
    logger.info(f"映射字典: {MAPPING_DICT_PATH}")
    
    if not INPUT_MAPPED_JSON.exists():
        logger.error(f"找不到输入文件: {INPUT_MAPPED_JSON}")
        return
    
    if not MAPPING_DICT_PATH.exists():
        logger.error(f"找不到映射字典: {MAPPING_DICT_PATH}")
        return

    # 加载映射字典
    with open(MAPPING_DICT_PATH, "r", encoding="utf-8") as f:
        mapping_cfg = json.load(f)

    # ---------- 构建反向查表 数字编码 → 可读文本 ----------
    # gm128 code -> instrument name
    gm_code2name = {v: k for k, v in mapping_cfg["instrument_gm128_map"].items()}
    # vad (v,a,d) tuple -> emotion name
    vad_tuple2name = {}
    for emo_name, vad_val in mapping_cfg["emotion_vad_map"].items():
        key = (vad_val["valence"], vad_val["arousal"], vad_val["dominance"])
        vad_tuple2name[key] = emo_name

    logger.info(f"GM128映射条目: {len(gm_code2name)}")
    logger.info(f"VAD情绪映射条目: {len(vad_tuple2name)}")

    # 读取映射完成的数据集
    with open(INPUT_MAPPED_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    logger.info(f"读取数据集: {len(dataset)} 条样本")

    ls_tasks = []

    for idx, item in enumerate(dataset):
        ytid = item["ytid"]
        start_s = item["start_s"]
        end_s = item["end_s"]
        caption_text = item["caption_text"]
        raw_aspect_tags = item["raw_aspect_tags"]
        gm128_codes = item["gm128_instrument_codes"]
        vad_emotion_list = item["vad_emotion"]
        genre_3level_list = item["genre_3level"]
        unmapped_tags = item["unmapped_original_tags"]

        # -------- GM128 数字翻译成可读乐器 --------
        instrument_text_parts = []
        for code in gm128_codes:
            name = gm_code2name.get(code, f"unknown_gm({code})")
            instrument_text_parts.append(f"{code}:{name}")
        gm128_instrument_text = ", ".join(instrument_text_parts)

        # -------- VAD数组翻译成可读情绪 --------
        vad_text_parts = []
        vad_value_str_parts = []
        for vad in vad_emotion_list:
            v = vad["valence"]
            a = vad["arousal"]
            d = vad["dominance"]
            vad_value_str_parts.append(f"V:{v},A:{a},D:{d}")
            emo_name = vad_tuple2name.get((v, a, d), "unknown_emotion")
            vad_text_parts.append(emo_name)
        vad_value_str = " | ".join(vad_value_str_parts)
        vad_emotion_text = ", ".join(vad_text_parts)

        # -------- 三级流派格式化 --------
        if len(genre_3level_list) > 0:
            three_level_genre_text = "; ".join([str(g) for g in genre_3level_list])
        else:
            three_level_genre_text = "无流派"

        # -------- youtube链接（自有音频ytid为文件名，链接会无效，不影响） --------
        yt_url = f"https://www.youtube.com/watch?v={ytid}&t={start_s}s"
        segment_time = f"{start_s}s ~ {end_s}s"

        # 组装LS data字段，完全匹配XML模板变量
        data_block = {
            "audio_url": item.get("local_audio_path", ""),
            "yt_url": yt_url,
            "segment_time": segment_time,
            "full_caption": caption_text,
            "raw_aspect_tags": ", ".join(raw_aspect_tags),
            "unmapped_tags": ", ".join(unmapped_tags),
            "gm128_instrument_codes": ", ".join(str(c) for c in gm128_codes),
            "gm128_instrument_text": gm128_instrument_text,
            "three_level_genre_text": three_level_genre_text,
            "vad_value": vad_value_str,
            "vad_emotion_text": vad_emotion_text,
        }

        # 如果存在本地音频路径（自有音频会带这个key，musiccaps无，不报错）
        if "local_audio_path" in item:
            data_block["local_audio_path"] = item["local_audio_path"]

        ls_task_item = {
            "data": data_block,
            "predictions": []
        }
        ls_tasks.append(ls_task_item)
        
        # 每1000条打个日志
        if (idx + 1) % 1000 == 0:
            logger.info(f"已转换 {idx + 1}/{len(dataset)} 条")

    # 输出LS任务json
    OUTPUT_LS_TASK.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LS_TASK, "w", encoding="utf-8") as fw:
        json.dump(ls_tasks, fw, ensure_ascii=False, indent=2)

    logger.info(f"✅ LabelStudio任务包输出完成：{OUTPUT_LS_TASK}")
    logger.info(f"总任务条数：{len(ls_tasks)}")
    logger.info(f"日志文件: {log_file}")
    logger.info("提示：请确认Label-Studio项目XML模板字段与data块字段一一对应。")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
