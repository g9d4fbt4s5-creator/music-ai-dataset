"""
custom_audio_preprocess.py
自有音频预标注：读取音频文件，提取基础信息，执行标签映射，输出标准格式JSON
"""
import json
import os
import sys
import csv
import logging
from datetime import datetime
from pathlib import Path
import librosa

# ===================== 全局路径配置（相对于项目根目录） =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 到路径，使用统一的路径工具函数
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from get_audio_physical_path import get_audio_physical_path, validate_audio_id

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"custom_preprocess_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 统一标签映射字典（和tag_mapping共用）
MAPPING_CFG_PATH = PROJECT_ROOT / "data/02_preannotation/label_mapping/label_mapping_dict.json"

# ⚠️ 新架构：优先从 audio_manifest.csv 读取音频列表
# 禁止直接 ls/find 扫描音频目录，永远读元数据表
AUDIO_MANIFEST_CSV = PROJECT_ROOT / "data/00_raw_collect/audio_manifest.csv"

# 兼容旧模式：如果 manifest 不存在，回退到扫描这个目录（仅用于测试）
LEGACY_INPUT_FOLDER = PROJECT_ROOT / "data/00_raw_collect/raw_audio/custom_test"

# 输出：和MusicCaps格式完全一致的预标注JSON
OUTPUT_MAPPED_JSON = PROJECT_ROOT / "data/02_preannotation/preann_csv/custom_audio_mapped.json"

# 加载全局统一映射字典
with open(MAPPING_CFG_PATH, "r", encoding="utf-8") as f:
    mapping_cfg = json.load(f)
INSTRUMENT_GM128_MAP = mapping_cfg["instrument_gm128_map"]
EMOTION_VAD_MAP = mapping_cfg["emotion_vad_map"]
GENRE_3LEVEL_MAP = mapping_cfg["genre_3level_map"]
BLACKLIST = mapping_cfg["blacklist_tags"]


def map_raw_tags(raw_tag_list):
    """
    复用和tag_mapping完全相同的映射逻辑
    输入原始标签列表，输出 gm_insts, vad_list, genre_list, unmapped_tags
    """
    gm_insts = []
    vad_list = []
    genre_list = []
    unmapped_tags = []
    for tag in raw_tag_list:
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
    # 去重排序
    gm_insts = sorted(list(set(gm_insts)))
    unmapped_tags = sorted(list(set(unmapped_tags)))
    return gm_insts, vad_list, genre_list, unmapped_tags


def get_audio_duration(audio_path: Path):
    """读取音频时长（librosa兼容wav/mp3/flac）"""
    try:
        duration = librosa.get_duration(path=str(audio_path))
        return round(duration, 1)
    except Exception as e:
        logger.warning(f"读取时长失败 {audio_path.name}: {e}")
        return 0.0


def extract_audio_basic_info(audio_file: Path):
    """获取音频基础唯一标识，替代MusicCaps ytid"""
    audio_id = audio_file.stem  # 用文件名作为唯一ID
    full_path = str(audio_file.resolve())
    duration = get_audio_duration(audio_file)
    return {
        "audio_id": audio_id,
        "audio_file_path": full_path,
        "total_duration": duration,
        "start_s": 0,
        "end_s": int(duration) if duration > 0 else 10
    }


def get_raw_audio_tags(audio_info):
    """
    ========== 【核心扩展点】两种方案二选一 ==========
    方案1：手动维护标签（少量音频直接写死列表，适合测试）
    方案2：调用MERT/VGGish音频模型推理，自动输出raw_tags（大规模数据集）
    """
    audio_id = audio_info["audio_id"]

    # ====== 方案1：手动填写标签（测试用，注释方案2启用这个）======
    raw_tags = ["piano", "calm", "pop"]
    caption = f"Custom audio: {audio_id}, soft pop piano music"

    # ====== 方案2：模型推理（大规模数据集，取消注释使用）======
    # raw_tags = model_infer(audio_info["audio_file_path"])
    # caption = f"Auto generated description from audio model: {raw_tags}"

    return raw_tags, caption


def load_audio_from_manifest(manifest_path: Path) -> list:
    """
    从 audio_manifest.csv 读取音频列表

    ⚠️ 新架构：禁止直接 ls/find 扫描音频目录，永远读元数据表
    返回：
        [{"audio_id": ..., "file_path": ..., "duration": ...}, ...]
    """
    audio_list = []

    if not manifest_path.exists():
        logger.warning(f"audio_manifest.csv 不存在: {manifest_path}")
        return audio_list

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_id = row.get("audio_id", "")
            rel_path = row.get("file_relative_path", "")
            duration = row.get("duration_sec", 0)

            if not audio_id or not rel_path:
                continue

            # 转换为绝对路径
            abs_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path

            audio_list.append({
                "audio_id": audio_id,
                "file_path": str(abs_path),
                "duration": float(duration) if duration else 0,
            })

    logger.info(f"从 manifest 读取到 {len(audio_list)} 个音频")
    return audio_list


def load_audio_from_folder(folder_path: Path) -> list:
    """
    兼容旧模式：从文件夹扫描音频（仅用于测试，不推荐）

    ⚠️ 警告：规模化时禁止使用此模式，会导致文件系统卡死
    """
    logger.warning("⚠️  使用旧模式（文件夹扫描），仅适用于少量测试音频")
    logger.warning("   规模化请使用 import_audio.py 入库，然后从 audio_manifest.csv 读取")

    audio_suffix = [".mp3", ".wav", ".flac", ".m4a"]
    audio_list = []

    for f in folder_path.iterdir():
        if f.suffix.lower() in audio_suffix:
            duration = get_audio_duration(f)
            audio_list.append({
                "audio_id": f.stem,  # 旧模式用文件名作为 ID
                "file_path": str(f.resolve()),
                "duration": duration,
            })

    return audio_list


def main():
    logger.info("=" * 60)
    logger.info("自有音频预标注开始")
    logger.info("=" * 60)

    logger.info(f"映射字典: {MAPPING_CFG_PATH}")

    # ⚠️ 新架构：优先从 audio_manifest.csv 读取
    audio_list = load_audio_from_manifest(AUDIO_MANIFEST_CSV)

    # 兼容旧模式：如果 manifest 为空，回退到扫描文件夹
    if len(audio_list) == 0:
        logger.info("manifest 为空，尝试旧模式（文件夹扫描）...")
        if LEGACY_INPUT_FOLDER.exists():
            audio_list = load_audio_from_folder(LEGACY_INPUT_FOLDER)
        else:
            logger.error(f"错误：音频文件夹不存在 {LEGACY_INPUT_FOLDER}")
            return

    if len(audio_list) == 0:
        logger.warning("没有找到任何音频文件")
        return

    logger.info(f"待处理音频数量: {len(audio_list)}")

    output_dataset = []
    for audio_info in audio_list:
        audio_id = audio_info["audio_id"]
        audio_path = Path(audio_info["file_path"])
        duration = audio_info["duration"]

        logger.info(f"处理: {audio_id} ({audio_path.name})")

        # 1. 构建基础信息
        info = {
            "audio_id": audio_id,
            "audio_file_path": str(audio_path),
            "total_duration": duration,
            "start_s": 0,
            "end_s": int(duration) if duration > 0 else 10,
        }

        # 2. 获取原始标签+描述
        raw_tags, caption = get_raw_audio_tags(info)

        # 3. 执行统一标签映射
        gm_insts, vad_list, genre_list, unmapped_tags = map_raw_tags(raw_tags)

        # 4. 结构完全对齐musiccaps_mapped_label.json
        item = {
            "ytid": info["audio_id"],  # 复用原有字段，自有音频用 audio_id 替代 ytid
            "start_s": info["start_s"],
            "end_s": info["end_s"],
            "caption_text": caption,
            "raw_aspect_tags": raw_tags,
            "gm128_instrument_codes": gm_insts,
            "vad_emotion": vad_list,
            "genre_3level": genre_list,
            "unmapped_original_tags": unmapped_tags,
            # 自有音频额外扩展字段（不影响原有转换脚本）
            "local_audio_path": info["audio_file_path"],
            "audio_total_duration": info["total_duration"]
        }
        output_dataset.append(item)

    # 输出标准JSON
    OUTPUT_MAPPED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MAPPED_JSON, "w", encoding="utf-8") as fw:
        json.dump(output_dataset, fw, ensure_ascii=False, indent=2)

    logger.info(f"✅ 自有音频预标注完成！输出路径：{OUTPUT_MAPPED_JSON}")
    logger.info(f"共处理音频数量：{len(output_dataset)}")
    logger.info(f"日志文件: {log_file}")
    logger.info("可直接运行 convert_mapped_to_labelstudio.py 生成标注任务")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
