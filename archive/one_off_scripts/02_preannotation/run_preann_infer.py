"""
run_preann_infer.py
读取本地 model_output_cache 推理缓存，转换为预标注结果

⚠️ 新架构约束（2026-08-20）：
- 本脚本只做推理结果读取 + 格式转换，绝不包含任何 OSS 上传/下载逻辑
- 音频路径使用本地磁盘路径，不使用 OSS URL
- OSS 备份由独立脚本 upload_cache_to_oss.py 负责
- 业务流水线绝不从 OSS 读取音频

输出：
- preann_all_samples.json（Label Studio 可导入的 tasks 格式）
- preann_musiccaps_format.csv（MusicCaps 兼容格式）
"""
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
log_file = LOG_DIR / f"run_preann_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- 输入输出路径（全部本地磁盘，不涉及 OSS） --------
# 输入：GPU 推理后通过 rsync 拉回本地的 model_output_cache 目录
INPUT_CACHE_DIR = PROJECT_ROOT / "data/02_preannotation/model_output_cache"

# 输出：LS 可导入的 tasks json
OUTPUT_LS_TASKS = PROJECT_ROOT / "data/02_preannotation/preann_csv/preann_all_samples.json"

# 输出：MusicCaps 格式 csv
OUTPUT_MUSICCAPS_CSV = PROJECT_ROOT / "data/02_preannotation/preann_csv/preann_musiccaps_format.csv"


def parse_inference_json(json_path: Path) -> Dict:
    """
    解析单个推理结果 json 文件
    返回标准化的样本字典

    预期推理缓存结构（GPU 端输出，rsync 拉回本地）：
    {
        "sample_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "audio_path": "/absolute/local/path/to/audio.wav",  # 本地磁盘路径
        "predictions": {
            "instruments": [...],
            "mood": [...],
            "genre": [...],
            "tempo": 120.0,
            "key": "C major",
            "confidence": {...}
        }
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 提取基础信息
    sample_id = raw_data.get("sample_id", raw_data.get("audio_id", json_path.stem))

    # 音频路径：直接使用本地磁盘路径
    # ⚠️ 新架构：不做任何 OSS URL 映射，业务只读本地磁盘
    local_audio_path = raw_data.get("audio_path", "")

    # 提取模型推理结果
    predictions = raw_data.get("predictions", {})

    # 构建标准化样本
    sample = {
        "sample_id": sample_id,
        "audio_path": local_audio_path,  # 本地磁盘绝对路径

        # 模型推理结果
        "instruments": predictions.get("instruments", []),
        "mood": predictions.get("mood", []),
        "genre": predictions.get("genre", []),
        "tempo": predictions.get("tempo", None),
        "key": predictions.get("key", None),

        # 置信度
        "confidence": predictions.get("confidence", {}),

        # 原始推理结果（完整保留，方便回溯）
        "raw_inference": raw_data
    }

    return sample


def build_ls_task(sample: Dict) -> Dict:
    """
    将标准化样本转换为 Label Studio task 格式

    ⚠️ 新架构：audio 字段使用本地磁盘路径
    Label Studio 需用 --allow-local-files 启动才能播放本地音频
    """
    ls_task = {
        "id": sample["sample_id"],
        "data": {
            "audio": sample["audio_path"],  # 本地磁盘路径
            "sample_id": sample["sample_id"],
        },
        "predictions": [],
        "predictions_ground_truth": [],
        "predictions_meta": {},
        "predictions_segments": [],
        "predictions_instruments": sample["instruments"],
        "predictions_mood": sample["mood"],
        "predictions_genre": sample["genre"],
        "predictions_tempo": sample["tempo"],
        "predictions_key": sample["key"],
        "predictions_confidence": sample["confidence"],
        "predictions_notes": []
    }

    return ls_task


def main():
    logger.info("=" * 60)
    logger.info("推理缓存 → 预标注结果转换（本地磁盘模式）")
    logger.info("=" * 60)
    logger.info("⚠️  新架构：本脚本不涉及任何 OSS 操作")
    logger.info("   - 音频路径：本地磁盘")
    logger.info("   - OSS 备份：由 upload_cache_to_oss.py 独立负责")
    logger.info(f"输入目录: {INPUT_CACHE_DIR}")

    # 检查输入目录
    if not INPUT_CACHE_DIR.exists():
        logger.error(f"找不到输入目录: {INPUT_CACHE_DIR}")
        logger.error("请先通过 rsync 从 GPU 拉回 model_output_cache 快照")
        return

    # 查找所有 json 文件
    json_files = list(INPUT_CACHE_DIR.glob("*.json"))
    logger.info(f"发现推理结果文件: {len(json_files)} 个")

    if len(json_files) == 0:
        logger.warning("没有找到任何 json 推理文件")
        return

    # 解析所有样本
    all_samples = []
    ls_tasks = []

    for idx, json_file in enumerate(json_files):
        try:
            sample = parse_inference_json(json_file)
            all_samples.append(sample)

            ls_task = build_ls_task(sample)
            ls_tasks.append(ls_task)

            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1}/{len(json_files)} 个")

        except Exception as e:
            logger.error(f"解析失败 {json_file.name}: {e}")
            continue

    # 输出 LS tasks json
    OUTPUT_LS_TASKS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_LS_TASKS, "w", encoding="utf-8") as f:
        json.dump(ls_tasks, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ LS tasks 输出: {OUTPUT_LS_TASKS}")
    logger.info(f"   共 {len(ls_tasks)} 条任务")

    # 输出 MusicCaps 格式 csv
    import pandas as pd

    df = pd.DataFrame([{
        "sample_id": s["sample_id"],
        "audio_path": s["audio_path"],
        "instruments": ", ".join(s["instruments"]),
        "mood": ", ".join(s["mood"]),
        "genre": ", ".join(s["genre"]),
        "tempo": s["tempo"],
        "key": s["key"],
    } for s in all_samples])

    df.to_csv(OUTPUT_MUSICCAPS_CSV, index=False, encoding="utf-8")
    logger.info(f"✅ MusicCaps 格式 csv 输出: {OUTPUT_MUSICCAPS_CSV}")

    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)
    logger.info("⚠️  后续步骤：")
    logger.info("   1. 如需备份到 OSS：运行 python scripts/utils/upload_cache_to_oss.py")
    logger.info("   2. Label Studio 需用 --allow-local-files 启动才能播放本地音频")
    logger.info("   3. 业务流水线绝不从 OSS 读取音频")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
