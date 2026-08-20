"""
convert_ls_jsonl.py
解析 LabelStudio 导出的 jsonl 文件，生成数据集切分 csv
输出：train / val / test / holdout_gold 四个子集

LabelStudio 导出选择：JSON-MIN / raw jsonl
每一行一条标注样本，包含音频路径、人工修正后的标签
"""
import json
import pandas as pd
import os
import random
from datetime import datetime
import logging
from pathlib import Path

# ===================== 路径配置（相对于项目根目录） =====================
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

# -------- 输入输出路径 --------
INPUT_JSONL = PROJECT_ROOT / "data/03_human_review/labelstudio_export/annotations.jsonl"
OUTPUT_FOLDER = PROJECT_ROOT / "data/04_final_dataset/final_metadata"


def parse_ls_jsonl(jsonl_path):
    """
    读取LabelStudio导出jsonl，提取真值
    返回：list of dict，每个dict是一条样本
    """
    records = []
    
    if not os.path.exists(jsonl_path):
        logger.error(f"找不到输入文件: {jsonl_path}")
        return records
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"第{line_num}行JSON解析失败: {e}")
                continue
            
            # 提取样本ID
            sample_id = item.get("id", f"sample_{line_num}")
            
            # 提取音频路径
            # Label Studio 不同导出格式key可能不同，这里做兼容
            audio_path = ""
            if "audio" in item:
                audio_path = item["audio"]
            elif "data" in item and "audio_url" in item["data"]:
                audio_path = item["data"]["audio_url"]
            elif "data" in item and "audio" in item["data"]:
                audio_path = item["data"]["audio"]
            
            # 提取人工标注结果
            annotations = item.get("annotations", [])
            if len(annotations) > 0:
                anno_result = annotations[0].get("result", [])
            else:
                anno_result = []
            
            # 提取质量校验结果（如果有）
            quality = ""
            for anno in anno_result:
                if anno.get("from_name") == "quality_check":
                    quality = anno.get("value", {}).get("choices", [""])[0]
                    break
            
            # 提取人工备注（如果有）
            human_note = ""
            for anno in anno_result:
                if anno.get("from_name") == "human_note":
                    human_note = anno.get("value", {}).get("text", [""])[0]
                    break
            
            records.append({
                "sample_id": sample_id,
                "audio_path": audio_path,
                "quality_check": quality,
                "human_note": human_note,
                "annotations_raw": json.dumps(anno_result, ensure_ascii=False)
            })
    
    return records


def split_dataset(records, holdout_ratio=0.1, test_ratio=0.1, val_ratio=0.1):
    """
    切分数据集：train / val / test / holdout_gold
    
    切分策略：
    1. 先切出 holdout_gold（黄金留出集，全程不参与训练调参）
    2. 剩下的再切 test / val / train
    
    holdout_gold 从最前面切分，保证不被训练污染
    固定随机种子 42，保证可复现
    """
    random.seed(42)  # 固定随机种子，复现切分
    shuffled = records.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    
    n_holdout = int(n * holdout_ratio)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    # holdout 从最前面切
    holdout_gold = shuffled[:n_holdout]
    rest = shuffled[n_holdout:]
    
    # test
    test = rest[:n_test]
    rest = rest[n_test:]
    
    # val
    val = rest[:n_val]
    train = rest[n_val:]
    
    return train, val, test, holdout_gold


def main():
    logger.info("=" * 60)
    logger.info("LabelStudio JSONL → 数据集切分")
    logger.info("=" * 60)
    
    # 1. 读取 jsonl
    logger.info(f"开始读取 LabelStudio jsonl: {INPUT_JSONL}")
    all_records = parse_ls_jsonl(INPUT_JSONL)
    logger.info(f"总样本数: {len(all_records)}")
    
    if len(all_records) == 0:
        logger.error("没有读取到任何样本，请检查输入文件路径和格式")
        return
    
    # 2. 数据集切分
    logger.info("开始数据集切分（随机种子=42）")
    train, val, test, holdout_gold = split_dataset(all_records)
    
    logger.info(f"  train:        {len(train)}  ({len(train)/len(all_records)*100:.1f}%)")
    logger.info(f"  val:          {len(val)}  ({len(val)/len(all_records)*100:.1f}%)")
    logger.info(f"  test:         {len(test)}  ({len(test)/len(all_records)*100:.1f}%)")
    logger.info(f"  holdout_gold: {len(holdout_gold)}  ({len(holdout_gold)/len(all_records)*100:.1f}%)")
    
    # 3. 创建输出目录
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 4. 写出 csv
    pd.DataFrame(all_records).to_csv(
        os.path.join(OUTPUT_FOLDER, "corpus_full_meta.csv"),
        index=False, encoding="utf-8"
    )
    logger.info(f"✅ 输出: corpus_full_meta.csv")
    
    pd.DataFrame(train).to_csv(
        os.path.join(OUTPUT_FOLDER, "train_split.csv"),
        index=False, encoding="utf-8"
    )
    logger.info(f"✅ 输出: train_split.csv")
    
    pd.DataFrame(val).to_csv(
        os.path.join(OUTPUT_FOLDER, "val_split.csv"),
        index=False, encoding="utf-8"
    )
    logger.info(f"✅ 输出: val_split.csv")
    
    pd.DataFrame(test).to_csv(
        os.path.join(OUTPUT_FOLDER, "test_split.csv"),
        index=False, encoding="utf-8"
    )
    logger.info(f"✅ 输出: test_split.csv")
    
    pd.DataFrame(holdout_gold).to_csv(
        os.path.join(OUTPUT_FOLDER, "holdout_gold.csv"),
        index=False, encoding="utf-8"
    )
    logger.info(f"✅ 输出: holdout_gold.csv")
    
    logger.info("=" * 60)
    logger.info("✅ 数据集切分csv全部输出完成")
    logger.info(f"输出目录: {OUTPUT_FOLDER}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
