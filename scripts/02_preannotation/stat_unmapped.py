"""
stat_unmapped.py
统计未映射标签的出现频次，生成报告
"""
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter

# ==========路径配置（相对于项目根目录）==========
PROJECT_ROOT = Path(__file__).parent.parent.parent

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"stat_unmapped_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INPUT_MAPPED_JSON = PROJECT_ROOT / "data/02_preannotation/preann_csv/musiccaps_mapped_label.json"
OUTPUT_UNMAPPED_REPORT = PROJECT_ROOT / "data/02_preannotation/label_mapping/unmapped_tag_report.json"


def main():
    logger.info("=" * 60)
    logger.info("未映射标签统计开始")
    logger.info("=" * 60)
    
    logger.info(f"输入文件: {INPUT_MAPPED_JSON}")
    
    if not INPUT_MAPPED_JSON.exists():
        logger.error(f"错误：找不到{INPUT_MAPPED_JSON}，请先运行 tag_mapping_musiccaps.py")
        return

    with open(INPUT_MAPPED_JSON, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    
    logger.info(f"读取完成，总样本数: {len(data_list)}")

    all_unmapped = []
    sample_ref = {}  # 记录每个未映射标签对应的一条样本作为参考

    for item in data_list:
        tags = item.get("unmapped_original_tags", [])
        for t in tags:
            t = t.strip()
            if t == "":
                continue
            all_unmapped.append(t)
            if t not in sample_ref:
                sample_ref[t] = {
                    "ytid": item["ytid"],
                    "start_s": item["start_s"],
                    "caption_text": item["caption_text"]
                }

    counter = Counter(all_unmapped)
    sorted_tags = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    report = {
        "total_unmapped_tag_count": len(counter),
        "total_occurrence": sum(counter.values()),
        "tag_stats": [{"tag": k, "count": v, "sample": sample_ref[k]} for k, v in sorted_tags]
    }

    OUTPUT_UNMAPPED_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_UNMAPPED_REPORT, "w", encoding="utf-8") as fw:
        json.dump(report, fw, ensure_ascii=False, indent=2)

    logger.info(f"✅ 未映射标签统计完成，报告输出：{OUTPUT_UNMAPPED_REPORT}")
    logger.info(f"未映射独立标签总数：{len(counter)}")
    logger.info(f"总出现次数：{sum(counter.values())}")
    logger.info(f"日志文件: {log_file}")
    logger.info("==== Top 20 高频未映射标签 ====")
    for tag, cnt in sorted_tags[:20]:
        logger.info(f"  {tag:40} 出现 {cnt}次")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
