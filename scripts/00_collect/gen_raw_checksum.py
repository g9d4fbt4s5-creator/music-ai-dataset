"""
gen_raw_checksum.py
生成 raw_audio_checksums.csv（完整性基线）+ audio_manifest.csv（全局索引）

⚠️ 重要约束：
- raw_audio_checksums.csv：只读基线，采集入库生成，**永不修改**
- audio_manifest.csv：业务索引，可更新 status 字段（active/deprecated），不删除历史行
- 禁止用 ls/find 扫描音频目录做业务遍历，本脚本是唯一允许全量扫描的场景（入库时一次性）
- 灾难恢复优先读 audio_manifest.csv 拿到要恢复哪些 audio_id；校验完整性用 raw_audio_checksums.csv 的 sha256

输出：
- data/00_raw_collect/raw_audio_checksums.csv
- data/00_raw_collect/audio_manifest.csv
"""
import os
import sys
import csv
import hashlib
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================== 路径配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from get_audio_physical_path import validate_audio_id, compute_hash

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"gen_checksum_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- 路径 --------
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio"
CHECKSUM_CSV = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio_checksums.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"

# 支持的音频格式
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".aiff", ".opus"}

# CSV 字段定义
CHECKSUM_FIELDS = [
    "audio_id",
    "file_relative_path",
    "original_filename",
    "sha256",
    "file_bytes",
    "import_timestamp",
]

MANIFEST_FIELDS = [
    "audio_id",
    "file_relative_path",
    "original_filename",
    "format",
    "sample_rate",
    "bit_depth",
    "channels",
    "duration_sec",
    "file_bytes",
    "sha256",
    "import_timestamp",
    "status",  # active | deprecated
]


def calculate_sha256(file_path: Path) -> str:
    """计算文件的 sha256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def parse_audio_id_from_filename(filename: str) -> Optional[str]:
    """
    从文件名解析 audio_id

    预期文件名格式：{hash_full}_{audio_id}.{ext}
    例如：aa7013c2..._01ARZ3NDEKTSV4RRFFQ69G5FAV.mp3

    返回：
        audio_id 或 None（解析失败）
    """
    stem = Path(filename).stem  # 去掉扩展名
    parts = stem.split("_", 1)  # 只分割第一个下划线

    if len(parts) != 2:
        return None

    possible_id = parts[1]
    if validate_audio_id(possible_id):
        return possible_id.upper()

    return None


def extract_audio_metadata(file_path: Path) -> Dict:
    """
    提取音频元数据（格式、采样率、位深、声道、时长）

    返回：
        包含元数据的字典，提取失败则返回默认值
    """
    metadata = {
        "format": file_path.suffix.lstrip(".").lower(),
        "sample_rate": None,
        "bit_depth": None,
        "channels": None,
        "duration_sec": None,
    }

    try:
        import soundfile as sf
        info = sf.info(str(file_path))
        metadata["sample_rate"] = info.samplerate
        metadata["bit_depth"] = info.subtype
        metadata["channels"] = info.channels
        metadata["duration_sec"] = round(info.duration, 3)
    except Exception as e:
        logger.warning(f"soundfile 提取元数据失败 {file_path.name}: {e}")

        # 尝试用 librosa
        try:
            import librosa
            duration = librosa.get_duration(path=str(file_path))
            metadata["duration_sec"] = round(duration, 3)
        except Exception as e2:
            logger.warning(f"librosa 提取时长也失败 {file_path.name}: {e2}")

    return metadata


def scan_audio_files(audio_dir: Path) -> List[Path]:
    """
    扫描音频目录下的所有音频文件

    ⚠️ 注意：这是唯一允许全量扫描的场景（入库时一次性）
    业务代码禁止用 ls/find 扫描音频目录
    """
    audio_files = []

    if not audio_dir.exists():
        logger.error(f"音频目录不存在: {audio_dir}")
        return audio_files

    for file_path in audio_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files.append(file_path)

    logger.info(f"扫描到 {len(audio_files)} 个音频文件")
    return audio_files


def load_existing_csv(csv_path: Path, fields: List[str]) -> Dict[str, Dict]:
    """
    加载已有的 CSV 文件，返回以 audio_id 为 key 的字典

    用于 --append 模式，避免重复处理
    """
    existing = {}
    if not csv_path.exists():
        return existing

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_id = row.get("audio_id", "")
                if audio_id:
                    existing[audio_id] = row
        logger.info(f"加载已有 {len(existing)} 条记录: {csv_path.name}")
    except Exception as e:
        logger.warning(f"加载已有 CSV 失败: {e}")

    return existing


def process_audio_file(file_path: Path, import_timestamp: str) -> Optional[Tuple[Dict, Dict]]:
    """
    处理单个音频文件，生成 checksum 记录和 manifest 记录

    返回：
        (checksum_record, manifest_record) 或 None（处理失败）
    """
    # 解析 audio_id
    audio_id = parse_audio_id_from_filename(file_path.name)
    if not audio_id:
        logger.warning(f"无法从文件名解析 audio_id: {file_path.name}，跳过")
        return None

    # 计算相对路径
    rel_path = str(file_path.relative_to(RAW_AUDIO_DIR.parent))

    # 计算 sha256
    try:
        sha256 = calculate_sha256(file_path)
    except Exception as e:
        logger.error(f"计算 sha256 失败 {file_path.name}: {e}")
        return None

    # 文件大小
    file_bytes = file_path.stat().st_size

    # 提取元数据
    metadata = extract_audio_metadata(file_path)

    # 构建 checksum 记录
    checksum_record = {
        "audio_id": audio_id,
        "file_relative_path": rel_path,
        "original_filename": file_path.name,
        "sha256": sha256,
        "file_bytes": file_bytes,
        "import_timestamp": import_timestamp,
    }

    # 构建 manifest 记录
    manifest_record = {
        "audio_id": audio_id,
        "file_relative_path": rel_path,
        "original_filename": file_path.name,
        "format": metadata["format"],
        "sample_rate": metadata["sample_rate"],
        "bit_depth": metadata["bit_depth"],
        "channels": metadata["channels"],
        "duration_sec": metadata["duration_sec"],
        "file_bytes": file_bytes,
        "sha256": sha256,
        "import_timestamp": import_timestamp,
        "status": "active",
    }

    return checksum_record, manifest_record


def write_csv(csv_path: Path, fields: List[str], records: List[Dict]):
    """写入 CSV 文件"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"✅ 写入 {len(records)} 条记录: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="生成 raw_audio 完整性基线和全局索引")
    parser.add_argument("--append", action="store_true",
                        help="追加模式，保留已有记录，只处理新文件")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("生成 raw_audio 完整性基线 + 全局索引")
    logger.info("=" * 60)
    logger.info(f"音频目录: {RAW_AUDIO_DIR}")
    logger.info(f"模式: {'追加' if args.append else '全量重建'}")

    import_timestamp = datetime.now().isoformat()

    # 扫描音频文件
    audio_files = scan_audio_files(RAW_AUDIO_DIR)
    if len(audio_files) == 0:
        logger.warning("没有找到音频文件")
        return

    # 追加模式：加载已有记录
    existing_checksums = {}
    existing_manifests = {}
    if args.append:
        existing_checksums = load_existing_csv(CHECKSUM_CSV, CHECKSUM_FIELDS)
        existing_manifests = load_existing_csv(MANIFEST_CSV, MANIFEST_FIELDS)

    # 处理所有文件
    checksum_records = list(existing_checksums.values()) if args.append else []
    manifest_records = list(existing_manifests.values()) if args.append else []
    existing_ids = set(existing_checksums.keys()) if args.append else set()

    new_count = 0
    skip_count = 0

    for idx, file_path in enumerate(audio_files):
        logger.info(f"[{idx + 1}/{len(audio_files)}] 处理: {file_path.name}")

        # 解析 audio_id 用于判断是否已存在
        audio_id = parse_audio_id_from_filename(file_path.name)
        if audio_id and audio_id in existing_ids:
            logger.info(f"  已存在，跳过")
            skip_count += 1
            continue

        # 处理文件
        result = process_audio_file(file_path, import_timestamp)
        if result:
            checksum_record, manifest_record = result
            checksum_records.append(checksum_record)
            manifest_records.append(manifest_record)
            new_count += 1
        else:
            skip_count += 1

    # 按 audio_id 排序
    checksum_records.sort(key=lambda x: x.get("audio_id", ""))
    manifest_records.sort(key=lambda x: x.get("audio_id", ""))

    # 写入 CSV
    write_csv(CHECKSUM_CSV, CHECKSUM_FIELDS, checksum_records)
    write_csv(MANIFEST_CSV, MANIFEST_FIELDS, manifest_records)

    logger.info("=" * 60)
    logger.info(f"✅ 完成")
    logger.info(f"  新增: {new_count} 条")
    logger.info(f"  跳过: {skip_count} 条")
    logger.info(f"  总计: {len(checksum_records)} 条")
    logger.info(f"  checksum 文件: {CHECKSUM_CSV}")
    logger.info(f"  manifest 文件: {MANIFEST_CSV}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)
    logger.info("⚠️  重要：raw_audio_checksums.csv 是只读基线，入库后永不修改")
    logger.info("   audio_manifest.csv 是业务索引，可更新 status，不删除历史行")


if __name__ == "__main__":
    main()
