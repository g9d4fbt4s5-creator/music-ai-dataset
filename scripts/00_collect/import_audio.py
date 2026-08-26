"""
import_audio.py
音频采集入库脚本：格式校验 + 生成ULID + 散列迁移

⚠️ 核心约束：
- 入库前强制格式校验，不符合规格的拒绝入库，移至 rejected/
- 符合规格的：生成 ULID audio_id、计算 sha256、存入散列目录
- raw_audio 目录只读，永不修改移动已有文件
- 散列规则：md5(audio_id)[0:4] 两层散列，统一调用 get_audio_physical_path()

流程：
1. 扫描源目录
2. 格式校验（采样率、位深、声道、时长、文件大小）
3. 生成 ULID audio_id
4. 计算 sha256
5. 迁移到散列目录（复制，不删除源文件）
6. 不符合的移到 rejected/ 并记录原因
7. 完成后提示运行 gen_raw_checksum.py 生成基线

用法：
    python import_audio.py --src /path/to/source_audio/
    python import_audio.py --src /path/to/source_audio/ --dry-run
"""
import os
import sys
import csv
import time
import random
import hashlib
import logging
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================== 路径配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from get_audio_physical_path import (
    validate_audio_id,
    compute_hash,
    get_audio_physical_path,
    ensure_directory_for_audio,
)

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"import_audio_{time_str}.log"
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
REJECTED_DIR = PROJECT_ROOT / "data" / "00_raw_collect" / "rejected"
REJECTED_LOG = REJECTED_DIR / "rejected_log.csv"

# 支持的音频格式
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac", ".aiff", ".opus"}

# 格式规格（对应 corpus_format_spec.yaml）
FORMAT_SPEC = {
    "sample_rate_min": 44100,
    "bit_depth_allowed": {16, 24, 32},
    "channels_allowed": {1, 2},
    "duration_min_sec": 1,
    "duration_max_sec": 1200,  # 20分钟
    "file_size_max_mb": 200,
}


# ===================== ULID 生成器（内置，不依赖外部库） =====================
# Crockford's Base32 字符表（排除 I,L,O,U）
CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_ulid() -> str:
    """
    生成 ULID（Universally Unique Lexicographically Sortable Identifier）

    ULID 格式：
    - 128位
    - 前48位：时间戳（毫秒级，UNIX epoch）
    - 后80位：随机数
    - Crockford's Base32 编码，26个字符

    返回：
        26字符的 ULID 字符串（大写）
    """
    # 时间戳（毫秒）
    timestamp_ms = int(time.time() * 1000)

    # 48位时间戳 + 80位随机数 = 128位
    # 用 Python 的大整数处理
    random_80 = random.getrandbits(80)
    ulid_int = (timestamp_ms << 80) | random_80

    # 转换为 Crockford's Base32（26个字符）
    ulid_chars = []
    for _ in range(26):
        ulid_chars.append(CROCKFORD_BASE32[ulid_int & 0x1F])
        ulid_int >>= 5

    # 反转（因为我们从低位开始取）
    ulid_str = "".join(reversed(ulid_chars))

    return ulid_str


# ===================== 格式校验 =====================
def extract_audio_info(file_path: Path) -> Dict:
    """
    提取音频信息（采样率、位深、声道、时长）

    返回：
        包含音频信息的字典
    """
    info = {
        "format": file_path.suffix.lstrip(".").lower(),
        "sample_rate": None,
        "bit_depth": None,
        "channels": None,
        "duration_sec": None,
        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
        "readable": True,
        "error": None,
    }

    try:
        import soundfile as sf
        sf_info = sf.info(str(file_path))
        info["sample_rate"] = sf_info.samplerate
        info["bit_depth"] = sf_info.subtype
        info["channels"] = sf_info.channels
        info["duration_sec"] = round(sf_info.duration, 3)
    except Exception as e:
        info["readable"] = False
        info["error"] = str(e)
        logger.warning(f"soundfile 读取失败 {file_path.name}: {e}")

        # 尝试用 librosa 提取时长
        try:
            import librosa
            duration = librosa.get_duration(path=str(file_path))
            info["duration_sec"] = round(duration, 3)
            info["readable"] = True  # 至少能读时长
        except Exception as e2:
            logger.warning(f"librosa 也失败 {file_path.name}: {e2}")

    return info


def validate_audio_format(file_path: Path) -> Tuple[bool, str, List[str]]:
    """
    校验音频格式是否符合入库标准

    返回：
        (is_valid, reason, low_quality_tags)
        - is_valid: 是否通过校验（可入库）
        - reason: 拒绝原因（如果拒绝）
        - low_quality_tags: 低质量标签（可入库但标记）
    """
    info = extract_audio_info(file_path)
    low_quality_tags = []

    # 1. 文件是否可读
    if not info["readable"]:
        return False, f"corrupted_file: {info['error']}", []

    # 2. 格式是否支持
    if info["format"] not in {"wav", "flac", "mp3", "m4a", "ogg"}:
        return False, f"unsupported_format: {info['format']}", []

    # 3. 时长校验
    if info["duration_sec"] is not None:
        if info["duration_sec"] < FORMAT_SPEC["duration_min_sec"]:
            return False, f"too_short: {info['duration_sec']}s", []
        if info["duration_sec"] > FORMAT_SPEC["duration_max_sec"]:
            return False, f"too_long: {info['duration_sec']}s", []

    # 4. 文件大小校验
    if info["file_size_mb"] > FORMAT_SPEC["file_size_max_mb"]:
        return False, f"too_large: {info['file_size_mb']:.1f}MB", []

    # 5. 采样率（低于标准标记为低质量，但仍可入库）
    if info["sample_rate"] is not None:
        if info["sample_rate"] < FORMAT_SPEC["sample_rate_min"]:
            low_quality_tags.append(f"low_sample_rate:{info['sample_rate']}")

    # 6. 单声道标记
    if info["channels"] == 1:
        low_quality_tags.append("mono_audio")

    # 7. 有损格式标记
    if info["format"] in {"mp3", "m4a", "ogg"}:
        low_quality_tags.append("lossy_format")

    return True, "", low_quality_tags


# ===================== 文件迁移 =====================
def calculate_sha256(file_path: Path) -> str:
    """计算文件的 sha256"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def import_single_audio(file_path: Path, dry_run: bool = False) -> Dict:
    """
    导入单个音频文件

    返回：
        包含导入结果的字典
    """
    result = {
        "source_file": str(file_path),
        "original_filename": file_path.name,
        "audio_id": None,
        "sha256": None,
        "target_path": None,
        "status": "pending",
        "reason": "",
        "low_quality_tags": [],
    }

    # 1. 格式校验
    is_valid, reason, low_quality_tags = validate_audio_format(file_path)
    result["low_quality_tags"] = low_quality_tags

    if not is_valid:
        result["status"] = "rejected"
        result["reason"] = reason
        return result

    # 2. 生成 ULID
    audio_id = generate_ulid()
    # 确保唯一（理论上 ULID 不会冲突，但保险起见）
    while not validate_audio_id(audio_id):
        audio_id = generate_ulid()
    result["audio_id"] = audio_id

    # 3. 计算 sha256
    sha256 = calculate_sha256(file_path)
    result["sha256"] = sha256

    # 4. 计算目标路径（散列目录）
    extension = file_path.suffix.lstrip(".")
    rel_path = get_audio_physical_path(audio_id, extension, "raw_audio")
    target_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path
    result["target_path"] = str(target_path)

    # 5. 执行迁移（复制，不删除源文件）
    if not dry_run:
        # 确保目标目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 复制文件
        shutil.copy2(file_path, target_path)
        logger.info(f"  已复制到: {rel_path}")

    result["status"] = "imported"
    return result


def move_to_rejected(file_path: Path, reason: str, dry_run: bool = False):
    """
    将不符合规格的文件移到 rejected/ 目录

    注意：rejected/ 是临时存放，需要定期人工清理
    """
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    target_path = REJECTED_DIR / file_path.name

    # 如果重名，加时间戳
    if target_path.exists():
        target_path = REJECTED_DIR / f"{file_path.stem}_{time_str}{file_path.suffix}"

    if not dry_run:
        shutil.copy2(file_path, target_path)  # 复制，不删除源文件

    # 记录拒绝日志
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "original_filename": file_path.name,
        "source_path": str(file_path),
        "rejected_path": str(target_path),
        "reason": reason,
    }

    if not dry_run:
        file_exists = REJECTED_LOG.exists()
        with open(REJECTED_LOG, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

    return target_path


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="音频采集入库：格式校验 + ULID生成 + 散列迁移")
    parser.add_argument("--src", required=True, help="源音频目录路径")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际复制文件")
    args = parser.parse_args()

    source_dir = Path(args.src)

    logger.info("=" * 60)
    logger.info("音频采集入库")
    logger.info("=" * 60)
    logger.info(f"源目录: {source_dir}")
    logger.info(f"目标目录: {RAW_AUDIO_DIR}")
    logger.info(f"模式: {'预览(不复制)' if args.dry_run else '正式入库'}")

    # 检查源目录
    if not source_dir.exists():
        logger.error(f"源目录不存在: {source_dir}")
        return

    # 扫描源目录
    source_files = []
    for file_path in source_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS:
            source_files.append(file_path)

    logger.info(f"扫描到 {len(source_files)} 个音频文件")

    if len(source_files) == 0:
        logger.warning("没有找到音频文件")
        return

    # 处理每个文件
    imported = []
    rejected = []

    for idx, file_path in enumerate(source_files):
        logger.info(f"[{idx + 1}/{len(source_files)}] 处理: {file_path.name}")

        result = import_single_audio(file_path, dry_run=args.dry_run)

        if result["status"] == "imported":
            imported.append(result)
            logger.info(f"  ✅ 入库成功: audio_id={result['audio_id']}")
            if result["low_quality_tags"]:
                logger.info(f"     低质量标记: {', '.join(result['low_quality_tags'])}")
        else:
            rejected.append(result)
            logger.info(f"  ❌ 拒绝入库: {result['reason']}")
            # 移到 rejected 目录
            move_to_rejected(file_path, result["reason"], dry_run=args.dry_run)

    # 汇总
    logger.info("=" * 60)
    logger.info("入库完成")
    logger.info(f"  成功入库: {len(imported)}")
    logger.info(f"  拒绝入库: {len(rejected)}")
    logger.info(f"  总计: {len(source_files)}")
    logger.info(f"  日志文件: {log_file}")

    if imported:
        logger.info("")
        logger.info("⚠️  下一步：运行 gen_raw_checksum.py 生成完整性基线")
        logger.info("   python3 scripts/00_collect/gen_raw_checksum.py")

    if rejected:
        logger.info("")
        logger.info("⚠️  rejected/ 目录需要定期人工清理，否则磁盘持续膨胀")
        logger.info(f"   拒绝日志: {REJECTED_LOG}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
