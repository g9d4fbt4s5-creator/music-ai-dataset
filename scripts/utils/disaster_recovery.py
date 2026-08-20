"""
disaster_recovery.py
灾难恢复工具：从 OSS 恢复本地数据

⚠️ 核心约束：
- 使用 OSS_RECOVERY 只读密钥（不是 OSS_BACKUP 只写密钥）
- OSS_RECOVERY 密钥只有 GetObject + ListObjects 权限，不能上传或删除
- 恢复后必须与 raw_audio_checksums.csv 比对校验 sha256
- 新协作者初始化项目时使用此脚本

使用场景：
1. 磁盘损坏，本地数据丢失
2. 新协作者加入项目，需要拉取数据
3. 误删除文件，需要从 OSS 恢复

用法：
    # 全量恢复 raw_audio
    python disaster_recovery.py --full-restore

    # 只恢复指定 audio_id
    python disaster_recovery.py --audio-ids ids.txt

    # 只校验本地文件完整性，不恢复
    python disaster_recovery.py --verify

    # 恢复指定类型的数据
    python disaster_recovery.py --full-restore --data-type raw_audio
    python disaster_recovery.py --full-restore --data-type model_output_cache
    python disaster_recovery.py --full-restore --data-type snapshots
"""
import os
import sys
import csv
import json
import hashlib
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Set

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径，导入统一配置加载器
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_oss_config

# 时区
TZ = timezone(timedelta(hours=8))

# -------- OSS 恢复配置（使用 OSS_RECOVERY 只读密钥） --------
# 三优先级：~/.config/music-corpus/.env → 项目.env → 环境变量
# 使用 OSS_RECOVERY 账号（只读权限：GetObject + ListObjects，不能上传或删除）
_oss_recovery_config = get_oss_config("recovery")
OSS_RECOVERY_ACCESS_KEY_ID = _oss_recovery_config["access_key_id"]
OSS_RECOVERY_ACCESS_KEY_SECRET = _oss_recovery_config["access_key_secret"]
OSS_BUCKET_NAME = _oss_recovery_config["bucket"]
OSS_REGION = _oss_recovery_config["region"]
OSS_ENDPOINT = _oss_recovery_config["endpoint"]

# 本地数据目录
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio"
CHECKSUM_CSV = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio_checksums.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"disaster_recovery_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def calculate_sha256(file_path: Path) -> str:
    """计算文件的 sha256"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def load_checksums() -> Dict[str, Dict]:
    """
    加载 raw_audio_checksums.csv

    返回：
        {audio_id: {"file_relative_path": ..., "sha256": ..., "file_bytes": ...}, ...}
    """
    checksums = {}
    if not CHECKSUM_CSV.exists():
        logger.warning(f"checksum 文件不存在: {CHECKSUM_CSV}")
        return checksums

    with open(CHECKSUM_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_id = row.get("audio_id", "")
            if audio_id:
                checksums[audio_id] = {
                    "file_relative_path": row.get("file_relative_path", ""),
                    "original_filename": row.get("original_filename", ""),
                    "sha256": row.get("sha256", ""),
                    "file_bytes": int(row.get("file_bytes", 0)),
                }

    logger.info(f"加载 checksum 记录: {len(checksums)} 条")
    return checksums


def get_oss_client():
    """获取 OSS 客户端（使用 OSS_RECOVERY 只读密钥）"""
    if not all([OSS_RECOVERY_ACCESS_KEY_ID, OSS_RECOVERY_ACCESS_KEY_SECRET]):
        logger.error("❌ OSS_RECOVERY 密钥未配置")
        logger.error("   请检查统一配置文件：~/.config/music-corpus/.env")
        logger.error("   需要配置以下字段：")
        logger.error("   OSS_RECOVERY_ACCESS_KEY_ID=你的只读密钥ID")
        logger.error("   OSS_RECOVERY_ACCESS_KEY_SECRET=你的只读密钥Secret")
        return None

    try:
        import boto3
        from botocore.config import Config
        # 阿里云 OSS 要求使用 virtual hosted style
        s3_config = Config(s3={'addressing_style': 'virtual'})
        return boto3.client(
            "s3",
            aws_access_key_id=OSS_RECOVERY_ACCESS_KEY_ID,
            aws_secret_access_key=OSS_RECOVERY_ACCESS_KEY_SECRET,
            endpoint_url=OSS_ENDPOINT,
            region_name=OSS_REGION,
            config=s3_config
        )
    except ImportError:
        logger.error("boto3 未安装，请运行: pip install boto3")
        return None
    except Exception as e:
        logger.error(f"OSS 客户端初始化失败: {e}")
        return None


def list_oss_objects(s3_client, prefix: str) -> List[Dict]:
    """列出 OSS 指定前缀下的所有对象"""
    objects = []
    continuation_token = None

    while True:
        kwargs = {"Bucket": OSS_BUCKET_NAME, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token

        try:
            response = s3_client.list_objects_v2(**kwargs)
        except Exception as e:
            logger.error(f"ListObjects 失败: {e}")
            return objects

        for obj in response.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
            })

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return objects


def download_oss_object(s3_client, key: str, local_path: Path) -> bool:
    """
    从 OSS 下载文件到本地

    返回：
        True 成功，False 失败
    """
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(OSS_BUCKET_NAME, key, str(local_path))
        return True
    except Exception as e:
        logger.error(f"下载失败 {key}: {e}")
        return False


def verify_local_files(checksums: Dict[str, Dict]) -> Dict:
    """
    校验本地文件完整性（与 checksum 比对）

    返回：
        {
            "total": int,
            "verified": int,
            "missing": [...],
            "corrupted": [...],
        }
    """
    result = {
        "total": len(checksums),
        "verified": 0,
        "missing": [],
        "corrupted": [],
    }

    for audio_id, info in checksums.items():
        rel_path = info["file_relative_path"]
        expected_sha256 = info["sha256"]
        local_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path

        if not local_path.exists():
            logger.warning(f"  ❌ 缺失: {audio_id} ({rel_path})")
            result["missing"].append(audio_id)
            continue

        # 计算实际 sha256
        actual_sha256 = calculate_sha256(local_path)
        if actual_sha256 != expected_sha256:
            logger.warning(f"  ❌ 损坏: {audio_id} (sha256 不匹配)")
            result["corrupted"].append(audio_id)
        else:
            result["verified"] += 1

    return result


def full_restore(data_type: str = "raw_audio", dry_run: bool = False) -> Dict:
    """
    全量恢复

    参数：
        data_type: 数据类型（raw_audio, model_output_cache, snapshots）
        dry_run: 预览模式

    返回：
        恢复结果
    """
    result = {
        "data_type": data_type,
        "total": 0,
        "downloaded": 0,
        "skipped": [],
        "errors": [],
    }

    # 确定 OSS 前缀和本地目录
    if data_type == "raw_audio":
        oss_prefix = "raw_audio/"
        local_base = PROJECT_ROOT / "data" / "00_raw_collect"
    elif data_type == "model_output_cache":
        oss_prefix = "model_output_cache/"
        local_base = PROJECT_ROOT / "data" / "02_preannotation"
    elif data_type == "snapshots":
        oss_prefix = "snapshots/"
        local_base = PROJECT_ROOT
    else:
        logger.error(f"不支持的数据类型: {data_type}")
        return result

    # 获取 OSS 客户端
    s3_client = get_oss_client()
    if not s3_client:
        return result

    # 列出 OSS 对象
    logger.info(f"列出 OSS 对象，前缀: {oss_prefix}")
    oss_objects = list_oss_objects(s3_client, oss_prefix)
    result["total"] = len(oss_objects)
    logger.info(f"OSS 对象数: {len(oss_objects)}")

    if len(oss_objects) == 0:
        logger.warning("OSS 上没有找到对象")
        return result

    # 逐个下载
    for idx, obj in enumerate(oss_objects):
        key = obj["key"]
        rel_path = key[len(oss_prefix):] if key.startswith(oss_prefix) else key
        local_path = local_base / rel_path

        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(f"  进度: {idx + 1}/{len(oss_objects)}")

        # 如果本地已存在，跳过
        if local_path.exists():
            logger.info(f"  跳过（已存在）: {rel_path}")
            result["skipped"].append(rel_path)
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] 将下载: {rel_path}")
            result["downloaded"] += 1
        else:
            if download_oss_object(s3_client, key, local_path):
                logger.info(f"  ✅ 已下载: {rel_path}")
                result["downloaded"] += 1
            else:
                result["errors"].append(rel_path)

    return result


def restore_by_audio_ids(audio_ids: Set[str], dry_run: bool = False) -> Dict:
    """
    根据 audio_id 列表恢复指定文件

    参数：
        audio_ids: audio_id 集合
        dry_run: 预览模式
    """
    result = {
        "requested": len(audio_ids),
        "found": 0,
        "downloaded": 0,
        "not_found": [],
        "errors": [],
    }

    # 加载 checksum
    checksums = load_checksums()

    # 获取 OSS 客户端
    s3_client = get_oss_client()
    if not s3_client:
        return result

    for audio_id in audio_ids:
        if audio_id not in checksums:
            logger.warning(f"  ❌ checksum 中未找到: {audio_id}")
            result["not_found"].append(audio_id)
            continue

        info = checksums[audio_id]
        rel_path = info["file_relative_path"]
        oss_key = "raw_audio/" + rel_path.replace("raw_audio/", "")
        local_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path

        result["found"] += 1

        if local_path.exists():
            logger.info(f"  跳过（已存在）: {audio_id}")
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] 将下载: {audio_id}")
            result["downloaded"] += 1
        else:
            if download_oss_object(s3_client, oss_key, local_path):
                # 校验 sha256
                actual_sha256 = calculate_sha256(local_path)
                if actual_sha256 == info["sha256"]:
                    logger.info(f"  ✅ 已下载并校验通过: {audio_id}")
                    result["downloaded"] += 1
                else:
                    logger.error(f"  ❌ 下载后校验失败: {audio_id} (sha256 不匹配)")
                    result["errors"].append(audio_id)
                    local_path.unlink(missing_ok=True)
            else:
                result["errors"].append(audio_id)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="灾难恢复工具：从 OSS 恢复本地数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 新协作者初始化：全量恢复
  python disaster_recovery.py --full-restore

  # 只校验本地文件完整性
  python disaster_recovery.py --verify

  # 恢复指定 audio_id
  python disaster_recovery.py --audio-ids ids.txt

  # 预览模式（不实际下载）
  python disaster_recovery.py --full-restore --dry-run
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--full-restore", action="store_true", help="全量恢复")
    group.add_argument("--verify", action="store_true", help="只校验本地文件完整性，不恢复")
    group.add_argument("--audio-ids", type=str, help="包含 audio_id 列表的文件（每行一个）")

    parser.add_argument("--data-type", type=str, default="raw_audio",
                        choices=["raw_audio", "model_output_cache", "snapshots"],
                        help="要恢复的数据类型（默认 raw_audio）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际下载")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("灾难恢复")
    logger.info("=" * 60)
    logger.info(f"模式: {'全量恢复' if args.full_restor else '校验' if args.verify else '按ID恢复'}")
    logger.info(f"数据类型: {args.data_type}")
    logger.info(f"预览模式: {'是' if args.dry_run else '否'}")
    logger.info(f"OSS Bucket: {OSS_BUCKET_NAME}")
    logger.info(f"OSS Endpoint: {OSS_ENDPOINT}")
    logger.info("")

    if args.verify:
        # 只校验
        logger.info("-" * 40)
        logger.info("校验本地文件完整性")
        checksums = load_checksums()
        if not checksums:
            logger.error("没有 checksum 数据，无法校验")
            sys.exit(1)

        result = verify_local_files(checksums)
        logger.info("")
        logger.info("校验结果:")
        logger.info(f"  总计: {result['total']}")
        logger.info(f"  校验通过: {result['verified']}")
        logger.info(f"  缺失: {len(result['missing'])}")
        logger.info(f"  损坏: {len(result['corrupted'])}")

        if result["missing"] or result["corrupted"]:
            logger.info("")
            logger.info("💡 如需恢复缺失/损坏的文件，请运行:")
            logger.info("   python disaster_recovery.py --full-restore")
            sys.exit(1)
        else:
            logger.info("")
            logger.info("✅ 所有文件校验通过")
            sys.exit(0)

    elif args.full_restore:
        # 全量恢复
        logger.info("-" * 40)
        logger.info(f"全量恢复: {args.data_type}")
        result = full_restore(args.data_type, dry_run=args.dry_run)

        logger.info("")
        logger.info("恢复结果:")
        logger.info(f"  OSS 对象总数: {result['total']}")
        logger.info(f"  已下载: {result['downloaded']}")
        logger.info(f"  跳过（已存在）: {len(result['skipped'])}")
        logger.info(f"  错误: {len(result['errors'])}")

        # 如果是 raw_audio，恢复后校验
        if args.data_type == "raw_audio" and not args.dry_run:
            logger.info("")
            logger.info("-" * 40)
            logger.info("恢复后校验...")
            checksums = load_checksums()
            if checksums:
                verify_result = verify_local_files(checksums)
                logger.info(f"  校验通过: {verify_result['verified']}/{verify_result['total']}")
                if verify_result["missing"] or verify_result["corrupted"]:
                    logger.warning("  ⚠️  部分文件校验失败，请检查日志")

        sys.exit(0 if len(result["errors"]) == 0 else 1)

    elif args.audio_ids:
        # 按 ID 恢复
        audio_ids_file = Path(args.audio_ids)
        if not audio_ids_file.exists():
            logger.error(f"文件不存在: {audio_ids_file}")
            sys.exit(1)

        audio_ids = set()
        with open(audio_ids_file, "r", encoding="utf-8") as f:
            for line in f:
                audio_id = line.strip()
                if audio_id:
                    audio_ids.add(audio_id)

        logger.info(f"待恢复 audio_id 数: {len(audio_ids)}")
        result = restore_by_audio_ids(audio_ids, dry_run=args.dry_run)

        logger.info("")
        logger.info("恢复结果:")
        logger.info(f"  请求: {result['requested']}")
        logger.info(f"  找到: {result['found']}")
        logger.info(f"  已下载: {result['downloaded']}")
        logger.info(f"  未找到: {len(result['not_found'])}")
        logger.info(f"  错误: {len(result['errors'])}")

        sys.exit(0 if len(result["errors"]) == 0 else 1)


if __name__ == "__main__":
    main()
