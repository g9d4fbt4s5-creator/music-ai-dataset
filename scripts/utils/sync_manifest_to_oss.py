"""
sync_manifest_to_oss.py
全局音频元数据索引同步到 OSS

⚠️ 核心约束：
- audio_manifest.csv 是灾难恢复的关键索引，必须定期同步到 OSS
- 灾难恢复时，先从 OSS 拉取此清单，再按需恢复音频文件
- 只上传元数据 CSV，不上传音频文件本身
- 使用 OSS_BACKUP 账号（只写权限）

用法：
    # 同步 audio_manifest.csv 到 OSS
    python sync_manifest_to_oss.py

    # 同步所有元数据文件（manifest + checksums + 配置）
    python sync_manifest_to_oss.py --all

    # 预览模式（不上传，只显示将上传的文件）
    python sync_manifest_to_oss.py --dry-run
"""
import os
import sys
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

# 添加 utils 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from oss_local_client import OSSLocalClient

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# OSS 前缀
OSS_MANIFEST_PREFIX = "metadata/manifests/"

# 需要同步的元数据文件
MANIFEST_FILES = [
    "data/00_raw_collect/audio_manifest.csv",
    "data/00_raw_collect/raw_audio_checksums.csv",
]

# --all 模式下额外同步的文件
ALL_METADATA_FILES = [
    "configs/cleaning_config.yaml",
    "configs/label_studio/labeling_interface.xml",
    "data/02_preannotation/label_mapping/label_mapping_dict.json",
    "environment.yml",
    "environment_gpu.yml",
    ".gitignore",
    "README.md",
]

# 时区
TZ = timezone(timedelta(hours=8))

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def sync_file_to_oss(client: OSSLocalClient, local_path: Path, oss_key: str, dry_run: bool = False) -> bool:
    """
    同步单个文件到 OSS

    Args:
        client: OSS 客户端
        local_path: 本地文件路径
        oss_key: OSS 对象键
        dry_run: 预览模式

    Returns:
        True 成功，False 失败
    """
    if not local_path.exists():
        logger.warning(f"  ⚠️  文件不存在，跳过: {local_path}")
        return False

    file_size = local_path.stat().st_size
    logger.info(f"  同步: {local_path.name} ({file_size:,} bytes) -> oss://{client.bucket_name}/{oss_key}")

    if dry_run:
        logger.info(f"  [DRY-RUN] 跳过上传")
        return True

    try:
        client.upload_file(str(local_path), oss_key)
        logger.info(f"  ✅ 上传成功")
        return True
    except Exception as e:
        logger.error(f"  ❌ 上传失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="同步元数据索引到 OSS")
    parser.add_argument("--all", action="store_true", help="同步所有元数据文件（包括配置、标签映射等）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不上传")
    parser.add_argument("--oss-prefix", type=str, default=OSS_MANIFEST_PREFIX, help="OSS 前缀")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("同步元数据索引到 OSS")
    logger.info(f"  项目根目录: {PROJECT_ROOT}")
    logger.info(f"  OSS 前缀: {args.oss_prefix}")
    logger.info(f"  预览模式: {args.dry_run}")
    logger.info("=" * 60)

    # 确定要同步的文件列表
    files_to_sync = list(MANIFEST_FILES)
    if args.all:
        files_to_sync.extend(ALL_METADATA_FILES)
        logger.info(f"  同步模式: 全部元数据文件 ({len(files_to_sync)} 个)")
    else:
        logger.info(f"  同步模式: 仅核心索引 ({len(files_to_sync)} 个)")

    # 初始化 OSS 客户端
    try:
        client = OSSLocalClient()
        logger.info(f"  OSS Bucket: {client.bucket_name}")
        logger.info(f"  OSS Region: {client.region}")
    except Exception as e:
        logger.error(f"❌ OSS 客户端初始化失败: {e}")
        sys.exit(1)

    # 同步文件
    logger.info("")
    logger.info("开始同步...")
    success_count = 0
    fail_count = 0
    skip_count = 0

    for rel_path in files_to_sync:
        local_path = PROJECT_ROOT / rel_path
        # OSS key: 前缀 + 相对路径（替换反斜杠）
        oss_key = args.oss_prefix + rel_path.replace("\\", "/")

        if not local_path.exists():
            logger.warning(f"  ⚠️  文件不存在，跳过: {rel_path}")
            skip_count += 1
            continue

        if sync_file_to_oss(client, local_path, oss_key, args.dry_run):
            success_count += 1
        else:
            fail_count += 1

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("同步完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  跳过（不存在）: {skip_count}")
    logger.info(f"  总计: {len(files_to_sync)}")
    logger.info("=" * 60)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
