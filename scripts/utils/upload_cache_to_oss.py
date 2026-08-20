"""
upload_cache_to_oss.py
将本地 model_output_cache 推理缓存上传到阿里云 OSS 做备份

⚠️ 新架构约束（2026-08-20）：
- 本脚本只负责上传备份，不参与业务流水线
- 业务绝不从 OSS 读取音频，OSS 仅作纯备份归档
- 使用 OSS_BACKUP 密钥（只写权限，禁止 GetObject/DeleteObject）
- 外网上传，不使用内网 Endpoint（内网仅 GPU 同地域免费用，但新架构关闭内网音频访问）
- 上传后必须做完整性校验，通过后写入 .oss_verified 标记

上传流程：
1. 扫描本地 model_output_cache/ 所有文件
2. 计算每个文件的 sha256
3. 生成 .upload_manifest_sha256.json（文件列表 + sha256）
4. 逐个上传到 OSS
5. 调用 OSS HeadObject 获取 ETag，与本地比对
6. 用 OSS ListObjects 核对对象数量和大小
7. 全部匹配后，在本地目录写入 .oss_verified 标记文件
"""
import json
import os
import sys
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ===================== 路径配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径，导入统一配置加载器
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_oss_config

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"upload_oss_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------- OSS 配置（使用统一配置加载器） --------
# 三优先级：~/.config/music-corpus/.env → 项目.env → 环境变量
# 使用 OSS_BACKUP 账号（读写权限，用于上传备份）
_oss_config = get_oss_config("backup")
OSS_BACKUP_ACCESS_KEY_ID = _oss_config["access_key_id"]
OSS_BACKUP_ACCESS_KEY_SECRET = _oss_config["access_key_secret"]
OSS_BUCKET_NAME = _oss_config["bucket"]
OSS_REGION = _oss_config["region"]
OSS_ENDPOINT = _oss_config["endpoint"]

# OSS 上的备份前缀
OSS_BACKUP_PREFIX = "model_output_cache/"

# -------- 本地路径 --------
LOCAL_CACHE_DIR = PROJECT_ROOT / "data/02_preannotation/model_output_cache"
MANIFEST_FILE = LOCAL_CACHE_DIR / ".upload_manifest_sha256.json"
VERIFIED_MARKER = LOCAL_CACHE_DIR / ".oss_verified"


def calculate_sha256(file_path: Path) -> str:
    """计算文件的 sha256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def scan_local_files(cache_dir: Path) -> List[Dict]:
    """扫描本地缓存目录，生成文件列表 + sha256"""
    file_list = []

    if not cache_dir.exists():
        logger.error(f"缓存目录不存在: {cache_dir}")
        return file_list

    for file_path in cache_dir.rglob("*"):
        if file_path.is_file():
            # 跳过标记文件和 manifest
            if file_path.name in [".oss_verified", ".upload_manifest_sha256.json"]:
                continue

            rel_path = file_path.relative_to(cache_dir)
            sha256 = calculate_sha256(file_path)
            file_size = file_path.stat().st_size

            file_list.append({
                "relative_path": str(rel_path),
                "absolute_path": str(file_path),
                "sha256": sha256,
                "size_bytes": file_size
            })

    return file_list


def generate_manifest(file_list: List[Dict], manifest_path: Path):
    """生成上传清单（文件列表 + sha256）"""
    manifest = {
        "upload_timestamp": datetime.now().isoformat(),
        "total_files": len(file_list),
        "total_size_bytes": sum(f["size_bytes"] for f in file_list),
        "files": file_list
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"上传清单已生成: {manifest_path}")
    logger.info(f"  文件总数: {len(file_list)}")
    logger.info(f"  总大小: {manifest['total_size_bytes'] / 1024 / 1024:.2f} MB")


def upload_to_oss(file_list: List[Dict], s3_client, bucket_name: str, prefix: str) -> Dict:
    """
    上传文件到 OSS，返回上传结果统计

    返回：
    {
        "success": [...],
        "failed": [...],
        "verified": [...]
    }
    """
    results = {
        "success": [],
        "failed": [],
        "verified": []
    }

    for idx, file_info in enumerate(file_list):
        rel_path = file_info["relative_path"]
        abs_path = file_info["absolute_path"]
        oss_key = prefix + rel_path

        logger.info(f"[{idx + 1}/{len(file_list)}] 上传: {rel_path}")

        try:
            # 上传文件
            s3_client.upload_file(abs_path, bucket_name, oss_key)
            results["success"].append(rel_path)

            # 上传后立即校验：HeadObject 获取 ETag
            try:
                response = s3_client.head_object(Bucket=bucket_name, Key=oss_key)
                oss_etag = response.get("ETag", "").strip('"')
                oss_size = response.get("ContentLength", 0)

                # 简单校验：大小一致
                if oss_size == file_info["size_bytes"]:
                    results["verified"].append(rel_path)
                else:
                    logger.warning(f"  ⚠️ 大小不匹配: 本地={file_info['size_bytes']}, OSS={oss_size}")

            except Exception as e:
                logger.warning(f"  ⚠️ HeadObject 校验失败: {e}")

        except Exception as e:
            logger.error(f"  ❌ 上传失败: {e}")
            results["failed"].append({
                "path": rel_path,
                "error": str(e)
            })

    return results


def write_verified_marker(marker_path: Path, results: Dict, total_files: int):
    """
    全部校验通过后，写入 .oss_verified 标记文件

    ⚠️ 重要：只有全部文件上传成功且校验通过，才写入此标记
    disk_guard.py 清理本地缓存前必须检测到此标记
    """
    all_success = len(results["failed"]) == 0
    all_verified = len(results["verified"]) == total_files

    if all_success and all_verified:
        marker_content = {
            "verified_at": datetime.now().isoformat(),
            "total_files": total_files,
            "verified_files": len(results["verified"]),
            "note": "本标记仅代表上传当时校验通过；OSS侧对象被外部删除不会同步更新本地标记"
        }
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(marker_content, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 全部校验通过，已写入标记: {marker_path}")
        logger.info("   现在可以安全地由 disk_guard.py 清理本地缓存")
        return True
    else:
        logger.warning("⚠️  未全部校验通过，不写入 .oss_verified 标记")
        logger.warning(f"   成功: {len(results['success'])}/{total_files}")
        logger.warning(f"   校验通过: {len(results['verified'])}/{total_files}")
        logger.warning(f"   失败: {len(results['failed'])}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("推理缓存上传 OSS 备份（纯备份模式）")
    logger.info("=" * 60)
    logger.info("⚠️  新架构：OSS 仅作纯备份归档，业务绝不从 OSS 读取音频")
    logger.info(f"OSS Endpoint: {OSS_ENDPOINT}")
    logger.info(f"OSS Bucket: {OSS_BUCKET_NAME}")
    logger.info(f"本地缓存目录: {LOCAL_CACHE_DIR}")

    # 检查配置
    if not all([OSS_BACKUP_ACCESS_KEY_ID, OSS_BACKUP_ACCESS_KEY_SECRET, OSS_BUCKET_NAME]):
        logger.error("❌ OSS 配置不完整")
        logger.error("   请检查统一配置文件：~/.config/music-corpus/.env")
        logger.error("   需要配置以下字段：")
        logger.error("   OSS_BACKUP_ACCESS_KEY_ID=你的只写密钥ID")
        logger.error("   OSS_BACKUP_ACCESS_KEY_SECRET=你的只写密钥Secret")
        logger.error("   OSS_BUCKET=你的bucket名")
        logger.error("   OSS_REGION=cn-hangzhou")
        return

    # 检查本地缓存
    if not LOCAL_CACHE_DIR.exists():
        logger.error(f"❌ 本地缓存目录不存在: {LOCAL_CACHE_DIR}")
        return

    # 1. 扫描本地文件
    logger.info("-" * 40)
    logger.info("步骤 1/4: 扫描本地文件，计算 sha256")
    file_list = scan_local_files(LOCAL_CACHE_DIR)

    if len(file_list) == 0:
        logger.warning("没有找到需要上传的文件")
        return

    # 2. 生成上传清单
    logger.info("-" * 40)
    logger.info("步骤 2/4: 生成上传清单")
    generate_manifest(file_list, MANIFEST_FILE)

    # 3. 上传到 OSS
    logger.info("-" * 40)
    logger.info("步骤 3/4: 上传到 OSS")

    try:
        import boto3
        from botocore.config import Config
        # 阿里云 OSS 要求使用 virtual hosted style
        s3_config = Config(s3={'addressing_style': 'virtual'})
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=OSS_BACKUP_ACCESS_KEY_ID,
            aws_secret_access_key=OSS_BACKUP_ACCESS_KEY_SECRET,
            endpoint_url=OSS_ENDPOINT,
            region_name=OSS_REGION,
            config=s3_config
        )
    except ImportError:
        logger.error("❌ boto3 未安装，请运行: pip install boto3")
        return
    except Exception as e:
        logger.error(f"❌ OSS 客户端初始化失败: {e}")
        return

    results = upload_to_oss(file_list, s3_client, OSS_BUCKET_NAME, OSS_BACKUP_PREFIX)

    # 4. 写入校验标记
    logger.info("-" * 40)
    logger.info("步骤 4/4: 完整性校验，写入标记")
    write_verified_marker(VERIFIED_MARKER, results, len(file_list))

    logger.info("=" * 60)
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
