"""
verify_oss_upload.py
OSS 上传完整性校验工具

⚠️ 核心约束：
- 上传后必须校验完整性，通过后才写入 .oss_verified 标记
- .oss_verified 仅代表上传当时校验通过；OSS侧对象被外部删除不会同步更新本地标记
- 高危快照清理可加 --recheck-oss 实时重校验
- disk_guard.py 清理本地快照/缓存前必须检测到 .oss_verified 标记

校验流程：
1. 扫描本地目录，生成文件列表 + sha256 + 大小
2. 调用 OSS ListObjects 获取远端文件列表
3. 比对文件数量
4. 逐个调用 HeadObject 比对大小和 ETag
5. 全部匹配后写入 .oss_verified 标记

用法：
    # 校验快照目录
    python verify_oss_upload.py --snapshot ./snapshots/gpu_backup_20260820_173500

    # 校验 model_output_cache 目录
    python verify_oss_upload.py --dir ./data/02_preannotation/model_output_cache --oss-prefix model_output_cache/

    # 实时重校验（即使已有 .oss_verified 标记）
    python verify_oss_upload.py --snapshot ./snapshots/gpu_backup_xxx --recheck-oss

    # 只校验不写入标记
    python verify_oss_upload.py --snapshot ./snapshots/gpu_backup_xxx --check-only
"""
import os
import sys
import json
import hashlib
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径，导入统一配置加载器
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_oss_config

# 时区
TZ = timezone(timedelta(hours=8))

# -------- OSS 配置（使用统一配置加载器） --------
# 三优先级：~/.config/music-corpus/.env → 项目.env → 环境变量
# 使用 OSS_BACKUP 账号（读写权限，用于上传备份和校验）
_oss_config = get_oss_config("backup")
OSS_BACKUP_ACCESS_KEY_ID = _oss_config["access_key_id"]
OSS_BACKUP_ACCESS_KEY_SECRET = _oss_config["access_key_secret"]
OSS_BUCKET_NAME = _oss_config["bucket"]
OSS_REGION = _oss_config["region"]
OSS_ENDPOINT = _oss_config["endpoint"]

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"verify_oss_{time_str}.log"
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


def calculate_md5(file_path: Path) -> str:
    """计算文件的 MD5（用于和 OSS ETag 比对）"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def scan_local_directory(dir_path: Path) -> List[Dict]:
    """
    扫描本地目录，生成文件列表

    返回：
        [{"relative_path": ..., "absolute_path": ..., "size": ..., "sha256": ..., "md5": ...}, ...]
    """
    files = []
    if not dir_path.exists():
        logger.error(f"目录不存在: {dir_path}")
        return files

    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue
        # 跳过标记文件
        if file_path.name in [".oss_verified", ".upload_manifest_sha256.json"]:
            continue

        rel_path = str(file_path.relative_to(dir_path))
        file_size = file_path.stat().st_size

        files.append({
            "relative_path": rel_path,
            "absolute_path": str(file_path),
            "size": file_size,
            "sha256": None,  # 延迟计算
            "md5": None,     # 延迟计算
        })

    return files


def get_oss_client():
    """获取 OSS 客户端"""
    try:
        import boto3
        from botocore.config import Config
        # 阿里云 OSS 要求使用 virtual hosted style
        s3_config = Config(s3={'addressing_style': 'virtual'})
        return boto3.client(
            "s3",
            aws_access_key_id=OSS_BACKUP_ACCESS_KEY_ID,
            aws_secret_access_key=OSS_BACKUP_ACCESS_KEY_SECRET,
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


def list_oss_objects(s3_client, bucket: str, prefix: str) -> List[Dict]:
    """
    列出 OSS 指定前缀下的所有对象

    返回：
        [{"key": ..., "size": ..., "etag": ...}, ...]
    """
    objects = []
    continuation_token = None

    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
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
                "etag": obj.get("ETag", "").strip('"'),
            })

        if response.get("IsTruncated"):
            continuation_token = response.get("NextContinuationToken")
        else:
            break

    return objects


def head_oss_object(s3_client, bucket: str, key: str) -> Dict:
    """
    获取 OSS 对象的元数据（HeadObject）

    返回：
        {"size": ..., "etag": ..., "exists": True/False}
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        return {
            "exists": True,
            "size": response.get("ContentLength", 0),
            "etag": response.get("ETag", "").strip('"'),
        }
    except Exception as e:
        return {
            "exists": False,
            "size": 0,
            "etag": "",
            "error": str(e),
        }


def verify_upload(local_dir: Path, oss_prefix: str,
                  recheck: bool = False, check_only: bool = False) -> Dict:
    """
    校验上传完整性

    参数：
        local_dir: 本地目录
        oss_prefix: OSS 前缀
        recheck: 是否实时重校验（即使已有 .oss_verified 标记）
        check_only: 只校验不写入标记

    返回：
        {
            "verified": bool,
            "total_files": int,
            "matched": int,
            "mismatched": [...],
            "missing": [...],
            "extra": [...],
        }
    """
    result = {
        "verified": False,
        "total_files": 0,
        "matched": 0,
        "mismatched": [],
        "missing": [],
        "extra": [],
    }

    # 检查是否已有 .oss_verified 标记
    marker_file = local_dir / ".oss_verified"
    if marker_file.exists() and not recheck:
        logger.info(f"✅ 已有 .oss_verified 标记，跳过校验")
        logger.info(f"   如需实时重校验，请添加 --recheck-oss 参数")
        result["verified"] = True
        return result

    # 检查 OSS 配置
    if not all([OSS_BACKUP_ACCESS_KEY_ID, OSS_BACKUP_ACCESS_KEY_SECRET, OSS_BUCKET_NAME]):
        logger.error("❌ OSS 配置不完整")
        logger.error("   请设置环境变量:")
        logger.error("   export OSS_BACKUP_ACCESS_KEY_ID=xxx")
        logger.error("   export OSS_BACKUP_ACCESS_KEY_SECRET=xxx")
        logger.error("   export OSS_BUCKET_NAME=xxx")
        logger.error("   export OSS_REGION=cn-hangzhou")
        return result

    # 扫描本地文件
    logger.info(f"扫描本地目录: {local_dir}")
    local_files = scan_local_directory(local_dir)
    result["total_files"] = len(local_files)
    logger.info(f"本地文件数: {len(local_files)}")

    if len(local_files) == 0:
        logger.warning("本地目录为空，无需校验")
        result["verified"] = True
        return result

    # 获取 OSS 客户端
    s3_client = get_oss_client()
    if not s3_client:
        return result

    # 列出 OSS 对象
    logger.info(f"列出 OSS 对象，前缀: {oss_prefix}")
    oss_objects = list_oss_objects(s3_client, OSS_BUCKET_NAME, oss_prefix)
    logger.info(f"OSS 对象数: {len(oss_objects)}")

    # 构建 OSS 对象字典（key 去掉前缀）
    oss_dict = {}
    for obj in oss_objects:
        rel_key = obj["key"][len(oss_prefix):] if obj["key"].startswith(oss_prefix) else obj["key"]
        oss_dict[rel_key] = obj

    # 逐个校验
    logger.info("开始逐个校验...")
    for idx, local_file in enumerate(local_files):
        rel_path = local_file["relative_path"]
        local_size = local_file["size"]

        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info(f"  进度: {idx + 1}/{len(local_files)}")

        # 检查 OSS 是否存在
        if rel_path not in oss_dict:
            logger.warning(f"  ❌ OSS 缺失: {rel_path}")
            result["missing"].append(rel_path)
            continue

        oss_obj = oss_dict[rel_path]
        oss_size = oss_obj["size"]

        # 比对大小
        if local_size != oss_size:
            logger.warning(f"  ❌ 大小不匹配: {rel_path} (本地={local_size}, OSS={oss_size})")
            result["mismatched"].append({
                "path": rel_path,
                "reason": f"size mismatch: local={local_size}, oss={oss_size}",
            })
            continue

        # 比对 ETag（MD5）
        # 注意：只有普通上传的 ETag 是 MD5，分片上传的不是
        local_md5 = calculate_md5(Path(local_file["absolute_path"]))
        oss_etag = oss_obj["etag"]

        # 如果 ETag 包含 "-"，说明是分片上传，不能直接比对 MD5
        if "-" in oss_etag:
            logger.info(f"  ⚠️  分片上传，跳过 ETag 比对: {rel_path} (大小匹配)")
            result["matched"] += 1
        elif local_md5 == oss_etag:
            result["matched"] += 1
        else:
            logger.warning(f"  ❌ ETag 不匹配: {rel_path} (本地MD5={local_md5}, OSS ETag={oss_etag})")
            result["mismatched"].append({
                "path": rel_path,
                "reason": f"etag mismatch: local_md5={local_md5}, oss_etag={oss_etag}",
            })

    # 检查 OSS 上是否有多余的文件
    local_keys = {f["relative_path"] for f in local_files}
    for rel_key in oss_dict:
        if rel_key not in local_keys:
            logger.warning(f"  ⚠️  OSS 多余文件: {rel_key}")
            result["extra"].append(rel_key)

    # 判断是否全部通过
    all_matched = (
        len(result["mismatched"]) == 0
        and len(result["missing"]) == 0
    )
    result["verified"] = all_matched

    # 写入 .oss_verified 标记
    if all_matched and not check_only:
        marker_content = {
            "verified_at": datetime.now(TZ).isoformat(),
            "total_files": result["total_files"],
            "matched": result["matched"],
            "oss_prefix": oss_prefix,
            "note": "本标记仅代表上传当时校验通过；OSS侧对象被外部删除不会同步更新本地标记",
        }
        with open(marker_file, "w", encoding="utf-8") as f:
            json.dump(marker_content, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 校验全部通过，已写入 .oss_verified 标记: {marker_file}")
    elif not all_matched:
        logger.error(f"❌ 校验未通过")
        logger.error(f"   匹配: {result['matched']}/{result['total_files']}")
        logger.error(f"   不匹配: {len(result['mismatched'])}")
        logger.error(f"   缺失: {len(result['missing'])}")
        logger.error(f"   OSS多余: {len(result['extra'])}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="OSS 上传完整性校验工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=str, help="快照目录路径")
    group.add_argument("--dir", type=str, help="要校验的本地目录路径")

    parser.add_argument("--oss-prefix", type=str, default=None,
                        help="OSS 前缀（默认根据目录类型自动推断）")
    parser.add_argument("--recheck-oss", action="store_true",
                        help="实时重校验（即使已有 .oss_verified 标记）")
    parser.add_argument("--check-only", action="store_true",
                        help="只校验不写入 .oss_verified 标记")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("OSS 上传完整性校验")
    logger.info("=" * 60)

    # 确定本地目录和 OSS 前缀
    if args.snapshot:
        local_dir = Path(args.snapshot).resolve()
        oss_prefix = args.oss_prefix or f"snapshots/{local_dir.name}/"
    else:
        local_dir = Path(args.dir).resolve()
        oss_prefix = args.oss_prefix or f"{local_dir.name}/"

    logger.info(f"本地目录: {local_dir}")
    logger.info(f"OSS 前缀: {oss_prefix}")
    logger.info(f"OSS Bucket: {OSS_BUCKET_NAME}")
    logger.info(f"OSS Endpoint: {OSS_ENDPOINT}")
    logger.info(f"实时重校验: {'是' if args.recheck_oss else '否'}")
    logger.info(f"只校验不写入: {'是' if args.check_only else '否'}")
    logger.info("")

    if not local_dir.exists():
        logger.error(f"❌ 目录不存在: {local_dir}")
        sys.exit(1)

    # 执行校验
    result = verify_upload(
        local_dir=local_dir,
        oss_prefix=oss_prefix,
        recheck=args.recheck_oss,
        check_only=args.check_only,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("校验结果汇总")
    logger.info("=" * 60)
    logger.info(f"  总文件数: {result['total_files']}")
    logger.info(f"  匹配: {result['matched']}")
    logger.info(f"  不匹配: {len(result['mismatched'])}")
    logger.info(f"  OSS 缺失: {len(result['missing'])}")
    logger.info(f"  OSS 多余: {len(result['extra'])}")
    logger.info(f"  校验结果: {'✅ 通过' if result['verified'] else '❌ 未通过'}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    sys.exit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
