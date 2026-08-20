"""
oss_local_client.py
本地 Mac 端 OSS 备份客户端（外网模式）

⚠️ 新架构约束（2026-08-20）：
- 本客户端只负责上传备份，不包含任何音频下载/读取方法
- 业务绝不从 OSS 读取音频，OSS 仅作纯备份归档
- 使用 OSS_BACKUP 密钥（只写权限：PutObject + ListObjects，禁止 GetObject/DeleteObject）
- 使用外网 Endpoint
- GPU ↔ Mac 的数据传输通过 rsync 直接进行，不走 OSS

使用场景：
- 本地完成预标注后，将 preann_csv 上传到 OSS 做备份
- 本地完成标注后，将 LS 导出结果上传到 OSS 做备份
- 定期备份配置文件、元数据到 OSS

注意：音频文件本身存储在本地磁盘，不通过 OSS 中转
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional

# 添加 utils 目录到路径，导入统一配置加载器
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_oss_config

# ===================== 配置 =====================
# 使用统一配置加载器（三优先级：~/.config/music-corpus/.env → 项目.env → 环境变量）
# 使用 OSS_BACKUP 账号（读写权限，用于上传备份）
_default_oss_config = get_oss_config("backup")

logger = logging.getLogger(__name__)


class OSSLocalClient:
    """
    本地 Mac 端 OSS 备份客户端

    只提供上传方法，不提供下载/读取音频的方法。
    灾难恢复请使用 disaster_recovery.py + OSS_RECOVERY 只读密钥。
    """

    def __init__(self,
                 access_key_id: Optional[str] = None,
                 access_key_secret: Optional[str] = None,
                 bucket_name: Optional[str] = None,
                 region: Optional[str] = None,
                 endpoint: Optional[str] = None):
        self.access_key_id = access_key_id or _default_oss_config["access_key_id"]
        self.access_key_secret = access_key_secret or _default_oss_config["access_key_secret"]
        self.bucket_name = bucket_name or _default_oss_config["bucket"]
        self.region = region or _default_oss_config["region"]
        self.endpoint = endpoint or _default_oss_config["endpoint"]
        self._s3_client = None

    def _get_client(self):
        if self._s3_client is None:
            try:
                import boto3
                from botocore.config import Config
                # 阿里云 OSS 要求使用 virtual hosted style
                s3_config = Config(s3={'addressing_style': 'virtual'})
                self._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.access_key_secret,
                    endpoint_url=self.endpoint,
                    region_name=self.region,
                    config=s3_config
                )
            except ImportError:
                raise ImportError("boto3 未安装，请运行: pip install boto3")
        return self._s3_client

    def upload_file(self, local_path: str, oss_key: str) -> bool:
        """上传单个文件到 OSS"""
        s3 = self._get_client()
        try:
            s3.upload_file(local_path, self.bucket_name, oss_key)
            logger.info(f"上传成功: {oss_key}")
            return True
        except Exception as e:
            logger.error(f"上传失败 {oss_key}: {e}")
            return False

    def upload_directory(self, local_dir: str, oss_prefix: str,
                         exclude_patterns: Optional[List[str]] = None) -> Dict:
        """上传整个目录到 OSS"""
        exclude_patterns = exclude_patterns or []
        local_path = Path(local_dir)

        if not local_path.exists():
            logger.error(f"目录不存在: {local_dir}")
            return {"success": [], "failed": [], "total": 0}

        results = {"success": [], "failed": [], "total": 0}

        for file_path in local_path.rglob("*"):
            if not file_path.is_file():
                continue
            if any(pattern in file_path.name for pattern in exclude_patterns):
                continue

            rel_path = file_path.relative_to(local_path)
            oss_key = oss_prefix.rstrip("/") + "/" + str(rel_path)
            results["total"] += 1

            if self.upload_file(str(file_path), oss_key):
                results["success"].append(oss_key)
            else:
                results["failed"].append(oss_key)

        logger.info(f"目录上传完成: 成功 {len(results['success'])}/{results['total']}, "
                    f"失败 {len(results['failed'])}")
        return results

    def verify_upload(self, oss_key: str, local_path: str) -> bool:
        """校验上传结果（HeadObject 比对大小）"""
        s3 = self._get_client()
        try:
            response = s3.head_object(Bucket=self.bucket_name, Key=oss_key)
            oss_size = response.get("ContentLength", 0)
            local_size = os.path.getsize(local_path)
            return oss_size == local_size
        except Exception as e:
            logger.error(f"校验异常 {oss_key}: {e}")
            return False

    def list_objects(self, prefix: str) -> List[str]:
        """列出 OSS 指定前缀下的对象"""
        s3 = self._get_client()
        try:
            response = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj["Key"] for obj in response.get("Contents", [])]
        except Exception as e:
            logger.error(f"列出对象失败: {e}")
            return []

    # ⚠️ 不提供 download_file / get_object 方法
    # 新架构禁止业务从 OSS 读取音频


# ===================== 命令行入口 =====================
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) < 2:
        print("用法:")
        print("  python oss_local_client.py upload <local_path> <oss_key>")
        print("  python oss_local_client.py upload-dir <local_dir> <oss_prefix>")
        print("  python oss_local_client.py list <prefix>")
        sys.exit(1)

    client = OSSLocalClient()
    command = sys.argv[1]

    if command == "upload":
        if len(sys.argv) != 4:
            print("用法: python oss_local_client.py upload <local_path> <oss_key>")
            sys.exit(1)
        success = client.upload_file(sys.argv[2], sys.argv[3])
        sys.exit(0 if success else 1)

    elif command == "upload-dir":
        if len(sys.argv) != 4:
            print("用法: python oss_local_client.py upload-dir <local_dir> <oss_prefix>")
            sys.exit(1)
        results = client.upload_directory(sys.argv[2], sys.argv[3])
        sys.exit(0 if len(results["failed"]) == 0 else 1)

    elif command == "list":
        if len(sys.argv) != 3:
            print("用法: python oss_local_client.py list <prefix>")
            sys.exit(1)
        for k in client.list_objects(sys.argv[2]):
            print(k)

    else:
        print(f"未知命令: {command}")
        sys.exit(1)
