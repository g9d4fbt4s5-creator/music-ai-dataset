"""
ls_backup_export.py
Label Studio 标注数据备份脚本

⚠️ 核心约束：
- 标注数据是项目核心资产，必须定期备份
- 备份文件带时间戳，不覆盖旧备份
- 备份后生成 sha256 校验，确保完整性
- 可选上传到 OSS 做异地备份
- 备份目录纳入 .gitignore（标注数据可能包含敏感信息）

功能：
1. 从 Label Studio API 导出标注数据（JSON/JSONL/CSV）
2. 从本地 Label Studio 导出文件备份
3. 备份 Label Studio 配置（labeling_interface.xml）
4. 生成备份清单（文件列表 + sha256 + 时间戳）
5. 可选上传到 OSS

用法：
    # 从 Label Studio API 导出并备份
    python ls_backup_export.py --api-url http://localhost:8080 --api-token xxx --project-id 1

    # 从本地导出文件备份
    python ls_backup_export.py --local-file ./export_ls/annotations.jsonl

    # 备份后上传到 OSS
    python ls_backup_export.py --local-file ./export_ls/annotations.jsonl --upload-oss

    # 列出所有备份
    python ls_backup_export.py --list

    # 验证备份完整性
    python ls_backup_export.py --verify ./backups/ls_backup_20260820_143000/
"""
import os
import sys
import json
import hashlib
import logging
import argparse
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径，导入统一配置加载器
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
from config_loader import get_oss_config

# 时区
TZ = timezone(timedelta(hours=8))

# 备份目录
BACKUP_DIR = PROJECT_ROOT / "data" / "03_human_review" / "backups"

# Label Studio 配置目录
LS_CONFIG_DIR = PROJECT_ROOT / "configs" / "label_studio"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"ls_backup_{time_str}.log"
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


def create_backup_directory() -> Path:
    """创建带时间戳的备份目录"""
    backup_name = datetime.now(TZ).strftime("ls_backup_%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / backup_name
    backup_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建备份目录: {backup_path}")
    return backup_path


def backup_ls_config(backup_path: Path) -> List[Dict]:
    """备份 Label Studio 配置文件"""
    backed_up = []

    if not LS_CONFIG_DIR.exists():
        logger.warning(f"Label Studio 配置目录不存在: {LS_CONFIG_DIR}")
        return backed_up

    for config_file in LS_CONFIG_DIR.rglob("*"):
        if not config_file.is_file():
            continue

        rel_path = config_file.relative_to(LS_CONFIG_DIR)
        target_path = backup_path / "config" / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(config_file, target_path)
        sha256 = calculate_sha256(target_path)

        backed_up.append({
            "file": str(rel_path),
            "size_bytes": target_path.stat().st_size,
            "sha256": sha256,
        })
        logger.info(f"  已备份配置: {rel_path}")

    return backed_up


def backup_from_local_file(local_file: Path, backup_path: Path) -> Optional[Dict]:
    """从本地 Label Studio 导出文件备份"""
    if not local_file.exists():
        logger.error(f"本地文件不存在: {local_file}")
        return None

    # 复制文件到备份目录
    target_file = backup_path / "annotations" / local_file.name
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_file, target_file)

    sha256 = calculate_sha256(target_file)
    file_size = target_file.stat().st_size

    # 尝试解析文件，统计标注数量
    annotation_count = 0
    try:
        if local_file.suffix == ".jsonl":
            with open(target_file, "r", encoding="utf-8") as f:
                annotation_count = sum(1 for line in f if line.strip())
        elif local_file.suffix == ".json":
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    annotation_count = len(data)
    except Exception as e:
        logger.warning(f"解析标注文件失败（不影响备份）: {e}")

    logger.info(f"  已备份标注文件: {local_file.name} ({annotation_count} 条标注)")

    return {
        "file": local_file.name,
        "size_bytes": file_size,
        "sha256": sha256,
        "annotation_count": annotation_count,
        "source": "local_file",
    }


def backup_from_api(api_url: str, api_token: str, project_id: int,
                    backup_path: Path, export_format: str = "JSON") -> Optional[Dict]:
    """
    从 Label Studio API 导出标注数据

    参数：
        api_url: Label Studio API 地址（如 http://localhost:8080）
        api_token: API Token
        project_id: 项目 ID
        export_format: 导出格式（JSON, JSON_MIN, CSV）
    """
    try:
        import requests
    except ImportError:
        logger.error("requests 未安装，请运行: pip install requests")
        return None

    # 导出标注
    export_url = f"{api_url.rstrip('/')}/api/projects/{project_id}/export"
    headers = {"Authorization": f"Token {api_token}"}
    params = {"exportType": export_format}

    logger.info(f"从 Label Studio API 导出: {export_url}")

    try:
        response = requests.get(export_url, headers=headers, params=params, timeout=300)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"API 导出失败: {e}")
        return None

    # 保存导出文件
    file_ext = export_format.lower().replace("_", ".")
    if file_ext == "json.min":
        file_ext = "jsonl"
    export_file = backup_path / "annotations" / f"annotations.{file_ext}"
    export_file.parent.mkdir(parents=True, exist_ok=True)

    with open(export_file, "wb") as f:
        f.write(response.content)

    sha256 = calculate_sha256(export_file)
    file_size = export_file.stat().st_size

    # 统计标注数量
    annotation_count = 0
    try:
        if file_ext == "jsonl":
            with open(export_file, "r", encoding="utf-8") as f:
                annotation_count = sum(1 for line in f if line.strip())
        elif file_ext == "json":
            data = response.json()
            if isinstance(data, list):
                annotation_count = len(data)
    except Exception:
        pass

    logger.info(f"  已导出标注: {annotation_count} 条")

    # 导出项目配置
    try:
        config_url = f"{api_url.rstrip('/')}/api/projects/{project_id}"
        config_response = requests.get(config_url, headers=headers, timeout=30)
        config_response.raise_for_status()
        project_config = config_response.json()

        config_file = backup_path / "config" / "project_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(project_config, f, ensure_ascii=False, indent=2)

        # 保存 labeling_config
        if "label_config" in project_config:
            lc_file = backup_path / "config" / "labeling_interface.xml"
            with open(lc_file, "w", encoding="utf-8") as f:
                f.write(project_config["label_config"])
            logger.info("  已备份项目配置和标注界面")
    except Exception as e:
        logger.warning(f"导出项目配置失败（不影响标注备份）: {e}")

    return {
        "file": f"annotations.{file_ext}",
        "size_bytes": file_size,
        "sha256": sha256,
        "annotation_count": annotation_count,
        "source": "api",
        "api_url": api_url,
        "project_id": project_id,
        "export_format": export_format,
    }


def generate_backup_manifest(backup_path: Path, annotations: Optional[Dict],
                              configs: List[Dict]) -> Path:
    """生成备份清单"""
    manifest = {
        "backup_name": backup_path.name,
        "created_at": datetime.now(TZ).isoformat(),
        "annotations": annotations,
        "configs": configs,
        "total_files": len(configs) + (1 if annotations else 0),
        "total_size_bytes": sum(c["size_bytes"] for c in configs) + (annotations["size_bytes"] if annotations else 0),
    }

    manifest_file = backup_path / "backup_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"生成备份清单: {manifest_file}")
    return manifest_file


def upload_backup_to_oss(backup_path: Path) -> bool:
    """上传备份到 OSS"""
    oss_config = get_oss_config("backup")

    if not all([oss_config["access_key_id"], oss_config["access_key_secret"], oss_config["bucket"]]):
        logger.error("OSS 配置不完整，跳过上传")
        return False

    try:
        import boto3
        from botocore.config import Config
        s3_config = Config(s3={'addressing_style': 'virtual'})
        s3 = boto3.client(
            "s3",
            aws_access_key_id=oss_config["access_key_id"],
            aws_secret_access_key=oss_config["access_key_secret"],
            endpoint_url=oss_config["endpoint"],
            region_name=oss_config["region"],
            config=s3_config
        )
    except Exception as e:
        logger.error(f"OSS 客户端初始化失败: {e}")
        return False

    oss_prefix = f"labelstudio_backups/{backup_path.name}/"
    uploaded = 0
    failed = 0

    for file_path in backup_path.rglob("*"):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(backup_path)
        oss_key = oss_prefix + str(rel_path)

        try:
            s3.upload_file(str(file_path), oss_config["bucket"], oss_key)
            uploaded += 1
            logger.info(f"  已上传: {rel_path}")
        except Exception as e:
            logger.error(f"  上传失败 {rel_path}: {e}")
            failed += 1

    logger.info(f"OSS 上传完成: 成功 {uploaded}, 失败 {failed}")
    return failed == 0


def list_backups():
    """列出所有备份"""
    if not BACKUP_DIR.exists():
        logger.info("暂无备份")
        return

    backups = sorted([d for d in BACKUP_DIR.iterdir() if d.is_dir()], reverse=True)

    print("=" * 80)
    print(f"{'备份名称':<35} {'创建时间':<25} {'标注数':<10} {'大小'}")
    print("-" * 80)

    for backup in backups:
        manifest_file = backup / "backup_manifest.json"
        annotation_count = "?"
        total_size = 0

        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                annotation_count = str(manifest.get("annotations", {}).get("annotation_count", "?"))
                total_size = manifest.get("total_size_bytes", 0)
            except Exception:
                pass

        # 计算目录大小
        if total_size == 0:
            total_size = sum(f.stat().st_size for f in backup.rglob("*") if f.is_file())

        created_at = backup.stat().st_mtime
        created_str = datetime.fromtimestamp(created_at, TZ).strftime("%Y-%m-%d %H:%M:%S")

        size_str = f"{total_size / 1024 / 1024:.2f} MB" if total_size > 1024 * 1024 else f"{total_size / 1024:.1f} KB"

        print(f"{backup.name:<35} {created_str:<25} {annotation_count:<10} {size_str}")

    print("=" * 80)
    print(f"共 {len(backups)} 个备份")


def verify_backup(backup_path: Path) -> bool:
    """验证备份完整性"""
    manifest_file = backup_path / "backup_manifest.json"

    if not manifest_file.exists():
        logger.error(f"备份清单不存在: {manifest_file}")
        return False

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    all_valid = True

    # 验证标注文件
    annotations = manifest.get("annotations")
    if annotations:
        file_path = backup_path / "annotations" / annotations["file"]
        if file_path.exists():
            actual_sha256 = calculate_sha256(file_path)
            if actual_sha256 == annotations["sha256"]:
                logger.info(f"✅ 标注文件校验通过: {annotations['file']}")
            else:
                logger.error(f"❌ 标注文件校验失败: {annotations['file']}")
                all_valid = False
        else:
            logger.error(f"❌ 标注文件缺失: {annotations['file']}")
            all_valid = False

    # 验证配置文件
    for config in manifest.get("configs", []):
        file_path = backup_path / "config" / config["file"]
        if file_path.exists():
            actual_sha256 = calculate_sha256(file_path)
            if actual_sha256 == config["sha256"]:
                logger.info(f"✅ 配置文件校验通过: {config['file']}")
            else:
                logger.error(f"❌ 配置文件校验失败: {config['file']}")
                all_valid = False
        else:
            logger.error(f"❌ 配置文件缺失: {config['file']}")
            all_valid = False

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Label Studio 标注数据备份脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从本地导出文件备份
  python ls_backup_export.py --local-file ./export_ls/annotations.jsonl

  # 从 Label Studio API 导出
  python ls_backup_export.py --api-url http://localhost:8080 --api-token xxx --project-id 1

  # 备份后上传到 OSS
  python ls_backup_export.py --local-file ./export.jsonl --upload-oss

  # 列出所有备份
  python ls_backup_export.py --list

  # 验证备份完整性
  python ls_backup_export.py --verify ./backups/ls_backup_20260820_143000
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local-file", type=str, help="本地 Label Studio 导出文件路径")
    group.add_argument("--api-url", type=str, help="Label Studio API 地址")
    group.add_argument("--list", action="store_true", help="列出所有备份")
    group.add_argument("--verify", type=str, help="验证指定备份的完整性")

    parser.add_argument("--api-token", type=str, help="Label Studio API Token")
    parser.add_argument("--project-id", type=int, help="Label Studio 项目 ID")
    parser.add_argument("--export-format", type=str, default="JSON",
                        choices=["JSON", "JSON_MIN", "CSV"], help="导出格式（默认 JSON）")
    parser.add_argument("--upload-oss", action="store_true", help="备份后上传到 OSS")
    parser.add_argument("--no-config-backup", action="store_true", help="不备份 Label Studio 配置")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Label Studio 标注数据备份")
    logger.info("=" * 60)

    if args.list:
        list_backups()
        return

    if args.verify:
        backup_path = Path(args.verify).resolve()
        logger.info(f"验证备份: {backup_path}")
        valid = verify_backup(backup_path)
        if valid:
            logger.info("✅ 备份完整性验证通过")
        else:
            logger.error("❌ 备份完整性验证失败")
        sys.exit(0 if valid else 1)

    # 创建备份目录
    backup_path = create_backup_directory()

    # 备份标注数据
    annotations = None
    if args.local_file:
        local_file = Path(args.local_file).resolve()
        logger.info(f"从本地文件备份: {local_file}")
        annotations = backup_from_local_file(local_file, backup_path)
    elif args.api_url:
        if not args.api_token or not args.project_id:
            logger.error("使用 API 导出必须提供 --api-token 和 --project-id")
            sys.exit(1)
        annotations = backup_from_api(
            api_url=args.api_url,
            api_token=args.api_token,
            project_id=args.project_id,
            backup_path=backup_path,
            export_format=args.export_format
        )
    else:
        logger.error("请指定 --local-file 或 --api-url")
        sys.exit(1)

    if not annotations:
        logger.error("标注数据备份失败")
        sys.exit(1)

    # 备份配置文件
    configs = []
    if not args.no_config_backup:
        logger.info("备份 Label Studio 配置...")
        configs = backup_ls_config(backup_path)

    # 生成备份清单
    generate_backup_manifest(backup_path, annotations, configs)

    # 上传到 OSS
    if args.upload_oss:
        logger.info("上传备份到 OSS...")
        upload_backup_to_oss(backup_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ 备份完成: {backup_path}")
    logger.info(f"   标注数: {annotations.get('annotation_count', '?')}")
    logger.info(f"   配置文件: {len(configs)} 个")
    logger.info(f"   日志文件: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
