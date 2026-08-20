"""
config_loader.py
统一配置加载器

三优先级加载（与 mtg_jamendo_meta 的 oss_pipeline_lib.py 保持一致）：
1. 全局配置：~/.config/music-corpus/.env（最高优先级）
2. 项目配置：{PROJECT_ROOT}/.env
3. 环境变量：os.environ（最低优先级，GPU 上 .bashrc 的环境变量继续生效）

设计原则：
- 密钥只存一份，改一处全局生效
- 项目目录只留 .env.example 模板，不留真实密钥
- 向后兼容：OSS_ACCESS_KEY_ID 等旧变量名继续支持
- 双账号支持：OSS_BACKUP_*（读写，上传用）/ OSS_RECOVERY_*（只读，灾难恢复用）

用法：
    from config_loader import load_config, get_oss_config

    # 加载全部配置
    config = load_config()

    # 获取 OSS 备份账号配置（读写）
    backup_config = get_oss_config("backup")
    # 返回: {"access_key_id": ..., "access_key_secret": ..., "bucket": ..., "region": ..., "endpoint": ...}

    # 获取 OSS 恢复账号配置（只读）
    recovery_config = get_oss_config("recovery")
"""
import os
import logging
from pathlib import Path
from typing import Dict, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 全局配置文件路径（统一凭据中心）
GLOBAL_CONFIG_PATH = Path.home() / ".config" / "music-corpus" / ".env"

# 项目配置文件路径
PROJECT_CONFIG_PATH = PROJECT_ROOT / ".env"

logger = logging.getLogger(__name__)


def parse_env_file(file_path: Path) -> Dict[str, str]:
    """
    解析 .env 文件

    支持格式：
        KEY=VALUE
        KEY="VALUE"
        KEY='VALUE'
        # 注释
        export KEY=VALUE
    """
    config = {}
    if not file_path.exists():
        return config

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith("#"):
                    continue
                # 去掉 export 前缀
                if line.startswith("export "):
                    line = line[7:].strip()
                # 解析 KEY=VALUE
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 去掉引号
                if len(value) >= 2:
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                config[key] = value
    except Exception as e:
        logger.warning(f"解析配置文件失败 {file_path}: {e}")

    return config


def load_config() -> Dict[str, str]:
    """
    加载配置（三优先级）

    优先级从高到低：
    1. 全局配置：~/.config/music-corpus/.env
    2. 项目配置：{PROJECT_ROOT}/.env
    3. 环境变量：os.environ

    返回：
        合并后的配置字典
    """
    config = {}

    # 优先级3：环境变量（最低）
    config.update(dict(os.environ))

    # 优先级2：项目 .env
    project_config = parse_env_file(PROJECT_CONFIG_PATH)
    if project_config:
        config.update(project_config)
        logger.debug(f"已加载项目配置: {PROJECT_CONFIG_PATH}")

    # 优先级1：全局配置（最高）
    global_config = parse_env_file(GLOBAL_CONFIG_PATH)
    if global_config:
        config.update(global_config)
        logger.debug(f"已加载全局配置: {GLOBAL_CONFIG_PATH}")

    return config


def get_oss_config(account_type: str = "backup") -> Dict[str, Optional[str]]:
    """
    获取 OSS 配置

    参数：
        account_type: "backup"（读写，上传用）或 "recovery"（只读，灾难恢复用）

    返回：
        {
            "access_key_id": str or None,
            "access_key_secret": str or None,
            "bucket": str or None,
            "region": str,
            "endpoint": str,
        }
    """
    config = load_config()

    if account_type == "backup":
        # 备份账号（读写）
        access_key_id = config.get("OSS_BACKUP_ACCESS_KEY_ID") or config.get("OSS_ACCESS_KEY_ID")
        access_key_secret = config.get("OSS_BACKUP_ACCESS_KEY_SECRET") or config.get("OSS_ACCESS_KEY_SECRET")
    elif account_type == "recovery":
        # 恢复账号（只读）
        access_key_id = config.get("OSS_RECOVERY_ACCESS_KEY_ID") or config.get("OSS_ACCESS_KEY_ID")
        access_key_secret = config.get("OSS_RECOVERY_ACCESS_KEY_SECRET") or config.get("OSS_ACCESS_KEY_SECRET")
    else:
        raise ValueError(f"不支持的账号类型: {account_type}，可选: backup, recovery")

    # 通用配置
    bucket = config.get("OSS_BUCKET") or config.get("OSS_BUCKET_NAME")
    region = config.get("OSS_REGION", "cn-hangzhou")

    # 规范化 region：兼容 "oss-cn-hangzhou" 和 "cn-hangzhou" 两种格式
    if region.startswith("oss-"):
        region = region[4:]  # 去掉 "oss-" 前缀

    # Endpoint 优先级：显式配置 > 按 region 拼接
    endpoint = config.get("OSS_ENDPOINT")
    if not endpoint:
        endpoint = f"https://oss-{region}.aliyuncs.com"

    # 规范化 endpoint：确保以 https:// 开头
    if endpoint and not endpoint.startswith("http"):
        endpoint = "https://" + endpoint

    return {
        "access_key_id": access_key_id,
        "access_key_secret": access_key_secret,
        "bucket": bucket,
        "region": region,
        "endpoint": endpoint,
    }


def get_autodl_config() -> Dict[str, Optional[str]]:
    """
    获取 AutoDL GPU 连接配置

    返回：
        {
            "host": str or None,
            "port": int or None,
            "password": str or None,
            "api_token": str or None,
            "ssh_key_path": str or None,
        }
    """
    config = load_config()

    port = config.get("AUTODL_PORT")
    if port:
        try:
            port = int(port)
        except (ValueError, TypeError):
            port = None

    return {
        "host": config.get("AUTODL_HOST"),
        "port": port,
        "password": config.get("AUTODL_PASSWORD"),
        "api_token": config.get("AUTODL_API_TOKEN"),
        "ssh_key_path": config.get("AUTODL_KEY") or config.get("AUTODL_SSH_KEY"),
    }


def check_config_status() -> Dict[str, bool]:
    """
    检查配置状态（用于启动时打印诊断信息）

    返回：
        {
            "global_config_exists": bool,
            "project_config_exists": bool,
            "oss_backup_ready": bool,
            "oss_recovery_ready": bool,
            "autodl_ready": bool,
        }
    """
    config = load_config()

    oss_backup = get_oss_config("backup")
    oss_recovery = get_oss_config("recovery")
    autodl = get_autodl_config()

    return {
        "global_config_exists": GLOBAL_CONFIG_PATH.exists(),
        "project_config_exists": PROJECT_CONFIG_PATH.exists(),
        "oss_backup_ready": bool(oss_backup["access_key_id"] and oss_backup["access_key_secret"] and oss_backup["bucket"]),
        "oss_recovery_ready": bool(oss_recovery["access_key_id"] and oss_recovery["access_key_secret"]),
        "autodl_ready": bool(autodl["host"] and autodl["port"]),
    }


def print_config_status():
    """打印配置状态诊断信息"""
    status = check_config_status()

    print("=" * 60)
    print("配置状态诊断")
    print("=" * 60)
    print(f"全局配置 (~/.config/music-corpus/.env): {'✅ 存在' if status['global_config_exists'] else '❌ 不存在'}")
    print(f"项目配置 (.env): {'✅ 存在' if status['project_config_exists'] else '⚠️  不存在（使用全局/环境变量）'}")
    print(f"OSS 备份账号 (读写): {'✅ 就绪' if status['oss_backup_ready'] else '❌ 未配置'}")
    print(f"OSS 恢复账号 (只读): {'✅ 就绪' if status['oss_recovery_ready'] else '⚠️  未配置'}")
    print(f"AutoDL 连接: {'✅ 就绪' if status['autodl_ready'] else '❌ 未配置'}")
    print("=" * 60)


if __name__ == "__main__":
    # 自测入口：打印配置状态
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    print_config_status()
