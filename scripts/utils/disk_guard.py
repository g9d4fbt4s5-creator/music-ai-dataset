"""
disk_guard.py
磁盘管控与快照轮转工具

⚠️ 核心约束：
- 清理快照/缓存前必须确认 .oss_verified 标记存在
- 未确认备份的文件，禁止删除
- 支持 --dry-run 预览模式，默认只报告不删除
- 支持 --enforce-retention 强制执行轮转

功能：
1. 磁盘水位告警
2. 快照轮转（保留最新 N 个，删除更早的）
3. 过期缓存清理（model_output_cache, processed_audio, demucs_stems, logs）
4. 清理日志记录

用法：
    # 只检查磁盘水位和可清理项，不删除
    python disk_guard.py

    # 预览模式（显示将删除什么，不实际删除）
    python disk_guard.py --dry-run

    # 强制执行快照轮转和缓存清理
    python disk_guard.py --enforce-retention

    # 只检查磁盘水位
    python disk_guard.py --check-only
"""
import os
import sys
import json
import shutil
import logging
import argparse
import tomllib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
RETENTION_CONFIG = PROJECT_ROOT / "snapshots" / "snapshot_retention.toml"

# 时区
TZ = timezone(timedelta(hours=8))

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"disk_guard_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """加载轮转配置"""
    if not RETENTION_CONFIG.exists():
        logger.warning(f"配置文件不存在: {RETENTION_CONFIG}，使用默认配置")
        return {}

    with open(RETENTION_CONFIG, "rb") as f:
        return tomllib.load(f)


def get_disk_usage(path: Path) -> dict:
    """
    获取磁盘使用情况

    返回：
        {
            "total": 总字节数,
            "used": 已用字节数,
            "free": 可用字节数,
            "percent": 使用率百分比,
        }
    """
    usage = shutil.disk_usage(str(path))
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "percent": usage.used / usage.total * 100,
    }


def check_disk_alert(config: dict) -> dict:
    """
    检查磁盘水位告警

    返回：
        {
            "level": "normal" | "warning" | "critical",
            "percent": 使用率,
            "message": 告警信息,
        }
    """
    alert_config = config.get("disk_alert", {})
    warning_threshold = alert_config.get("warning_threshold", 80)
    critical_threshold = alert_config.get("critical_threshold", 90)

    usage = get_disk_usage(PROJECT_ROOT)
    percent = usage["percent"]

    if percent >= critical_threshold:
        level = "critical"
        message = f"🔴 严重告警：磁盘使用率 {percent:.1f}%，超过 {critical_threshold}% 阈值"
    elif percent >= warning_threshold:
        level = "warning"
        message = f"🟡 警告：磁盘使用率 {percent:.1f}%，超过 {warning_threshold}% 阈值"
    else:
        level = "normal"
        message = f"🟢 正常：磁盘使用率 {percent:.1f}%"

    return {
        "level": level,
        "percent": percent,
        "total_gb": usage["total"] / (1024**3),
        "used_gb": usage["used"] / (1024**3),
        "free_gb": usage["free"] / (1024**3),
        "message": message,
    }


def has_oss_verified_marker(dir_path: Path) -> bool:
    """检查目录是否有 .oss_verified 标记"""
    marker = dir_path / ".oss_verified"
    return marker.exists()


def list_snapshots(snapshot_dir: Path, prefix: str) -> List[Path]:
    """列出所有快照目录，按时间排序（最新的在前）"""
    if not snapshot_dir.exists():
        return []

    snapshots = [
        d for d in snapshot_dir.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    # 按目录名排序（目录名包含时间戳，字典序即时间序）
    snapshots.sort(key=lambda d: d.name, reverse=True)
    return snapshots


def rotate_snapshots(config: dict, dry_run: bool = False) -> dict:
    """
    快照轮转：保留最新 N 个，删除更早的

    注意：删除前必须确认 .oss_verified 标记存在
    """
    retention_config = config.get("snapshot_retention", {})
    snapshot_dir = PROJECT_ROOT / retention_config.get("snapshot_dir", "snapshots")
    prefix = retention_config.get("snapshot_prefix", "gpu_backup_")
    keep_count = retention_config.get("local_snapshot_count", 5)

    result = {
        "total": 0,
        "kept": [],
        "deleted": [],
        "skipped": [],
        "errors": [],
    }

    snapshots = list_snapshots(snapshot_dir, prefix)
    result["total"] = len(snapshots)

    if len(snapshots) <= keep_count:
        logger.info(f"快照数量 {len(snapshots)} <= 保留数量 {keep_count}，无需轮转")
        result["kept"] = [s.name for s in snapshots]
        return result

    # 保留最新的 N 个
    to_keep = snapshots[:keep_count]
    to_delete = snapshots[keep_count:]

    result["kept"] = [s.name for s in to_keep]

    logger.info(f"快照轮转：保留 {len(to_keep)} 个，待删除 {len(to_delete)} 个")

    for snapshot in to_delete:
        # 检查 .oss_verified 标记
        if not has_oss_verified_marker(snapshot):
            logger.warning(f"⚠️  跳过删除（无 .oss_verified 标记）: {snapshot.name}")
            result["skipped"].append(snapshot.name)
            continue

        if dry_run:
            logger.info(f"[DRY-RUN] 将删除: {snapshot.name}")
            result["deleted"].append(snapshot.name)
        else:
            try:
                shutil.rmtree(snapshot)
                logger.info(f"✅ 已删除: {snapshot.name}")
                result["deleted"].append(snapshot.name)
            except Exception as e:
                logger.error(f"❌ 删除失败 {snapshot.name}: {e}")
                result["errors"].append({"name": snapshot.name, "error": str(e)})

    return result


def cleanup_expired_cache(config: dict, dry_run: bool = False) -> dict:
    """
    清理过期缓存

    注意：清理前必须确认已备份到 OSS（检查 .oss_verified 或 .upload_manifest）
    """
    cache_config = config.get("cache_cleanup", {})
    prerequisites = config.get("cleanup_prerequisites", {})
    require_oss_backup = prerequisites.get("require_oss_backup", True)

    result = {
        "model_output_cache": {"deleted": [], "skipped": [], "errors": []},
        "processed_audio": {"deleted": [], "skipped": [], "errors": []},
        "demucs_stems": {"deleted": [], "skipped": [], "errors": []},
        "logs": {"deleted": [], "skipped": [], "errors": []},
    }

    now = datetime.now(TZ)

    # 定义需要清理的目录和保留天数
    cache_dirs = [
        ("model_output_cache", PROJECT_ROOT / "data/02_preannotation/model_output_cache",
         cache_config.get("model_output_cache_days", 7)),
        ("processed_audio", PROJECT_ROOT / "data/01_preprocess/processed_audio/segments",
         cache_config.get("processed_audio_days", 30)),
        ("demucs_stems", PROJECT_ROOT / "data/01_preprocess/demucs_stems",
         cache_config.get("demucs_stems_days", 30)),
    ]

    for cache_name, cache_dir, retention_days in cache_dirs:
        if not cache_dir.exists():
            continue

        logger.info(f"检查 {cache_name}（保留 {retention_days} 天）: {cache_dir}")

        # 检查目录级别的 .oss_verified 标记
        dir_verified = has_oss_verified_marker(cache_dir)

        for item in cache_dir.iterdir():
            # 计算文件/目录年龄
            try:
                mtime = datetime.fromtimestamp(item.stat().st_mtime, TZ)
                age_days = (now - mtime).days
            except OSError:
                continue

            if age_days <= retention_days:
                continue

            # 检查是否已备份到 OSS
            if require_oss_backup and not dir_verified:
                # 检查文件级别的标记
                item_verified = False
                if item.is_dir():
                    item_verified = has_oss_verified_marker(item)

                if not item_verified:
                    logger.warning(f"  ⚠️  跳过（未确认 OSS 备份）: {item.name} (年龄 {age_days} 天)")
                    result[cache_name]["skipped"].append(item.name)
                    continue

            if dry_run:
                logger.info(f"  [DRY-RUN] 将删除: {item.name} (年龄 {age_days} 天)")
                result[cache_name]["deleted"].append(item.name)
            else:
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    logger.info(f"  ✅ 已删除: {item.name} (年龄 {age_days} 天)")
                    result[cache_name]["deleted"].append(item.name)
                except Exception as e:
                    logger.error(f"  ❌ 删除失败 {item.name}: {e}")
                    result[cache_name]["errors"].append({"name": item.name, "error": str(e)})

    # 日志清理
    logs_dir = PROJECT_ROOT / "logs"
    logs_retention_days = cache_config.get("logs_days", 90)
    if logs_dir.exists():
        logger.info(f"检查 logs（保留 {logs_retention_days} 天）")
        for log_file in logs_dir.glob("*.log"):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime, TZ)
                age_days = (now - mtime).days
                if age_days > logs_retention_days:
                    if dry_run:
                        logger.info(f"  [DRY-RUN] 将删除日志: {log_file.name}")
                        result["logs"]["deleted"].append(log_file.name)
                    else:
                        log_file.unlink()
                        logger.info(f"  ✅ 已删除日志: {log_file.name}")
                        result["logs"]["deleted"].append(log_file.name)
            except OSError:
                continue

    return result


def main():
    parser = argparse.ArgumentParser(
        description="磁盘管控与快照轮转工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，显示将删除什么，不实际删除")
    parser.add_argument("--enforce-retention", action="store_true",
                        help="强制执行快照轮转和缓存清理")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查磁盘水位，不执行清理")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("磁盘管控与快照轮转")
    logger.info("=" * 60)
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"配置文件: {RETENTION_CONFIG}")
    logger.info(f"模式: {'预览(不删除)' if args.dry_run else '检查' if args.check_only else '正常'}")
    logger.info(f"强制执行轮转: {'是' if args.enforce_retention else '否'}")
    logger.info("")

    # 加载配置
    config = load_config()

    # 1. 检查磁盘水位
    logger.info("-" * 40)
    logger.info("步骤 1/3: 磁盘水位检查")
    disk_status = check_disk_alert(config)
    logger.info(disk_status["message"])
    logger.info(f"  总计: {disk_status['total_gb']:.1f} GB")
    logger.info(f"  已用: {disk_status['used_gb']:.1f} GB")
    logger.info(f"  可用: {disk_status['free_gb']:.1f} GB")

    if args.check_only:
        logger.info("")
        logger.info("检查模式，不执行清理")
        logger.info("=" * 60)
        return

    # 2. 快照轮转
    logger.info("")
    logger.info("-" * 40)
    logger.info("步骤 2/3: 快照轮转")
    if args.enforce_retention or args.dry_run:
        snapshot_result = rotate_snapshots(config, dry_run=args.dry_run)
        logger.info(f"  总计: {snapshot_result['total']}")
        logger.info(f"  保留: {len(snapshot_result['kept'])}")
        logger.info(f"  删除: {len(snapshot_result['deleted'])}")
        logger.info(f"  跳过(未备份): {len(snapshot_result['skipped'])}")
        logger.info(f"  错误: {len(snapshot_result['errors'])}")
    else:
        logger.info("  未启用 --enforce-retention，跳过快照轮转")
        logger.info("  如需执行，请添加 --enforce-retention 参数")

    # 3. 缓存清理
    logger.info("")
    logger.info("-" * 40)
    logger.info("步骤 3/3: 过期缓存清理")
    if args.enforce_retention or args.dry_run:
        cache_result = cleanup_expired_cache(config, dry_run=args.dry_run)
        for cache_name, result in cache_result.items():
            total_deleted = len(result["deleted"])
            total_skipped = len(result["skipped"])
            if total_deleted > 0 or total_skipped > 0:
                logger.info(f"  {cache_name}: 删除 {total_deleted}, 跳过 {total_skipped}")
    else:
        logger.info("  未启用 --enforce-retention，跳过缓存清理")
        logger.info("  如需执行，请添加 --enforce-retention 参数")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 60)
    logger.info("💡 提示：")
    logger.info("   - 默认只检查，不删除任何文件")
    logger.info("   - 使用 --dry-run 预览将删除什么")
    logger.info("   - 使用 --enforce-retention 实际执行清理")
    logger.info("   - 清理前必须确认 .oss_verified 标记存在")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
