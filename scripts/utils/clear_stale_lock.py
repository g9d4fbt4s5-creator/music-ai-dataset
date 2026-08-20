"""
clear_stale_lock.py
显式清理僵尸锁工具

⚠️ 核心约束：
- 默认只检查和告警，不自动删除
- 必须带 --force-stale-clean 参数才执行删除
- 防止误中断长时任务（机器负载高时任务可能超过2小时）
- fcntl.flock 内核锁已经可以处理进程崩溃；时间阈值只做告警提示

用法：
    # 只检查，不删除
    python clear_stale_lock.py

    # 检查并强制清理僵尸锁
    python clear_stale_lock.py --force-stale-clean

    # 自定义超时阈值（秒）
    python clear_stale_lock.py --stale-threshold 3600 --force-stale-clean
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCK_FILE = PROJECT_ROOT / "data" / ".pipeline_lock"

# 默认超时阈值（秒）
DEFAULT_STALE_THRESHOLD = 2 * 60 * 60  # 2小时

# 时区
TZ = timezone(timedelta(hours=8))


def is_pid_running(pid: int) -> bool:
    """检查 PID 是否还在运行"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def read_lock_file() -> dict:
    """读取锁文件内容"""
    if not LOCK_FILE.exists():
        return {}
    try:
        with open(LOCK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def check_lock_status(lock_content: dict, stale_threshold: int) -> dict:
    """
    检查锁状态

    返回：
        {
            "exists": bool,
            "pid": int,
            "pid_running": bool,
            "stage": str,
            "started": str,
            "elapsed_hours": float,
            "is_stale": bool,
            "stale_reasons": [str],
            "can_force_clean": bool,
        }
    """
    result = {
        "exists": False,
        "pid": None,
        "pid_running": False,
        "stage": None,
        "started": None,
        "elapsed_hours": 0,
        "is_stale": False,
        "stale_reasons": [],
        "can_force_clean": False,
    }

    if not lock_content:
        return result

    result["exists"] = True
    result["pid"] = lock_content.get("pid")
    result["stage"] = lock_content.get("stage")
    result["started"] = lock_content.get("started")

    stale_reasons = []

    # 检查 PID
    pid = lock_content.get("pid")
    if pid:
        pid_running = is_pid_running(pid)
        result["pid_running"] = pid_running
        if not pid_running:
            stale_reasons.append(f"PID {pid} 已不存在（进程可能已崩溃）")

    # 检查时间
    started_str = lock_content.get("started", "")
    if started_str:
        try:
            started = datetime.fromisoformat(started_str)
            elapsed = (datetime.now(TZ) - started).total_seconds()
            result["elapsed_hours"] = elapsed / 3600
            if elapsed > stale_threshold:
                stale_reasons.append(
                    f"锁已存在 {result['elapsed_hours']:.1f} 小时"
                    f"（超过 {stale_threshold/3600:.0f} 小时阈值）"
                )
        except (ValueError, TypeError):
            stale_reasons.append("无法解析 started 时间字段")

    result["stale_reasons"] = stale_reasons
    result["is_stale"] = len(stale_reasons) > 0

    # 是否可以强制清理：
    # - PID 不存在 → 可以清理
    # - PID 存在但超时 → 可以清理（但需用户确认）
    result["can_force_clean"] = result["is_stale"]

    return result


def force_clean_lock(lock_content: dict) -> bool:
    """
    强制清理锁文件

    注意：这是危险操作，只有在确认是僵尸锁时才执行
    """
    pid = lock_content.get("pid")

    # 如果 PID 还在运行，警告但仍执行（用户已明确 --force-stale-clean）
    if pid and is_pid_running(pid):
        print(f"⚠️  警告：PID {pid} 仍在运行！")
        print(f"   强制删除锁可能导致两个流水线并行写 data/，产生脏数据")
        print(f"   建议先确认该进程是否真的是僵尸进程")
        response = input("   确认继续删除？(yes/no): ")
        if response.lower() != "yes":
            print("已取消")
            return False

    # 删除锁文件
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
            print(f"✅ 锁文件已删除: {LOCK_FILE}")
            return True
        else:
            print("锁文件不存在")
            return True
    except OSError as e:
        print(f"❌ 删除锁文件失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="显式清理僵尸锁工具（默认只检查，不删除）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 只检查锁状态
  python clear_stale_lock.py

  # 强制清理僵尸锁
  python clear_stale_lock.py --force-stale-clean

  # 自定义超时阈值（1小时）
  python clear_stale_lock.py --stale-threshold 3600 --force-stale-clean
        """
    )
    parser.add_argument(
        "--force-stale-clean",
        action="store_true",
        help="强制清理僵尸锁（默认只检查，不删除）"
    )
    parser.add_argument(
        "--stale-threshold",
        type=int,
        default=DEFAULT_STALE_THRESHOLD,
        help=f"超时阈值（秒），默认 {DEFAULT_STALE_THRESHOLD}（2小时）"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("流水线锁检查")
    print("=" * 60)
    print(f"锁文件: {LOCK_FILE}")
    print(f"超时阈值: {args.stale_threshold/3600:.0f} 小时")
    print(f"模式: {'强制清理' if args.force_stale_clean else '只检查（不删除）'}")
    print()

    # 读取锁文件
    lock_content = read_lock_file()

    # 检查状态
    status = check_lock_status(lock_content, args.stale_threshold)

    if not status["exists"]:
        print("✅ 当前无锁，一切正常")
        print("=" * 60)
        sys.exit(0)

    # 打印锁信息
    print("📋 当前锁信息:")
    print(f"  PID:        {status['pid']}")
    print(f"  PID运行中:  {'是' if status['pid_running'] else '否'}")
    print(f"  阶段:       {status['stage']}")
    print(f"  开始时间:   {status['started']}")
    print(f"  已运行:     {status['elapsed_hours']:.1f} 小时")
    print()

    if status["is_stale"]:
        print("⚠️  检测到可能的僵尸锁:")
        for reason in status["stale_reasons"]:
            print(f"   - {reason}")
        print()

        if args.force_stale_clean:
            print("🔧 执行强制清理...")
            success = force_clean_lock(lock_content)
            print()
            if success:
                print("✅ 清理完成")
            else:
                print("❌ 清理失败或已取消")
            print("=" * 60)
            sys.exit(0 if success else 1)
        else:
            print("💡 如需强制清理，请运行:")
            print(f"   python {sys.argv[0]} --force-stale-clean")
            print()
            print("⚠️  注意：")
            print("   - fcntl.flock 内核锁会在进程崩溃时自动释放")
            print("   - 时间阈值只做告警，不代表锁一定是僵尸")
            print("   - 长时任务（大模型推理、大量音频处理）可能超过2小时")
            print("   - 强制清理前请确认 PID 对应的进程确实已崩溃")
    else:
        print("✅ 锁状态正常（PID 仍在运行，未超时）")

    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
