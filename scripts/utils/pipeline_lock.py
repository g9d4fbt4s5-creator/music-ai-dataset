"""
pipeline_lock.py
流水线并发锁工具

⚠️ 核心约束：
- 使用 fcntl.flock（进程级锁，进程崩溃时操作系统自动释放）
- 锁超时仅告警，不无条件自动删除僵尸锁
- 提供 --force-stale-clean 显式清理入口，防止误中断长时任务
- 任何修改 data/ 的流水线前后必须调用 acquire/release

锁文件位置：data/.pipeline_lock
锁文件内容（JSON）：
{
    "pid": 12345,
    "stage": "01_preprocess",
    "started": "2026-08-20T08:00:00+08:00",
    "host": "macbook-pro"
}

用法（Python API）：
    from pipeline_lock import PipelineLock
    lock = PipelineLock(stage="01_preprocess")
    lock.acquire()
    try:
        # 执行流水线
        pass
    finally:
        lock.release()

用法（命令行）：
    python pipeline_lock.py acquire --stage 01_preprocess
    python pipeline_lock.py release
    python pipeline_lock.py status
"""
import os
import sys
import json
import time
import fcntl
import socket
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCK_FILE = PROJECT_ROOT / "data" / ".pipeline_lock"

# 超时阈值（秒）：超过此时间的锁被视为可能的僵尸锁
# 注意：仅用于告警，不自动删除
STALE_THRESHOLD_SECONDS = 2 * 60 * 60  # 2小时

# 时区
TZ = timezone(timedelta(hours=8))  # UTC+8

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineLock:
    """流水线并发锁"""

    def __init__(self, stage: str = "unknown", lock_file: Path = None):
        """
        初始化锁

        参数：
            stage: 流水线阶段名称，如 "01_preprocess", "02_preannotation"
            lock_file: 锁文件路径，默认 data/.pipeline_lock
        """
        self.stage = stage
        self.lock_file = lock_file or LOCK_FILE
        self._fd = None
        self._acquired = False

    def _read_lock_content(self) -> dict:
        """读取锁文件内容"""
        if not self.lock_file.exists():
            return {}
        try:
            with open(self.lock_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _write_lock_content(self, content: dict):
        """写入锁文件内容"""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    def _is_pid_running(self, pid: int) -> bool:
        """检查 PID 是否还在运行"""
        try:
            os.kill(pid, 0)  # signal 0 不发送信号，只检查进程是否存在
            return True
        except (OSError, ProcessLookupError):
            return False

    def _check_stale(self, lock_content: dict) -> tuple:
        """
        检查锁是否为僵尸锁

        返回：
            (is_stale, reason)
        """
        pid = lock_content.get("pid")
        started_str = lock_content.get("started", "")

        # 检查 PID 是否还在运行
        if pid and not self._is_pid_running(pid):
            return True, f"PID {pid} 已不存在（进程可能已崩溃）"

        # 检查创建时间
        if started_str:
            try:
                started = datetime.fromisoformat(started_str)
                elapsed = (datetime.now(TZ) - started).total_seconds()
                if elapsed > STALE_THRESHOLD_SECONDS:
                    hours = elapsed / 3600
                    return True, f"锁已存在 {hours:.1f} 小时（超过 {STALE_THRESHOLD_SECONDS/3600:.0f} 小时阈值）"
            except (ValueError, TypeError):
                pass

        return False, ""

    def acquire(self, timeout: int = 0) -> bool:
        """
        获取锁

        参数：
            timeout: 等待超时时间（秒），0 表示不等待，获取失败立即返回 False

        返回：
            True 获取成功，False 获取失败
        """
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        # 打开锁文件
        self._fd = open(self.lock_file, "w")

        try:
            # 尝试获取文件锁（非阻塞）
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break  # 获取成功
                except (IOError, OSError):
                    # 获取失败，检查是否等待
                    if timeout <= 0 or (time.time() - start_time) >= timeout:
                        # 读取现有锁信息用于告警
                        existing = self._read_lock_content()
                        logger.warning(f"❌ 获取锁失败，锁被占用")
                        if existing:
                            logger.warning(f"   占用者: PID={existing.get('pid')}, "
                                         f"阶段={existing.get('stage')}, "
                                         f"开始={existing.get('started')}")
                            is_stale, reason = self._check_stale(existing)
                            if is_stale:
                                logger.warning(f"   ⚠️  可能是僵尸锁: {reason}")
                                logger.warning(f"   如需强制清理，请运行:")
                                logger.warning(f"   python scripts/utils/clear_stale_lock.py --force-stale-clean")
                        self._fd.close()
                        self._fd = None
                        return False
                    time.sleep(1)  # 等待1秒后重试

            # 获取成功，写入锁信息
            lock_content = {
                "pid": os.getpid(),
                "stage": self.stage,
                "started": datetime.now(TZ).isoformat(),
                "host": socket.gethostname(),
            }
            self._write_lock_content(lock_content)
            self._acquired = True

            logger.info(f"✅ 获取锁成功: stage={self.stage}, pid={os.getpid()}")
            return True

        except Exception as e:
            logger.error(f"获取锁时发生异常: {e}")
            if self._fd:
                self._fd.close()
                self._fd = None
            return False

    def release(self) -> bool:
        """
        释放锁

        返回：
            True 释放成功，False 释放失败
        """
        if not self._acquired or not self._fd:
            logger.warning("锁未被获取，无需释放")
            return False

        try:
            # 清空锁文件内容
            self._fd.seek(0)
            self._fd.truncate()
            # 释放文件锁
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None
            self._acquired = False

            # 删除锁文件
            if self.lock_file.exists():
                try:
                    self.lock_file.unlink()
                except OSError:
                    pass

            logger.info("✅ 锁已释放")
            return True

        except Exception as e:
            logger.error(f"释放锁时发生异常: {e}")
            return False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False  # 不抑制异常


@contextmanager
def pipeline_lock(stage: str = "unknown"):
    """
    上下文管理器，方便 with 语句使用

    用法：
        with pipeline_lock("01_preprocess"):
            # 执行流水线
            pass
    """
    lock = PipelineLock(stage=stage)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


def get_lock_status() -> dict:
    """
    获取当前锁状态

    返回：
        包含锁状态的字典
    """
    lock = PipelineLock()
    content = lock._read_lock_content()

    if not content:
        return {
            "locked": False,
            "message": "当前无锁"
        }

    is_stale, reason = lock._check_stale(content)
    pid_running = lock._is_pid_running(content.get("pid", 0)) if content.get("pid") else False

    return {
        "locked": True,
        "pid": content.get("pid"),
        "pid_running": pid_running,
        "stage": content.get("stage"),
        "started": content.get("started"),
        "host": content.get("host"),
        "is_stale": is_stale,
        "stale_reason": reason,
    }


# ===================== 命令行入口 =====================
def main():
    parser = argparse.ArgumentParser(description="流水线并发锁工具")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # acquire
    acquire_parser = subparsers.add_parser("acquire", help="获取锁")
    acquire_parser.add_argument("--stage", default="unknown", help="流水线阶段名称")
    acquire_parser.add_argument("--timeout", type=int, default=0, help="等待超时时间（秒），0=不等待")

    # release
    subparsers.add_parser("release", help="释放锁")

    # status
    subparsers.add_parser("status", help="查看锁状态")

    args = parser.parse_args()

    if args.command == "acquire":
        lock = PipelineLock(stage=args.stage)
        success = lock.acquire(timeout=args.timeout)
        if success:
            print("锁已获取，按 Ctrl+C 释放...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                lock.release()
        sys.exit(0 if success else 1)

    elif args.command == "release":
        lock = PipelineLock()
        # 注意：命令行 release 只能释放自己获取的锁
        # 如果是其他进程的锁，需要用 clear_stale_lock.py
        if lock._fd:
            lock.release()
            sys.exit(0)
        else:
            print("当前进程未持有锁，无法释放")
            print("如需清理其他进程的僵尸锁，请使用 clear_stale_lock.py")
            sys.exit(1)

    elif args.command == "status":
        status = get_lock_status()
        print(json.dumps(status, ensure_ascii=False, indent=2))
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
