#!/usr/bin/env python3
"""
并行 FLAC/WAV → MP3 转换（带超时，替代 shell 循环防止 ffmpeg 卡住）

用法：
    python convert_to_mp3_parallel.py --input-dir DIR --output-dir DIR [--workers 4] [--timeout 120] [--bitrate 320k]

特性：
    - 保持输入目录的相对路径结构
    - 线程池并行转换（默认4线程）
    - 单文件超时（默认120秒，防止 ffmpeg 死锁）
    - 跳过已存在的 MP3（幂等）
    - 详细进度输出（OK/SKIP/FAIL/TIMEOUT）
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".flac", ".wav", ".flac", ".ogg", ".m4a"}


def convert_one(args_tuple):
    """转换单个文件，带超时。返回 (status, src_path, error_msg)"""
    src_path, dst_path, bitrate, timeout = args_tuple

    if dst_path.exists():
        return ("skip", str(src_path), "already exists")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_path),
        "-c:a", "libmp3lame", "-b:a", bitrate, "-ar", "48000",
        "-map_metadata", "-1",
        str(dst_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0 and dst_path.exists():
            return ("ok", str(src_path), None)
        else:
            err = result.stderr.strip()[:200] if result.stderr else "unknown error"
            return ("fail", str(src_path), err)
    except subprocess.TimeoutExpired:
        # 清理可能的部分文件
        if dst_path.exists():
            try:
                dst_path.unlink()
            except OSError:
                pass
        return ("timeout", str(src_path), f"exceeded {timeout}s")
    except Exception as e:
        return ("fail", str(src_path), str(e)[:200])


def collect_files(input_dir: Path, output_dir: Path):
    """收集所有音频文件，保持相对路径结构。返回 [(src, dst, bitrate, timeout), ...]"""
    tasks = []
    for src in sorted(input_dir.rglob("*")):
        if src.is_file() and src.suffix.lower() in AUDIO_EXTS:
            rel = src.relative_to(input_dir)
            dst = output_dir / rel.with_suffix(".mp3")
            tasks.append((src, dst))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="并行 FLAC/WAV → MP3 转换（带超时）"
    )
    parser.add_argument("--input-dir", required=True, help="输入目录（含 FLAC/WAV）")
    parser.add_argument("--output-dir", required=True, help="输出目录（保持相对路径结构）")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数（默认4）")
    parser.add_argument("--timeout", type=int, default=120, help="单文件超时秒数（默认120）")
    parser.add_argument("--bitrate", type=str, default="320k", help="MP3 码率（默认320k）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集文件
    file_pairs = collect_files(input_dir, output_dir)
    total = len(file_pairs)
    if total == 0:
        logger.warning(f"未找到音频文件（{AUDIO_EXTS}）于 {input_dir}")
        sys.exit(0)

    logger.info(
        f"找到 {total} 个音频文件 | workers={args.workers} | "
        f"timeout={args.timeout}s | bitrate={args.bitrate}"
    )
    logger.info(f"输入: {input_dir}")
    logger.info(f"输出: {output_dir}")

    # 构建任务参数
    tasks = [(src, dst, args.bitrate, args.timeout) for src, dst in file_pairs]

    # 并行执行
    ok = skip = fail = timeout = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_one, t): t for t in tasks}
        for future in as_completed(futures):
            status, src, err = future.result()
            name = Path(src).name
            if status == "ok":
                ok += 1
                logger.info(f"OK  [{ok + skip}/{total}] {name}")
            elif status == "skip":
                skip += 1
                logger.info(f"SKIP [{ok + skip}/{total}] {name} (已存在)")
            elif status == "timeout":
                timeout += 1
                logger.warning(f"TIMEOUT {name}: {err}")
            else:
                fail += 1
                logger.error(f"FAIL {name}: {err}")

    # 汇总
    logger.info("=" * 50)
    logger.info(f"完成: ok={ok}, skip={skip}, fail={fail}, timeout={timeout}, total={total}")
    if fail > 0 or timeout > 0:
        logger.warning(f"有 {fail + timeout} 个文件未成功转换")
        sys.exit(1)
    else:
        logger.info("全部转换成功")
        sys.exit(0)


if __name__ == "__main__":
    main()
