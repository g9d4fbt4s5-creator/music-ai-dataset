#!/usr/bin/env python3
"""
按时间戳切片合集音频

从配置文件读取时间戳，用 ffmpeg 将合集音频切成单独的歌曲。

用法：
  python scripts/utils/split_by_timestamps.py --config configs/split_timestamps.json
  python scripts/utils/split_by_timestamps.py --config configs/split_timestamps.json --input-dir data/00_raw_collect/raw_audio_bilibili --output-dir data/00_raw_collect/raw_audio_sliced
  python scripts/utils/split_by_timestamps.py --config configs/split_timestamps.json --import-manifest
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def time_to_seconds(time_str: str) -> float:
    """将 MM:SS 或 HH:MM:SS 转换为秒"""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"无效时间格式: {time_str}")


def get_audio_duration(file_path: Path) -> float:
    """获取音频时长（秒）"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(result.stdout.strip())


def sanitize_filename(name: str) -> str:
    """清理文件名中的特殊字符"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")
    return name.strip()


def split_audio(
    input_file: Path,
    output_dir: Path,
    tracks: List[Dict],
    output_prefix: str,
    dry_run: bool = False,
) -> List[Path]:
    """
    按时间戳切片音频。

    Args:
        input_file: 输入音频文件
        output_dir: 输出目录
        tracks: 歌曲列表（含 start, title, artist）
        output_prefix: 输出文件名前缀
        dry_run: 只打印不执行

    Returns:
        切片后的文件路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取总时长
    total_duration = get_audio_duration(input_file)
    logger.info(f"输入: {input_file.name} ({total_duration:.0f}秒)")
    logger.info(f"歌曲数: {len(tracks)}")

    sliced_files = []

    for i, track in enumerate(tracks):
        start_sec = time_to_seconds(track["start"])

        # 结束时间 = 下一首的开始时间，或文件末尾
        if i < len(tracks) - 1:
            end_sec = time_to_seconds(tracks[i + 1]["start"])
        else:
            end_sec = total_duration

        duration = end_sec - start_sec

        # 构建输出文件名
        artist = sanitize_filename(track.get("artist", "unknown"))
        title = sanitize_filename(track.get("title", f"track_{i+1}"))
        output_filename = f"{output_prefix}_{i+1:02d}_{artist}-{title}.flac"
        output_path = output_dir / output_filename

        logger.info(f"  [{i+1}/{len(tracks)}] {track['start']} - {artist} - {title} ({duration:.0f}秒)")

        if dry_run:
            sliced_files.append(output_path)
            continue

        if output_path.exists():
            logger.info(f"    已存在，跳过: {output_filename}")
            sliced_files.append(output_path)
            continue

        # 用 ffmpeg 切片（-ss 在 -i 之前表示快速定位，-c copy 表示直接复制流不重新编码）
        # 但 FLAC 用 -c copy 可能有问题，改用 -c:a flac 重新编码
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", str(input_file),
            "-t", str(duration),
            "-c:a", "flac",
            "-compression_level", "5",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and output_path.exists():
                size_mb = output_path.stat().st_size / 1024 / 1024
                logger.info(f"    ✅ {output_filename} ({size_mb:.1f}MB)")
                sliced_files.append(output_path)
            else:
                logger.error(f"    ❌ 切片失败: {result.stderr[-200:] if result.stderr else '未知错误'}")
        except subprocess.TimeoutExpired:
            logger.error(f"    ❌ 超时")
        except Exception as e:
            logger.error(f"    ❌ 出错: {e}")

    return sliced_files


def main():
    parser = argparse.ArgumentParser(description="按时间戳切片合集音频")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/split_timestamps.json",
        help="时间戳配置文件路径（默认 configs/split_timestamps.json）",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/00_raw_collect/raw_audio_bilibili",
        help="输入音频目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/00_raw_collect/raw_audio_sliced",
        help="输出切片目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印切片计划，不实际执行",
    )
    args = parser.parse_args()

    # 读取配置
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    input_dir = PROJECT_ROOT / args.input_dir
    output_dir = PROJECT_ROOT / args.output_dir

    logger.info(f"配置: {config_path}")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"待切片合集: {len(config)} 个")

    all_sliced = []

    for filename, playlist_config in config.items():
        input_file = input_dir / filename
        if not input_file.exists():
            logger.warning(f"文件不存在，跳过: {input_file}")
            continue

        logger.info(f"\n{'='*60}")
        sliced = split_audio(
            input_file=input_file,
            output_dir=output_dir,
            tracks=playlist_config["tracks"],
            output_prefix=playlist_config["output_prefix"],
            dry_run=args.dry_run,
        )
        all_sliced.extend(sliced)

    # 总结
    logger.info(f"\n{'='*60}")
    logger.info("切片完成")
    logger.info(f"  成功切片: {len(all_sliced)} 首")
    if all_sliced:
        total_size = sum(f.stat().st_size for f in all_sliced if f.exists())
        logger.info(f"  总大小: {total_size/1024/1024:.1f} MB")
        logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
