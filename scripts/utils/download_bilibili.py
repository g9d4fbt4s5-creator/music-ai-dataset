#!/usr/bin/env python3
"""
Bilibili 音频批量下载脚本（基于 yt-dlp）

从 Bilibili 视频链接提取音频，保存为 FLAC 或 MP3 格式。
适用于 50 首试点数据采集。

用法：
  # 下载单个视频的音频
  python scripts/utils/download_bilibili.py --url "https://www.bilibili.com/video/BV1xx411c7mD"

  # 从文件批量下载（每行一个链接或 BV 号）
  python scripts/utils/download_bilibili.py --list urls.txt

  # 指定输出格式和目录
  python scripts/utils/download_bilibili.py --url "BV1xx411c7mD" --format mp3 --output data/00_raw_collect/raw_audio_bilibili

  # 下载整个播放列表
  python scripts/utils/download_bilibili.py --url "https://www.bilibili.com/medialist/play/ml123456" --playlist

依赖：
  pip install yt-dlp
  brew install ffmpeg
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 支持的输出格式
SUPPORTED_FORMATS = ["flac", "mp3", "m4a", "wav", "opus"]

# 默认输出目录
DEFAULT_OUTPUT_DIR = "data/00_raw_collect/raw_audio_bilibili"


def check_dependencies():
    """检查 yt-dlp 和 ffmpeg 是否安装"""
    missing = []

    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append("yt-dlp")

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append("ffmpeg")

    if missing:
        logger.error(f"缺少依赖: {', '.join(missing)}")
        logger.info("安装方法:")
        logger.info("  pip install yt-dlp")
        logger.info("  brew install ffmpeg")
        return False
    return True


def normalize_url(url: str) -> str:
    """标准化 Bilibili 链接"""
    url = url.strip()
    if not url:
        return url

    # 如果是 BV 号，补全链接
    if url.startswith("BV") or url.startswith("bv"):
        return f"https://www.bilibili.com/video/{url}"

    # 如果是 av 号，补全链接
    if url.startswith("av") or url.startswith("AV"):
        return f"https://www.bilibili.com/video/{url}"

    return url


def download_audio(
    url: str,
    output_dir: Path,
    audio_format: str = "flac",
    playlist: bool = False,
) -> Optional[Path]:
    """
    下载单个 Bilibili 视频的音频。

    Args:
        url: 视频链接或 BV 号
        output_dir: 输出目录
        audio_format: 输出音频格式（flac/mp3/m4a/wav/opus）
        playlist: 是否下载整个播放列表

    Returns:
        下载后的文件路径，失败返回 None
    """
    url = normalize_url(url)
    if not url:
        logger.warning("空链接，跳过")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 yt-dlp 命令
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", audio_format,
        "--audio-quality", "0",  # 最佳质量
        "--output", str(output_dir / "%(title)s.%(ext)s"),
        "--no-playlist" if not playlist else "--yes-playlist",
        "--restrict-filenames",  # 限制文件名（避免特殊字符）
        "--no-warnings",
        url,
    ]

    logger.info(f"下载: {url}")
    logger.info(f"格式: {audio_format} | 输出: {output_dir}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
        )

        if result.returncode == 0:
            # 查找下载的文件
            downloaded = list(output_dir.glob(f"*.{audio_format}"))
            if downloaded:
                # 取最新修改的文件
                latest = max(downloaded, key=lambda p: p.stat().st_mtime)
                size_mb = latest.stat().st_size / 1024 / 1024
                logger.info(f"✅ 下载完成: {latest.name} ({size_mb:.1f} MB)")
                return latest
            else:
                logger.warning("下载完成但未找到输出文件")
                return None
        else:
            logger.error(f"下载失败 (返回码 {result.returncode})")
            if result.stderr:
                # 只显示最后几行错误
                error_lines = result.stderr.strip().split("\n")[-5:]
                for line in error_lines:
                    logger.error(f"  {line}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("下载超时（5分钟）")
        return None
    except Exception as e:
        logger.error(f"下载出错: {e}")
        return None


def batch_download(
    urls: List[str],
    output_dir: Path,
    audio_format: str = "flac",
    playlist: bool = False,
) -> List[Path]:
    """
    批量下载多个视频的音频。

    Args:
        urls: 视频链接列表
        output_dir: 输出目录
        audio_format: 输出音频格式
        playlist: 是否下载播放列表

    Returns:
        成功下载的文件路径列表
    """
    success = []
    failed = []

    logger.info(f"开始批量下载: {len(urls)} 个视频")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"音频格式: {audio_format}")

    for i, url in enumerate(urls, 1):
        logger.info(f"\n--- [{i}/{len(urls)}] ---")
        result = download_audio(url, output_dir, audio_format, playlist)
        if result:
            success.append(result)
        else:
            failed.append(url)

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("批量下载完成")
    logger.info(f"  成功: {len(success)}/{len(urls)}")
    logger.info(f"  失败: {len(failed)}")
    if failed:
        logger.info("  失败链接:")
        for url in failed:
            logger.info(f"    {url}")
    logger.info("=" * 60)

    return success


def load_urls_from_file(file_path: Path) -> List[str]:
    """从文件加载 URL 列表（每行一个链接或 BV 号）"""
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return []

    urls = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    logger.info(f"从文件加载 {len(urls)} 个链接")
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Bilibili 音频批量下载脚本（基于 yt-dlp）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 下载单个视频
  python download_bilibili.py --url "https://www.bilibili.com/video/BV1xx411c7mD"

  # 用 BV 号下载
  python download_bilibili.py --url "BV1xx411c7mD"

  # 批量下载
  python download_bilibili.py --list urls.txt

  # 指定格式和目录
  python download_bilibili.py --url "BV1xx411c7mD" --format mp3 --output downloads/
        """,
    )
    parser.add_argument("--url", type=str, help="视频链接或 BV 号")
    parser.add_argument("--list", type=str, help="包含链接列表的文件路径")
    parser.add_argument(
        "--format",
        type=str,
        default="flac",
        choices=SUPPORTED_FORMATS,
        help="输出音频格式（默认 flac）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--playlist",
        action="store_true",
        help="下载整个播放列表（默认只下载单个视频）",
    )
    args = parser.parse_args()

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    output_dir = Path(args.output)

    # 收集 URL
    urls = []
    if args.url:
        urls.append(args.url)
    if args.list:
        urls.extend(load_urls_from_file(Path(args.list)))

    if not urls:
        parser.error("请提供 --url 或 --list 参数")

    # 下载
    if len(urls) == 1:
        download_audio(urls[0], output_dir, args.format, args.playlist)
    else:
        batch_download(urls, output_dir, args.format, args.playlist)


if __name__ == "__main__":
    main()
