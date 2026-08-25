#!/usr/bin/env python3
"""
从 Internet Archive 下载公共领域音频（免注册，CC0/公共领域）

ADR-003 50首试点采集工具：用于采集老爵士/古典老录音（1950s前）

用法：
  # 搜索并下载老爵士 78rpm 录音
  python scripts/utils/download_archive.py --query "jazz 78rpm" --limit 5

  # 搜索古典钢琴录音
  python scripts/utils/download_archive.py --query "classical piano 78rpm" --limit 3

  # 只搜索不下载（查看结果）
  python scripts/utils/download_archive.py --query "jazz 78rpm" --limit 10 --dry-run

  # 指定输出目录
  python scripts/utils/download_archive.py --query "jazz 78rpm" --limit 5 --output data/00_raw_collect/raw_audio_archive

Internet Archive API 文档：
  搜索: https://archive.org/advancedsearch.php
  元数据: https://archive.org/metadata/{identifier}
  下载: https://archive.org/download/{identifier}/{filename}
"""
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 支持的音频格式（按优先级排序）
AUDIO_FORMATS = {
    "Flac": ".flac",
    "VBR MP3": ".mp3",
    "MP3": ".mp3",
    "Ogg Vorbis": ".ogg",
    "128Kbps MP3": ".mp3",
    "64Kbps MP3": ".mp3",
}

# 请求间隔（避免被限流）
REQUEST_DELAY = 1.0


def search_archive(
    query: str,
    limit: int = 10,
    mediatype: str = "audio",
) -> List[Dict]:
    """
    搜索 Internet Archive。

    Args:
        query: 搜索关键词
        limit: 返回结果数量
        mediatype: 媒体类型（默认 audio）

    Returns:
        搜索结果列表，每个包含 identifier/title/creator 等
    """
    url = "https://archive.org/advancedsearch.php"
    params = {
        "q": f"{query} AND mediatype:{mediatype}",
        "fl[]": ["identifier", "title", "creator", "date", "description"],
        "rows": limit,
        "output": "json",
        "sort[]": "downloads desc",  # 按下载量排序，优先热门资源
    }

    logger.info(f"搜索 Internet Archive: query='{query}', limit={limit}")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    docs = resp.json().get("response", {}).get("docs", [])
    logger.info(f"找到 {len(docs)} 个条目")
    return docs


def get_item_files(identifier: str) -> List[Dict]:
    """
    获取某个 identifier 下的所有文件列表。

    Args:
        identifier: Archive 条目 ID

    Returns:
        文件列表，每个包含 name/format/size 等
    """
    url = f"https://archive.org/metadata/{identifier}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def download_file(
    identifier: str,
    filename: str,
    output_dir: Path,
    prefix: str = "",
) -> Optional[Path]:
    """
    下载单个文件。

    Args:
        identifier: Archive 条目 ID
        filename: 文件名
        output_dir: 输出目录
        prefix: 文件名前缀（避免重名）

    Returns:
        下载后的文件路径，跳过则返回 None
    """
    file_url = f"https://archive.org/download/{identifier}/{filename}"
    safe_prefix = prefix.replace("/", "_").replace(" ", "_")[:50] if prefix else identifier[:30]
    output_path = output_dir / f"{safe_prefix}_{filename}"

    if output_path.exists():
        logger.info(f"  已存在，跳过: {output_path.name}")
        return output_path

    logger.info(f"  下载: {filename}")
    try:
        resp = requests.get(file_url, stream=True, timeout=60)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

        if total_size > 0:
            logger.info(f"  ✅ 完成: {output_path.name} ({downloaded / 1024 / 1024:.1f} MB)")
        else:
            logger.info(f"  ✅ 完成: {output_path.name}")
        return output_path

    except Exception as e:
        logger.error(f"  ❌ 下载失败: {filename} - {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def download_item(
    item: Dict,
    output_dir: Path,
    formats: Optional[List[str]] = None,
    max_files_per_item: int = 1,
) -> List[Path]:
    """
    下载单个 Archive 条目的音频文件。

    Args:
        item: 搜索结果条目（含 identifier/title/creator）
        output_dir: 输出目录
        formats: 优先下载的格式列表（默认按 AUDIO_FORMATS 优先级）
        max_files_per_item: 每个条目最多下载几个文件

    Returns:
        下载成功的文件路径列表
    """
    identifier = item["identifier"]
    title = item.get("title", identifier)[:60]
    creator = item.get("creator", "Unknown")[:40]
    date = item.get("date", "Unknown")

    logger.info(f"\n📀 {title}")
    logger.info(f"   艺术家: {creator} | 年代: {date} | ID: {identifier}")

    try:
        files = get_item_files(identifier)
    except Exception as e:
        logger.error(f"  获取文件列表失败: {e}")
        return []

    time.sleep(REQUEST_DELAY)

    # 按格式优先级筛选音频文件
    if formats is None:
        formats = list(AUDIO_FORMATS.keys())

    audio_files = []
    for fmt in formats:
        for f in files:
            if f.get("format") == fmt and f.get("name"):
                # 排除衍生文件（如 _sample.mp3）
                name = f["name"].lower()
                if not any(skip in name for skip in ["_sample", "_thumb", "archive_banner"]):
                    audio_files.append(f)
        if len(audio_files) >= max_files_per_item:
            break

    if not audio_files:
        logger.warning(f"  未找到可下载的音频文件")
        return []

    # 限制每个条目下载数量
    audio_files = audio_files[:max_files_per_item]

    downloaded = []
    prefix = f"{creator}_{title}"
    for f in audio_files:
        path = download_file(identifier, f["name"], output_dir, prefix=prefix)
        if path:
            downloaded.append(path)
        time.sleep(REQUEST_DELAY)

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="从 Internet Archive 下载公共领域音频（免注册）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 下载5首老爵士
  python download_archive.py --query "jazz 78rpm" --limit 5

  # 只搜索不下载
  python download_archive.py --query "classical piano" --limit 10 --dry-run

  # 指定输出目录和格式
  python download_archive.py --query "jazz 78rpm" --limit 5 --output downloads/archive --formats "Flac,VBR MP3"
        """,
    )
    parser.add_argument("--query", required=True, help="搜索关键词，如 'jazz 78rpm'")
    parser.add_argument("--limit", type=int, default=5, help="搜索结果数量（默认5）")
    parser.add_argument("--output", default="data/00_raw_collect/raw_audio_archive",
                        help="输出目录（默认 data/00_raw_collect/raw_audio_archive）")
    parser.add_argument("--formats", default=None,
                        help="优先格式，逗号分隔，如 'Flac,VBR MP3'（默认按内置优先级）")
    parser.add_argument("--max-files", type=int, default=1,
                        help="每个条目最多下载几个文件（默认1）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只搜索不下载，打印结果")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 解析格式
    formats = [f.strip() for f in args.formats.split(",")] if args.formats else None

    # 搜索
    items = search_archive(args.query, args.limit)

    if not items:
        logger.warning("未找到结果，请尝试其他关键词")
        return

    # 打印搜索结果
    logger.info("\n" + "=" * 60)
    logger.info("搜索结果")
    logger.info("=" * 60)
    for i, item in enumerate(items, 1):
        title = item.get("title", "Unknown")[:50]
        creator = item.get("creator", "Unknown")[:30]
        date = item.get("date", "?")
        logger.info(f"  {i:2d}. {title} | {creator} | {date}")

    if args.dry_run:
        logger.info("\n--dry-run 模式，不下载")
        return

    # 下载
    logger.info("\n" + "=" * 60)
    logger.info("开始下载")
    logger.info("=" * 60)

    all_downloaded = []
    for item in items:
        downloaded = download_item(
            item,
            output_dir,
            formats=formats,
            max_files_per_item=args.max_files,
        )
        all_downloaded.extend(downloaded)

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("下载完成")
    logger.info("=" * 60)
    logger.info(f"  搜索条目: {len(items)}")
    logger.info(f"  下载文件: {len(all_downloaded)}")
    logger.info(f"  输出目录: {output_dir}")
    if all_downloaded:
        total_size = sum(p.stat().st_size for p in all_downloaded if p.exists())
        logger.info(f"  总大小: {total_size / 1024 / 1024:.1f} MB")

    logger.info("\n下一步：")
    logger.info("  1. 听 5 秒/首，填写 pilot_50_checklist.csv 的六个维度")
    logger.info("  2. 用 import_audio.py 入库到 audio_manifest.csv")
    logger.info("  3. 运行 check_pilot_gaps.py 统计分布缺口")


if __name__ == "__main__":
    main()
