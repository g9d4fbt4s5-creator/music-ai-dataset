#!/usr/bin/env python3
"""
从文件名解析 artist 和 title，填充 artist_id 和 song_group_id

解析规则：
1. 切片文件: {prefix}_{序号}_{artist}-{title}.flac
2. Apple Music录制: {artist} - {title}.flac
3. Bilibili合集: ...{artist} - {title} [BV...]_pXX.flac

artist_id 格式: artist:{name}
song_group_id 格式: song:{artist}:{title}
"""
import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"


def normalize_artist(name: str) -> str:
    """标准化艺术家名：去首尾空格、统一分隔符"""
    name = name.strip()
    # 统一 " _ " 为 " & "
    name = re.sub(r'\s+_\s+', ' & ', name)
    # 合并多个空格
    name = re.sub(r'\s+', ' ', name)
    return name


def parse_sliced_filename(filename: str) -> tuple[str, str] | None:
    """
    解析切片文件名: {prefix}_{序号}_{artist}-{title}.ext
    例如: rb_playlist_01_Whitney Houston-Call You Tonight.flac
    """
    # 去掉扩展名
    name = Path(filename).stem

    # 匹配前缀_序号_ 后面的部分
    # 前缀可能是: rb_playlist, cantonese_playlist, jobim_wave, 1st_ep, kira_linn_illusion, hiromi_alive
    match = re.match(
        r'^(?:rb_playlist|cantonese_playlist|jobim_wave|1st_ep|kira_linn_illusion|hiromi_alive)_\d+_(.+)$',
        name,
        re.IGNORECASE
    )
    if not match:
        return None

    rest = match.group(1)
    # 按第一个 "-" 分割 artist 和 title
    if '-' in rest:
        artist, title = rest.split('-', 1)
        return normalize_artist(artist), title.strip()
    return None


def parse_apple_music_filename(filename: str) -> tuple[str, str] | None:
    """
    解析 Apple Music 录制文件名: {artist} - {title}.ext
    例如: 椎名林檎 - The Creamy Season.flac
    """
    name = Path(filename).stem
    if ' - ' in name:
        artist, title = name.split(' - ', 1)
        return normalize_artist(artist), title.strip()
    return None


def parse_bilibili_compilation(filename: str) -> tuple[str, str] | None:
    """
    解析 Bilibili 合集文件名: ...{artist} - {title} [BV...]_pXX.ext
    例如: ...p05 B2 Deep Choice - Children Trip [BV1Sh4y1z7Yv_p5]_p05.flac
    """
    name = Path(filename).stem

    # 匹配 " - " 前面的 artist（从 pXX 后面开始）
    # 格式: ...pXX {序号} {artist} - {title} [BV...]_pXX
    match = re.search(r'p\d+\s+[A-Z]\d+\s+(.+?)\s+-\s+(.+?)\s+\[BV', name)
    if match:
        artist = match.group(1).strip()
        title = match.group(2).strip()
        return normalize_artist(artist), title

    # 备用：匹配 " - " 前面的部分（去掉 [BV 后面的）
    match = re.search(r'(.+?)\s+-\s+(.+?)\s+\[BV', name)
    if match:
        # artist 可能包含前缀，取最后几个词
        artist_part = match.group(1).strip()
        # 去掉前缀（pXX A1 等）
        artist_part = re.sub(r'^.*?p\d+\s+[A-Z]\d+\s+', '', artist_part)
        title = match.group(2).strip()
        if artist_part:
            return normalize_artist(artist_part), title

    return None


def parse_filename(filename: str) -> tuple[str, str] | None:
    """综合解析文件名，返回 (artist, title)"""
    # 1. 切片文件
    result = parse_sliced_filename(filename)
    if result:
        return result

    # 2. Bilibili 合集
    result = parse_bilibili_compilation(filename)
    if result:
        return result

    # 3. Apple Music（最后尝试，因为格式最通用）
    result = parse_apple_music_filename(filename)
    if result:
        return result

    return None


def main():
    parser = argparse.ArgumentParser(description="从文件名解析 artist 和 title，填充 manifest")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_PATH))
    parser.add_argument("--dry-run", action="store_true", help="只预览不写入")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    logger.info(f"加载 manifest: {len(df)} 首")

    parsed_count = 0
    unknown_before = df['artist_id'].str.startswith('unknown_').sum()

    for idx, row in df.iterrows():
        filename = row['original_filename']
        result = parse_filename(filename)

        if result:
            artist, title = result
            artist_id = f"artist:{artist}"
            song_group_id = f"song:{artist}:{title}"

            df.at[idx, 'artist_id'] = artist_id
            df.at[idx, 'song_group_id'] = song_group_id
            parsed_count += 1
            logger.info(f"  ✅ {artist} - {title}")
        else:
            logger.debug(f"  ⏭️  无法解析: {filename[:60]}")

    unknown_after = df['artist_id'].str.startswith('unknown_').sum()

    logger.info(f"\n解析完成:")
    logger.info(f"  成功解析: {parsed_count} 首")
    logger.info(f"  仍为 unknown: {unknown_after} 首 (之前 {unknown_before} 首)")
    logger.info(f"  唯一 artist_id: {df['artist_id'].nunique()} 个")

    # 显示 artist 分布
    logger.info(f"\nArtist 分布 (Top 20):")
    artist_counts = df[~df['artist_id'].str.startswith('unknown_')]['artist_id'].value_counts()
    for artist, count in artist_counts.head(20).items():
        logger.info(f"  {artist}: {count} 首")

    if not args.dry_run:
        df.to_csv(args.manifest, index=False)
        logger.info(f"\nManifest 已更新: {args.manifest}")
    else:
        logger.info(f"\n[dry-run] 未写入文件")


if __name__ == "__main__":
    main()
