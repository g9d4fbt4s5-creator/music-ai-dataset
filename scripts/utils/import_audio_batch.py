#!/usr/bin/env python3
"""
批量导入音频文件到项目

从指定目录扫描音频文件，复制到 raw_audio/ 目录，
计算元数据（sha256、时长、格式等），添加到 audio_manifest.csv。

用法：
  python scripts/utils/import_audio_batch.py --dir "/path/to/audio"
  python scripts/utils/import_audio_batch.py --dirs "/dir1" "/dir2" "/dir3"
  python scripts/utils/import_audio_batch.py --dirs "/dir1" "/dir2" --source-type normal
"""
import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 支持的音频格式
SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".flac", ".wav", ".ogg", ".opus", ".aac"}

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 原始音频目录
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio"

# Manifest 路径
MANIFEST_PATH = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"


def generate_audio_id() -> str:
    """生成类似 ULID 的 audio_id（26位大写字母数字）"""
    # 使用 uuid4 的前 26 位，转大写
    return uuid.uuid4().hex[:26].upper()


def compute_sha256(file_path: Path) -> str:
    """计算文件的 SHA256 哈希"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_audio_metadata(file_path: Path) -> Dict:
    """
    使用 ffprobe 获取音频元数据。
    返回 format, sample_rate, bit_depth, channels, duration_sec
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration:stream=codec_name,sample_rate,bits_per_sample,channels",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        import json
        data = json.loads(result.stdout)

        # 从第一个音频流获取信息
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        codec = stream.get("codec_name", "unknown")
        sample_rate = int(stream.get("sample_rate", 0))
        bits_per_sample = stream.get("bits_per_sample", 0)
        channels = int(stream.get("channels", 0))
        duration = float(fmt.get("duration", 0))

        # 格式映射
        ext = file_path.suffix.lower().lstrip(".")
        if ext == "flac":
            format_name = "flac"
            bit_depth_str = f"PCM_{bits_per_sample}" if bits_per_sample else "PCM_16"
        elif ext == "wav":
            format_name = "wav"
            bit_depth_str = f"PCM_{bits_per_sample}" if bits_per_sample else "PCM_16"
        elif ext == "mp3":
            format_name = "mp3"
            bit_depth_str = "lossy"
        elif ext == "m4a":
            format_name = "m4a"
            bit_depth_str = "lossy"
        else:
            format_name = ext
            bit_depth_str = str(bits_per_sample) if bits_per_sample else "unknown"

        return {
            "format": format_name,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth_str,
            "channels": channels,
            "duration_sec": round(duration, 2),
        }
    except Exception as e:
        logger.warning(f"无法获取元数据 {file_path.name}: {e}")
        ext = file_path.suffix.lower().lstrip(".")
        return {
            "format": ext,
            "sample_rate": 0,
            "bit_depth": "unknown",
            "channels": 0,
            "duration_sec": 0,
        }


def copy_to_raw_audio(file_path: Path, audio_id: str, sha256: str) -> Path:
    """
    复制音频文件到 raw_audio/ 目录，按 hash 前两位分子目录。
    文件名格式: {sha256}_{audio_id}.{ext}
    """
    ext = file_path.suffix.lower()
    sub_dir = RAW_AUDIO_DIR / sha256[:2] / sha256[2:4]
    sub_dir.mkdir(parents=True, exist_ok=True)

    new_filename = f"{sha256}_{audio_id}{ext}"
    dest_path = sub_dir / new_filename

    if not dest_path.exists():
        shutil.copy2(file_path, dest_path)

    # 返回相对路径
    return Path("raw_audio") / sha256[:2] / sha256[2:4] / new_filename


def scan_audio_files(dirs: List[Path]) -> List[Path]:
    """扫描目录中的音频文件"""
    audio_files = []
    for d in dirs:
        if not d.exists():
            logger.warning(f"目录不存在: {d}")
            continue
        for root, _, files in os.walk(d):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    audio_files.append(Path(root) / f)
    logger.info(f"扫描到 {len(audio_files)} 个音频文件")
    return audio_files


def load_existing_manifest() -> pd.DataFrame:
    """加载现有 manifest"""
    if MANIFEST_PATH.exists():
        return pd.read_csv(MANIFEST_PATH)
    # 创建空 manifest
    columns = [
        "audio_id", "file_relative_path", "original_filename",
        "format", "sample_rate", "bit_depth", "channels",
        "duration_sec", "file_bytes", "sha256",
        "import_timestamp", "status", "source_type",
        "artist_id", "song_group_id", "is_golden",
    ]
    return pd.DataFrame(columns=columns)


def import_audio_files(
    dirs: List[Path],
    source_type: str = "normal",
    dry_run: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    批量导入音频文件。

    Returns:
        (imported_list, skipped_list)
    """
    # 加载现有 manifest
    manifest_df = load_existing_manifest()
    existing_sha256 = set(manifest_df["sha256"].tolist()) if "sha256" in manifest_df.columns else set()
    logger.info(f"现有 manifest: {len(manifest_df)} 首")

    # 扫描音频文件
    audio_files = scan_audio_files(dirs)

    imported = []
    skipped = []

    for file_path in audio_files:
        logger.info(f"\n处理: {file_path.name}")

        # 计算 sha256
        sha256 = compute_sha256(file_path)

        # 检查是否已存在
        if sha256 in existing_sha256:
            logger.info(f"  跳过（已存在）: {file_path.name}")
            skipped.append({"file": str(file_path), "reason": "duplicate"})
            continue

        # 获取元数据
        metadata = get_audio_metadata(file_path)
        file_bytes = file_path.stat().st_size

        # 生成 audio_id
        audio_id = generate_audio_id()

        # 复制文件
        if dry_run:
            relative_path = f"raw_audio/{sha256[:2]}/{sha256[2:4]}/{sha256}_{audio_id}{file_path.suffix.lower()}"
        else:
            relative_path = copy_to_raw_audio(file_path, audio_id, sha256)

        # 构建记录
        record = {
            "audio_id": audio_id,
            "file_relative_path": str(relative_path),
            "original_filename": file_path.name,
            "format": metadata["format"],
            "sample_rate": metadata["sample_rate"],
            "bit_depth": metadata["bit_depth"],
            "channels": metadata["channels"],
            "duration_sec": metadata["duration_sec"],
            "file_bytes": file_bytes,
            "sha256": sha256,
            "import_timestamp": datetime.now().isoformat(),
            "status": "active",
            "source_type": source_type,
            "artist_id": f"unknown_{audio_id[:8]}",
            "song_group_id": f"unknown_song_{audio_id[:8]}",
            "is_golden": False,
        }

        imported.append(record)
        existing_sha256.add(sha256)

        logger.info(f"  ✅ 导入: {audio_id} | {metadata['format']} | {metadata['duration_sec']}s | {file_bytes/1024/1024:.1f}MB")

    # 保存 manifest
    if imported and not dry_run:
        new_df = pd.DataFrame(imported)
        manifest_df = pd.concat([manifest_df, new_df], ignore_index=True)
        manifest_df.to_csv(MANIFEST_PATH, index=False)
        logger.info(f"\nManifest 已更新: {len(manifest_df)} 首（新增 {len(imported)} 首）")

    return imported, skipped


def main():
    parser = argparse.ArgumentParser(description="批量导入音频文件到项目")
    parser.add_argument("--dir", type=str, help="单个音频目录")
    parser.add_argument("--dirs", nargs="+", help="多个音频目录")
    parser.add_argument(
        "--source-type",
        type=str,
        default="normal",
        help="source_type 标签（默认 normal）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描不实际导入",
    )
    args = parser.parse_args()

    # 收集目录
    dirs = []
    if args.dir:
        dirs.append(Path(args.dir))
    if args.dirs:
        dirs.extend([Path(d) for d in args.dirs])

    if not dirs:
        parser.error("请提供 --dir 或 --dirs 参数")

    # 导入
    imported, skipped = import_audio_files(dirs, args.source_type, args.dry_run)

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("导入完成")
    logger.info(f"  成功导入: {len(imported)} 首")
    logger.info(f"  跳过（重复）: {len(skipped)} 首")
    if imported:
        logger.info(f"  总时长: {sum(r['duration_sec'] for r in imported):.0f} 秒")
        logger.info(f"  总大小: {sum(r['file_bytes'] for r in imported)/1024/1024:.1f} MB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
