#!/usr/bin/env python3
"""
generate_master.py
母版生成脚本：从原始采集文件转码为统一 FLAC 母版（48kHz/24bit/立体声）

架构原则：
- 00_raw_collect/raw_audio/：原始采集物（mp3/flac/m4a混存），只读永不修改
- 01_preprocess/processed_master/：统一母版 FLAC 48k/24bit/stereo，所有派生从这里出
- 重采样实时做，不永久存（ffmpeg 开销极小）
- Demucs stems / segments 必须缓存（分离/切分成本高）

用法：
    # 生成所有音频的母版
    python3 scripts/01_preprocess/generate_master.py

    # 只处理指定 audio_id
    python3 scripts/01_preprocess/generate_master.py --audio-id 01M0E9X162CTB4D15WZQ5D8FVX

    # 预览模式（不实际转码）
    python3 scripts/01_preprocess/generate_master.py --dry-run

    # 强制重新生成（覆盖已有母版）
    python3 scripts/01_preprocess/generate_master.py --force
"""
import os
import sys
import hashlib
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import pandas as pd

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 母版规格
MASTER_SAMPLE_RATE = 48000
MASTER_BIT_DEPTH = 24
MASTER_CHANNELS = 2
MASTER_FORMAT = "flac"

# 目录
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio"
MASTER_DIR = PROJECT_ROOT / "data" / "01_preprocess" / "processed_master"
MANIFEST_CSV = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"

# 日志
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"generate_master_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_hash_dir(audio_id: str) -> Tuple[str, str]:
    """
    根据 audio_id 计算两层散列目录

    Args:
        audio_id: 音频ID

    Returns:
        Tuple[str, str]: (第一层目录, 第二层目录)
            例如 md5(audio_id) = "a1b2c3d4..." → ("a1", "b2")
    """
    hash_full = hashlib.md5(audio_id.encode()).hexdigest()
    return hash_full[0:2], hash_full[2:4]


def get_master_path(audio_id: str) -> Path:
    """
    获取母版文件路径

    Args:
        audio_id: 音频ID

    Returns:
        Path: 母版文件路径
            例如 processed_master/a1/b2/a1b2c3d4_audioid.flac
    """
    dir1, dir2 = get_hash_dir(audio_id)
    hash_full = hashlib.md5(audio_id.encode()).hexdigest()
    filename = f"{hash_full}_{audio_id}.{MASTER_FORMAT}"
    return MASTER_DIR / dir1 / dir2 / filename


def get_raw_path(audio_id: str, ext: str) -> Path:
    """
    获取原始音频文件路径

    Args:
        audio_id: 音频ID
        ext: 文件扩展名

    Returns:
        Path: 原始音频文件路径
    """
    dir1, dir2 = get_hash_dir(audio_id)
    hash_full = hashlib.md5(audio_id.encode()).hexdigest()
    filename = f"{hash_full}_{audio_id}.{ext.lower()}"
    return RAW_AUDIO_DIR / dir1 / dir2 / filename


def compute_md5(file_path: Path) -> str:
    """计算文件 MD5"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def transcode_to_master(
    raw_path: Path,
    master_path: Path,
    sample_rate: int = MASTER_SAMPLE_RATE,
    bit_depth: int = MASTER_BIT_DEPTH,
    channels: int = MASTER_CHANNELS,
) -> Tuple[bool, str]:
    """
    使用 ffmpeg 将原始音频转码为 FLAC 母版

    Args:
        raw_path: 原始音频路径
        master_path: 母版输出路径
        sample_rate: 目标采样率
        bit_depth: 目标位深
        channels: 目标声道数

    Returns:
        Tuple[bool, str]: (是否成功, 错误信息)
    """
    master_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg 命令：转 FLAC，指定采样率/位深/声道
    # -acodec flac: FLAC 编码
    # -ar: 采样率
    # -ac: 声道数
    # -sample_fmt s32: 32位整数采样格式（FLAC 支持 24-bit，用 s32 容器）
    # -compression_level 8: 最高压缩
    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出
        "-i", str(raw_path),
        "-acodec", "flac",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s32",
        "-compression_level", "8",
        str(master_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2分钟超时
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "未知错误"
            return False, f"ffmpeg 转码失败: {error_msg}"

        # 验证输出文件存在且非空
        if not master_path.exists() or master_path.stat().st_size == 0:
            return False, "输出文件不存在或为空"

        return True, ""

    except subprocess.TimeoutExpired:
        return False, "ffmpeg 转码超时（>120秒）"
    except Exception as e:
        return False, f"转码异常: {str(e)}"


def generate_master_for_audio(
    audio_id: str,
    raw_ext: str,
    force: bool = False,
    dry_run: bool = False,
    raw_path_override: Path = None,
) -> Dict:
    """
    为单个音频生成母版

    Args:
        audio_id: 音频ID
        raw_ext: 原始文件扩展名
        force: 是否强制重新生成
        dry_run: 预览模式
        raw_path_override: 直接指定原始文件路径（优先于自动计算）

    Returns:
        Dict: 处理结果
            - audio_id: 音频ID
            - raw_path: 原始路径
            - master_path: 母版路径
            - status: success / skipped / failed
            - master_md5: 母版MD5
            - error: 错误信息
    """
    if raw_path_override is not None:
        raw_path = raw_path_override
    else:
        raw_path = get_raw_path(audio_id, raw_ext)
    master_path = get_master_path(audio_id)

    result = {
        "audio_id": audio_id,
        "raw_path": str(raw_path),
        "master_path": str(master_path),
        "status": "pending",
        "master_md5": "",
        "error": "",
    }

    # 检查原始文件是否存在
    if not raw_path.exists():
        result["status"] = "failed"
        result["error"] = f"原始文件不存在: {raw_path}"
        logger.warning(f"  ❌ {audio_id[:25]}... 原始文件不存在")
        return result

    # 检查母版是否已存在
    if master_path.exists() and not force:
        result["status"] = "skipped"
        result["master_md5"] = compute_md5(master_path)
        logger.info(f"  ⏭️  {audio_id[:25]}... 母版已存在，跳过")
        return result

    if dry_run:
        result["status"] = "dry_run"
        logger.info(f"  📋 [预览] {audio_id[:25]}... 将转码 {raw_ext} → FLAC {MASTER_SAMPLE_RATE}Hz/{MASTER_BIT_DEPTH}bit/{MASTER_CHANNELS}ch")
        return result

    # 执行转码
    logger.info(f"  🔄 {audio_id[:25]}... 转码中 ({raw_ext} → FLAC)")
    success, error = transcode_to_master(raw_path, master_path)

    if success:
        result["status"] = "success"
        result["master_md5"] = compute_md5(master_path)
        file_size_mb = master_path.stat().st_size / (1024 * 1024)
        logger.info(f"  ✅ {audio_id[:25]}... 母版生成成功 ({file_size_mb:.2f}MB)")
    else:
        result["status"] = "failed"
        result["error"] = error
        logger.error(f"  ❌ {audio_id[:25]}... 母版生成失败: {error}")

    return result


def load_manifest() -> pd.DataFrame:
    """加载 audio_manifest.csv"""
    if not MANIFEST_CSV.exists():
        logger.error(f"audio_manifest.csv 不存在: {MANIFEST_CSV}")
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_CSV}")

    df = pd.read_csv(MANIFEST_CSV)
    logger.info(f"加载 audio_manifest.csv: {len(df)} 条记录")
    return df


def update_manifest(results: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
    """
    更新 audio_manifest.csv，添加 master_path 和 master_md5 字段

    Args:
        results: 母版生成结果列表
        df: 原始 manifest DataFrame

    Returns:
        pd.DataFrame: 更新后的 DataFrame
    """
    # 创建结果映射
    result_map = {r["audio_id"]: r for r in results}

    # 添加 master_path 字段
    if "master_path" not in df.columns:
        df["master_path"] = ""
    if "master_md5" not in df.columns:
        df["master_md5"] = ""
    if "master_status" not in df.columns:
        df["master_status"] = ""

    for idx, row in df.iterrows():
        audio_id = row["audio_id"]
        if audio_id in result_map:
            r = result_map[audio_id]
            df.at[idx, "master_path"] = r["master_path"]
            df.at[idx, "master_md5"] = r["master_md5"]
            df.at[idx, "master_status"] = r["status"]

    return df


def main():
    parser = argparse.ArgumentParser(description="母版生成脚本：从原始采集文件转码为统一 FLAC 母版")
    parser.add_argument("--audio-id", type=str, help="只处理指定 audio_id")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际转码")
    parser.add_argument("--force", action="store_true", help="强制重新生成，覆盖已有母版")
    parser.add_argument("--manifest", type=str, default=str(MANIFEST_CSV), help="audio_manifest.csv 路径")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("母版生成脚本启动")
    logger.info(f"  母版规格: FLAC {MASTER_SAMPLE_RATE}Hz/{MASTER_BIT_DEPTH}bit/{MASTER_CHANNELS}ch")
    logger.info(f"  原始目录: {RAW_AUDIO_DIR}")
    logger.info(f"  母版目录: {MASTER_DIR}")
    logger.info(f"  预览模式: {args.dry_run}")
    logger.info(f"  强制重生成: {args.force}")
    logger.info("=" * 60)

    # 加载 manifest
    df = load_manifest()

    # P0: source_type 过滤 — 排除域外样本（AI生成、分轨人声等，ADR-003 第7节）
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from utils.source_type_filter import filter_by_source_type
        df, _ = filter_by_source_type(df, report_path=None)
    except ImportError as e:
        logger.warning(f"source_type_filter 导入失败: {e}，跳过 source_type 过滤")

    # 过滤要处理的音频
    if args.audio_id:
        df = df[df["audio_id"] == args.audio_id]
        if len(df) == 0:
            logger.error(f"未找到 audio_id: {args.audio_id}")
            return

    logger.info(f"待处理音频: {len(df)} 个")
    logger.info("")

    # 逐个生成母版
    results = []
    for idx, row in df.iterrows():
        audio_id = row["audio_id"]
        raw_ext = row.get("format", "wav")
        if pd.isna(raw_ext) or raw_ext == "":
            raw_ext = "wav"

        # 优先使用 manifest 中的 file_relative_path（支持 sha256 命名规则）
        raw_path_override = None
        file_rel = row.get("file_relative_path", "")
        if pd.notna(file_rel) and file_rel != "":
            raw_path_override = PROJECT_ROOT / "data" / "00_raw_collect" / file_rel

        result = generate_master_for_audio(
            audio_id=audio_id,
            raw_ext=str(raw_ext),
            force=args.force,
            dry_run=args.dry_run,
            raw_path_override=raw_path_override,
        )
        results.append(result)

    # 统计
    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    dry_run_count = sum(1 for r in results if r["status"] == "dry_run")

    logger.info("")
    logger.info("=" * 60)
    logger.info("母版生成完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  跳过(已存在): {skipped_count}")
    logger.info(f"  失败: {failed_count}")
    if args.dry_run:
        logger.info(f"  预览: {dry_run_count}")
    logger.info(f"  总计: {len(results)}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 更新 manifest（非预览模式）
    if not args.dry_run and results:
        df_updated = update_manifest(results, df)
        df_updated.to_csv(args.manifest, index=False, encoding="utf-8")
        logger.info(f"audio_manifest.csv 已更新: {args.manifest}")

    # 输出失败列表
    if failed_count > 0:
        logger.info("")
        logger.info("失败列表:")
        for r in results:
            if r["status"] == "failed":
                logger.info(f"  - {r['audio_id'][:30]}...: {r['error']}")


if __name__ == "__main__":
    main()
