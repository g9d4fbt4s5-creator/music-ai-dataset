#!/usr/bin/env python3
"""
batch_denoise.py
降噪候选批量处理脚本（人工审核后批量跑 noisereduce）

工作流程：
1. quality_check.py 标记降噪候选（SNR<15dB 或 频谱平坦度>0.5），生成候选列表 CSV
2. 人工审核候选列表，确认哪些需要降噪
3. 运行本脚本，对确认的音频批量跑 noisereduce
4. 输出降噪后的音频到指定目录，生成降噪报告

用法：
    # 批量降噪所有候选
    python3 scripts/01_preprocess/batch_denoise.py

    # 只降噪指定 audio_id
    python3 scripts/01_preprocess/batch_denoise.py --audio-id 01M0E9X162CTB4D15WZQ5D8FVX

    # 指定候选列表文件
    python3 scripts/01_preprocess/batch_denoise.py --candidates data/00.5_cleaned/reports/quality_check_report_noise_candidates.csv

    # 自定义降噪参数
    python3 scripts/01_preprocess/batch_denoise.py --strength 0.5 --stationary

    # 预览模式（不实际降噪）
    python3 scripts/01_preprocess/batch_denoise.py --dry-run
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

# noisereduce 降噪库
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 默认路径
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "00.5_cleaned" / "reports" / "quality_check_report_noise_candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "01_preprocess" / "denoised_audio"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "00.5_cleaned" / "reports" / "denoise_report.csv"

# 原始音频目录
RAW_AUDIO_DIR = PROJECT_ROOT / "data" / "00_raw_collect" / "raw_audio"

# 日志
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"batch_denoise_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_hash_dir(audio_id: str) -> tuple:
    """根据 audio_id 计算两层散列目录"""
    import hashlib
    hash_full = hashlib.md5(audio_id.encode()).hexdigest()
    return hash_full[0:2], hash_full[2:4]


def get_raw_path(audio_id: str, ext: str = "wav") -> Path:
    """获取原始音频路径"""
    dir1, dir2 = get_hash_dir(audio_id)
    import hashlib
    hash_full = hashlib.md5(audio_id.encode()).hexdigest()
    filename = f"{hash_full}_{audio_id}.{ext.lower()}"
    return RAW_AUDIO_DIR / dir1 / dir2 / filename


def get_output_path(audio_id: str, output_dir: Path, ext: str = "wav") -> Path:
    """获取降噪后音频输出路径（保持散列目录结构）"""
    dir1, dir2 = get_hash_dir(audio_id)
    import hashlib
    hash_full = hashlib.md5(audio_id.encode()).hexdigest()
    filename = f"{hash_full}_{audio_id}_denoised.{ext.lower()}"
    return output_dir / dir1 / dir2 / filename


def denoise_audio(
    input_path: Path,
    output_path: Path,
    strength: float = 0.5,
    stationary: bool = False,
    sample_rate: Optional[int] = None,
) -> Dict:
    """
    对单个音频进行降噪

    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        strength: 降噪强度 0.0-1.0
        stationary: 是否为稳态噪声（持续的背景噪声）
        sample_rate: 目标采样率（None 保持原采样率）

    Returns:
        Dict: 降噪结果
            - audio_id: 音频ID
            - input_path: 输入路径
            - output_path: 输出路径
            - status: success / failed
            - original_sr: 原始采样率
            - original_duration: 原始时长
            - output_sr: 输出采样率
            - output_duration: 输出时长
            - strength: 降噪强度
            - stationary: 是否稳态噪声
            - error: 错误信息
    """
    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "pending",
        "original_sr": 0,
        "original_duration": 0.0,
        "output_sr": 0,
        "output_duration": 0.0,
        "strength": strength,
        "stationary": stationary,
        "error": "",
    }

    if not HAS_NOISEREDUCE:
        result["status"] = "failed"
        result["error"] = "noisereduce 未安装，请运行 pip install noisereduce"
        return result

    try:
        # 加载音频
        y, sr = librosa.load(str(input_path), sr=sample_rate, mono=False)
        result["original_sr"] = sr
        result["original_duration"] = len(y[0]) / sr if y.ndim > 1 else len(y) / sr

        # 降噪
        if y.ndim > 1:
            # 立体声：逐声道降噪
            denoised_channels = []
            for channel in y:
                denoised = nr.reduce_noise(
                    y=channel,
                    sr=sr,
                    prop_decrease=strength,
                    stationary=stationary,
                )
                denoised_channels.append(denoised)
            denoised_y = np.array(denoised_channels)
        else:
            # 单声道
            denoised_y = nr.reduce_noise(
                y=y,
                sr=sr,
                prop_decrease=strength,
                stationary=stationary,
            )

        # 保存降噪后音频
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), denoised_y.T if denoised_y.ndim > 1 else denoised_y, sr)

        result["status"] = "success"
        result["output_sr"] = sr
        result["output_duration"] = len(denoised_y[0]) / sr if denoised_y.ndim > 1 else len(denoised_y) / sr

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result


def load_candidates(candidates_path: Path) -> pd.DataFrame:
    """加载降噪候选列表"""
    if not candidates_path.exists():
        logger.error(f"降噪候选列表不存在: {candidates_path}")
        raise FileNotFoundError(f"Candidates not found: {candidates_path}")

    df = pd.read_csv(candidates_path)
    logger.info(f"加载降噪候选列表: {len(df)} 条")
    return df


def main():
    parser = argparse.ArgumentParser(description="降噪候选批量处理脚本（人工审核后批量跑 noisereduce）")
    parser.add_argument("--candidates", type=str, default=str(DEFAULT_CANDIDATES), help="降噪候选列表 CSV 路径")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="降噪后音频输出目录")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT), help="降噪报告输出路径")
    parser.add_argument("--audio-id", type=str, help="只降噪指定 audio_id")
    parser.add_argument("--strength", type=float, default=0.5, help="降噪强度 0.0-1.0（默认0.5）")
    parser.add_argument("--stationary", action="store_true", help="稳态噪声模式（持续的背景噪声）")
    parser.add_argument("--sample-rate", type=int, default=None, help="目标采样率（默认保持原采样率）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际降噪")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("降噪候选批量处理启动")
    logger.info(f"  候选列表: {args.candidates}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  降噪强度: {args.strength}")
    logger.info(f"  稳态噪声: {args.stationary}")
    logger.info(f"  目标采样率: {args.sample_rate or '保持原样'}")
    logger.info(f"  预览模式: {args.dry_run}")
    logger.info("=" * 60)

    # 检查 noisereduce
    if not HAS_NOISEREDUCE:
        logger.error("noisereduce 未安装，请运行: pip install noisereduce")
        return

    # 加载候选列表
    try:
        candidates_df = load_candidates(Path(args.candidates))
    except FileNotFoundError:
        return

    # 过滤指定 audio_id
    if args.audio_id:
        candidates_df = candidates_df[candidates_df["audio_id"] == args.audio_id]
        if len(candidates_df) == 0:
            logger.error(f"未找到 audio_id: {args.audio_id}")
            return

    logger.info(f"待降噪音频: {len(candidates_df)} 个")
    logger.info("")

    # 逐个降噪
    results = []
    for idx, row in candidates_df.iterrows():
        # 优先从 audio_path 列读取路径，其次从 audio_id 构造
        if "audio_path" in row and pd.notna(row["audio_path"]):
            raw_path = Path(row["audio_path"])
            # 从路径中提取 audio_id（文件名格式：hash_audioid.ext）
            filename = raw_path.stem
            parts = filename.split("_", 1)
            audio_id = parts[1] if len(parts) > 1 else filename
        elif "audio_id" in row and pd.notna(row["audio_id"]):
            audio_id = row["audio_id"]
            ext = row.get("format", "wav")
            if pd.isna(ext) or ext == "":
                ext = "wav"
            raw_path = get_raw_path(audio_id, str(ext))
        else:
            logger.warning(f"  ⚠️ 第 {idx+1} 行没有 audio_path 或 audio_id，跳过")
            continue

        output_path = get_output_path(audio_id, Path(args.output_dir), "wav")

        if not raw_path.exists():
            logger.warning(f"  ❌ {audio_id[:25]}... 原始文件不存在: {raw_path}")
            results.append({
                "audio_id": audio_id,
                "status": "failed",
                "error": f"原始文件不存在: {raw_path}",
            })
            continue

        if args.dry_run:
            logger.info(f"  📋 [预览] {audio_id[:25]}... 将降噪 (strength={args.strength}, stationary={args.stationary})")
            results.append({
                "audio_id": audio_id,
                "status": "dry_run",
                "input_path": str(raw_path),
                "output_path": str(output_path),
            })
            continue

        # 执行降噪
        logger.info(f"  🔄 {audio_id[:25]}... 降噪中")
        result = denoise_audio(
            input_path=raw_path,
            output_path=output_path,
            strength=args.strength,
            stationary=args.stationary,
            sample_rate=args.sample_rate,
        )
        result["audio_id"] = audio_id

        if result["status"] == "success":
            file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
            logger.info(f"  ✅ {audio_id[:25]}... 降噪成功 ({file_size_mb:.2f}MB)")
        else:
            logger.error(f"  ❌ {audio_id[:25]}... 降噪失败: {result['error']}")

        results.append(result)

    # 统计
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    dry_run_count = sum(1 for r in results if r["status"] == "dry_run")

    logger.info("")
    logger.info("=" * 60)
    logger.info("降噪批量处理完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    if args.dry_run:
        logger.info(f"  预览: {dry_run_count}")
    logger.info(f"  总计: {len(results)}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 保存报告
    if results and not args.dry_run:
        report_df = pd.DataFrame(results)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False, encoding="utf-8")
        logger.info(f"降噪报告已保存: {report_path}")

    # 输出失败列表
    if failed_count > 0:
        logger.info("")
        logger.info("失败列表:")
        for r in results:
            if r["status"] == "failed":
                logger.info(f"  - {r.get('audio_id', 'unknown')[:30]}...: {r.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
