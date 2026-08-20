#!/usr/bin/env python3
"""
batch_denoise_gpu.py
GPU 服务器版降噪候选批量处理脚本

与本地版 batch_denoise.py 的区别：
- 输入是平铺目录（如 /root/autodl-tmp/jazz_500_audio-low/*.mp3），不是散列目录
- 输出也是平铺目录（如 /root/autodl-tmp/denoised_audio/）
- 支持多进程并行（GPU 实例 CPU 核数多）
- 支持从 quality_check_report.csv 读取降噪候选列表
- 降噪后通过 rsync 回传本地或上传 OSS 备份（可选）

用法：
    # 方式1：从质量检查报告读取降噪候选
    python3 batch_denoise_gpu.py \
        --candidates /root/autodl-tmp/reports/quality_check_report_noise_candidates.csv \
        --output-dir /root/autodl-tmp/denoised_audio

    # 方式2：直接指定输入目录，降噪所有音频
    python3 batch_denoise_gpu.py \
        --input-dir /root/autodl-tmp/jazz_500_audio-low \
        --output-dir /root/autodl-tmp/denoised_audio

    # 方式3：只降噪指定文件
    python3 batch_denoise_gpu.py \
        --input-dir /root/autodl-tmp/jazz_500_audio-low \
        --file-list /root/autodl-tmp/denoise_list.txt \
        --output-dir /root/autodl-tmp/denoised_audio

    # 并行处理（GPU实例CPU核数多）
    python3 batch_denoise_gpu.py --input-dir ... --output-dir ... --workers 8

    # 自定义降噪参数
    python3 batch_denoise_gpu.py --input-dir ... --strength 0.5 --stationary

    # 预览模式（不实际降噪）
    python3 batch_denoise_gpu.py --input-dir ... --dry-run
"""
import os
import sys
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

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
# GPU 服务器默认路径
DEFAULT_INPUT_DIR = "/root/autodl-tmp/jazz_500_audio-low"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/denoised_audio"
DEFAULT_REPORT = "/root/autodl-tmp/reports/denoise_report.csv"

# 支持的音频格式
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

# 日志
LOG_DIR = "/root/autodl-tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"batch_denoise_gpu_{time_str}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def denoise_single_audio(args_tuple: Tuple) -> Dict:
    """
    降噪单个音频（多进程 worker 函数）

    Args:
        args_tuple: (input_path, output_path, strength, stationary, sample_rate)

    Returns:
        Dict: 降噪结果
    """
    input_path, output_path, strength, stationary, sample_rate = args_tuple

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "filename": os.path.basename(input_path),
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
        result["error"] = "noisereduce 未安装"
        return result

    try:
        # 加载音频
        y, sr = librosa.load(str(input_path), sr=sample_rate, mono=False)
        result["original_sr"] = sr
        if y.ndim > 1:
            result["original_duration"] = len(y[0]) / sr
        else:
            result["original_duration"] = len(y) / sr

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

        # 保存降噪后音频（统一输出为 wav，避免 mp3 编码损失）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(str(output_path), denoised_y.T if denoised_y.ndim > 1 else denoised_y, sr)

        result["status"] = "success"
        result["output_sr"] = sr
        if denoised_y.ndim > 1:
            result["output_duration"] = len(denoised_y[0]) / sr
        else:
            result["output_duration"] = len(denoised_y) / sr

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        return result


def load_candidates(candidates_path: str, input_dir: str) -> List[str]:
    """
    从质量检查报告读取降噪候选列表

    Args:
        candidates_path: 候选列表 CSV 路径
        input_dir: 音频输入目录（用于补全路径）

    Returns:
        List[str]: 候选音频文件路径列表
    """
    if not os.path.exists(candidates_path):
        logger.error(f"候选列表不存在: {candidates_path}")
        return []

    df = pd.read_csv(candidates_path)
    logger.info(f"加载候选列表: {len(df)} 条")

    audio_paths = []
    for _, row in df.iterrows():
        # 优先从 audio_path 列读取
        if "audio_path" in row and pd.notna(row["audio_path"]):
            path = str(row["audio_path"])
            # 如果路径不存在，尝试在 input_dir 中找
            if not os.path.exists(path):
                filename = os.path.basename(path)
                alt_path = os.path.join(input_dir, filename)
                if os.path.exists(alt_path):
                    path = alt_path
            audio_paths.append(path)
        # 其次从 filename 列读取
        elif "filename" in row and pd.notna(row["filename"]):
            path = os.path.join(input_dir, str(row["filename"]))
            if os.path.exists(path):
                audio_paths.append(path)
        # 最后从 track_id 列读取
        elif "track_id" in row and pd.notna(row["track_id"]):
            # 尝试常见命名格式
            track_id = str(row["track_id"])
            for ext in [".mp3", ".wav", ".flac"]:
                path = os.path.join(input_dir, f"{track_id}{ext}")
                if os.path.exists(path):
                    audio_paths.append(path)
                    break

    # 去重并过滤不存在的文件
    audio_paths = list(set(audio_paths))
    audio_paths = [p for p in audio_paths if os.path.exists(p)]
    logger.info(f"有效候选音频: {len(audio_paths)} 个")

    return audio_paths


def scan_input_dir(input_dir: str) -> List[str]:
    """
    扫描输入目录，获取所有音频文件

    Args:
        input_dir: 输入目录路径

    Returns:
        List[str]: 音频文件路径列表
    """
    audio_paths = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_FORMATS:
                audio_paths.append(os.path.join(root, f))

    logger.info(f"扫描输入目录: {input_dir}，找到 {len(audio_paths)} 个音频文件")
    return sorted(audio_paths)


def load_file_list(file_list_path: str, input_dir: str) -> List[str]:
    """
    从文件列表读取待降噪音频

    Args:
        file_list_path: 文件列表路径（每行一个文件名或路径）
        input_dir: 音频输入目录（用于补全路径）

    Returns:
        List[str]: 音频文件路径列表
    """
    if not os.path.exists(file_list_path):
        logger.error(f"文件列表不存在: {file_list_path}")
        return []

    audio_paths = []
    with open(file_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 如果是完整路径，直接使用
            if os.path.exists(line):
                audio_paths.append(line)
            # 否则在 input_dir 中找
            else:
                path = os.path.join(input_dir, line)
                if os.path.exists(path):
                    audio_paths.append(path)

    logger.info(f"从文件列表加载: {len(audio_paths)} 个音频")
    return audio_paths


def main():
    parser = argparse.ArgumentParser(description="GPU 服务器版降噪候选批量处理脚本")
    # 输入方式（三选一）
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help=f"音频输入目录（默认: {DEFAULT_INPUT_DIR}）")
    parser.add_argument("--candidates", type=str, default=None, help="降噪候选列表 CSV 路径（从质量检查报告生成）")
    parser.add_argument("--file-list", type=str, default=None, help="待降噪文件列表路径（每行一个文件名）")
    # 输出
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"降噪后音频输出目录（默认: {DEFAULT_OUTPUT_DIR}）")
    parser.add_argument("--report", type=str, default=DEFAULT_REPORT, help=f"降噪报告输出路径（默认: {DEFAULT_REPORT}）")
    # 降噪参数
    parser.add_argument("--strength", type=float, default=0.5, help="降噪强度 0.0-1.0（默认0.5）")
    parser.add_argument("--stationary", action="store_true", help="稳态噪声模式（持续的背景噪声）")
    parser.add_argument("--sample-rate", type=int, default=None, help="目标采样率（默认保持原采样率）")
    # 并行
    parser.add_argument("--workers", type=int, default=1, help="并行进程数（默认1，GPU实例建议4-8）")
    # 其他
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际降噪")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个音频（用于测试）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GPU 版降噪批量处理启动")
    logger.info(f"  输入目录: {args.input_dir}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  降噪强度: {args.strength}")
    logger.info(f"  稳态噪声: {args.stationary}")
    logger.info(f"  目标采样率: {args.sample_rate or '保持原样'}")
    logger.info(f"  并行进程数: {args.workers}")
    logger.info(f"  预览模式: {args.dry_run}")
    logger.info("=" * 60)

    # 检查 noisereduce
    if not HAS_NOISEREDUCE and not args.dry_run:
        logger.error("noisereduce 未安装，请运行: pip install noisereduce")
        return

    # 获取待降噪音频列表（三选一）
    if args.candidates:
        logger.info(f"使用候选列表模式: {args.candidates}")
        audio_paths = load_candidates(args.candidates, args.input_dir)
    elif args.file_list:
        logger.info(f"使用文件列表模式: {args.file_list}")
        audio_paths = load_file_list(args.file_list, args.input_dir)
    else:
        logger.info("使用输入目录扫描模式")
        audio_paths = scan_input_dir(args.input_dir)

    if not audio_paths:
        logger.error("没有找到待降噪音频")
        return

    # 限制数量
    if args.limit:
        audio_paths = audio_paths[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个音频")

    logger.info(f"待降噪音频: {len(audio_paths)} 个")

    # 生成输出路径
    tasks = []
    for input_path in audio_paths:
        filename = os.path.basename(input_path)
        # 统一输出为 wav，避免 mp3 编码损失
        output_filename = os.path.splitext(filename)[0] + "_denoised.wav"
        output_path = os.path.join(args.output_dir, output_filename)
        tasks.append((input_path, output_path, args.strength, args.stationary, args.sample_rate))

    # 预览模式
    if args.dry_run:
        logger.info("")
        logger.info("[预览模式] 待处理音频:")
        for i, (input_path, output_path, _, _, _) in enumerate(tasks[:10]):
            logger.info(f"  [{i+1}] {os.path.basename(input_path)} → {os.path.basename(output_path)}")
        if len(tasks) > 10:
            logger.info(f"  ... 共 {len(tasks)} 个")
        logger.info("")
        logger.info("预览完成，未实际降噪")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    # 执行降噪
    logger.info("")
    logger.info(f"开始降噪处理（{args.workers} 进程并行）...")
    start_time = datetime.now()

    if args.workers > 1:
        # 多进程并行
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(denoise_single_audio, tasks)
    else:
        # 串行
        results = []
        for i, task in enumerate(tasks):
            logger.info(f"[{i+1}/{len(tasks)}] 处理: {os.path.basename(task[0])}")
            result = denoise_single_audio(task)
            results.append(result)
            if result["status"] == "success":
                logger.info(f"  ✅ 成功 ({result['output_duration']:.1f}s)")
            else:
                logger.error(f"  ❌ 失败: {result['error']}")

    elapsed = (datetime.now() - start_time).total_seconds()

    # 统计
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    logger.info("")
    logger.info("=" * 60)
    logger.info("降噪批量处理完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"  总计: {len(results)}")
    logger.info(f"  耗时: {elapsed:.1f}秒")
    if len(results) > 0:
        logger.info(f"  平均每首: {elapsed/len(results):.2f}秒")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 保存报告
    if results:
        report_df = pd.DataFrame(results)
        report_df.to_csv(args.report, index=False, encoding="utf-8")
        logger.info(f"降噪报告已保存: {args.report}")

    # 输出失败列表
    if failed_count > 0:
        logger.info("")
        logger.info("失败列表:")
        for r in results:
            if r["status"] == "failed":
                logger.info(f"  - {r['filename']}: {r['error']}")


if __name__ == "__main__":
    main()
