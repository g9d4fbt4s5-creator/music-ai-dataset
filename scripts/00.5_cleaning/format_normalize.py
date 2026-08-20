"""
format_normalize.py
格式标准化脚本（Stage 2）

功能：
- 从 audio_manifest.csv 读取音频列表
- 批量格式转换（mp3/ogg/m4a → wav/flac）
- 采样率转换（统一到 44.1kHz 或 48kHz）
- 位深转换（16-bit / 24-bit）
- 声道转换（单声道 / 立体声）
- 可选降噪（noisereduce）
- 生成转换报告

用法：
    # 全部转换
    python format_normalize.py

    # 只转换指定 audio_id
    python format_normalize.py --audio-id 01M0E9X162CTB4D15WZQ5D8FVX

    # 指定输出目录
    python format_normalize.py --output-dir ./data/01_preprocess/processed_audio

    # 启用降噪
    python format_normalize.py --enable-noise-reduction

    # 预览模式（不实际转换）
    python format_normalize.py --dry-run

    # 限制处理数量
    python format_normalize.py --limit 10
"""
import os
import sys
import yaml
import logging
import argparse
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "00.5_cleaning"))

from get_audio_physical_path import get_audio_absolute_path

# 默认配置文件
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "cleaning_config.yaml"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"format_normalize_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """加载配置文件"""
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_path}")
    return config


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """加载音频清单"""
    if not manifest_path.exists():
        logger.error(f"音频清单不存在: {manifest_path}")
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info(f"加载音频清单: {len(df)} 条记录")
    return df


def normalize_audio(
    input_path: str,
    output_path: str,
    target_sample_rate: int = 44100,
    target_bit_depth: int = 16,
    target_channels: int = 2,
    enable_noise_reduction: bool = False,
    noise_reduction_strength: float = 0.5,
    noise_reduction_stationary: bool = True,
) -> Tuple[bool, Dict]:
    """
    标准化单个音频

    Args:
        input_path: 输入音频路径
        output_path: 输出音频路径
        target_sample_rate: 目标采样率
        target_bit_depth: 目标位深
        target_channels: 目标声道数
        enable_noise_reduction: 是否启用降噪
        noise_reduction_strength: 降噪强度
        noise_reduction_stationary: 是否稳态噪声

    Returns:
        (success, info): 是否成功，转换信息
    """
    info = {
        "input_path": input_path,
        "output_path": output_path,
        "original_sample_rate": 0,
        "original_channels": 0,
        "original_duration": 0,
        "target_sample_rate": target_sample_rate,
        "target_bit_depth": target_bit_depth,
        "target_channels": target_channels,
        "noise_reduction_applied": False,
        "error": None,
    }

    try:
        # 加载音频
        y, sr = librosa.load(input_path, sr=None, mono=False)
        info["original_sample_rate"] = sr
        info["original_channels"] = y.shape[0] if y.ndim > 1 else 1
        info["original_duration"] = len(y[0]) / sr if y.ndim > 1 else len(y) / sr

        # 声道处理
        if y.ndim > 1:
            if target_channels == 1:
                y = librosa.to_mono(y)
            elif y.shape[0] == 1 and target_channels == 2:
                y = np.vstack([y, y])
        elif target_channels == 2:
            y = np.vstack([y, y])

        # 采样率转换
        if sr != target_sample_rate:
            if y.ndim > 1:
                y_resampled = []
                for ch in range(y.shape[0]):
                    y_ch = librosa.resample(y[ch], orig_sr=sr, target_sr=target_sample_rate)
                    y_resampled.append(y_ch)
                y = np.array(y_resampled)
            else:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sample_rate)
            sr = target_sample_rate

        # 降噪
        if enable_noise_reduction:
            try:
                import noisereduce as nr
                if y.ndim > 1:
                    y_denoised = []
                    for ch in range(y.shape[0]):
                        y_ch = nr.reduce_noise(
                            y=y[ch],
                            sr=sr,
                            stationary=noise_reduction_stationary,
                            prop_decrease=noise_reduction_strength
                        )
                        y_denoised.append(y_ch)
                    y = np.array(y_denoised)
                else:
                    y = nr.reduce_noise(
                        y=y,
                        sr=sr,
                        stationary=noise_reduction_stationary,
                        prop_decrease=noise_reduction_strength
                    )
                info["noise_reduction_applied"] = True
            except ImportError:
                logger.warning("noisereduce 未安装，跳过降噪")
            except Exception as e:
                logger.warning(f"降噪失败: {str(e)}")

        # 确定位深的 subtype
        if target_bit_depth == 16:
            subtype = "PCM_16"
        elif target_bit_depth == 24:
            subtype = "PCM_24"
        elif target_bit_depth == 32:
            subtype = "PCM_32"
        else:
            subtype = "PCM_16"

        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y.T if y.ndim > 1 else y, sr, subtype=subtype)

        return True, info

    except Exception as e:
        info["error"] = str(e)
        logger.error(f"转换失败: {input_path} -> {str(e)}")
        return False, info


def main():
    parser = argparse.ArgumentParser(
        description="格式标准化脚本（Stage 2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="配置文件路径")
    parser.add_argument("--manifest", type=str, default=None,
                        help="音频清单 CSV 路径（默认使用配置中的路径）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认使用配置中的路径）")
    parser.add_argument("--audio-id", type=str, default=None,
                        help="只处理指定的 audio_id")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    parser.add_argument("--target-format", type=str, default=None,
                        help="目标格式（wav/flac）")
    parser.add_argument("--target-sample-rate", type=int, default=None,
                        help="目标采样率")
    parser.add_argument("--target-bit-depth", type=int, default=None,
                        help="目标位深（16/24）")
    parser.add_argument("--target-channels", type=int, default=None,
                        help="目标声道数（1/2）")
    parser.add_argument("--enable-noise-reduction", action="store_true",
                        help="启用降噪")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不实际转换")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("格式标准化（Stage 2）")
    logger.info("=" * 60)

    # 加载配置
    config = load_config(Path(args.config))
    stage2_config = config.get("stage2_format", {})

    # 目标参数（命令行优先，其次配置文件，最后默认值）
    target_format = args.target_format or stage2_config.get("target_format", "wav")
    target_sample_rate = args.target_sample_rate or stage2_config.get("target_sample_rate", 44100)
    target_bit_depth = args.target_bit_depth or stage2_config.get("target_bit_depth", 16)
    target_channels = args.target_channels or stage2_config.get("target_channels", 2)

    # 降噪配置
    noise_cfg = stage2_config.get("noise_reduction", {})
    enable_noise_reduction = args.enable_noise_reduction or noise_cfg.get("enabled", False)
    noise_strength = noise_cfg.get("strength", 0.5)
    noise_stationary = noise_cfg.get("stationary", True)

    # 输出目录
    output_dir = Path(args.output_dir) if args.output_dir else \
        PROJECT_ROOT / "data" / "01_preprocess" / "processed_audio"

    logger.info(f"目标格式: {target_format}")
    logger.info(f"目标采样率: {target_sample_rate} Hz")
    logger.info(f"目标位深: {target_bit_depth}-bit")
    logger.info(f"目标声道: {target_channels}")
    logger.info(f"降噪: {'开启' if enable_noise_reduction else '关闭'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"预览模式: {'是' if args.dry_run else '否'}")

    # 加载音频清单
    manifest_path = Path(args.manifest) if args.manifest else \
        PROJECT_ROOT / config.get("global", {}).get("manifest_csv", "data/00_raw_collect/audio_manifest.csv")
    df = load_manifest(manifest_path)

    # 过滤
    if args.audio_id:
        df = df[df["audio_id"] == args.audio_id]
        logger.info(f"只处理 audio_id: {args.audio_id} ({len(df)} 个)")

    if args.limit:
        df = df.head(args.limit)
        logger.info(f"限制处理数量: {len(df)}")

    logger.info(f"待处理音频: {len(df)} 个")

    if args.dry_run:
        logger.info("预览模式，不实际转换")
        for _, row in df.iterrows():
            audio_id = row["audio_id"]
            ext = row.get("format", "wav").lower()
            input_path = get_audio_absolute_path(audio_id, ext)
            output_filename = f"{audio_id}.{target_format}"
            output_path = output_dir / output_filename
            logger.info(f"  {audio_id}: {input_path.name} -> {output_filename}")
        return

    # 批量转换
    results = []
    success_count = 0
    fail_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        audio_id = row["audio_id"]
        ext = row.get("format", "wav").lower()

        # 输入路径
        input_path = get_audio_absolute_path(audio_id, ext)
        if not input_path.exists():
            logger.warning(f"[{i+1}/{len(df)}] 文件不存在，跳过: {audio_id}")
            fail_count += 1
            results.append({
                "audio_id": audio_id,
                "success": False,
                "error": "文件不存在",
            })
            continue

        # 输出路径
        output_filename = f"{audio_id}.{target_format}"
        output_path = output_dir / output_filename

        logger.info(f"[{i+1}/{len(df)}] 转换: {audio_id}")

        # 转换
        success, info = normalize_audio(
            input_path=str(input_path),
            output_path=str(output_path),
            target_sample_rate=target_sample_rate,
            target_bit_depth=target_bit_depth,
            target_channels=target_channels,
            enable_noise_reduction=enable_noise_reduction,
            noise_reduction_strength=noise_strength,
            noise_reduction_stationary=noise_stationary,
        )

        result = {
            "audio_id": audio_id,
            "success": success,
            **info,
        }
        results.append(result)

        if success:
            success_count += 1
            logger.info(f"  ✅ 完成: {output_filename}")
        else:
            fail_count += 1
            logger.error(f"  ❌ 失败: {info.get('error', '未知错误')}")

    # 生成报告
    report_df = pd.DataFrame(results)
    report_dir = PROJECT_ROOT / "data" / "00.5_cleaned" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_csv = report_dir / f"format_normalize_report_{time_str}.csv"
    report_df.to_csv(report_csv, index=False, encoding="utf-8")

    logger.info("")
    logger.info("=" * 60)
    logger.info("格式标准化完成")
    logger.info(f"  成功: {success_count}/{len(df)}")
    logger.info(f"  失败: {fail_count}/{len(df)}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  报告: {report_csv}")
    logger.info(f"  日志: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
