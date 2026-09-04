"""
audio_chunker.py
音频切片分块模块（Stage 6 预处理输出）

功能：
- 将长音频切分为固定长度的片段（5-30秒）
- 支持滑动窗口重叠（默认50%）
- 最小片段长度过滤（默认5秒，丢弃过短的末尾片段）
- 生成切片元数据 CSV（chunk_id, audio_id, start, end, duration, path）
- 支持重采样到目标采样率
- 支持单声道/立体声输出
- 支持批量处理 manifest 中的音频

用法：
    # 批量处理 manifest 中的所有音频
    python 03_audio_chunker.py --manifest data/00_raw_collect/audio_manifest.csv

    # 自定义切片参数
    python 03_audio_chunker.py --manifest manifest.csv --chunk-sec 15 --overlap 0.5

    # 只处理前N个
    python 03_audio_chunker.py --manifest manifest.csv --limit 10

    # 预览模式
    python 03_audio_chunker.py --manifest manifest.csv --dry-run

    # 作为模块导入
    from audio_chunker import AudioChunker
    chunker = AudioChunker(chunk_size=30, overlap=0.5, min_chunk_length=5)
    chunks = chunker.chunk(audio_path, output_dir)
"""
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from get_audio_physical_path import get_audio_absolute_path

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """切片信息"""
    chunk_id: str
    audio_id: str
    source_path: str
    chunk_path: str
    start_time: float  # 秒
    end_time: float  # 秒
    duration: float  # 秒
    sample_rate: int
    channels: int
    bit_depth: int

    def to_dict(self) -> Dict:
        return asdict(self)


class AudioChunker:
    """音频切片器"""

    def __init__(
        self,
        chunk_size: float = 30.0,
        overlap: float = 0.5,
        min_chunk_length: float = 5.0,
        target_sample_rate: Optional[int] = None,
        target_channels: Optional[int] = None,
        output_format: str = "wav",
        output_bit_depth: str = "PCM_16",
    ):
        """
        初始化音频切片器

        Args:
            chunk_size: 每个切片的长度（秒），默认30秒
            overlap: 重叠比例（0-1），默认0.5（50%重叠）
            min_chunk_length: 最小切片长度（秒），低于此值的末尾片段丢弃，默认5秒
            target_sample_rate: 目标采样率，None表示保持原采样率
            target_channels: 目标声道数，None表示保持原声道数
            output_format: 输出格式（wav/flac）
            output_bit_depth: 输出位深（PCM_16/PCM_24/PCM_32）
        """
        if not (0 <= overlap < 1):
            raise ValueError(f"overlap 必须在 [0, 1) 范围内，当前: {overlap}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须大于0，当前: {chunk_size}")
        if min_chunk_length <= 0:
            raise ValueError(f"min_chunk_length 必须大于0，当前: {min_chunk_length}")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.output_format = output_format
        self.output_bit_depth = output_bit_depth

        # 计算步长
        self.hop_size = chunk_size * (1 - overlap)

        logger.info("音频切片器初始化完成")
        logger.info(f"  切片长度: {chunk_size}秒")
        logger.info(f"  重叠比例: {overlap*100:.0f}%")
        logger.info(f"  步长: {self.hop_size:.1f}秒")
        logger.info(f"  最小切片长度: {min_chunk_length}秒")
        logger.info(f"  目标采样率: {target_sample_rate if target_sample_rate else '保持原样'}")
        logger.info(f"  目标声道: {target_channels if target_channels else '保持原样'}")
        logger.info(f"  输出格式: {output_format} ({output_bit_depth})")

    def chunk(
        self,
        audio_path: str,
        output_dir: str,
        audio_id: Optional[str] = None,
    ) -> List[ChunkInfo]:
        """
        对单个音频进行切片

        Args:
            audio_path: 输入音频路径
            output_dir: 输出目录
            audio_id: 音频ID（用于生成 chunk_id 和文件名），None则用文件名

        Returns:
            List[ChunkInfo]: 切片信息列表
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if audio_id is None:
            audio_id = audio_path.stem

        logger.info(f"切片: {audio_path.name} (audio_id={audio_id})")

        # 加载音频
        try:
            y, sr = librosa.load(str(audio_path), sr=self.target_sample_rate, mono=False)
        except Exception as e:
            logger.error(f"加载音频失败: {audio_path} -> {e}")
            return []

        # 处理声道
        if y.ndim == 1:
            y = y.reshape(1, -1)
            channels = 1
        else:
            channels = y.shape[0]

        if self.target_channels == 1 and channels > 1:
            y = np.mean(y, axis=0, keepdims=True)
            channels = 1
        elif self.target_channels == 2 and channels == 1:
            y = np.vstack([y, y])
            channels = 2

        duration = y.shape[1] / sr
        logger.info(f"  时长: {duration:.1f}秒, 采样率: {sr}Hz, 声道: {channels}")

        # 计算切片数量
        chunk_samples = int(self.chunk_size * sr)
        hop_samples = int(self.hop_size * sr)
        min_chunk_samples = int(self.min_chunk_length * sr)

        if duration < self.min_chunk_length:
            logger.warning(f"  音频过短（{duration:.1f}秒 < {self.min_chunk_length}秒），跳过切片")
            return []

        # 生成切片
        chunks = []
        start_sample = 0
        chunk_index = 0

        while start_sample < y.shape[1]:
            end_sample = min(start_sample + chunk_samples, y.shape[1])
            chunk_duration = (end_sample - start_sample) / sr

            # 跳过过短的末尾片段
            if chunk_duration < self.min_chunk_length and start_sample > 0:
                logger.info(f"  跳过末尾过短片段: {chunk_duration:.1f}秒 < {self.min_chunk_length}秒")
                break

            # 提取切片
            chunk_data = y[:, start_sample:end_sample]

            # 生成 chunk_id 和文件名
            chunk_id = f"{audio_id}_chunk{chunk_index:04d}"
            start_time = start_sample / sr
            end_time = end_sample / sr
            chunk_filename = f"{chunk_id}.{self.output_format}"
            chunk_path = output_dir / chunk_filename

            # 保存切片
            try:
                # soundfile 需要 (samples, channels) 格式
                save_data = chunk_data.T if channels > 1 else chunk_data[0]
                sf.write(
                    str(chunk_path),
                    save_data,
                    sr,
                    subtype=self.output_bit_depth,
                )

                chunk_info = ChunkInfo(
                    chunk_id=chunk_id,
                    audio_id=audio_id,
                    source_path=str(audio_path),
                    chunk_path=str(chunk_path),
                    start_time=round(start_time, 3),
                    end_time=round(end_time, 3),
                    duration=round(chunk_duration, 3),
                    sample_rate=sr,
                    channels=channels,
                    bit_depth=int(self.output_bit_depth.split("_")[1]),
                )
                chunks.append(chunk_info)
                chunk_index += 1

            except Exception as e:
                logger.error(f"保存切片失败: {chunk_path} -> {e}")

            start_sample += hop_samples

        logger.info(f"  生成 {len(chunks)} 个切片")
        return chunks

    def batch_chunk(
        self,
        audio_paths: List[str],
        output_dir: str,
        audio_ids: Optional[List[str]] = None,
        manifest_csv: Optional[str] = None,
    ) -> Tuple[List[ChunkInfo], pd.DataFrame]:
        """
        批量切片

        Args:
            audio_paths: 音频路径列表
            output_dir: 输出目录
            audio_ids: 音频ID列表，None则用文件名
            manifest_csv: 切片元数据 CSV 输出路径

        Returns:
            (all_chunks, manifest_df): 所有切片信息列表和元数据 DataFrame
        """
        all_chunks = []

        for i, audio_path in enumerate(audio_paths):
            audio_id = audio_ids[i] if audio_ids else None
            chunks = self.chunk(audio_path, output_dir, audio_id)
            all_chunks.extend(chunks)

        # 生成元数据 DataFrame
        manifest_df = pd.DataFrame([c.to_dict() for c in all_chunks])

        if manifest_csv and len(manifest_df) > 0:
            os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)
            manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8")
            logger.info(f"切片元数据已保存: {manifest_csv} ({len(manifest_df)} 条)")

        # 统计
        total_duration = sum(c.duration for c in all_chunks)
        logger.info(f"批量切片完成: {len(audio_paths)} 个音频 -> {len(all_chunks)} 个切片, 总时长 {total_duration:.1f}秒")

        return all_chunks, manifest_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="音频切片分块脚本（Stage 6 预处理输出）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=str, required=True,
                        help="输入 audio_manifest.csv 路径")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "01_preprocess" / "segments"),
                        help="切片输出目录（默认 data/01_preprocess/segments）")
    parser.add_argument("--chunk-sec", type=float, default=15.0,
                        help="切片长度（秒），默认15秒")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="滑动窗口重叠比例（0-1），默认0.5（50%%）")
    parser.add_argument("--min-chunk-length", type=float, default=5.0,
                        help="最小切片长度（秒），默认5秒，丢弃过短的末尾片段")
    parser.add_argument("--target-sr", type=int, default=None,
                        help="目标采样率（Hz），None则保持原采样率")
    parser.add_argument("--target-channels", type=int, default=None,
                        help="目标声道数（1=单声道，2=立体声），None则保持原声道")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前N个音频（用于测试）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不实际生成切片文件")
    parser.add_argument("--splits", type=str,
                        default=str(PROJECT_ROOT / "data" / "04_final_dataset" / "splits"),
                        help="数据划分目录（用于只切 train/val），默认 data/04_final_dataset/splits")
    parser.add_argument("--only-train-val", action="store_true",
                        help="只切 train/val 的样本（Stage 5 正确用法），跳过 test/holdout/ood")
    args = parser.parse_args()

    # 日志配置
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"audio_chunker_{time_str}.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logger.info("=" * 60)
    logger.info("音频切片分块（Stage 6 预处理输出）")
    logger.info(f"manifest: {args.manifest}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"切片长度: {args.chunk_sec}秒")
    logger.info(f"重叠比例: {args.overlap*100:.0f}%")
    logger.info(f"最小切片长度: {args.min_chunk_length}秒")
    if args.target_sr:
        logger.info(f"目标采样率: {args.target_sr}Hz")
    if args.target_channels:
        logger.info(f"目标声道数: {args.target_channels}")
    logger.info("=" * 60)

    # 加载 manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"加载音频清单: {len(manifest_df)} 首")

    # P0: 只切 train/val（Stage 5 正确用法，PIPELINE_OVERVIEW_V2.md）
    if args.only_train_val:
        splits_dir = Path(args.splits)
        if not splits_dir.is_absolute():
            splits_dir = PROJECT_ROOT / splits_dir
        
        target_ids = set()
        for split_name in ["train.csv", "val.csv"]:
            split_file = splits_dir / split_name
            if split_file.exists():
                split_df = pd.read_csv(split_file)
                if "audio_id" in split_df.columns:
                    target_ids.update(split_df["audio_id"].tolist())
                    logger.info(f"  加载 {split_name}: {len(split_df)} 首")
        
        if target_ids:
            before = len(manifest_df)
            manifest_df = manifest_df[manifest_df["audio_id"].isin(target_ids)]
            after = len(manifest_df)
            logger.info(f"只切 train/val: {before} → {after} 首（排除 {before - after} 首 test/holdout/ood）")
        else:
            logger.warning(f"未在 {splits_dir} 找到 train.csv/val.csv，跳过过滤（处理全部样本）")

    if args.limit:
        manifest_df = manifest_df.head(args.limit)
        logger.info(f"限制处理: 前 {args.limit} 首")

    # 解析音频路径
    audio_paths = []
    audio_ids = []
    skipped = 0

    for _, row in manifest_df.iterrows():
        audio_id = row.get("audio_id", "")
        if not audio_id:
            skipped += 1
            continue

        # ADR-004: 排除不参与训练的样本（黄金集、challenge_set、val）
        # in_train_training=False 的样本只做KNN种子/真值/压力测试，不切成训练样本
        in_train = row.get("in_train_training", True)
        if in_train is False or (isinstance(in_train, str) and in_train.lower() == "false"):
            sample_type = row.get("sample_type", "unknown")
            logger.debug(f"跳过非训练样本: {audio_id} (sample_type={sample_type}, in_train_training=False)")
            skipped += 1
            continue

        # 优先从 file_relative_path 读取
        rel_path = row.get("file_relative_path", "")
        if rel_path and isinstance(rel_path, str):
            abs_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path
            if abs_path.exists():
                audio_paths.append(str(abs_path))
                audio_ids.append(audio_id)
                continue

        # 尝试用 audio_id 查找
        abs_path = get_audio_absolute_path(audio_id)
        if abs_path and Path(abs_path).exists():
            audio_paths.append(abs_path)
            audio_ids.append(audio_id)
        else:
            logger.warning(f"音频文件不存在: {audio_id}")
            skipped += 1

    logger.info(f"有效音频: {len(audio_paths)} 首，跳过: {skipped} 首")

    if not audio_paths:
        logger.error("没有有效音频，退出")
        sys.exit(1)

    # 创建切片器
    chunker = AudioChunker(
        chunk_size=args.chunk_sec,
        overlap=args.overlap,
        min_chunk_length=args.min_chunk_length,
        target_sample_rate=args.target_sr,
        target_channels=args.target_channels,
    )

    # 输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 切片元数据 CSV 路径
    manifest_csv = output_dir / f"chunks_manifest_{time_str}.csv"

    if args.dry_run:
        logger.info("DRY-RUN 模式：预览，不实际生成切片文件")
        # 只计算预计切片数量
        total_duration = 0
        for ap in audio_paths:
            try:
                duration = librosa.get_duration(path=ap)
                total_duration += duration
            except Exception:
                pass
        step = args.chunk_sec * (1 - args.overlap)
        estimated_chunks = int(total_duration / step) if step > 0 else 0
        logger.info(f"预计总时长: {total_duration:.1f}秒")
        logger.info(f"预计切片数量: ~{estimated_chunks} 个")
    else:
        # 批量切片
        all_chunks, chunks_df = chunker.batch_chunk(
            audio_paths=audio_paths,
            output_dir=str(output_dir),
            audio_ids=audio_ids,
            manifest_csv=str(manifest_csv),
        )

        # 总结
        logger.info("")
        logger.info("=" * 60)
        logger.info("切片完成")
        logger.info(f"  输入音频: {len(audio_paths)} 首")
        logger.info(f"  生成切片: {len(all_chunks)} 个")
        logger.info(f"  总时长: {sum(c.duration for c in all_chunks):.1f}秒")
        logger.info(f"  输出目录: {output_dir}")
        logger.info(f"  元数据: {manifest_csv}")
        logger.info("=" * 60)
