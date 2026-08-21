#!/usr/bin/env python3
"""
yamnet_infer.py
YAMNet 音频事件检测推理脚本（yamnet_env 专用，不 import 到主环境）

原则：
- yamnet_env 只输出 CSV/Parquet
- 主环境（labelstudio-env）只读 CSV/Parquet，绝不 import tensorflow
- YAMNet 输入硬约束：16kHz 单声道，波形值 [-1.0, 1.0]

用法（在 yamnet_env 环境下运行）：
    conda activate yamnet_env
    python3 yamnet_infer.py --input-list input_list.csv --output yamnet_output.parquet

输入：
    input_list.csv：每行一个音频路径（或 track_id,path 两列）

输出：
    yamnet_output.parquet：曲目级 YAMNet 检测结果
        - track_id: 曲目ID
        - yamnet_top_tags: top5标签（JSON字符串）
        - is_music: 是否音乐
        - has_speech: 是否有语音
        - has_noise: 是否有噪声
        - vocals_ratio_estimate: 人声占比估算
        - total_frames: 总帧数
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

# librosa 作为备选（soundfile 加载失败时使用）
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ===================== 配置 =====================
SAMPLE_RATE = 16000  # YAMNet 硬约束：16kHz 单声道
FRAME_HOP = 0.48  # YAMNet 每帧跳步 0.48s
FRAME_DURATION = 0.96  # YAMNet 每帧时长 0.96s

# YAMNet 类别分组（用于曲目级聚合）
# 修正版：移除 Silence 从 NOISE_TAGS，拆分 SPEECH/VOCALS，新增 SILENCE_TAGS
MUSIC_TAGS = {
    "Music", "Musical instrument", "Singing", "Male singing", "Female singing",
    "Choir", "Orchestra", "Piano", "Guitar", "Drum", "Bass guitar",
    "Synthesizer", "Pop music", "Rock music", "Hip hop music", "Electronic music",
    "Jazz", "Classical music", "Country music", "Reggae", "Blues", "R&B",
    "Folk music", "World music", "Latin music", "Funk", "Disco", "Techno",
    "House music", "Trance", "Dubstep", "Ambient music", "New age music",
    "Vocalization", "Hum", "Beatboxing", "Rapping",
}

SPEECH_TAGS = {
    "Speech", "Conversation", "Narration, monologue", "Babble",
    "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell",
    "Children shouting", "Woman speech", "Man speech", "Child speech",
    "Infant cry", "Cough", "Sneeze", "Throat clearing",
}

# 演唱标签（人声乐，区别于说话）
VOCALS_TAGS = {
    "Singing", "Male singing", "Female singing", "Choir",
    "Vocalization", "Hum", "Beatboxing", "Rapping",
}

# 噪声标签（修正：移除 Silence！）
NOISE_TAGS = {
    "Environmental noise", "White noise", "Pink noise",
    "Throbbing", "Static", "Distortion", "Sound effect",
    "Traffic noise", "Aircraft noise", "Engine noise", "Wind noise",
    "Rain", "Thunder", "Water", "Waves", "Bird", "Insect",
    "Dog", "Cat", "Horse", "Cow", "Frog", "Cricket",
}

# 静音标签（单独处理，不算噪声，音乐结构的一部分）
SILENCE_TAGS = {"Silence"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_yamnet_model():
    """
    加载 YAMNet 模型（首次会联网下载 ~15MB）

    Returns:
        tuple: (model, class_names)
            - model: YAMNet 模型
            - class_names: 521 个类别名称列表
    """
    logger.info("加载 YAMNet 模型...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    # 获取类别名称（通过 class_map_path，不是 model.class_names）
    class_map_path = model.class_map_path().numpy()
    if isinstance(class_map_path, bytes):
        class_map_path = class_map_path.decode()
    class_names = pd.read_csv(class_map_path)['display_name'].tolist()

    logger.info(f"YAMNet 模型加载完成，{len(class_names)} 个类别")
    return model, class_names


def load_audio_for_yamnet(audio_path: str, target_sr: int = 16000) -> tuple:
    """
    加载音频为 YAMNet 格式（16kHz 单声道 float32 [-1, 1]）

    优先用 soundfile（轻量，无 numba/llvmlite 依赖），
    soundfile 失败时回退到 librosa（重采样质量更高）。

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率（YAMNet 要求 16000）

    Returns:
        tuple: (audio_array, sample_rate)
    """
    # 优先用 soundfile
    try:
        y, orig_sr = sf.read(audio_path, dtype='float32')

        # 转单声道
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # 重采样到目标采样率
        if orig_sr != target_sr:
            if HAS_LIBROSA:
                # librosa 重采样质量更高
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
            else:
                # 无 librosa 时用 scipy 简单重采样
                from scipy.signal import resample
                new_length = int(len(y) * target_sr / orig_sr)
                y = resample(y, new_length).astype(np.float32)

        # 峰值归一化（解决低音量音频被 YAMNet 误判为静音的问题）
        # 逻辑：如果最大振幅 < 0.3（音量偏小），归一化到 0.9（约 -0.9dB）
        # 这样 track_0048594 这种低音量录音，YAMNet 也能正确识别为音乐
        max_val = np.max(np.abs(y))
        if max_val > 0 and max_val < 0.3:
            y = y * (0.9 / max_val)
            logger.debug(f"峰值归一化: max={max_val:.4f} → 0.9 (放大 {0.9/max_val:.1f}x)")

        # 确保范围 [-1, 1]
        max_val = np.max(np.abs(y))
        if max_val > 1.0:
            y = y / max_val

        return y, target_sr

    except Exception as e:
        # soundfile 失败，回退到 librosa
        if HAS_LIBROSA:
            logger.debug(f"soundfile 加载失败 ({e})，回退到 librosa")
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return y.astype(np.float32), target_sr
        else:
            raise


def run_yamnet_single(model, class_names: List[str], audio_path: str, track_id: str) -> Optional[List[Dict]]:
    """
    对单个音频运行 YAMNet，返回帧级结果

    Args:
        model: YAMNet 模型
        class_names: 类别名称列表（从 model.class_map_path 获取）
        audio_path: 音频文件路径
        track_id: 曲目ID

    Returns:
        List[Dict]: 帧级结果列表，失败返回 None
    """
    try:
        # 加载音频（16kHz 单声道，soundfile 优先）
        y, sr = load_audio_for_yamnet(audio_path, SAMPLE_RATE)

        if len(y) < SAMPLE_RATE:  # 小于1秒
            logger.warning(f"音频过短（{len(y)/sr:.2f}s），跳过: {track_id}")
            return None

        # YAMNet 推理
        scores, embeddings, spectrogram = model(y)

        # 逐帧取 top-1 标签
        results = []
        for i, frame_scores in enumerate(scores):
            top_idx = int(tf.argmax(frame_scores))
            confidence = float(frame_scores[top_idx])
            timestamp = i * FRAME_HOP
            results.append({
                "track_id": track_id,
                "frame_idx": i,
                "timestamp": round(timestamp, 2),
                "class_name": class_names[top_idx],
                "confidence": round(confidence, 4),
            })

        return results

    except Exception as e:
        logger.error(f"YAMNet 推理失败: {track_id} -> {e}")
        return None


def aggregate_track_level(
    frame_results: List[Dict],
    confidence_threshold: float = 0.3,
) -> Optional[Dict]:
    """
    聚合帧级结果为曲目级标签（修正版：占比阈值判定）

    修正内容：
    - 从 top-10 判定改为占比阈值判定
    - NOISE_TAGS 移除 Silence，新增 SILENCE_TAGS
    - 拆分 has_speech（说话）和 has_vocals（演唱）为独立指标
    - 新增各类占比字段，用于人工校验阈值

    Args:
        frame_results: 帧级结果列表
        confidence_threshold: 置信度阈值，低于此值的帧不计入统计

    Returns:
        Dict: 曲目级聚合结果，失败返回 None
    """
    if not frame_results:
        return None

    track_id = frame_results[0]["track_id"]

    # 1. 过滤低置信度帧
    high_conf = [r for r in frame_results if r["confidence"] > confidence_threshold]
    if not high_conf:
        # 兜底：如果没有高置信度帧，用所有帧
        high_conf = frame_results

    total = len(high_conf)
    counter = Counter(r["class_name"] for r in high_conf)

    # 2. 计算各类占比（替代 top-10 判定）
    def calc_ratio(tag_set):
        """计算某组标签的帧数占比"""
        return sum(counter[c] for c in tag_set if c in counter) / total if total else 0.0

    music_ratio = calc_ratio(MUSIC_TAGS)
    speech_ratio = calc_ratio(SPEECH_TAGS)
    vocals_ratio = calc_ratio(VOCALS_TAGS)
    noise_ratio = calc_ratio(NOISE_TAGS)
    silence_ratio = calc_ratio(SILENCE_TAGS)

    # 3. 阈值判定（经验值，可人工校验后调整）
    # - is_music: 音乐帧 >30%（一首3分钟歌至少54秒被识别为音乐）
    # - has_speech: 说话帧 >5%（约9秒，可能是电台采样/采访）
    # - has_vocals: 演唱帧 >5%（约9秒，有人声演唱）
    # - has_noise: 噪声帧 >5%（真实的底噪/环境声，不含静音）
    # - has_silence: 静音帧 >15%（音乐结构，不剔除）
    is_music = music_ratio > 0.30
    has_speech = speech_ratio > 0.05
    has_vocals = vocals_ratio > 0.05
    has_noise = noise_ratio > 0.05
    has_silence = silence_ratio > 0.15

    # top5 标签
    top5 = counter.most_common(5)
    top5_str = "; ".join(f"{name}:{count}" for name, count in top5)

    return {
        "track_id": track_id,
        "yamnet_top_tags": top5_str,
        # 核心布尔判定
        "is_music": is_music,
        "has_speech": has_speech,
        "has_vocals": has_vocals,
        "has_noise": has_noise,
        "has_silence": has_silence,
        # 占比数值（用于人工校验阈值）
        "music_ratio": round(music_ratio, 4),
        "speech_ratio": round(speech_ratio, 4),
        "vocals_ratio": round(vocals_ratio, 4),
        "noise_ratio": round(noise_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        # 兼容旧字段名
        "vocals_ratio_estimate": round(vocals_ratio, 4),
        # 帧统计
        "total_frames": len(frame_results),
        "high_confidence_frames": total,
    }


def batch_inference(
    model,
    class_names: List[str],
    input_list: str,
    output_path: str,
    confidence_threshold: float = 0.3,
):
    """
    批量 YAMNet 推理

    Args:
        model: YAMNet 模型
        input_list: 输入列表文件路径（CSV，每行一个路径或 track_id,path）
        output_path: 输出 Parquet 路径
        confidence_threshold: 置信度阈值
    """
    # 读取输入列表
    logger.info(f"读取输入列表: {input_list}")
    if input_list.endswith(".csv"):
        df = pd.read_csv(input_list)
        if "path" in df.columns and "track_id" in df.columns:
            tracks = list(zip(df["track_id"].astype(str), df["path"].astype(str)))
        elif "path" in df.columns:
            tracks = [(Path(p).stem, p) for p in df["path"].astype(str)]
        else:
            # 假设第一列是路径
            tracks = [(Path(p).stem, p) for p in df.iloc[:, 0].astype(str)]
    else:
        # 纯文本，每行一个路径
        with open(input_list, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        tracks = [(Path(p).stem, p) for p in paths]

    logger.info(f"待处理音频: {len(tracks)} 个")

    # 批量推理
    all_track_meta = []
    success_count = 0
    fail_count = 0

    for i, (track_id, audio_path) in enumerate(tracks):
        logger.info(f"[{i+1}/{len(tracks)}] 处理: {track_id}")

        if not os.path.exists(audio_path):
            logger.warning(f"文件不存在，跳过: {audio_path}")
            fail_count += 1
            continue

        frame_results = run_yamnet_single(model, class_names, audio_path, track_id)
        if frame_results is None:
            fail_count += 1
            continue

        track_meta = aggregate_track_level(frame_results, confidence_threshold)
        if track_meta:
            all_track_meta.append(track_meta)
            success_count += 1
        else:
            fail_count += 1

    # 保存结果
    if all_track_meta:
        result_df = pd.DataFrame(all_track_meta)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if output_path.endswith(".parquet"):
            result_df.to_parquet(output_path, index=False)
        elif output_path.endswith(".csv"):
            result_df.to_csv(output_path, index=False, encoding="utf-8")
        else:
            result_df.to_parquet(output_path + ".parquet", index=False)

        logger.info(f"结果已保存: {output_path} ({len(result_df)} 条)")
    else:
        logger.error("没有成功处理的音频")

    logger.info(f"批量推理完成: 成功 {success_count}, 失败 {fail_count}")

    return all_track_meta


# ===================== 并行处理（方案 B：>100 首时使用）=====================

def _parallel_worker(args_tuple):
    """
    多进程 worker 函数：每个进程单独加载 YAMNet 模型，处理分配的音频

    Args:
        args_tuple: (tracks_chunk, confidence_threshold)
            tracks_chunk: List[(track_id, audio_path)]
            confidence_threshold: float

    Returns:
        List[Dict]: 处理结果列表
    """
    tracks_chunk, confidence_threshold = args_tuple

    # 每个 worker 单独加载模型（避免跨进程共享 TF 模型的问题）
    try:
        model, class_names = load_yamnet_model()
    except Exception as e:
        logger.error(f"Worker 加载模型失败: {e}")
        return []

    results = []
    for track_id, audio_path in tracks_chunk:
        if not os.path.exists(audio_path):
            logger.warning(f"文件不存在，跳过: {audio_path}")
            continue

        frame_results = run_yamnet_single(model, class_names, audio_path, track_id)
        if frame_results is None:
            continue

        track_meta = aggregate_track_level(frame_results, confidence_threshold)
        if track_meta:
            results.append(track_meta)

    return results


def batch_inference_parallel(
    input_list: str,
    output_path: str,
    confidence_threshold: float = 0.3,
    num_workers: int = 4,
):
    """
    并行批量 YAMNet 推理（方案 B：>100 首时使用，multiprocessing）

    每个 worker 进程单独加载 YAMNet 模型，处理分配的音频分片。
    适用于大批量处理（>100 首），小批量（<20 首）建议用串行版本。

    Args:
        input_list: 输入列表文件路径
        output_path: 输出文件路径
        confidence_threshold: 置信度阈值
        num_workers: 并行进程数
    """
    import multiprocessing as mp

    # 读取输入列表（复用 batch_inference 的读取逻辑）
    logger.info(f"读取输入列表: {input_list}")
    if input_list.endswith(".csv"):
        df = pd.read_csv(input_list)
        if "path" in df.columns and "track_id" in df.columns:
            tracks = list(zip(df["track_id"].astype(str), df["path"].astype(str)))
        elif "path" in df.columns:
            tracks = [(Path(p).stem, p) for p in df["path"].astype(str)]
        else:
            tracks = [(Path(p).stem, p) for p in df.iloc[:, 0].astype(str)]
    else:
        with open(input_list, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        tracks = [(Path(p).stem, p) for p in paths]

    logger.info(f"待处理音频: {len(tracks)} 个")
    logger.info(f"并行进程数: {num_workers}")

    if len(tracks) < num_workers:
        num_workers = max(1, len(tracks))
        logger.info(f"音频数少于进程数，调整为 {num_workers} 个进程")

    # 分片：将 tracks 均分给 num_workers 个 worker
    chunk_size = (len(tracks) + num_workers - 1) // num_workers
    chunks = [tracks[i:i + chunk_size] for i in range(0, len(tracks), chunk_size)]
    logger.info(f"分片: {len(chunks)} 个，每片约 {chunk_size} 个音频")

    # 多进程处理
    logger.info("开始并行推理...")
    start_time = datetime.now()

    with mp.Pool(processes=num_workers) as pool:
        worker_args = [(chunk, confidence_threshold) for chunk in chunks]
        results_list = pool.map(_parallel_worker, worker_args)

    # 合并结果
    all_track_meta = []
    for results in results_list:
        all_track_meta.extend(results)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"并行推理完成: 耗时 {elapsed:.1f}秒, 成功 {len(all_track_meta)}/{len(tracks)}")

    # 保存结果
    if all_track_meta:
        result_df = pd.DataFrame(all_track_meta)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        if output_path.endswith(".parquet"):
            result_df.to_parquet(output_path, index=False)
        elif output_path.endswith(".csv"):
            result_df.to_csv(output_path, index=False, encoding="utf-8")
        else:
            result_df.to_parquet(output_path + ".parquet", index=False)

        logger.info(f"结果已保存: {output_path} ({len(result_df)} 条)")
    else:
        logger.error("没有成功处理的音频")

    return all_track_meta


def main():
    parser = argparse.ArgumentParser(description="YAMNet 音频事件检测批量推理（yamnet_env 专用）")
    parser.add_argument("--input-list", type=str, required=True, help="输入列表文件路径（CSV）")
    parser.add_argument("--output", type=str, default="yamnet_output.parquet", help="输出文件路径（parquet/csv）")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="帧级置信度阈值")
    parser.add_argument("--parallel", type=int, default=0, help="并行进程数（0=串行，>0启用multiprocessing并行，建议>100首时使用）")
    args = parser.parse_args()

    if args.parallel > 0:
        # 方案 B：并行处理（每个 worker 单独加载模型）
        logger.info("使用并行模式（方案 B）")
        batch_inference_parallel(
            input_list=args.input_list,
            output_path=args.output,
            confidence_threshold=args.confidence_threshold,
            num_workers=args.parallel,
        )
    else:
        # 方案 A：串行处理（20 首以内直接跑）
        logger.info("使用串行模式（方案 A）")
        model, class_names = load_yamnet_model()
        batch_inference(model, class_names, args.input_list, args.output, args.confidence_threshold)


if __name__ == "__main__":
    main()
