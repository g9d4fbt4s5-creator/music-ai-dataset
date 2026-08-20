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
MUSIC_TAGS = {
    "Music", "Musical instrument", "Singing", "Male singing", "Female singing",
    "Choir", "Orchestra", "Piano", "Guitar", "Drum", "Bass guitar",
    "Synthesizer", "Pop music", "Rock music", "Hip hop music", "Electronic music",
    "Jazz", "Classical music", "Country music", "Reggae", "Blues", "R&B",
    "Folk music", "World music", "Latin music", "Funk", "Disco", "Techno",
    "House music", "Trance", "Dubstep", "Ambient music", "New age music",
}

SPEECH_TAGS = {
    "Speech", "Conversation", "Narration, monologue", "Babble",
    "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell",
    "Children shouting", "Woman speech", "Man speech", "Child speech",
    "Infant cry", "Cough", "Sneeze", "Throat clearing",
}

NOISE_TAGS = {
    "Environmental noise", "Silence", "White noise", "Pink noise",
    "Throbbing", "Static", "Distortion", "Sound effect",
    "Traffic noise", "Aircraft noise", "Engine noise", "Wind noise",
    "Rain", "Thunder", "Water", "Waves", "Bird", "Insect",
    "Dog", "Cat", "Horse", "Cow", "Frog", "Cricket",
}

VOCALS_TAGS = {
    "Singing", "Male singing", "Female singing", "Choir",
    "Vocalization", "Hum", "Beatboxing", "Rapping",
}

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
    聚合帧级结果为曲目级标签

    Args:
        frame_results: 帧级结果列表
        confidence_threshold: 置信度阈值，低于此值的帧不计入统计

    Returns:
        Dict: 曲目级聚合结果，失败返回 None
    """
    if not frame_results:
        return None

    track_id = frame_results[0]["track_id"]

    # 统计高置信度帧的类别
    classes = [r["class_name"] for r in frame_results if r["confidence"] > confidence_threshold]
    if not classes:
        # 如果没有高置信度帧，用所有帧
        classes = [r["class_name"] for r in frame_results]

    counter = Counter(classes)
    total = len(classes)

    # 判断是否音乐、语音、噪声
    is_music = any(c in MUSIC_TAGS for c, _ in counter.most_common(10))
    has_speech = any(c in SPEECH_TAGS for c, _ in counter.most_common(10))
    has_noise = any(c in NOISE_TAGS for c, _ in counter.most_common(10))

    # 人声占比估算
    vocals_count = sum(count for c, count in counter.items() if c in VOCALS_TAGS)
    vocals_ratio = vocals_count / total if total > 0 else 0.0

    # top5 标签
    top5 = counter.most_common(5)
    top5_str = "; ".join(f"{name}:{count}" for name, count in top5)

    return {
        "track_id": track_id,
        "yamnet_top_tags": top5_str,
        "is_music": is_music,
        "has_speech": has_speech,
        "has_noise": has_noise,
        "vocals_ratio_estimate": round(vocals_ratio, 4),
        "total_frames": len(frame_results),
        "high_confidence_frames": len(classes),
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


def main():
    parser = argparse.ArgumentParser(description="YAMNet 音频事件检测批量推理（yamnet_env 专用）")
    parser.add_argument("--input-list", type=str, required=True, help="输入列表文件路径（CSV）")
    parser.add_argument("--output", type=str, default="yamnet_output.parquet", help="输出文件路径（parquet/csv）")
    parser.add_argument("--confidence-threshold", type=float, default=0.3, help="帧级置信度阈值")
    args = parser.parse_args()

    # 加载模型和类别名称
    model, class_names = load_yamnet_model()

    # 批量推理
    batch_inference(model, class_names, args.input_list, args.output, args.confidence_threshold)


if __name__ == "__main__":
    main()
