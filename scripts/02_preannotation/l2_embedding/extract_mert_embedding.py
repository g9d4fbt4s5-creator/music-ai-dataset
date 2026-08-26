"""
extract_mert_embedding.py
提取 MERT 音乐理解模型的嵌入向量

功能：
- 加载 MERT-v1-95M 模型
- 对音频提取 768维 嵌入向量
- 支持长音频分块（30秒）
- 输出：每个音频一个 .npy 文件

用法：
    python extract_mert_embedding.py \
        --input-dir /root/autodl-tmp/jazz_25_audio \
        --output /root/autodl-tmp/preannotation/l2_mert_embedding \
        --device cuda \
        --chunk-sec 30
"""
import os
import sys
import argparse
import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24000


def load_mert_model(device: str = "cuda"):
    from transformers import AutoModel, AutoFeatureExtractor
    logger.info(f"加载 MERT 模型: {MODEL_NAME}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    logger.info(f"✅ MERT 模型加载完成，设备: {device}")
    return model, feature_extractor


def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE, max_duration: int = 30) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze().numpy()
    max_samples = max_duration * target_sr
    if len(audio) > max_samples:
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]
    return audio.astype(np.float32)


def extract_embedding(model, feature_extractor, audio_path: str, device: str = "cuda", chunk_sec: int = 30) -> np.ndarray:
    audio = load_audio(audio_path, max_duration=chunk_sec)
    inputs = feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding


def main():
    parser = argparse.ArgumentParser(description="提取 MERT 音乐理解模型嵌入向量")
    parser.add_argument("--input-dir", type=str, required=True, help="输入音频目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备")
    parser.add_argument("--chunk-sec", type=int, default=30, help="分块时长")
    parser.add_argument("--limit", type=int, default=None, help="限制处理数量")
    args = parser.parse_args()

    # 自动创建日志目录（防止 nohup 重定向失败导致进程立即退出）
    Path("logs").mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
    audio_files = sorted(audio_files)

    if args.limit:
        audio_files = audio_files[:args.limit]

    logger.info(f"输入目录: {input_dir}")
    logger.info(f"音频文件数: {len(audio_files)}")

    if not audio_files:
        logger.error("未找到音频文件")
        return

    model, feature_extractor = load_mert_model(args.device)

    results = []
    for idx, audio_path in enumerate(audio_files):
        try:
            audio_id = audio_path.stem
            embedding = extract_embedding(model, feature_extractor, str(audio_path), args.device, args.chunk_sec)
            np.save(output_dir / f"{audio_id}_mert_embedding.npy", embedding)
            results.append(audio_id)
            if (idx + 1) % 5 == 0 or idx == 0:
                logger.info(f"[{idx+1}/{len(audio_files)}] ✅ {audio_id}: shape={embedding.shape}, norm={np.linalg.norm(embedding):.2f}")
        except Exception as e:
            logger.error(f"[{idx+1}/{len(audio_files)}] ❌ {audio_path.name}: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("MERT 嵌入提取完成")
    logger.info("=" * 60)
    logger.info(f"  总数: {len(audio_files)}")
    logger.info(f"  成功: {len(results)}")
    logger.info(f"  失败: {len(audio_files) - len(results)}")
    logger.info(f"  嵌入维度: 768")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
