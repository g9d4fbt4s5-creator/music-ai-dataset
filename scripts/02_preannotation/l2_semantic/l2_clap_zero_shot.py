"""
l2_clap_zero_shot.py
L2 语义候选层：CLAP 嵌入提取 + zero-shot 分类

功能：
- 读取音频文件，用 CLAP 模型提取嵌入向量
- 对预定义标签（流派/情绪/乐器/场景）做 zero-shot 分类
- 输出：嵌入向量 (.npy) + 语义候选 (.json)

用法：
    python l2_clap_zero_shot.py \
        --input-dir /root/autodl-tmp/jazz_20_audio \
        --output /root/autodl-tmp/preannotation/l2_semantic \
        --embedding-output /root/autodl-tmp/preannotation/l2_embedding \
        --model-path /root/autodl-tmp/models/clap_fusion/630k-audioset-fusion-best.pt

    # 只提取嵌入，不做 zero-shot 分类
    python l2_clap_zero_shot.py \
        --input-dir /root/autodl-tmp/jazz_20_audio \
        --embedding-output /root/autodl-tmp/preannotation/l2_embedding \
        --no-zero-shot
"""
import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================== 配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ===================== 标签定义 =====================
GENRE_LABELS = [
    "jazz", "classical", "pop", "rock", "electronic", "hip hop",
    "folk", "world", "blues", "R&B", "country", "reggae",
    "bebop", "cool jazz", "free jazz", "fusion", "swing",
    "latin jazz", "smooth jazz", "hard bop",
]

MOOD_LABELS = [
    "happy", "sad", "energetic", "calm", "melancholic", "uplifting",
    "tense", "relaxed", "romantic", "angry", "mysterious", "peaceful",
    "excited", "nostalgic", "dreamy", "dark", "bright", "introspective",
]

INSTRUMENT_LABELS = [
    "piano", "guitar", "saxophone", "trumpet", "drums", "bass",
    "violin", "cello", "flute", "clarinet", "trombone", "vibraphone",
    "harp", "organ", "synthesizer", "double bass", "electric guitar",
    "acoustic guitar", "vocals", "no vocals",
]

SCENE_LABELS = [
    "background music", "concert live", "studio recording",
    "dance", "meditation", "study", "workout", "party",
    "romantic dinner", "coffee shop", "elevator music",
    "movie soundtrack", "video game music",
]


# ===================== CLAP 模型加载 =====================

def load_clap_model(model_path: Optional[str] = None, device: str = "cuda"):
    """
    加载 CLAP 模型

    Args:
        model_path: 本地模型权重路径（可选）
        device: 运行设备

    Returns:
        model, audio_encoder, tokenizer
    """
    import laion_clap
    import torch

    logger.info(f"加载 CLAP 模型，设备: {device}")

    if model_path and Path(model_path).exists():
        logger.info(f"使用本地模型: {model_path}")
        # fusion 版本的权重需要 enable_fusion=True
        enable_fusion = "fusion" in str(model_path).lower()
        model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, device=device)
        model.load_ckpt(model_path)
    else:
        logger.info("使用默认模型（从 HuggingFace 下载）")
        model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
        model.load_ckpt()  # 默认下载 630k-audioset-fusion-best.pt

    model.eval()
    logger.info("✅ CLAP 模型加载完成")
    return model


# ===================== 音频处理 =====================

def load_audio(audio_path: str, sr: int = 48000, max_duration: int = 30) -> np.ndarray:
    """
    加载音频，重采样，截断到最大时长

    Args:
        audio_path: 音频文件路径
        sr: 采样率
        max_duration: 最大时长（秒）

    Returns:
        音频数组 (shape: [samples])
    """
    import librosa

    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    # 截断到最大时长
    max_samples = max_duration * sr
    if len(audio) > max_samples:
        # 取中间片段
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]

    return audio.astype(np.float32)


# ===================== 嵌入提取 =====================

def extract_embedding(model, audio_path: str, device: str = "cuda") -> np.ndarray:
    """
    提取音频的 CLAP 嵌入向量

    关键修复（2026-08-25）：
    - laion_clap 的 get_audio_embedding_from_data(use_tensor=False) 期望输入是
      一个**列表**的 int16 numpy 数组，而不是单个 float32 数组或 torch tensor。
    - 之前用 float32 tensor + use_tensor=True 会报 "expected np.ndarray (got numpy.float32)"。
    - 正确做法：wav * 32767 → int16 → [wav_int16] → use_tensor=False

    Args:
        model: CLAP 模型
        audio_path: 音频文件路径
        device: 运行设备

    Returns:
        嵌入向量 (shape: [512])
    """
    import torch

    audio = load_audio(audio_path)  # float32, sr=48000

    # 关键修复1: 转为int16格式（CLAP内部期望16-bit PCM）
    audio_int16 = (audio * 32767).astype(np.int16)

    with torch.no_grad():
        # 关键修复2: 传入列表[audio_int16]而不是单个数组，use_tensor=False
        embedding = model.get_audio_embedding_from_data(
            x=[audio_int16], use_tensor=False
        )

    return np.array(embedding).flatten()


# ===================== Zero-shot 分类 =====================

def zero_shot_classify(model, audio_embedding: np.ndarray,
                        labels: List[str], device: str = "cuda",
                        top_k: int = 5) -> List[Dict]:
    """
    对一组标签做 zero-shot 分类

    Args:
        model: CLAP 模型
        audio_embedding: 音频嵌入向量
        labels: 标签列表
        device: 运行设备
        top_k: 返回 top-k 结果

    Returns:
        标签列表 [{"label": str, "score": float}, ...]
    """
    import torch

    # 文本编码
    with torch.no_grad():
        text_embedding = model.get_text_embedding(labels, use_tensor=True)

    # 计算相似度
    audio_tensor = torch.from_numpy(audio_embedding).unsqueeze(0).to(device)
    similarities = torch.nn.functional.cosine_similarity(audio_tensor, text_embedding, dim=1)
    scores = similarities.cpu().numpy()

    # 排序，取 top-k
    indices = np.argsort(scores)[::-1][:top_k]
    results = [{"label": labels[i], "score": float(scores[i])} for i in indices]

    return results


# ===================== 主流程 =====================

def process_audio(model, audio_path: str, output_dir: Path,
                  embedding_dir: Optional[Path], device: str = "cuda",
                  do_zero_shot: bool = True, top_k: int = 5) -> Dict:
    """
    处理单个音频文件

    Args:
        model: CLAP 模型
        audio_path: 音频文件路径
        output_dir: 语义候选输出目录
        embedding_dir: 嵌入向量输出目录（可选）
        device: 运行设备
        do_zero_shot: 是否做 zero-shot 分类
        top_k: top-k 结果数

    Returns:
        处理结果字典
    """
    audio_id = Path(audio_path).stem
    logger.info(f"处理: {audio_id}")

    # 提取嵌入
    embedding = extract_embedding(model, str(audio_path), device)

    # 保存嵌入
    if embedding_dir:
        embedding_dir.mkdir(parents=True, exist_ok=True)
        np.save(embedding_dir / f"{audio_id}_embedding.npy", embedding)

    # Zero-shot 分类
    result = {
        "audio_id": audio_id,
        "embedding_dim": embedding.shape[0],
        "embedding_norm": float(np.linalg.norm(embedding)),
    }

    if do_zero_shot:
        result["genre"] = zero_shot_classify(model, embedding, GENRE_LABELS, device, top_k)
        result["mood"] = zero_shot_classify(model, embedding, MOOD_LABELS, device, top_k)
        result["instrumentation"] = zero_shot_classify(model, embedding, INSTRUMENT_LABELS, device, top_k)
        result["scene"] = zero_shot_classify(model, embedding, SCENE_LABELS, device, top_k)

        # 保存语义候选
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"{audio_id}_semantic.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"  ✅ {audio_id}: top genre={result.get('genre', [{}])[0].get('label', 'N/A')} "
                f"score={result.get('genre', [{}])[0].get('score', 0):.3f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="L2 语义候选层：CLAP 嵌入提取 + zero-shot 分类",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="输入音频目录")
    parser.add_argument("--output", type=str, default=None,
                        help="语义候选输出目录（默认: input-dir/../l2_semantic）")
    parser.add_argument("--embedding-output", type=str, default=None,
                        help="嵌入向量输出目录（默认: input-dir/../l2_embedding）")
    parser.add_argument("--model-path", type=str, default=None,
                        help="CLAP 模型权重路径（可选，默认从 HuggingFace 下载）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备（cuda/cpu）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="zero-shot 分类返回 top-k 结果")
    parser.add_argument("--no-zero-shot", action="store_true",
                        help="只提取嵌入，不做 zero-shot 分类")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output) if args.output else input_dir.parent / "l2_semantic"
    embedding_dir = Path(args.embedding_output) if args.embedding_output else input_dir.parent / "l2_embedding"

    # 查找音频文件
    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_dir.glob(f"*{ext}"))
    audio_files = sorted(audio_files)

    if args.limit:
        audio_files = audio_files[:args.limit]

    logger.info(f"输入目录: {input_dir}")
    logger.info(f"音频文件数: {len(audio_files)}")
    logger.info(f"语义候选输出: {output_dir}")
    logger.info(f"嵌入输出: {embedding_dir}")
    logger.info(f"Zero-shot: {'禁用' if args.no_zero_shot else '启用'}")

    if not audio_files:
        logger.error("未找到音频文件")
        return

    # 加载模型
    model = load_clap_model(args.model_path, args.device)

    # 处理每个音频
    results = []
    for idx, audio_path in enumerate(audio_files):
        try:
            result = process_audio(
                model, audio_path, output_dir, embedding_dir,
                args.device, not args.no_zero_shot, args.top_k
            )
            results.append(result)
        except Exception as e:
            logger.error(f"  ❌ {audio_path.name}: {e}")

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("L2 处理完成")
    logger.info("=" * 60)
    logger.info(f"  总数: {len(audio_files)}")
    logger.info(f"  成功: {len(results)}")
    logger.info(f"  失败: {len(audio_files) - len(results)}")
    logger.info(f"  语义候选目录: {output_dir}")
    logger.info(f"  嵌入目录: {embedding_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
