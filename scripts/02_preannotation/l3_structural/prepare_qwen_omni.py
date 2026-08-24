#!/usr/bin/env python3
"""
Qwen-Omni 音频预处理脚本

将 FLAC 母版预处理为适合 Qwen-Omni API 上传的格式:
- FLAC < 8MB → 直接使用
- FLAC ≥ 8MB → 转 MP3 320kbps
- MP3 仍 > 10MB → 取中间代表性片段(前3分钟)
- 临时文件用完即删，不保留

Qwen-Omni Base64 限制: < 10MB
Qwen3.5-Omni-Flash 音频上限: 20分钟

使用:
    from prepare_qwen_omni import prepare_for_qwen_omni
    output_path = prepare_for_qwen_omni("path/to/audio.flac")
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def get_audio_duration(file_path: str) -> float:
    """获取音频时长(秒)"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_file_size_mb(file_path: str) -> float:
    """获取文件大小(MB)"""
    return os.path.getsize(file_path) / (1024 * 1024)


def convert_to_mp3(input_path: str, output_path: str, bitrate: str = "320k",
                    sample_rate: int = 44100, channels: int = 2) -> bool:
    """转 MP3 320kbps"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-b:a", bitrate, "-ac", str(channels), "-ar", str(sample_rate),
             "-loglevel", "error", output_path],
            check=True, timeout=120
        )
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        print(f"  转MP3失败: {e}")
        return False


def extract_segment(input_path: str, output_path: str,
                    start_sec: float = 0, duration_sec: float = 180,
                    bitrate: str = "320k") -> bool:
    """提取音频片段(默认前3分钟)"""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ss", str(start_sec), "-t", str(duration_sec),
             "-b:a", bitrate, "-ac", "2", "-ar", "44100",
             "-loglevel", "error", output_path],
            check=True, timeout=60
        )
        return os.path.exists(output_path)
    except subprocess.CalledProcessError as e:
        print(f"  提取片段失败: {e}")
        return False


def prepare_for_qwen_omni(
    input_path: str,
    max_flac_mb: float = 8.0,
    max_final_mb: float = 10.0,
    segment_duration_sec: float = 180,
    output_dir: Optional[str] = None,
    cleanup: bool = True,
) -> Tuple[str, dict]:
    """
    预处理音频为 Qwen-Omni 可上传格式。

    Args:
        input_path: 输入音频路径(FLAC/MP3/WAV等)
        max_flac_mb: FLAC直接上传的大小阈值(默认8MB)
        max_final_mb: 最终文件大小上限(默认10MB，Base64限制)
        segment_duration_sec: 超限后提取片段时长(默认180秒=3分钟)
        output_dir: 输出目录(默认临时目录)
        cleanup: 是否清理临时文件

    Returns:
        (output_path, metadata) — 预处理后的音频路径和元数据
    """
    input_path = str(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    original_size = get_file_size_mb(input_path)
    original_duration = get_audio_duration(input_path)
    metadata = {
        "original_path": input_path,
        "original_size_mb": round(original_size, 2),
        "original_duration_sec": round(original_duration, 1),
        "processing_steps": [],
        "final_format": None,
        "is_segment": False,
    }

    # Step 1: 检查是否可以直接上传
    ext = Path(input_path).suffix.lower()
    if ext in [".flac", ".wav"] and original_size < max_flac_mb:
        metadata["processing_steps"].append("direct_upload(FLAC<8MB)")
        metadata["final_format"] = ext
        metadata["final_size_mb"] = round(original_size, 2)
        print(f"  直接上传: {original_size:.1f}MB {ext}")
        return input_path, metadata

    # Step 2: 转 MP3 320kbps
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        mp3_path = os.path.join(output_dir, f"{Path(input_path).stem}_qwen.mp3")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="qwen_omni_")
        mp3_path = os.path.join(tmp_dir, f"{Path(input_path).stem}_qwen.mp3")

    print(f"  转MP3 320kbps...")
    if not convert_to_mp3(input_path, mp3_path):
        raise RuntimeError("MP3转换失败")

    mp3_size = get_file_size_mb(mp3_path)
    metadata["processing_steps"].append(f"convert_mp3_320k({mp3_size:.1f}MB)")

    # Step 3: 检查MP3大小
    if mp3_size <= max_final_mb:
        metadata["final_format"] = "mp3"
        metadata["final_size_mb"] = round(mp3_size, 2)
        print(f"  MP3可用: {mp3_size:.1f}MB")
        return mp3_path, metadata

    # Step 4: MP3仍太大，提取片段
    print(f"  MP3仍超限({mp3_size:.1f}MB>10MB)，提取前{segment_duration_sec}秒...")
    segment_path = mp3_path.replace("_qwen.mp3", "_segment.mp3")
    if not extract_segment(input_path, segment_path,
                           start_sec=0, duration_sec=segment_duration_sec):
        raise RuntimeError("片段提取失败")

    segment_size = get_file_size_mb(segment_path)
    metadata["processing_steps"].append(f"extract_segment_{segment_duration_sec}s({segment_size:.1f}MB)")
    metadata["final_format"] = "mp3_segment"
    metadata["final_size_mb"] = round(segment_size, 2)
    metadata["is_segment"] = True
    metadata["segment_duration_sec"] = segment_duration_sec

    # 清理中间MP3
    if cleanup and os.path.exists(mp3_path) and mp3_path != segment_path:
        os.remove(mp3_path)

    print(f"  片段可用: {segment_size:.1f}MB (前{segment_duration_sec}秒)")
    return segment_path, metadata


def batch_prepare(
    input_paths: list,
    output_dir: str,
    **kwargs,
) -> list:
    """
    批量预处理音频为 Qwen-Omni 格式。

    Args:
        input_paths: 输入音频路径列表
        output_dir: 输出目录
        **kwargs: 传给 prepare_for_qwen_omni 的额外参数

    Returns:
        [(output_path, metadata), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, input_path in enumerate(input_paths):
        print(f"[{i+1}/{len(input_paths)}] {Path(input_path).name}")
        try:
            output_path, metadata = prepare_for_qwen_omni(
                input_path, output_dir=output_dir, **kwargs
            )
            results.append((output_path, metadata))
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            results.append((None, {"error": str(e), "original_path": input_path}))

    # 统计
    success = sum(1 for r in results if r[0] is not None)
    segments = sum(1 for r in results if r[1].get("is_segment"))
    print(f"\n批量预处理完成: {success}/{len(input_paths)} 成功, {segments} 个提取了片段")

    return results


# ========== 使用示例 ==========

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python prepare_qwen_omni.py <audio_path> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    output_path, metadata = prepare_for_qwen_omni(input_path, output_dir=output_dir)

    print(f"\n{'='*60}")
    print(f"预处理完成")
    print(f"{'='*60}")
    print(f"  原始: {metadata['original_size_mb']}MB, {metadata['original_duration_sec']}s")
    print(f"  处理步骤: {' → '.join(metadata['processing_steps'])}")
    print(f"  最终: {metadata['final_format']}, {metadata['final_size_mb']}MB")
    print(f"  输出: {output_path}")
    if metadata.get("is_segment"):
        print(f"  ⚠️ 注意: 这是片段(前{metadata['segment_duration_sec']}秒)，非完整音频")
