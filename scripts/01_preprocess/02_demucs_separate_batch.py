#!/usr/bin/env python3
"""
demucs_separate_batch.py
Demucs 批量分轨脚本（GPU 端运行）

架构原则：
- stems 是"不可再生资产"，分离一次 10-30s，必须永久保存
- stems 不删除，分离完成后回传 Mac 永久存档
- 至少保存 vocals + other（other 包含钢琴/吉他/合成器，对乐器标注和多轨生成最有价值）
- drums + bass 按需保存（省空间）
- Mac 有 600GB+ 可用，500首 stems 约 100GB，完全存得下

输出目录结构：
    output_dir/
    └── {track_id}/
        ├── vocals.wav
        ├── drums.wav
        ├── bass.wav
        └── other.wav

用法：
    # 全量4轨分离
    python3 demucs_separate_batch.py --input-dir /root/autodl-tmp/jazz_500_audio-low --output-dir /root/autodl-tmp/demucs_stems

    # 只保存vocals+other（省一半空间）
    python3 demucs_separate_batch.py --input-dir ... --output-dir ... --stems vocals,other

    # 指定文件列表
    python3 demucs_separate_batch.py --file-list list.txt --output-dir ...

    # 分离完成后自动回传Mac
    python3 demucs_separate_batch.py --input-dir ... --output-dir ... --rsync-dest m.jian@mac:/path/to/demucs_stems/

    # 预览模式
    python3 demucs_separate_batch.py --input-dir ... --output-dir ... --dry-run
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np

# ===================== 配置 =====================
# Demucs 模型（htdemucs 是最新版，4轨分离）
DEFAULT_MODEL = "htdemucs"

# 支持的 stems
ALL_STEMS = ["vocals", "drums", "bass", "other"]

# 日志
LOG_DIR = "/root/autodl-tmp/logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"demucs_separate_{time_str}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_demucs():
    """检查 demucs 是否安装"""
    try:
        import demucs
        from demucs.pretrained import get_model
        return True
    except ImportError:
        return False


def load_demucs_model(model_name: str = DEFAULT_MODEL):
    """加载 Demucs 模型"""
    import torch
    from demucs.pretrained import get_model

    logger.info(f"加载 Demucs 模型: {model_name}")
    model = get_model(model_name)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        logger.info("使用 GPU 加速")
    else:
        logger.warning("未检测到 GPU，使用 CPU（速度慢）")
    return model


def separate_audio(
    model,
    audio_path: str,
    output_dir: str,
    track_id: str,
    stems_to_save: List[str] = None,
) -> Dict:
    """
    对单个音频进行 Demucs 分轨

    Args:
        model: Demucs 模型
        audio_path: 输入音频路径
        output_dir: 输出根目录
        track_id: 音频ID（用于创建子目录）
        stems_to_save: 要保存的 stems 列表（None=全部）

    Returns:
        Dict: 分离结果
            - track_id: 音频ID
            - input_path: 输入路径
            - output_dir: 输出目录
            - status: success/failed
            - stems: 保存的 stems 列表
            - duration: 音频时长（秒）
            - error: 错误信息
    """
    import torch
    import librosa
    from demucs.apply import apply_model

    result = {
        "track_id": track_id,
        "input_path": audio_path,
        "output_dir": os.path.join(output_dir, track_id),
        "status": "pending",
        "stems": [],
        "duration": 0.0,
        "error": "",
    }

    if stems_to_save is None:
        stems_to_save = ALL_STEMS

    try:
        # 加载音频（demucs 需要 44100Hz 立体声）
        y, sr = librosa.load(audio_path, sr=44100, mono=False)
        if y.ndim == 1:
            y = np.vstack([y, y])  # 转立体声
        result["duration"] = len(y[0]) / sr

        # 转为 tensor
        wav = torch.tensor(y).float().unsqueeze(0)
        if torch.cuda.is_available():
            wav = wav.cuda()

        # 分离
        with torch.no_grad():
            sources = apply_model(model, wav, progress=False)[0]

        # sources: [drums, bass, other, vocals]
        stem_names = ["drums", "bass", "other", "vocals"]
        stem_arrays = {name: sources[i].cpu().numpy() for i, name in enumerate(stem_names)}

        # 保存指定的 stems
        track_output_dir = os.path.join(output_dir, track_id)
        os.makedirs(track_output_dir, exist_ok=True)

        saved_stems = []
        import soundfile as sf
        for stem_name in stems_to_save:
            if stem_name in stem_arrays:
                output_path = os.path.join(track_output_dir, f"{stem_name}.wav")
                sf.write(output_path, stem_arrays[stem_name].T, 44100)
                saved_stems.append(stem_name)

        result["status"] = "success"
        result["stems"] = saved_stems
        logger.info(f"  ✅ {track_id}: 分离成功 ({result['duration']:.1f}s), 保存: {', '.join(saved_stems)}")

        return result

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error(f"  ❌ {track_id}: 分离失败: {e}")
        return result


def scan_input_dir(input_dir: str) -> List[str]:
    """扫描输入目录，获取所有音频文件"""
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
    audio_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in audio_extensions:
                audio_files.append(os.path.join(root, f))
    logger.info(f"扫描输入目录: {input_dir}，找到 {len(audio_files)} 个音频文件")
    return sorted(audio_files)


def load_file_list(file_list_path: str) -> List[str]:
    """从文件列表加载音频路径"""
    audio_files = []
    with open(file_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # 支持 CSV 格式（track_id,path）或纯路径
                if "," in line:
                    parts = line.split(",", 1)
                    path = parts[1].strip()
                else:
                    path = line
                if os.path.exists(path):
                    audio_files.append(path)
    logger.info(f"从文件列表加载: {len(audio_files)} 个音频")
    return audio_files


def rsync_to_mac(
    local_dir: str,
    remote_dest: str,
    ssh_port: int = 22,
) -> bool:
    """
    将 stems 回传到 Mac

    Args:
        local_dir: 本地 stems 目录
        remote_dest: 远程目标路径（如 user@mac:/path/to/demucs_stems/）
        ssh_port: SSH 端口

    Returns:
        bool: 是否成功
    """
    logger.info(f"回传 stems 到 Mac: {local_dir} → {remote_dest}")

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e", f"ssh -p {ssh_port}",
        local_dir + "/",
        remote_dest,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logger.info("✅ 回传成功")
            return True
        else:
            logger.error(f"❌ 回传失败: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("❌ 回传超时（>1小时）")
        return False
    except Exception as e:
        logger.error(f"❌ 回传异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Demucs 批量分轨脚本（GPU 端运行，stems 不删除，支持回传 Mac）")
    # 输入方式（三选一）
    parser.add_argument("--input-dir", type=str, help="输入音频目录")
    parser.add_argument("--file-list", type=str, help="音频文件列表路径（每行一个路径或 track_id,path）")
    parser.add_argument("--input-file", type=str, help="单个音频文件路径")
    # 输出
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/demucs_stems", help="输出 stems 根目录")
    # stems 选择
    parser.add_argument("--stems", type=str, default="all", help="要保存的 stems（all/vocals,other/vocals,drums,bass,other），默认 all")
    # 模型
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Demucs 模型（默认 {DEFAULT_MODEL}）")
    # 回传
    parser.add_argument("--rsync-dest", type=str, default=None, help="分离完成后回传 Mac 的目标路径（如 user@mac:/path/to/demucs_stems/）")
    parser.add_argument("--rsync-port", type=int, default=22, help="SSH 端口（默认 22）")
    # 其他
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个音频（用于测试）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际分离")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Demucs 批量分轨启动")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  模型: {args.model}")
    logger.info(f"  保存 stems: {args.stems}")
    logger.info(f"  回传 Mac: {args.rsync_dest or '不回传'}")
    logger.info(f"  预览模式: {args.dry_run}")
    logger.info("=" * 60)

    # 检查 demucs
    if not check_demucs():
        logger.error("demucs 未安装，请运行: pip install demucs")
        return

    # 解析要保存的 stems
    if args.stems == "all":
        stems_to_save = ALL_STEMS
    else:
        stems_to_save = [s.strip() for s in args.stems.split(",") if s.strip() in ALL_STEMS]
        if not stems_to_save:
            logger.error(f"无效的 stems: {args.stems}，支持: {', '.join(ALL_STEMS)}")
            return
    logger.info(f"将保存 stems: {', '.join(stems_to_save)}")

    # 获取音频列表
    if args.input_dir:
        audio_files = scan_input_dir(args.input_dir)
    elif args.file_list:
        audio_files = load_file_list(args.file_list)
    elif args.input_file:
        audio_files = [args.input_file]
    else:
        logger.error("请指定 --input-dir / --file-list / --input-file 之一")
        return

    if not audio_files:
        logger.error("没有找到音频文件")
        return

    # 限制数量
    if args.limit:
        audio_files = audio_files[:args.limit]
        logger.info(f"限制处理前 {args.limit} 个音频")

    logger.info(f"待处理音频: {len(audio_files)} 个")

    # 预览模式
    if args.dry_run:
        logger.info("")
        logger.info("[预览模式] 待处理音频:")
        for i, path in enumerate(audio_files[:10]):
            track_id = os.path.splitext(os.path.basename(path))[0]
            logger.info(f"  [{i+1}] {track_id}: {path}")
        if len(audio_files) > 10:
            logger.info(f"  ... 共 {len(audio_files)} 个")
        logger.info("")
        logger.info("预览完成，未实际分离")
        return

    # 加载模型
    model = load_demucs_model(args.model)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 批量分离
    logger.info("")
    logger.info("开始批量分离...")
    start_time = datetime.now()

    results = []
    success_count = 0
    failed_count = 0

    for i, audio_path in enumerate(audio_files):
        track_id = os.path.splitext(os.path.basename(audio_path))[0]
        logger.info(f"[{i+1}/{len(audio_files)}] 处理: {track_id}")

        # 检查是否已经分离过（不删除，跳过已存在的）
        track_output_dir = os.path.join(args.output_dir, track_id)
        if os.path.exists(track_output_dir):
            existing_stems = [s for s in stems_to_save if os.path.exists(os.path.join(track_output_dir, f"{s}.wav"))]
            if len(existing_stems) == len(stems_to_save):
                logger.info(f"  ⏭️  已存在，跳过: {track_id}")
                results.append({
                    "track_id": track_id,
                    "status": "skipped",
                    "stems": existing_stems,
                })
                success_count += 1
                continue

        result = separate_audio(
            model=model,
            audio_path=audio_path,
            output_dir=args.output_dir,
            track_id=track_id,
            stems_to_save=stems_to_save,
        )
        results.append(result)

        if result["status"] == "success":
            success_count += 1
        else:
            failed_count += 1

    elapsed = (datetime.now() - start_time).total_seconds()

    # 统计
    logger.info("")
    logger.info("=" * 60)
    logger.info("Demucs 批量分轨完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"  跳过(已存在): {sum(1 for r in results if r.get('status') == 'skipped')}")
    logger.info(f"  总计: {len(results)}")
    logger.info(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 保存分离报告
    import pandas as pd
    report_path = os.path.join(args.output_dir, "demucs_separate_report.csv")
    pd.DataFrame(results).to_csv(report_path, index=False, encoding="utf-8")
    logger.info(f"分离报告已保存: {report_path}")

    # 回传 Mac
    if args.rsync_dest and success_count > 0:
        logger.info("")
        rsync_success = rsync_to_mac(
            local_dir=args.output_dir,
            remote_dest=args.rsync_dest,
            ssh_port=args.rsync_port,
        )
        if rsync_success:
            logger.info("✅ stems 已回传 Mac")
        else:
            logger.warning("⚠️ stems 回传失败，请手动运行 rsync 命令:")
            logger.warning(f"  rsync -avz --progress -e 'ssh -p {args.rsync_port}' {args.output_dir}/ {args.rsync_dest}")

    # 输出失败列表
    if failed_count > 0:
        logger.info("")
        logger.info("失败列表:")
        for r in results:
            if r["status"] == "failed":
                logger.info(f"  - {r['track_id']}: {r.get('error', '未知错误')}")


if __name__ == "__main__":
    main()
