#!/usr/bin/env python3
"""
rsync_stems_to_mac.py
Demucs stems 批量回传脚本（Mac 端运行，从 GPU 拉取）

架构原则：
- stems 是"不可再生资产"，分离一次10-30s，必须永久保存
- 从 GPU 拉取到 Mac 本地永久存档，不删除
- 至少存 vocals + other（other 含钢琴/吉他/合成器，对乐器标注和多轨生成最有价值）
- drums + bass 按需保存（省空间）

用法（在 Mac 上运行）：
    # 全量4轨回传
    python3 rsync_stems_to_mac.py --gpu-host root@connect.westb.seetacloud.com --gpu-port 43107 --gpu-dir /root/autodl-tmp/demucs_stems --local-dir /Users/m.jian/music_corpus_project/data/01_preprocess/demucs_stems

    # 只回传 vocals + other（省一半空间）
    python3 rsync_stems_to_mac.py --stems vocals,other ...

    # 增量回传（只拉取本地没有的 track_id）
    python3 rsync_stems_to_mac.py --incremental ...

    # 预览模式（不实际传输）
    python3 rsync_stems_to_mac.py --dry-run ...

目录结构：
    GPU: /root/autodl-tmp/demucs_stems/{track_id}/vocals.wav, drums.wav, bass.wav, other.wav
    Mac: data/01_preprocess/demucs_stems/{track_id}/vocals.wav, drums.wav, bass.wav, other.wav
"""
import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional

# ===================== 配置 =====================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(LOG_DIR, f"rsync_stems_{time_str}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 支持的 stems
ALL_STEMS = ["vocals", "drums", "bass", "other"]


def get_remote_track_ids(gpu_host: str, gpu_port: int, gpu_dir: str) -> Set[str]:
    """获取 GPU 上的 track_id 列表"""
    cmd = [
        "ssh", "-p", str(gpu_port), gpu_host,
        f"ls -1 {gpu_dir} 2>/dev/null | head -10000"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            track_ids = set(line.strip() for line in result.stdout.strip().split("\n") if line.strip())
            logger.info(f"GPU 上有 {len(track_ids)} 个 track_id")
            return track_ids
        else:
            logger.warning(f"获取 GPU track_id 失败: {result.stderr[-200:]}")
            return set()
    except Exception as e:
        logger.error(f"SSH 连接失败: {e}")
        return set()


def get_local_track_ids(local_dir: str) -> Set[str]:
    """获取本地已有的 track_id 列表"""
    local_path = Path(local_dir)
    if not local_path.exists():
        return set()
    track_ids = set(d.name for d in local_path.iterdir() if d.is_dir())
    logger.info(f"本地已有 {len(track_ids)} 个 track_id")
    return track_ids


def rsync_single_track(
    gpu_host: str,
    gpu_port: int,
    gpu_dir: str,
    local_dir: str,
    track_id: str,
    stems_to_get: List[str],
    dry_run: bool = False,
) -> bool:
    """
    回传单个 track_id 的 stems

    Args:
        gpu_host: GPU 主机地址
        gpu_port: SSH 端口
        gpu_dir: GPU 上的 stems 根目录
        local_dir: 本地 stems 根目录
        track_id: 音频ID
        stems_to_get: 要回传的 stems 列表
        dry_run: 预览模式

    Returns:
        bool: 是否成功
    """
    local_track_dir = Path(local_dir) / track_id
    local_track_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for stem in stems_to_get:
        remote_path = f"{gpu_host}:{gpu_dir}/{track_id}/{stem}.wav"
        local_path = local_track_dir / f"{stem}.wav"

        # 检查本地是否已存在（增量）
        if local_path.exists() and local_path.stat().st_size > 0:
            logger.debug(f"  {stem}.wav 已存在，跳过")
            continue

        if dry_run:
            logger.info(f"  [DRY-RUN] 将回传 {stem}.wav")
            continue

        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", f"ssh -p {gpu_port} -o StrictHostKeyChecking=no",
            remote_path,
            str(local_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and local_path.exists():
                size_mb = local_path.stat().st_size / 1024 / 1024
                logger.info(f"  ✅ {stem}.wav ({size_mb:.1f}MB)")
            else:
                logger.warning(f"  ⚠️ {stem}.wav 回传可能失败: {result.stderr[-100:]}")
                success = False
        except subprocess.TimeoutExpired:
            logger.error(f"  ❌ {stem}.wav 回传超时")
            success = False
        except Exception as e:
            logger.error(f"  ❌ {stem}.wav 回传异常: {e}")
            success = False

    return success


def verify_local_stems(local_dir: str, track_ids: Set[str], stems_to_check: List[str]) -> dict:
    """验证本地 stems 完整性"""
    stats = {
        "total_tracks": len(track_ids),
        "complete_tracks": 0,
        "partial_tracks": 0,
        "missing_tracks": 0,
        "total_size_mb": 0,
        "missing_files": [],
    }

    local_path = Path(local_dir)
    for track_id in track_ids:
        track_dir = local_path / track_id
        if not track_dir.exists():
            stats["missing_tracks"] += 1
            continue

        has_all = True
        has_any = False
        for stem in stems_to_check:
            stem_file = track_dir / f"{stem}.wav"
            if stem_file.exists() and stem_file.stat().st_size > 0:
                has_any = True
                stats["total_size_mb"] += stem_file.stat().st_size / 1024 / 1024
            else:
                has_all = False
                stats["missing_files"].append(f"{track_id}/{stem}.wav")

        if has_all:
            stats["complete_tracks"] += 1
        elif has_any:
            stats["partial_tracks"] += 1
        else:
            stats["missing_tracks"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Demucs stems 批量回传脚本（Mac 端运行，从 GPU 拉取）")
    # GPU 连接
    parser.add_argument("--gpu-host", type=str, default="root@connect.westb.seetacloud.com", help="GPU 主机地址")
    parser.add_argument("--gpu-port", type=int, default=43107, help="SSH 端口")
    parser.add_argument("--gpu-dir", type=str, default="/root/autodl-tmp/demucs_stems", help="GPU 上的 stems 根目录")
    # 本地
    parser.add_argument("--local-dir", type=str, default="data/01_preprocess/demucs_stems", help="本地 stems 根目录")
    # stems 选择
    parser.add_argument("--stems", type=str, default="all", help="要回传的 stems（all/vocals,other/vocals,drums,bass,other），默认 all")
    # 模式
    parser.add_argument("--incremental", action="store_true", help="增量回传（只拉取本地没有的 track_id）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际传输")
    parser.add_argument("--limit", type=int, default=None, help="只回传前 N 个 track_id（用于测试）")
    # 验证
    parser.add_argument("--verify-only", action="store_true", help="只验证本地完整性，不回传")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Demucs stems 批量回传启动")
    logger.info(f"  GPU: {args.gpu_host}:{args.gpu_port}")
    logger.info(f"  GPU 目录: {args.gpu_dir}")
    logger.info(f"  本地目录: {args.local_dir}")
    logger.info(f"  回传 stems: {args.stems}")
    logger.info(f"  增量模式: {args.incremental}")
    logger.info(f"  预览模式: {args.dry_run}")
    logger.info("=" * 60)

    # 解析要回传的 stems
    if args.stems == "all":
        stems_to_get = ALL_STEMS
    else:
        stems_to_get = [s.strip() for s in args.stems.split(",") if s.strip() in ALL_STEMS]
        if not stems_to_get:
            logger.error(f"无效的 stems: {args.stems}，支持: {', '.join(ALL_STEMS)}")
            return
    logger.info(f"将回传 stems: {', '.join(stems_to_get)}")

    # 创建本地目录
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    # 只验证模式
    if args.verify_only:
        logger.info("")
        logger.info("=== 只验证本地完整性 ===")
        local_track_ids = get_local_track_ids(args.local_dir)
        stats = verify_local_stems(args.local_dir, local_track_ids, stems_to_get)
        logger.info(f"总 track_id: {stats['total_tracks']}")
        logger.info(f"完整: {stats['complete_tracks']}")
        logger.info(f"部分: {stats['partial_tracks']}")
        logger.info(f"缺失: {stats['missing_tracks']}")
        logger.info(f"总大小: {stats['total_size_mb']:.1f}MB")
        if stats['missing_files']:
            logger.info(f"缺失文件数: {len(stats['missing_files'])}")
            for f in stats['missing_files'][:10]:
                logger.info(f"  - {f}")
        return

    # 获取 GPU 上的 track_id
    logger.info("")
    logger.info("获取 GPU 上的 track_id 列表...")
    remote_track_ids = get_remote_track_ids(args.gpu_host, args.gpu_port, args.gpu_dir)
    if not remote_track_ids:
        logger.error("GPU 上没有 track_id，或连接失败")
        return

    # 增量模式：过滤掉本地已有的
    track_ids_to_get = remote_track_ids
    if args.incremental:
        local_track_ids = get_local_track_ids(args.local_dir)
        track_ids_to_get = remote_track_ids - local_track_ids
        logger.info(f"增量模式：跳过 {len(local_track_ids)} 个本地已有的 track_id")
        logger.info(f"待回传: {len(track_ids_to_get)} 个 track_id")

    # 限制数量
    if args.limit:
        track_ids_to_get = set(list(track_ids_to_get)[:args.limit])
        logger.info(f"限制回传前 {args.limit} 个 track_id")

    if not track_ids_to_get:
        logger.info("没有需要回传的 track_id")
        return

    # 批量回传
    logger.info("")
    logger.info(f"开始回传 {len(track_ids_to_get)} 个 track_id...")
    start_time = datetime.now()

    success_count = 0
    failed_count = 0

    for i, track_id in enumerate(sorted(track_ids_to_get)):
        logger.info(f"[{i+1}/{len(track_ids_to_get)}] {track_id}")
        success = rsync_single_track(
            gpu_host=args.gpu_host,
            gpu_port=args.gpu_port,
            gpu_dir=args.gpu_dir,
            local_dir=args.local_dir,
            track_id=track_id,
            stems_to_get=stems_to_get,
            dry_run=args.dry_run,
        )
        if success:
            success_count += 1
        else:
            failed_count += 1

    elapsed = (datetime.now() - start_time).total_seconds()

    # 汇总
    logger.info("")
    logger.info("=" * 60)
    logger.info("回传完成")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {failed_count}")
    logger.info(f"  总计: {len(track_ids_to_get)}")
    logger.info(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    logger.info(f"  本地目录: {args.local_dir}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    # 验证本地完整性
    if not args.dry_run:
        logger.info("")
        logger.info("验证本地完整性...")
        local_track_ids = get_local_track_ids(args.local_dir)
        stats = verify_local_stems(args.local_dir, local_track_ids, stems_to_get)
        logger.info(f"本地总 track_id: {stats['total_tracks']}")
        logger.info(f"完整: {stats['complete_tracks']}")
        logger.info(f"部分: {stats['partial_tracks']}")
        logger.info(f"缺失: {stats['missing_tracks']}")
        logger.info(f"总大小: {stats['total_size_mb']:.1f}MB ({stats['total_size_mb']/1024:.1f}GB)")


if __name__ == "__main__":
    main()
