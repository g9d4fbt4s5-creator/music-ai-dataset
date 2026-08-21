"""
mac_batch_orchestrator.py
Mac端批次编排脚本 — 自动化 Mac↔GPU 批次处理流水线

功能：
1. 读取待处理清单（prefilter_passed.csv 或 audio_manifest.csv）
2. 按 batch_size 分组（默认100首/批）
3. 准备批次目录（复制音频到 data/gpu_batches/batch_XXX/）
4. rsync 上传到 GPU
5. SSH 触发 GPU 处理（tmux 后台运行）
6. 轮询等待完成（检查 tmux session 是否存在）
7. rsync 回传产物到 Mac
8. Mac 本地合并（元数据/segments/features）
9. 循环直到所有批次完成

用法：
    # 完整流水线（从清单到合并）
    python mac_batch_orchestrator.py --input prefilter_passed.csv --batch-size 100

    # 只准备批次（不上传）
    python mac_batch_orchestrator.py --input prefilter_passed.csv --prepare-only

    # 只上传指定批次
    python mac_batch_orchestrator.py --upload-only --batch-id 0

    # 只回传指定批次
    python mac_batch_orchestrator.py --download-only --batch-id 0

    # 从指定批次开始（断点续跑）
    python mac_batch_orchestrator.py --input prefilter_passed.csv --start-from 3

    # 预览模式（不实际执行）
    python mac_batch_orchestrator.py --input prefilter_passed.csv --dry-run
"""
import os
import sys
import json
import time
import argparse
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"mac_batch_orchestrator_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU 连接配置
GPU_HOST = "root@connect.westb.seetacloud.com"
GPU_PORT = "43107"
REMOTE_TMP = "/root/autodl-tmp"
REMOTE_PROJECT = "/root/music_corpus_project"

# 批次状态文件（幂等恢复机制）
BATCH_STATE_DIR = PROJECT_ROOT / "data" / "gpu_batches" / ".state"
BATCH_STATE_FILE = BATCH_STATE_DIR / "batch_states.json"

# 批次状态定义
BATCH_STATUSES = [
    "pending",       # 待处理
    "prepared",      # 已准备（音频已复制到批次目录）
    "uploaded",      # 已上传到GPU
    "processing",    # GPU处理中
    "completed",     # GPU处理完成
    "fetched",       # 产物已回传Mac
    "verified",      # 回传完整性已校验
    "merged",        # 已合并到全局
    "cleaned",       # GPU已清理
    "failed",        # 失败
]

# 本地目录
BATCHES_DIR = PROJECT_ROOT / "data" / "gpu_batches"
GPU_OUTPUT_DIR = PROJECT_ROOT / "data" / "01_preprocess" / "gpu_batches"
MANIFEST_PATH = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"

# 默认参数
DEFAULT_BATCH_SIZE = 100
POLL_INTERVAL = 60  # 轮询间隔（秒）


# ===================== 工具函数 =====================
def run_command(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """运行 shell 命令"""
    logger.debug(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return result


def ssh_run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """通过 SSH 在 GPU 上运行命令"""
    ssh_cmd = ["ssh", "-p", GPU_PORT, GPU_HOST, cmd]
    return run_command(ssh_cmd, check=check)


# ===================== 批次状态管理（幂等恢复） =====================
def load_batch_state() -> Dict:
    """加载批次状态文件"""
    if BATCH_STATE_FILE.exists():
        with open(BATCH_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_batch_state(state: Dict) -> None:
    """保存批次状态文件"""
    BATCH_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def update_batch_status(batch_id: int, status: str, **kwargs) -> None:
    """
    更新批次状态

    Args:
        batch_id: 批次ID
        status: 状态（必须在 BATCH_STATUSES 中）
        **kwargs: 额外信息（如 gpu_pid, verified, error 等）
    """
    if status not in BATCH_STATUSES:
        logger.warning(f"未知状态: {status}（允许但未在预定义列表中）")

    state = load_batch_state()
    batch_key = f"batch_{batch_id:03d}"

    if batch_key not in state:
        state[batch_key] = {}

    state[batch_key]["status"] = status
    state[batch_key]["updated_at"] = datetime.now().isoformat()
    for k, v in kwargs.items():
        state[batch_key][k] = v

    save_batch_state(state)
    logger.info(f"  📝 状态更新: {batch_key} → {status}")


def get_batch_status(batch_id: int) -> Optional[Dict]:
    """获取批次状态"""
    state = load_batch_state()
    return state.get(f"batch_{batch_id:03d}")


def rsync_upload(local_dir: Path, remote_dir: str) -> None:
    """rsync 上传到 GPU"""
    logger.info(f"上传: {local_dir} → {GPU_HOST}:{remote_dir}")
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -p {GPU_PORT}",
        f"{local_dir}/",
        f"{GPU_HOST}:{remote_dir}/"
    ]
    run_command(cmd)
    logger.info("上传完成")


def rsync_download(remote_dir: str, local_dir: Path) -> None:
    """rsync 从 GPU 回传"""
    logger.info(f"回传: {GPU_HOST}:{remote_dir} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -p {GPU_PORT}",
        f"{GPU_HOST}:{remote_dir}/",
        f"{local_dir}/"
    ]
    run_command(cmd)
    logger.info("回传完成")


def get_audio_path(audio_id: str, manifest_df: pd.DataFrame) -> Optional[Path]:
    """从 manifest 获取音频绝对路径"""
    row = manifest_df[manifest_df["audio_id"] == audio_id]
    if row.empty:
        logger.warning(f"未找到 audio_id={audio_id}")
        return None

    file_relative_path = row.iloc[0].get("file_relative_path", "")
    if pd.isna(file_relative_path) or not file_relative_path:
        logger.warning(f"audio_id={audio_id} 无 file_relative_path")
        return None

    abs_path = PROJECT_ROOT / "data" / "00_raw_collect" / file_relative_path
    if not abs_path.exists():
        logger.warning(f"音频文件不存在: {abs_path}")
        return None

    return abs_path


# ===================== 批次准备 =====================
def prepare_batch(batch_id: int, track_ids: List[str], manifest_df: pd.DataFrame, dry_run: bool = False) -> Path:
    """准备批次目录（复制音频）"""
    batch_dir = BATCHES_DIR / f"batch_{batch_id:03d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"准备批次 {batch_id:03d}: {len(track_ids)} 首音频")

    manifest_rows = []
    copied = 0
    skipped = 0

    for audio_id in track_ids:
        src_path = get_audio_path(audio_id, manifest_df)
        if src_path is None:
            skipped += 1
            continue

        dst_path = batch_dir / src_path.name
        if not dst_path.exists():
            if dry_run:
                logger.debug(f"[预览] 将复制: {src_path.name}")
            else:
                shutil.copy2(src_path, dst_path)
        copied += 1

        # 记录到批次 manifest
        row = manifest_df[manifest_df["audio_id"] == audio_id].iloc[0].to_dict()
        manifest_rows.append(row)

    # 保存批次 manifest
    batch_manifest = pd.DataFrame(manifest_rows)
    batch_manifest.to_csv(batch_dir / "batch_manifest.csv", index=False)

    logger.info(f"批次 {batch_id:03d} 准备完成: 复制 {copied} 首, 跳过 {skipped} 首")
    return batch_dir


# ===================== GPU 处理触发 =====================
def trigger_gpu_processing(batch_id: int, dry_run: bool = False) -> str:
    """
    触发 GPU 处理（tmux 后台）

    加固：GPU 端处理完成后写原子完成标记 .done 文件，
    Mac 端轮询 .done 文件而不是 tmux session（tmux 可能因 SSH 断开/重启而消失）。
    """
    session_name = f"batch_{batch_id:03d}"
    remote_in = f"{REMOTE_TMP}/batch_{batch_id:03d}_in"
    remote_out = f"{REMOTE_TMP}/batch_{batch_id:03d}_out"
    done_file = f"{remote_out}/.done"

    # GPU 端处理命令（完成后写 .done 原子标记）
    gpu_cmd = (
        f"mkdir -p {remote_out} && "
        f"rm -f {done_file} && "
        f"source /opt/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate labelstudio-env && "
        f"cd {REMOTE_PROJECT} && "
        f"python scripts/00.5_cleaning/clean_pipeline.py "
        f"--input {remote_in}/ "
        f"--output {remote_out}/ "
        f"--stages 2,3,5,6 && "
        f"echo '{{\"status\":\"completed\",\"batch_id\":\"{batch_id:03d}\","
        f"\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}}' > {done_file} && "
        f"echo 'BATCH_{batch_id:03d}_DONE' >> {remote_out}/processing_status.log"
    )

    # tmux 命令
    tmux_cmd = (
        f"tmux new-session -d -s {session_name} "
        f"\"bash -c '{gpu_cmd} 2>&1 | tee {remote_out}/processing.log'\""
    )

    if dry_run:
        logger.info(f"[预览] 将在 GPU 触发处理: session={session_name}")
        return session_name

    logger.info(f"触发 GPU 处理: session={session_name}")
    ssh_run(tmux_cmd)
    logger.info("GPU 处理已启动（tmux 后台，完成后写 .done 标记）")

    return session_name


def wait_for_gpu_completion(batch_id: int, session_name: str, poll_interval: int = POLL_INTERVAL) -> bool:
    """
    轮询等待 GPU 处理完成

    加固：检查 .done 原子完成标记，而不是 tmux session。
    tmux session 可能因 SSH 断开、GPU 重启、手动误操作而消失，但处理可能只跑了一半。
    """
    remote_out = f"{REMOTE_TMP}/batch_{batch_id:03d}_out"
    done_file = f"{remote_out}/.done"

    logger.info(f"等待 GPU 处理完成: batch_{batch_id:03d}（轮询 .done 标记，每 {poll_interval}s）")

    start_time = time.time()
    while True:
        # 检查 .done 原子完成标记（比 tmux session 更可靠）
        result = ssh_run(f"test -f {done_file} && echo 'DONE' || echo 'PENDING'", check=False)
        is_done = result.stdout.strip() == "DONE"

        if is_done:
            elapsed = time.time() - start_time
            logger.info(f"GPU 处理完成: batch_{batch_id:03d}, 耗时 {elapsed/60:.1f} 分钟（.done 标记确认）")
            return True

        # 辅助检查：tmux session 是否还在（如果不在但 .done 也不在，可能异常退出）
        tmux_result = ssh_run(f"tmux has-session -t {session_name} 2>/dev/null; echo $?", check=False)
        session_exists = tmux_result.stdout.strip() == "0"

        elapsed = time.time() - start_time
        if not session_exists and not is_done:
            logger.warning(f"  ⚠️  tmux session 已消失但 .done 标记不存在，可能异常退出！已耗时 {elapsed/60:.1f} 分钟")
            logger.warning(f"     检查 GPU 日志: {remote_out}/processing.log")
            # 继续等待一段时间，可能是 tmux 崩溃但进程还在跑
            if elapsed > 3600:  # 超过1小时还没完成，报错
                logger.error(f"  ❌ 超时：1小时未完成，可能需要手动检查")
                return False
        else:
            logger.info(f"  处理中... 已耗时 {elapsed/60:.1f} 分钟")

        time.sleep(poll_interval)


def wait_for_preannotation_completion(batch_id: int, poll_interval: int = POLL_INTERVAL, timeout: int = 3600) -> bool:
    """
    轮询等待 GPU 预标注完成（.preannotation_done 标记）

    GPU 清理时机延后：预标注完成后才清理，因为 MOSS 等预标注模型需要 stems。
    如果预标注不需要 stems（如 CLAP），可以跳过此等待，直接清理。

    Args:
        batch_id: 批次ID
        poll_interval: 轮询间隔（秒）
        timeout: 超时时间（秒），默认1小时

    Returns:
        是否完成
    """
    remote_out = f"{REMOTE_TMP}/batch_{batch_id:03d}_out"
    preannotation_done_file = f"{remote_out}/.preannotation_done"

    logger.info(f"等待 GPU 预标注完成: batch_{batch_id:03d}（轮询 .preannotation_done 标记，每 {poll_interval}s，超时 {timeout}s）")

    start_time = time.time()
    while True:
        # 检查 .preannotation_done 原子完成标记
        result = ssh_run(f"test -f {preannotation_done_file} && echo 'DONE' || echo 'PENDING'", check=False)
        is_done = result.stdout.strip() == "DONE"

        if is_done:
            elapsed = time.time() - start_time
            logger.info(f"GPU 预标注完成: batch_{batch_id:03d}, 耗时 {elapsed/60:.1f} 分钟（.preannotation_done 标记确认）")
            return True

        elapsed = time.time() - start_time
        if elapsed > timeout:
            logger.error(f"  ❌ 超时：{timeout/3600:.1f}小时未完成预标注，可能需要手动检查")
            return False

        logger.info(f"  预标注中... 已耗时 {elapsed/60:.1f} 分钟")
        time.sleep(poll_interval)


# ===================== 回传与合并 =====================
def download_batch_output(batch_id: int, dry_run: bool = False) -> Path:
    """回传 GPU 产物到 Mac"""
    remote_out = f"{REMOTE_TMP}/batch_{batch_id:03d}_out"
    local_out = GPU_OUTPUT_DIR / f"batch_{batch_id:03d}"

    if dry_run:
        logger.info(f"[预览] 将回传: {remote_out} → {local_out}")
        return local_out

    rsync_download(remote_out, local_out)
    return local_out


def merge_batch_to_global(batch_id: int, local_out: Path) -> None:
    """Mac 本地合并批次产物到全局"""
    logger.info(f"合并批次 {batch_id:03d} 到全局")

    # 1. 合并 segments
    segments_dir = local_out / "segments"
    if segments_dir.exists():
        global_segments = PROJECT_ROOT / "data" / "01_preprocess" / "segments"
        global_segments.mkdir(parents=True, exist_ok=True)
        for seg_file in segments_dir.glob("*"):
            dst = global_segments / seg_file.name
            if not dst.exists():
                shutil.copy2(seg_file, dst)
        logger.info(f"  合并 segments: {len(list(segments_dir.glob('*')))} 个")

    # 2. 合并 features
    features_dir = local_out / "features"
    if features_dir.exists():
        global_features = PROJECT_ROOT / "data" / "01_preprocess" / "features"
        global_features.mkdir(parents=True, exist_ok=True)
        for feat_file in features_dir.glob("*"):
            dst = global_features / feat_file.name
            if not dst.exists():
                shutil.copy2(feat_file, dst)
        logger.info(f"  合并 features: {len(list(features_dir.glob('*')))} 个")

    # 3. 合并元数据
    meta_dir = local_out / "meta"
    if meta_dir.exists():
        global_meta = PROJECT_ROOT / "data" / "00.5_cleaned" / "reports"
        global_meta.mkdir(parents=True, exist_ok=True)
        for meta_file in meta_dir.glob("*"):
            dst = global_meta / f"batch_{batch_id:03d}_{meta_file.name}"
            if not dst.exists():
                shutil.copy2(meta_file, dst)
        logger.info(f"  合并 meta: {len(list(meta_dir.glob('*')))} 个")

    # 4. 合并 demucs_vocals
    vocals_dir = local_out / "demucs_vocals"
    if vocals_dir.exists():
        global_vocals = PROJECT_ROOT / "data" / "01_preprocess" / "demucs_stems"
        global_vocals.mkdir(parents=True, exist_ok=True)
        for vocal_dir in vocals_dir.iterdir():
            if vocal_dir.is_dir():
                dst = global_vocals / vocal_dir.name
                if not dst.exists():
                    shutil.copytree(vocal_dir, dst)
        logger.info(f"  合并 demucs_vocals: {len(list(vocals_dir.iterdir()))} 个")

    logger.info(f"批次 {batch_id:03d} 合并完成")


def cleanup_gpu_batch(batch_id: int, dry_run: bool = False) -> bool:
    """
    SSH 触发 GPU 清理（方案A：Mac回传完成后安全清理）

    只删音频/产物，不删模型/代码/环境。
    必须在 Mac 回传产物完成后调用，否则会删掉还没回传的文件！

    Args:
        batch_id: 批次ID
        dry_run: 预览模式

    Returns:
        是否清理成功
    """
    logger.info(f"触发 GPU 清理: batch_{batch_id:03d}（只删音频/产物，保留模型/代码/环境）")

    # GPU 上的清理脚本路径
    cleanup_script = f"{REMOTE_PROJECT}/scripts/utils/gpu_cleanup.sh"

    # 清理命令
    cleanup_cmd = f"bash {cleanup_script} batch_{batch_id:03d}"

    if dry_run:
        logger.info(f"[预览] 将在 GPU 执行: {cleanup_cmd}")
        return True

    try:
        result = ssh_run(cleanup_cmd, check=False)
        if result.returncode == 0:
            logger.info(f"GPU 清理完成: batch_{batch_id:03d}")
            return True
        else:
            logger.warning(f"GPU 清理返回非零状态: {result.returncode}")
            logger.warning(f"stderr: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"GPU 清理失败: {e}")
        return False


# ===================== 主流水线 =====================
def run_full_pipeline(
    input_csv: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    start_from: int = 0,
    dry_run: bool = False,
    auto_cleanup: bool = True,
    wait_preannotation: bool = False,
) -> None:
    """
    完整流水线

    Args:
        input_csv: 待处理清单 CSV
        batch_size: 每批音频数量
        start_from: 从指定批次开始（断点续跑）
        dry_run: 预览模式
        auto_cleanup: 是否自动清理GPU
        wait_preannotation: 是否等待预标注完成后再清理（MOSS等需要stems的预标注模型）
            - True: 等待 .preannotation_done 标记，预标注完成后才清理
            - False: 不等待预标注，处理完成后即可清理（CLAP等不需要stems的模型）
    """
    logger.info("=" * 60)
    logger.info("Mac↔GPU 批次编排流水线")
    logger.info(f"自动清理: {'开启' if auto_cleanup else '关闭'}（回传完成后清理GPU音频/产物）")
    logger.info(f"等待预标注: {'开启' if wait_preannotation else '关闭'}（{'预标注完成后才清理' if wait_preannotation else '处理完成后即可清理'}）")
    logger.info("=" * 60)

    # 1. 读取待处理清单
    logger.info(f"读取待处理清单: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"共 {len(df)} 首待处理")

    # 读取全局 manifest（用于获取音频路径）
    manifest_df = pd.read_csv(MANIFEST_PATH)

    # 2. 分批
    track_ids = df["audio_id"].tolist() if "audio_id" in df.columns else df.iloc[:, 0].tolist()
    total_batches = (len(track_ids) + batch_size - 1) // batch_size
    logger.info(f"分批: {len(track_ids)} 首 / {batch_size} 首/批 = {total_batches} 批")

    # 3. 逐批处理
    for batch_idx in range(start_from, total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(track_ids))
        batch_tracks = track_ids[start:end]

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"批次 {batch_idx:03d}/{total_batches - 1:03d}: {len(batch_tracks)} 首")
        logger.info("=" * 60)

        # Step 1: 准备批次
        batch_dir = prepare_batch(batch_idx, batch_tracks, manifest_df, dry_run)
        update_batch_status(batch_idx, "prepared", track_count=len(batch_tracks))

        # Step 2: 上传到 GPU
        remote_in = f"{REMOTE_TMP}/batch_{batch_idx:03d}_in"
        if not dry_run:
            rsync_upload(batch_dir, remote_in)
        update_batch_status(batch_idx, "uploaded")

        # Step 3: 触发 GPU 处理
        session_name = trigger_gpu_processing(batch_idx, dry_run)
        update_batch_status(batch_idx, "processing", session_name=session_name)

        # Step 4: 等待完成（轮询 .done 标记）
        if not dry_run:
            wait_for_gpu_completion(batch_idx, session_name)
        update_batch_status(batch_idx, "completed")

        # Step 4.5: 可选等待预标注完成（MOSS等需要stems的预标注模型）
        # GPU清理时机延后：预标注完成后才清理，因为MOSS等预标注模型需要stems
        # 如果预标注不需要stems（如CLAP），可以跳过此等待，直接清理
        if wait_preannotation and not dry_run:
            preannotation_success = wait_for_preannotation_completion(batch_idx)
            if preannotation_success:
                update_batch_status(batch_idx, "preannotation_completed")
            else:
                logger.warning(f"批次 {batch_idx:03d} 预标注超时，继续后续流程（可能需要手动检查）")
                update_batch_status(batch_idx, "preannotation_timeout")

        # Step 4.6: 分层清理1 - 删除原始音频（母版FLAC已生成，原始音频已备份在Mac+OSS）
        # GPU数据生命周期优化：及时删除大文件，节省磁盘空间
        if not dry_run:
            logger.info(f"分层清理1: 删除原始音频（母版FLAC已生成）")
            cleanup_script = f"{REMOTE_PROJECT}/scripts/utils/gpu_cleanup.sh"
            ssh_run(f"bash {cleanup_script} batch_{batch_idx:03d} --raw-only", check=False)
            update_batch_status(batch_idx, "raw_cleaned")

        # Step 4.7: 分层清理2 - 删除母版FLAC（stems/segments/嵌入已生成）
        if not dry_run:
            logger.info(f"分层清理2: 删除母版FLAC（stems/segments/嵌入已生成）")
            ssh_run(f"bash {cleanup_script} batch_{batch_idx:03d} --master-only", check=False)
            update_batch_status(batch_idx, "master_cleaned")

        # Step 5: 回传产物（此时原始音频和母版FLAC已删除，只回传stems/segments/嵌入/预标注结果）
        local_out = download_batch_output(batch_idx, dry_run)
        update_batch_status(batch_idx, "fetched")

        # Step 5.5: 回传完整性校验（只有校验通过才触发清理）
        if not dry_run:
            gpu_out = f"{REMOTE_TMP}/batch_{batch_idx:03d}_out"
            # 注意：verify_batch_transfer 需要本地访问GPU目录，这里通过SSH生成清单后校验
            # 简化版：检查关键文件是否存在
            logger.info(f"  校验回传完整性: {local_out}")
            key_files = ["meta", "segments", "features"]
            missing = [f for f in key_files if not (local_out / f).exists()]
            if missing:
                logger.warning(f"  ⚠️  回传不完整，缺少: {missing}")
                logger.warning(f"     跳过GPU清理，请手动检查后重新回传")
                update_batch_status(batch_idx, "failed", error=f"missing files: {missing}")
                continue
            else:
                logger.info(f"  ✅ 回传完整性校验通过")
                update_batch_status(batch_idx, "verified")

        # Step 6: 合并到全局
        if not dry_run:
            merge_batch_to_global(batch_idx, local_out)
        update_batch_status(batch_idx, "merged")

        # Step 7: 清理GPU批次（回传完成且校验通过后安全清理，只删音频/产物，保留模型/代码/环境）
        if auto_cleanup:
            if not dry_run:
                cleanup_success = cleanup_gpu_batch(batch_idx, dry_run)
                if cleanup_success:
                    update_batch_status(batch_idx, "cleaned")
                else:
                    logger.warning(f"批次 {batch_idx:03d} GPU清理失败，需手动清理")
                    update_batch_status(batch_idx, "failed", error="gpu cleanup failed")
            else:
                cleanup_gpu_batch(batch_idx, dry_run)

        logger.info(f"批次 {batch_idx:03d} 完成！")

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ 全部 {total_batches} 批处理完成！")
    logger.info("=" * 60)


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser(
        description="Mac↔GPU 批次编排脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, help="待处理清单 CSV（audio_id 列）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="每批音频数量（默认100）")
    parser.add_argument("--start-from", type=int, default=0, help="从指定批次开始（断点续跑）")
    parser.add_argument("--batch-id", type=int, help="指定批次ID（用于 --upload-only/--download-only）")
    parser.add_argument("--prepare-only", action="store_true", help="只准备批次，不上传")
    parser.add_argument("--upload-only", action="store_true", help="只上传指定批次")
    parser.add_argument("--download-only", action="store_true", help="只回传指定批次")
    parser.add_argument("--no-cleanup", action="store_true",
                        help="关闭自动清理（默认回传完成后自动清理GPU音频/产物）")
    parser.add_argument("--wait-preannotation", action="store_true",
                        help="等待预标注完成后再清理（MOSS等需要stems的预标注模型；CLAP等不需要stems的模型不用开）")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际执行")
    args = parser.parse_args()

    if args.prepare_only:
        if not args.input:
            logger.error("--prepare-only 需要 --input 参数")
            sys.exit(1)
        df = pd.read_csv(args.input)
        manifest_df = pd.read_csv(MANIFEST_PATH)
        track_ids = df["audio_id"].tolist() if "audio_id" in df.columns else df.iloc[:, 0].tolist()
        total_batches = (len(track_ids) + args.batch_size - 1) // args.batch_size
        for i in range(total_batches):
            start = i * args.batch_size
            end = min(start + args.batch_size, len(track_ids))
            prepare_batch(i, track_ids[start:end], manifest_df, args.dry_run)
        logger.info("所有批次准备完成")
        return

    if args.upload_only:
        if args.batch_id is None:
            logger.error("--upload-only 需要 --batch-id 参数")
            sys.exit(1)
        batch_dir = BATCHES_DIR / f"batch_{args.batch_id:03d}"
        if not batch_dir.exists():
            logger.error(f"批次目录不存在: {batch_dir}")
            sys.exit(1)
        remote_in = f"{REMOTE_TMP}/batch_{args.batch_id:03d}_in"
        rsync_upload(batch_dir, remote_in)
        return

    if args.download_only:
        if args.batch_id is None:
            logger.error("--download-only 需要 --batch-id 参数")
            sys.exit(1)
        local_out = download_batch_output(args.batch_id, args.dry_run)
        if not args.dry_run:
            merge_batch_to_global(args.batch_id, local_out)
        return

    # 默认：完整流水线
    if not args.input:
        logger.error("完整流水线需要 --input 参数")
        parser.print_help()
        sys.exit(1)

    run_full_pipeline(
        input_csv=Path(args.input),
        batch_size=args.batch_size,
        start_from=args.start_from,
        dry_run=args.dry_run,
        auto_cleanup=not args.no_cleanup,
        wait_preannotation=args.wait_preannotation,
    )


if __name__ == "__main__":
    main()
