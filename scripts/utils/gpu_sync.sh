#!/bin/bash
# ============================================================
# GPU实例数据同步脚本
# 用法：
#   ./scripts/utils/gpu_sync.sh <端口> <本地路径> <远程路径> [--pull|--mp3]
# 示例：
#   # 同步音频数据到GPU（默认rsync）
#   ./scripts/utils/gpu_sync.sh 49530 \
#     data/01_preprocess/processed_master/ \
#     /workspace/music-ai-dataset/data/01_preprocess/processed_master/
#
#   # 同步母版时临时转MP3（减少传输量，约75%）
#   ./scripts/utils/gpu_sync.sh 49530 \
#     data/01_preprocess/processed_master/ \
#     /workspace/music-ai-dataset/data/01_preprocess/processed_master/ \
#     --mp3
#
#   # 从GPU拉回结果
#   ./scripts/utils/gpu_sync.sh 49530 \
#     /workspace/music-ai-dataset/data/02_preannotation/ \
#     data/02_preannotation/ \
#     --pull
# ============================================================

set -euo pipefail

# 脚本所在目录（用于定位 Python 工具脚本）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 配置
GPU_HOST="connect.westb.seetacloud.com"
GPU_USER="root"
SSH_KEY="$HOME/.ssh/id_rsa"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"
MP3_BITRATE="320k"

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <端口> <本地路径> <远程路径> [--pull|--mp3]"
    echo "  默认推送（本地→远程），加 --pull 拉取（远程→本地）"
    echo "  加 --mp3 推送前将FLAC临时转为320kbps MP3（减少传输量）"
    exit 1
fi

PORT="$1"
LOCAL_PATH="$2"
REMOTE_PATH="$3"
DIRECTION="${4:-push}"

# 创建远程目录
echo "=== 创建远程目录 ==="
ssh -p "$PORT" -i "$SSH_KEY" $SSH_OPTS "$GPU_USER@$GPU_HOST" "mkdir -p $(dirname "$REMOTE_PATH")"

if [ "$DIRECTION" = "--pull" ]; then
    echo "=== 拉取数据: 远程→本地 ==="
    echo "  远程: $GPU_USER@$GPU_HOST:$REMOTE_PATH"
    echo "  本地: $LOCAL_PATH"
    rsync -avz --progress -e "ssh -p $PORT -i $SSH_KEY $SSH_OPTS" \
        "$GPU_USER@$GPU_HOST:$REMOTE_PATH" "$LOCAL_PATH"
elif [ "$DIRECTION" = "--mp3" ]; then
    echo "=== MP3优化推送: FLAC→320kbps MP3→远程 ==="
    echo "  本地: $LOCAL_PATH"
    echo "  远程: $GPU_USER@$GPU_HOST:$REMOTE_PATH"
    echo "  码率: ${MP3_BITRATE}"

    # 创建临时MP3目录
    TMP_MP3_DIR=$(mktemp -d /tmp/gpu_sync_mp3_XXXXXX)
    echo "  临时目录: $TMP_MP3_DIR"

    # 统计原始大小
    if [ -d "$LOCAL_PATH" ]; then
        ORIG_SIZE=$(du -sh "$LOCAL_PATH" | cut -f1)
        FILE_COUNT=$(find "$LOCAL_PATH" -name "*.flac" -o -name "*.wav" | wc -l)
        echo "  原始: ${ORIG_SIZE} (${FILE_COUNT} 个音频文件)"
    fi

    # 转换FLAC/WAV为MP3（并行，带超时，防止ffmpeg卡住）
    echo "  正在转换（并行4线程，单文件120秒超时）..."
    python "$SCRIPT_DIR/convert_to_mp3_parallel.py" \
        --input-dir "$LOCAL_PATH" \
        --output-dir "$TMP_MP3_DIR" \
        --workers 4 \
        --timeout 120 \
        --bitrate "$MP3_BITRATE"
    CONVERTED=$(find "$TMP_MP3_DIR" -name "*.mp3" 2>/dev/null | wc -l)

    # 复制非音频文件（CSV、JSON等）
    while IFS= read -r -d '' src_file; do
        rel_path="${src_file#$LOCAL_PATH}"
        dst_file="$TMP_MP3_DIR/$rel_path"
        mkdir -p "$(dirname "$dst_file")"
        cp "$src_file" "$dst_file"
    done < <(find "$LOCAL_PATH" -type f ! \( -name "*.flac" -o -name "*.wav" \) -print0)

    # 统计MP3大小
    MP3_SIZE=$(du -sh "$TMP_MP3_DIR" | cut -f1)
    echo "  转换完成: ${CONVERTED} 个文件，MP3大小: ${MP3_SIZE}"

    # rsync推送MP3
    echo "  正在同步..."
    rsync -avz --progress -e "ssh -p $PORT -i $SSH_KEY $SSH_OPTS" \
        "$TMP_MP3_DIR/" "$GPU_USER@$GPU_HOST:$REMOTE_PATH"

    # 清理临时目录
    rm -rf "$TMP_MP3_DIR"
    echo "  临时目录已清理"
else
    echo "=== 推送数据: 本地→远程 ==="
    echo "  本地: $LOCAL_PATH"
    echo "  远程: $GPU_USER@$GPU_HOST:$REMOTE_PATH"
    rsync -avz --progress -e "ssh -p $PORT -i $SSH_KEY $SSH_OPTS" \
        "$LOCAL_PATH" "$GPU_USER@$GPU_HOST:$REMOTE_PATH"
fi

echo "=== 同步完成 ==="
