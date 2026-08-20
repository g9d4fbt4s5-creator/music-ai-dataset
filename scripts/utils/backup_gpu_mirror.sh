#!/bin/bash
# ============================================================
# backup_gpu_mirror.sh
# 通过 rsync 直接从 GPU 服务器拉回快照到本地
#
# ⚠️ 新架构约束（2026-08-20）：
# - GPU ↔ Mac 的数据传输通过 rsync 直接进行，不走 OSS
# - OSS 仅作纯备份归档，业务绝不从 OSS 读取音频
# - 本地磁盘是唯一业务数据源
#
# 用法：
#   ./scripts/utils/backup_gpu_mirror.sh <gpu_ssh_host> <gpu_snapshot_path>
#
# 示例：
#   ./scripts/utils/backup_gpu_mirror.sh root@123.45.67.89 /workspace/data/snapshots/run_20260820
#
# 输出：
#   ./snapshots/gpu_backup_YYYYMMDD_HHMMSS/
# ============================================================

set -euo pipefail

# ===================== 配置 =====================
# 项目根目录（脚本所在目录的上两级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 本地快照目录
LOCAL_SNAPSHOT_DIR="$PROJECT_ROOT/snapshots"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SNAPSHOT_NAME="gpu_backup_${TIMESTAMP}"
SNAPSHOT_PATH="$LOCAL_SNAPSHOT_DIR/$SNAPSHOT_NAME"

# rsync 参数
# -a: 归档模式（保留权限、时间戳等）
# -v: 详细输出
# -z: 压缩传输
# -h: 人类可读的输出
# --progress: 显示进度
# --partial: 断点续传
RSYNC_OPTS="-avzh --progress --partial"

# SSH 端口（默认 22，AutoDL 通常是自定义端口）
SSH_PORT="${SSH_PORT:-22}"

# ===================== 参数检查 =====================
if [ $# -lt 2 ]; then
    echo "用法: $0 <gpu_ssh_host> <gpu_snapshot_path>"
    echo ""
    echo "参数:"
    echo "  gpu_ssh_host      GPU SSH 地址，如 root@123.45.67.89"
    echo "  gpu_snapshot_path GPU 上的快照目录路径"
    echo ""
    echo "环境变量:"
    echo "  SSH_PORT           SSH 端口，默认 22"
    echo ""
    echo "示例:"
    echo "  $0 root@123.45.67.89 /workspace/data/snapshots/run_20260820"
    echo "  SSH_PORT=12345 $0 root@123.45.67.89 /workspace/data/model_output_cache"
    exit 1
fi

GPU_SSH_HOST="$1"
GPU_SNAPSHOT_PATH="$2"

# ===================== 前置检查 =====================
echo "============================================================"
echo "GPU 快照拉回（rsync 直接传输，不走 OSS）"
echo "============================================================"
echo ""
echo "GPU SSH:        $GPU_SSH_HOST"
echo "SSH 端口:       $SSH_PORT"
echo "GPU 快照路径:   $GPU_SNAPSHOT_PATH"
echo "本地快照目录:   $SNAPSHOT_PATH"
echo ""

# 检查 rsync 是否安装
if ! command -v rsync &> /dev/null; then
    echo "❌ 错误: rsync 未安装"
    echo "   Mac 安装: brew install rsync"
    exit 1
fi

# 检查 SSH 连通性
echo "检查 SSH 连通性..."
if ! ssh -p "$SSH_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$GPU_SSH_HOST" "test -d '$GPU_SNAPSHOT_PATH'" 2>/dev/null; then
    echo "❌ 错误: 无法连接到 GPU 或快照目录不存在"
    echo "   请检查:"
    echo "   1. SSH 地址和端口是否正确"
    echo "   2. GPU 上的快照路径是否存在: $GPU_SNAPSHOT_PATH"
    echo "   3. SSH 密钥是否配置正确"
    exit 1
fi
echo "✅ SSH 连通性正常，快照目录存在"
echo ""

# 创建本地快照目录
mkdir -p "$SNAPSHOT_PATH"

# ===================== 执行 rsync =====================
echo "开始 rsync 传输..."
echo "命令: rsync $RSYNC_OPTS -e 'ssh -p $SSH_PORT' $GPU_SSH_HOST:$GPU_SNAPSHOT_PATH/ $SNAPSHOT_PATH/"
echo ""

rsync $RSYNC_OPTS \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$GPU_SSH_HOST:$GPU_SNAPSHOT_PATH/" \
    "$SNAPSHOT_PATH/"

echo ""
echo "✅ rsync 传输完成"
echo ""

# ===================== 传输后处理 =====================
# 统计快照大小
SNAPSHOT_SIZE=$(du -sh "$SNAPSHOT_PATH" | cut -f1)
FILE_COUNT=$(find "$SNAPSHOT_PATH" -type f | wc -l)

echo "============================================================"
echo "快照拉回完成"
echo "============================================================"
echo "快照名称:   $SNAPSHOT_NAME"
echo "快照路径:   $SNAPSHOT_PATH"
echo "快照大小:   $SNAPSHOT_SIZE"
echo "文件数量:   $FILE_COUNT"
echo ""

# 检查是否有 .oss_verified 标记（说明 GPU 端已上传 OSS 并校验）
if [ -f "$SNAPSHOT_PATH/.oss_verified" ]; then
    echo "✅ 快照包含 .oss_verified 标记，GPU 端已上传 OSS 并校验通过"
    echo "   本地可以安全使用此快照"
else
    echo "⚠️  快照不包含 .oss_verified 标记"
    echo "   如需备份到 OSS，请运行:"
    echo "   python3 $PROJECT_ROOT/scripts/utils/upload_cache_to_oss.py"
    echo ""
    echo "   如需校验 OSS 上传完整性，请运行:"
    echo "   python3 $PROJECT_ROOT/scripts/utils/verify_oss_upload.py --snapshot $SNAPSHOT_PATH"
fi

echo ""
echo "============================================================"
echo "⚠️  新架构提醒:"
echo "   - 本次传输通过 rsync 直接进行，未经过 OSS"
echo "   - OSS 仅作纯备份归档，业务绝不从 OSS 读取音频"
echo "   - 本地磁盘是唯一业务数据源"
echo "   - 快照轮转清理请使用 disk_guard.py"
echo "============================================================"
