#!/bin/bash
# ============================================================
# GPU实例便捷连接脚本
# 用法：
#   ./scripts/utils/gpu_ssh.sh <端口> [命令]
# 示例：
#   ./scripts/utils/gpu_ssh.sh 49530                    # 交互式登录
#   ./scripts/utils/gpu_ssh.sh 49530 "nvidia-smi"      # 执行命令
#   ./scripts/utils/gpu_ssh.sh 49530 "python script.py" # 执行脚本
#
# 免密登录：使用 ~/.ssh/id_rsa 密钥（已配置到GPU实例authorized_keys）
# 端口每次开机都会变，从AutoDL控制台复制新端口
# ============================================================

set -euo pipefail

# 配置
GPU_HOST="connect.westb.seetacloud.com"
GPU_USER="root"
SSH_KEY="$HOME/.ssh/id_rsa"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=60"

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <端口> [命令]"
    echo "示例: $0 49530"
    echo "      $0 49530 'nvidia-smi'"
    exit 1
fi

PORT="$1"
shift

# 如果有命令，执行命令；否则交互式登录
if [ $# -gt 0 ]; then
    CMD="$*"
    echo "=== 连接 GPU 实例 ($GPU_HOST:$PORT) ==="
    echo "=== 执行: $CMD ==="
    ssh -p "$PORT" -i "$SSH_KEY" $SSH_OPTS "$GPU_USER@$GPU_HOST" "$CMD"
else
    echo "=== 连接 GPU 实例 ($GPU_HOST:$PORT) ==="
    echo "=== 输入 exit 退出 ==="
    ssh -p "$PORT" -i "$SSH_KEY" $SSH_OPTS "$GPU_USER@$GPU_HOST"
fi
