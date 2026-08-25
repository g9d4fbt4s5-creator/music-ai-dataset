#!/bin/bash
# ============================================================
# Label Studio 启动脚本 - 自动启用本地文件服务
# 用法: bash scripts/utils/start_labelstudio.sh
# ============================================================

set -e

PROJECT_ROOT="/Users/m.jian/music_corpus_project"
CONDA_ENV="labelstudio-env"
PORT=8080

cd "$PROJECT_ROOT"

# 激活 conda 环境
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# ============================================================
# 关键环境变量：让 Label Studio 能访问本地音频文件
# 没有这些变量，audio 字段的本地路径会被当成 HTTP URL，返回 404
# ============================================================
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$PROJECT_ROOT"

# Label Studio API Token（从环境变量读取，未设置则使用默认值）
export LABEL_STUDIO_TOKEN="${LABEL_STUDIO_TOKEN:-0c42572cb998a04808267af748b96fb88cde6fc3}"

echo "============================================================"
echo "🚀 启动 Label Studio"
echo "📂 项目根目录: $PROJECT_ROOT"
echo "🔧 Conda 环境: $CONDA_ENV"
echo "🌐 访问地址: http://localhost:$PORT"
echo "🔑 API Token: ${LABEL_STUDIO_TOKEN:0:8}...（已设置）"
echo "📁 本地文件服务: 已启用（recursive_scan）"
echo "============================================================"
echo ""
echo "提示：创建听检项目时使用 scripts/utils/ls_create_task.py"
echo "      它会自动处理本地存储创建和音频路径转换"
echo ""

label-studio start --port $PORT
