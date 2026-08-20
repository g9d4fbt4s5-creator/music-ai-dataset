#!/usr/bin/env bash
# ============================================================
# install_hooks.sh - 安装 Git 钩子
# ============================================================
# 用法：
#   bash scripts/git_hooks/install_hooks.sh
#
# 功能：
#   将 scripts/git_hooks/ 下的钩子脚本链接到 .git/hooks/
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HOOKS_SRC_DIR="$SCRIPT_DIR"
HOOKS_DST_DIR="$PROJECT_ROOT/.git/hooks"

echo "========================================"
echo "  安装 Git 钩子"
echo "========================================"
echo "  项目根目录: $PROJECT_ROOT"
echo "  钩子源目录: $HOOKS_SRC_DIR"
echo "  钩子目标目录: $HOOKS_DST_DIR"
echo ""

# 检查是否在 git 仓库中
if [ ! -d "$HOOKS_DST_DIR" ]; then
    echo -e "${RED}❌ 错误: .git/hooks 目录不存在，请先运行 git init${NC}"
    exit 1
fi

# 要安装的钩子列表
hooks=("pre-commit")

installed=0
skipped=0

for hook in "${hooks[@]}"; do
    src_file="$HOOKS_SRC_DIR/$hook"
    dst_file="$HOOKS_DST_DIR/$hook"

    if [ ! -f "$src_file" ]; then
        echo -e "  ${YELLOW}⚠️  跳过: $hook (源文件不存在)${NC}"
        skipped=$((skipped + 1))
        continue
    fi

    # 如果目标已存在，备份
    if [ -f "$dst_file" ] || [ -L "$dst_file" ]; then
        backup_file="${dst_file}.backup.$(date +%Y%m%d_%H%M%S)"
        mv "$dst_file" "$backup_file"
        echo -e "  ${YELLOW}已备份旧钩子: $hook -> $backup_file${NC}"
    fi

    # 创建符号链接
    ln -s "$src_file" "$dst_file"
    chmod +x "$src_file"

    echo -e "  ${GREEN}✅ 已安装: $hook${NC}"
    installed=$((installed + 1))
done

echo ""
echo "========================================"
echo -e "  ${GREEN}安装完成: $installed 个钩子, $skipped 个跳过${NC}"
echo "========================================"
echo ""
echo "钩子说明："
echo "  pre-commit: 提交前检查大文件、密钥、音频、模型权重"
echo ""
echo "如需跳过检查（不推荐）："
echo "  git commit --no-verify"
echo ""
echo "如需卸载钩子："
echo "  rm .git/hooks/pre-commit"
