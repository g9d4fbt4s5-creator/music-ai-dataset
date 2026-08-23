#!/bin/bash
# ============================================================
# push.sh — 推送Docker镜像到仓库
# ============================================================
# 用法：
#   bash docker/push.sh              # 推送所有镜像
#   bash docker/push.sh gpu          # 只推送GPU主镜像
#   bash docker/push.sh yamnet       # 只推送YAMNet镜像
#
# 前置条件：
#   1. 已登录GitHub Container Registry:
#      echo "YOUR_GITHUB_TOKEN" | docker login ghcr.io -u YOUR_USERNAME --password-stdin
#   2. 已运行 bash docker/build.sh 构建镜像
# ============================================================

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 镜像配置（和build.sh保持一致）
REGISTRY="${REGISTRY:-ghcr.io}"
NAMESPACE="${NAMESPACE:-g9d4fbt4s5-creator}"
IMAGE_PREFIX="${IMAGE_PREFIX:-music-corpus}"
VERSION="${VERSION:-latest}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  AI音乐数据集流水线 — Docker镜像推送${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "  目标仓库: ${REGISTRY}/${NAMESPACE}"
echo -e "  版本标签: ${VERSION}"
echo ""

# 检查是否已登录
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}⚠️  未检测到Docker登录信息${NC}"
    echo ""
    echo "请先登录GitHub Container Registry:"
    echo "  echo \"YOUR_GITHUB_TOKEN\" | docker login ghcr.io -u ${NAMESPACE} --password-stdin"
    echo ""
    echo "获取GitHub Token: https://github.com/settings/tokens"
    echo "需要的权限: write:packages, read:packages"
    echo ""
    read -p "是否继续？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消。"
        exit 1
    fi
fi

# 推送函数
push_image() {
    local name="$1"
    local full_image="${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-${name}:${VERSION}"
    local latest_image="${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-${name}:latest"

    echo -e "${YELLOW}▶ 推送镜像: ${full_image}${NC}"
    echo ""

    # 检查镜像是否存在
    if ! docker image inspect "$full_image" >/dev/null 2>&1; then
        echo -e "${RED}❌ 镜像不存在: ${full_image}${NC}"
        echo "请先运行: bash docker/build.sh ${name}"
        exit 1
    fi

    # 推送版本标签
    docker push "$full_image"
    echo -e "${GREEN}✅ 已推送: ${full_image}${NC}"

    # 推送latest标签
    docker push "$latest_image"
    echo -e "${GREEN}✅ 已推送: ${latest_image}${NC}"
    echo ""
}

# 根据参数决定推送哪些镜像
case "${1:-all}" in
    gpu)
        push_image "gpu"
        ;;
    yamnet)
        push_image "yamnet"
        ;;
    all)
        push_image "gpu"
        push_image "yamnet"
        ;;
    *)
        echo -e "${RED}❌ 未知参数: $1${NC}"
        echo "用法: bash docker/push.sh [gpu|yamnet|all]"
        exit 1
        ;;
esac

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}  所有镜像推送完成！${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "镜像地址："
echo "  ${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-gpu:${VERSION}"
echo "  ${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-yamnet:${VERSION}"
echo ""
echo "在AutoDL上使用："
echo "  docker pull ${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-gpu:${VERSION}"
echo "  nvidia-docker run --rm -v /root/autodl-tmp:/data ${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-gpu:${VERSION} python scripts/01_preprocess/04_extract_features.py --input /data/manifest.csv"
echo ""
