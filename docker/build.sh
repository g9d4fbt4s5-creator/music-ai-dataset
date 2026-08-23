#!/bin/bash
# ============================================================
# build.sh — 构建Docker镜像
# ============================================================
# 用法：
#   bash docker/build.sh              # 构建所有镜像
#   bash docker/build.sh gpu          # 只构建GPU主镜像
#   bash docker/build.sh yamnet       # 只构建YAMNet镜像
#   bash docker/build.sh --no-cache   # 无缓存构建
# ============================================================

set -e  # 遇到错误立即退出

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 镜像配置
REGISTRY="${REGISTRY:-ghcr.io}"                    # 默认GitHub Container Registry
NAMESPACE="${NAMESPACE:-g9d4fbt4s5-creator}"       # GitHub用户名
IMAGE_PREFIX="${IMAGE_PREFIX:-music-corpus}"         # 镜像名前缀
VERSION="${VERSION:-latest}"                          # 版本标签

# 构建参数
BUILD_ARGS=""
if [[ "$*" == *"--no-cache"* ]]; then
    BUILD_ARGS="--no-cache"
fi

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  AI音乐数据集流水线 — Docker镜像构建${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "  项目根目录: ${PROJECT_ROOT}"
echo -e "  镜像仓库:   ${REGISTRY}/${NAMESPACE}"
echo -e "  版本标签:   ${VERSION}"
echo -e "  构建参数:   ${BUILD_ARGS:-无}"
echo ""

# 构建函数
build_image() {
    local name="$1"
    local dockerfile="$2"
    local full_image="${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-${name}:${VERSION}"

    echo -e "${YELLOW}▶ 构建镜像: ${full_image}${NC}"
    echo -e "  Dockerfile: ${dockerfile}"
    echo ""

    docker build \
        $BUILD_ARGS \
        -f "$dockerfile" \
        -t "$full_image" \
        -t "${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-${name}:latest" \
        .

    echo ""
    echo -e "${GREEN}✅ 构建成功: ${full_image}${NC}"
    echo ""
}

# 根据参数决定构建哪些镜像
case "${1:-all}" in
    gpu)
        build_image "gpu" "docker/Dockerfile.gpu"
        ;;
    yamnet)
        build_image "yamnet" "docker/Dockerfile.yamnet"
        ;;
    all)
        build_image "gpu" "docker/Dockerfile.gpu"
        build_image "yamnet" "docker/Dockerfile.yamnet"
        ;;
    *)
        echo -e "${RED}❌ 未知参数: $1${NC}"
        echo "用法: bash docker/build.sh [gpu|yamnet|all] [--no-cache]"
        exit 1
        ;;
esac

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}  所有镜像构建完成！${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "下一步："
echo "  1. 本地测试:  nvidia-docker run --rm ${REGISTRY}/${NAMESPACE}/${IMAGE_PREFIX}-gpu:latest python --version"
echo "  2. 推送镜像:  bash docker/push.sh"
echo "  3. AutoDL使用: 见 docker/README.md"
echo ""
