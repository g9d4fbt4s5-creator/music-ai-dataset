#!/bin/bash
# ============================================================
# gpu_compress_stems.sh
# GPU 端 Demucs stems WAV → FLAC 压缩脚本
#
# 功能：
# - 将 Demucs 输出的 WAV stems 转为 FLAC 最高压缩（-compression_level 8）
# - 音乐类音频通常能压到 45-55%，节省 40-50% 传输带宽和存储空间
# - 转换完成后删除原 WAV 文件
#
# 用法：
#   # 压缩单个批次的所有 stems
#   bash gpu_compress_stems.sh /root/autodl-tmp/batch_000_out/stems
#
#   # 压缩指定目录（递归）
#   bash gpu_compress_stems.sh /path/to/stems
#
#   # 预览模式（不实际转换）
#   bash gpu_compress_stems.sh /path/to/stems --dry-run
#
# 注意：
# - FLAC 是无损压缩，解码后与原 WAV 完全一致
# - 训练时如需 WAV，可实时解码（ffmpeg 解码 FLAC 几乎无开销）
# - -compression_level 8 是最高压缩，编码稍慢但压缩率最高
# ============================================================

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ===================== 参数解析 =====================
STEMS_DIR="${1:-}"
DRY_RUN=false

if [[ -z "$STEMS_DIR" ]]; then
    echo -e "${RED}❌ 错误：请指定 stems 目录路径${NC}"
    echo "用法: bash gpu_compress_stems.sh /path/to/stems [--dry-run]"
    exit 1
fi

if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# 检查目录是否存在
if [[ ! -d "$STEMS_DIR" ]]; then
    echo -e "${RED}❌ 错误：目录不存在: $STEMS_DIR${NC}"
    exit 1
fi

# 检查 ffmpeg 是否可用
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ 错误：ffmpeg 未安装${NC}"
    echo "请先安装 ffmpeg: conda install -c conda-forge ffmpeg"
    exit 1
fi

# ===================== 统计 =====================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GPU Stems WAV → FLAC 压缩${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "目标目录: ${CYAN}$STEMS_DIR${NC}"
echo -e "压缩级别: ${CYAN}8（最高）${NC}"
echo -e "预览模式: ${CYAN}$DRY_RUN${NC}"
echo ""

# 统计 WAV 文件
WAV_FILES=$(find "$STEMS_DIR" -type f -name "*.wav" 2>/dev/null | sort)
WAV_COUNT=$(echo "$WAV_FILES" | grep -c . || echo 0)

if [[ "$WAV_COUNT" -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  未找到 WAV 文件，无需压缩${NC}"
    exit 0
fi

# 计算原始总大小
ORIGINAL_SIZE=0
while IFS= read -r wav; do
    if [[ -f "$wav" ]]; then
        SIZE=$(stat -f%z "$wav" 2>/dev/null || stat -c%s "$wav" 2>/dev/null || echo 0)
        ORIGINAL_SIZE=$((ORIGINAL_SIZE + SIZE))
    fi
done <<< "$WAV_FILES"

ORIGINAL_SIZE_MB=$(echo "scale=2; $ORIGINAL_SIZE / 1024 / 1024" | bc)
echo -e "找到 ${CYAN}$WAV_COUNT${NC} 个 WAV 文件，总计 ${CYAN}${ORIGINAL_SIZE_MB} MB${NC}"
echo ""

# ===================== 压缩 =====================
echo -e "${BLUE}开始压缩...${NC}"
echo ""

SUCCESS_COUNT=0
FAIL_COUNT=0
COMPRESSED_SIZE=0

while IFS= read -r wav; do
    if [[ ! -f "$wav" ]]; then
        continue
    fi

    flac="${wav%.wav}.flac"
    FILENAME=$(basename "$wav")

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${BLUE}  [预览] 将压缩: $FILENAME → $(basename "$flac")${NC}"
        continue
    fi

    # 如果 FLAC 已存在，跳过
    if [[ -f "$flac" ]]; then
        echo -e "${YELLOW}  ⏭️  跳过（已存在）: $FILENAME${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        continue
    fi

    # ffmpeg 转 FLAC（最高压缩级别 8）
    if ffmpeg -y -i "$wav" -c:a flac -compression_level 8 "$flac" \
        -loglevel error -nostdin 2>/dev/null; then
        # 验证 FLAC 文件
        if [[ -f "$flac" ]] && [[ $(stat -f%z "$flac" 2>/dev/null || stat -c%s "$flac" 2>/dev/null || echo 0) -gt 0 ]]; then
            # 删除原 WAV
            rm -f "$wav"
            FLAC_SIZE=$(stat -f%z "$flac" 2>/dev/null || stat -c%s "$flac" 2>/dev/null || echo 0)
            COMPRESSED_SIZE=$((COMPRESSED_SIZE + FLAC_SIZE))
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo -e "${GREEN}  ✅ $FILENAME → $(basename "$flac")${NC}"
        else
            echo -e "${RED}  ❌ 验证失败: $FILENAME（FLAC 文件为空）${NC}"
            rm -f "$flac"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo -e "${RED}  ❌ 转换失败: $FILENAME${NC}"
        rm -f "$flac"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done <<< "$WAV_FILES"

# ===================== 总结 =====================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}压缩完成${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "预览模式：未实际转换"
else
    echo -e "成功: ${GREEN}$SUCCESS_COUNT${NC} / $WAV_COUNT"
    if [[ "$FAIL_COUNT" -gt 0 ]]; then
        echo -e "失败: ${RED}$FAIL_COUNT${NC}"
    fi

    if [[ "$COMPRESSED_SIZE" -gt 0 ]] && [[ "$ORIGINAL_SIZE" -gt 0 ]]; then
        COMPRESSED_SIZE_MB=$(echo "scale=2; $COMPRESSED_SIZE / 1024 / 1024" | bc)
        RATIO=$(echo "scale=2; $COMPRESSED_SIZE * 100 / $ORIGINAL_SIZE" | bc)
        SAVED_MB=$(echo "scale=2; ($ORIGINAL_SIZE - $COMPRESSED_SIZE) / 1024 / 1024" | bc)
        echo ""
        echo -e "原始大小: ${CYAN}${ORIGINAL_SIZE_MB} MB${NC}"
        echo -e "压缩后: ${CYAN}${COMPRESSED_SIZE_MB} MB${NC}"
        echo -e "压缩率: ${CYAN}${RATIO}%${NC}（节省 ${SAVED_MB} MB）"
    fi
fi

echo -e "${BLUE}========================================${NC}"

# 如果有失败，返回非零退出码
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
fi

exit 0
