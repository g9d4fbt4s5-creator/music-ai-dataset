#!/bin/bash
# gpu_cleanup.sh
# GPU磁盘清理脚本 — 只删音频/产物，不删模型/代码/环境
#
# 用法:
#   bash gpu_cleanup.sh <batch_id>
#   bash gpu_cleanup.sh batch_000
#   bash gpu_cleanup.sh --all          # 清理所有批次（保留模型/代码）
#   bash gpu_cleanup.sh --dry-run      # 预览要删除的内容，不实际删除
#
# 安全规则:
#   ✅ 删除: 原始音频(batch_*_in/)、母版FLAC(master/)、segments、features
#   ❌ 不删除: 模型权重(MERT/CLAP/Demucs/Whisper/FunASR)、conda环境、代码、wheel缓存
#   ❌ 不删除: /root/ 根目录、/root/autodl-tmp/ 根目录
#   ❌ 不删除: meta/（元数据，待确认Mac已合并后手动删除）

set -euo pipefail

# ===================== 配置 =====================
GPU_TMP="/root/autodl-tmp"
PROJECT_DIR="/root/music_corpus_project"
MODELS_DIR="${GPU_TMP}/models"
WHEEL_CACHE="${PROJECT_DIR}/envs/wheel_cache"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ===================== 安全检查 =====================
safety_check() {
    local target="$1"

    # 禁止删除根目录
    if [[ "$target" == "/" || "$target" == "/root" || "$target" == "/root/" ]]; then
        echo -e "${RED}❌ 安全拦截: 禁止删除根目录 $target${NC}"
        exit 1
    fi

    # 禁止删除 autodl-tmp 根目录
    if [[ "$target" == "$GPU_TMP" || "$target" == "${GPU_TMP}/" ]]; then
        echo -e "${RED}❌ 安全拦截: 禁止删除 $GPU_TMP 根目录${NC}"
        exit 1
    fi

    # 禁止删除项目代码目录
    if [[ "$target" == "$PROJECT_DIR" || "$target" == "${PROJECT_DIR}/" ]]; then
        echo -e "${RED}❌ 安全拦截: 禁止删除项目代码目录 $PROJECT_DIR${NC}"
        exit 1
    fi

    # 禁止删除模型目录
    if [[ "$target" == "$MODELS_DIR" || "$target" == "${MODELS_DIR}/" ]]; then
        echo -e "${RED}❌ 安全拦截: 禁止删除模型目录 $MODELS_DIR${NC}"
        exit 1
    fi

    # 禁止删除 conda 环境
    if [[ "$target" == /opt/conda* || "$target" == /root/miniconda* ]]; then
        echo -e "${RED}❌ 安全拦截: 禁止删除 conda 环境${NC}"
        exit 1
    fi
}

# ===================== 批次目录验证（并发安全） =====================
validate_batch_dir() {
    local batch_id="$1"
    local dir_type="$2"  # "in" 或 "out"

    # 构造完整路径
    local batch_dir="${GPU_TMP}/batch_${batch_id}_${dir_type}"

    # 严格验证路径格式：必须是 /root/autodl-tmp/batch_XXX_in 或 batch_XXX_out
    # 拒绝通配符、相对路径、父目录遍历
    if [[ ! "$batch_dir" =~ ^${GPU_TMP}/batch_[0-9]+_(in|out)$ ]]; then
        echo -e "${RED}❌ 安全拦截: 无效的批次目录路径: $batch_dir${NC}"
        echo -e "${RED}   必须符合格式: ${GPU_TMP}/batch_XXX_in 或 batch_XXX_out${NC}"
        exit 1
    fi

    # 拒绝包含通配符的路径
    if [[ "$batch_dir" == *"*"* || "$batch_dir" == *"?"* || "$batch_dir" == *"["* ]]; then
        echo -e "${RED}❌ 安全拦截: 路径包含通配符: $batch_dir${NC}"
        exit 1
    fi

    # 拒绝父目录遍历
    if [[ "$batch_dir" == *".."* ]]; then
        echo -e "${RED}❌ 安全拦截: 路径包含父目录遍历: $batch_dir${NC}"
        exit 1
    fi

    echo "$batch_dir"
}

# ===================== 安全删除函数 =====================
safe_delete() {
    local target="$1"
    local description="$2"
    local dry_run="${3:-false}"

    if [[ ! -e "$target" ]]; then
        echo -e "${YELLOW}  ⏭️  跳过（不存在）: $description ($target)${NC}"
        return 0
    fi

    safety_check "$target"

    local size=$(du -sh "$target" 2>/dev/null | cut -f1)

    if [[ "$dry_run" == "true" ]]; then
        echo -e "${BLUE}  [预览] 将删除: $description ($target) [${size}]${NC}"
    else
        echo -e "${GREEN}  🗑️  删除: $description ($target) [${size}]${NC}"
        rm -rf "$target"
    fi
}

# ===================== 分层清理：原始音频 =====================
# GPU数据生命周期优化：母版FLAC生成并校验后即可删除原始音频
# 原始音频已备份在 Mac + OSS，所有下游产物都从母版FLAC生成
cleanup_raw() {
    local batch_id="$1"
    local dry_run="${2:-false}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}分层清理：删除原始音频（母版FLAC已生成）${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 并发安全：严格验证批次目录路径
    local batch_in=$(validate_batch_dir "$batch_id" "in")

    echo -e "${CYAN}  验证通过:${NC}"
    echo -e "     输入目录: $batch_in"
    echo ""

    # 删除原始音频输入目录
    safe_delete "$batch_in" "原始音频（已备份在Mac+OSS，母版FLAC已生成）" "$dry_run"

    echo ""
    echo -e "${GREEN}✅ 原始音频清理完成${NC}"
    echo ""

    # 显示磁盘使用情况
    echo -e "${BLUE}磁盘使用情况:${NC}"
    df -h "$GPU_TMP" | tail -1
    echo ""
}

# ===================== 分层清理：母版FLAC =====================
# GPU数据生命周期优化：stems+segments+嵌入向量全部生成并校验后即可删除母版FLAC
# 母版FLAC的使命已完成，所有下游产物（stems/segments/嵌入）都已生成
cleanup_master() {
    local batch_id="$1"
    local dry_run="${2:-false}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}分层清理：删除母版FLAC（stems/segments/嵌入已生成）${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 并发安全：严格验证批次目录路径
    local batch_out=$(validate_batch_dir "$batch_id" "out")

    echo -e "${CYAN}  验证通过:${NC}"
    echo -e "     输出目录: $batch_out"
    echo ""

    # 安全检查：确认 stems/segments/嵌入目录非空（母版FLAC的下游产物已生成）
    local stems_ready=false
    local segments_ready=false
    local embeddings_ready=false

    # 检查 stems（可能在 demucs_stems/ 或 stems/ 目录）
    if [[ -d "${batch_out}/demucs_stems" ]] && [[ -n "$(ls -A ${batch_out}/demucs_stems 2>/dev/null)" ]]; then
        stems_ready=true
    elif [[ -d "${batch_out}/stems" ]] && [[ -n "$(ls -A ${batch_out}/stems 2>/dev/null)" ]]; then
        stems_ready=true
    fi

    # 检查 segments
    if [[ -d "${batch_out}/segments" ]] && [[ -n "$(ls -A ${batch_out}/segments 2>/dev/null)" ]]; then
        segments_ready=true
    fi

    # 检查嵌入/特征（可能在 features/ 或 model_output_cache/ 目录）
    if [[ -d "${batch_out}/features" ]] && [[ -n "$(ls -A ${batch_out}/features 2>/dev/null)" ]]; then
        embeddings_ready=true
    elif [[ -d "${batch_out}/model_output_cache" ]] && [[ -n "$(ls -A ${batch_out}/model_output_cache 2>/dev/null)" ]]; then
        embeddings_ready=true
    fi

    echo -e "${CYAN}  下游产物检查:${NC}"
    echo -e "     stems:      $([[ "$stems_ready" == "true" ]] && echo '✅ 已生成' || echo '❌ 未生成')"
    echo -e "     segments:   $([[ "$segments_ready" == "true" ]] && echo '✅ 已生成' || echo '❌ 未生成')"
    echo -e "     embeddings: $([[ "$embeddings_ready" == "true" ]] && echo '✅ 已生成' || echo '❌ 未生成')"
    echo ""

    # 安全检查：至少需要 stems 或 segments 或 embeddings 其中之一已生成
    # （不同任务可能不需要全部，但至少需要一个下游产物，否则母版FLAC可能还在用）
    if [[ "$stems_ready" == "false" ]] && [[ "$segments_ready" == "false" ]] && [[ "$embeddings_ready" == "false" ]]; then
        echo -e "${RED}❌ 安全拦截: stems/segments/embeddings 均未生成，母版FLAC可能还在使用中，拒绝删除${NC}"
        echo -e "${RED}   请确认下游产物已生成后再执行母版FLAC清理${NC}"
        exit 1
    fi

    # 删除母版FLAC
    safe_delete "${batch_out}/master" "母版FLAC（stems/segments/嵌入已生成）" "$dry_run"
    safe_delete "${batch_out}/processed_master" "母版FLAC(备用路径)" "$dry_run"

    echo ""
    echo -e "${GREEN}✅ 母版FLAC清理完成${NC}"
    echo ""

    # 显示磁盘使用情况
    echo -e "${BLUE}磁盘使用情况:${NC}"
    df -h "$GPU_TMP" | tail -1
    echo ""
}

# ===================== 清理单个批次 =====================
cleanup_batch() {
    local batch_id="$1"
    local dry_run="${2:-false}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}清理批次: $batch_id${NC}"
    echo -e "${BLUE}========================================${NC}"

    # 并发安全：严格验证批次目录路径，拒绝通配符和无效路径
    local batch_in=$(validate_batch_dir "$batch_id" "in")
    local batch_out=$(validate_batch_dir "$batch_id" "out")

    echo -e "${CYAN}  验证通过:${NC}"
    echo -e "     输入目录: $batch_in"
    echo -e "     输出目录: $batch_out"
    echo ""

    # 1. 删除原始音频输入
    safe_delete "$batch_in" "原始音频输入" "$dry_run"

    # 2. 删除母版FLAC（Mac可重新生成）
    safe_delete "${batch_out}/master" "母版FLAC" "$dry_run"
    safe_delete "${batch_out}/processed_master" "母版FLAC(备用路径)" "$dry_run"

    # 3. 删除segments（已回传Mac）
    safe_delete "${batch_out}/segments" "切片segments" "$dry_run"

    # 4. 删除features（已回传Mac）
    safe_delete "${batch_out}/features" "特征文件" "$dry_run"

    # 5. 删除demucs中间产物（drums/bass/other，vocals已回传Mac）
    # 注意: 不删除 demucs_vocals/，因为可能还没回传
    safe_delete "${batch_out}/demucs_stems" "Demucs分轨(全量)" "$dry_run"
    safe_delete "${batch_out}/stems" "Demucs分轨(备用路径)" "$dry_run"

    # 6. 删除临时处理目录
    safe_delete "${batch_out}/tmp" "临时处理文件" "$dry_run"
    safe_delete "${batch_out}/work" "工作目录" "$dry_run"

    # ⚠️ 保留: meta/（元数据，待确认Mac已合并）
    # ⚠️ 保留: demucs_vocals/（人声，可能还没回传）
    # ⚠️ 保留: preannotation/（预标注结果）

    echo ""
    echo -e "${YELLOW}  ⚠️  保留（需手动确认后删除）:${NC}"
    echo -e "     - ${batch_out}/meta/          (元数据，确认Mac已合并)"
    echo -e "     - ${batch_out}/demucs_vocals/ (人声，确认已回传Mac)"
    echo -e "     - ${batch_out}/preannotation/  (预标注结果)"
    echo ""

    # 显示磁盘使用情况
    echo -e "${BLUE}磁盘使用情况:${NC}"
    df -h "$GPU_TMP" | tail -1
    echo ""
}

# ===================== 清理所有批次 =====================
cleanup_all() {
    local dry_run="${1:-false}"

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}清理所有批次（保留模型/代码/环境）${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # 查找所有批次目录
    local batch_dirs=$(find "$GPU_TMP" -maxdepth 1 -type d -name "batch_*" 2>/dev/null | sort)

    if [[ -z "$batch_dirs" ]]; then
        echo -e "${YELLOW}未找到批次目录${NC}"
    else
        for dir in $batch_dirs; do
            local batch_id=$(basename "$dir" | sed 's/_in$//' | sed 's/_out$//')
            # 避免重复处理（_in和_out对应同一个batch_id）
            if [[ ! " ${processed_batches:-} " =~ " ${batch_id} " ]]; then
                cleanup_batch "$batch_id" "$dry_run"
                processed_batches="${processed_batches:-} $batch_id"
            fi
        done
    fi

    # 清理通用临时目录
    echo -e "${BLUE}清理通用临时目录:${NC}"
    safe_delete "${GPU_TMP}/tmp" "全局临时目录" "$dry_run"
    safe_delete "${GPU_TMP}/cache" "全局缓存" "$dry_run"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ 所有批次清理完成${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # 显示保留的内容
    echo -e "${BLUE}保留的内容（不删除）:${NC}"
    echo -e "  - 模型权重: $MODELS_DIR"
    echo -e "  - 项目代码: $PROJECT_DIR"
    echo -e "  - conda环境: /opt/conda/"
    echo -e "  - wheel缓存: $WHEEL_CACHE"
    echo -e "  - 各批次 meta/ demucs_vocals/ preannotation/"
    echo ""

    df -h "$GPU_TMP"
}

# ===================== 主函数 =====================
main() {
    local dry_run="false"
    local target=""
    local cleanup_mode="all"  # all / raw / master

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                dry_run="true"
                shift
                ;;
            --all)
                target="all"
                shift
                ;;
            --raw-only)
                cleanup_mode="raw"
                shift
                ;;
            --master-only)
                cleanup_mode="master"
                shift
                ;;
            -h|--help)
                echo "用法: bash gpu_cleanup.sh <batch_id> [--dry-run] [--raw-only|--master-only]"
                echo "      bash gpu_cleanup.sh --all [--dry-run]"
                echo ""
                echo "分层清理模式:"
                echo "  --raw-only      只删除原始音频（母版FLAC已生成后调用）"
                echo "  --master-only   只删除母版FLAC（stems/segments/嵌入已生成后调用）"
                echo "  (默认)          全量清理（预标注完成并回传后调用）"
                echo ""
                echo "示例:"
                echo "  bash gpu_cleanup.sh batch_000 --raw-only      # 删原始音频"
                echo "  bash gpu_cleanup.sh batch_000 --master-only   # 删母版FLAC"
                echo "  bash gpu_cleanup.sh batch_000                  # 全量清理"
                echo "  bash gpu_cleanup.sh --all                       # 清理所有批次"
                echo "  bash gpu_cleanup.sh batch_000 --dry-run        # 预览，不实际删除"
                exit 0
                ;;
            *)
                target="$1"
                shift
                ;;
        esac
    done

    if [[ -z "$target" ]]; then
        echo -e "${RED}❌ 错误: 请指定批次ID或使用 --all${NC}"
        echo "用法: bash gpu_cleanup.sh <batch_id> [--dry-run] [--raw-only|--master-only]"
        echo "      bash gpu_cleanup.sh --all [--dry-run]"
        exit 1
    fi

    if [[ "$dry_run" == "true" ]]; then
        echo -e "${YELLOW}⚠️  DRY-RUN 模式: 只预览，不实际删除${NC}"
        echo ""
    fi

    # 根据清理模式执行
    if [[ "$target" == "all" ]]; then
        if [[ "$cleanup_mode" != "all" ]]; then
            echo -e "${YELLOW}⚠️  --all 模式下忽略 --raw-only/--master-only，执行全量清理${NC}"
        fi
        cleanup_all "$dry_run"
    elif [[ "$cleanup_mode" == "raw" ]]; then
        cleanup_raw "$target" "$dry_run"
    elif [[ "$cleanup_mode" == "master" ]]; then
        cleanup_master "$target" "$dry_run"
    else
        cleanup_batch "$target" "$dry_run"
    fi
}

main "$@"
