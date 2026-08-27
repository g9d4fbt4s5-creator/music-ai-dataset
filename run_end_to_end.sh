#!/bin/bash
# =============================================================================
# run_end_to_end.sh - 音乐语料库端到端流水线
#
# 流程：QC Gate → 母版 → L1物理 → L2 MERT/CLAP嵌入 → 划分 → L3标注 → L4传播 → Stage5切片
#
# 约束：
# - 黄金集5首 + Challenge3首标记保持不变，不重新采样
# - L2嵌入需要GPU（AutoDL远程实例）
# - L3标注需要Qwen-Omni API
# - 每步执行后验证输出，失败即停
# =============================================================================

set -euo pipefail

# 配置
PROJECT_ROOT="/Users/m.jian/music_corpus_project"
MANIFEST="${PROJECT_ROOT}/data/00_raw_collect/audio_manifest.csv"
QC_REPORT="${PROJECT_ROOT}/data/00.5_cleaned/reports/qc_gate_report.csv"
SPLITS_DIR="${PROJECT_ROOT}/data/04_final_dataset/splits/splits"
LOG_DIR="${PROJECT_ROOT}/logs/end_to_end_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# GPU 实例配置（AutoDL，L2 嵌入需要）
# SSH 免密已配置（id_rsa + id_ed25519），无需密码
GPU_SSH_PORT="49530"
GPU_SSH_HOST="root@connect.westb.seetacloud.com"
GPU_REMOTE_ROOT="/root/autodl-tmp/music-ai-dataset"
# SSH 命令前缀（免密登录，自动接受主机密钥）
SSH_CMD="ssh -p ${GPU_SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15 ${GPU_SSH_HOST}"
RSYNC_CMD="rsync -avz -e 'ssh -p ${GPU_SSH_PORT} -o StrictHostKeyChecking=no'"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_step() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}[$(date '+%H:%M:%S')] STEP $1: $2${NC}"
    echo -e "${GREEN}========================================${NC}"
}

log_warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

log_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

verify_output() {
    local desc="$1"
    local expected_count="$2"
    local actual_count="$3"
    if [ "${actual_count}" -ge "${expected_count}" ]; then
        echo -e "${GREEN}[PASS] ${desc}: ${actual_count} (expected >= ${expected_count})${NC}"
    else
        echo -e "${RED}[FAIL] ${desc}: ${actual_count} (expected >= ${expected_count})${NC}"
        exit 1
    fi
}

# =============================================================================
# GPU 数据同步函数（P0 根治：避免手动同步数据）
# =============================================================================

sync_to_gpu() {
    """同步 L2 步骤所需的文件到 GPU 实例：manifest + QC报告 + 母版文件"""
    echo "同步数据到 GPU 实例..."
    echo "  1/3 manifest"
    eval ${RSYNC_CMD} "${MANIFEST}" "${GPU_SSH_HOST}:${GPU_REMOTE_ROOT}/data/00_raw_collect/audio_manifest.csv" 2>&1 | tail -2
    echo "  2/3 QC报告"
    eval ${RSYNC_CMD} "${QC_REPORT}" "${GPU_SSH_HOST}:${GPU_REMOTE_ROOT}/data/00.5_cleaned/reports/qc_gate_report.csv" 2>&1 | tail -2
    echo "  3/3 母版文件（可能较大，请耐心等待）"
    eval ${RSYNC_CMD} "${PROJECT_ROOT}/data/01_preprocess/processed_master/" "${GPU_SSH_HOST}:${GPU_REMOTE_ROOT}/data/01_preprocess/processed_master/" 2>&1 | tail -3
    echo "✅ 数据同步完成"
}

sync_from_gpu() {
    """将 L2 结果从 GPU 实例同步回本地：MERT嵌入 + CLAP嵌入 + CLAP语义"""
    local l2_type="$1"  # mert / clap / all
    echo "从 GPU 实例同步 ${l2_type} 结果..."
    if [ "${l2_type}" = "mert" ] || [ "${l2_type}" = "all" ]; then
        eval ${RSYNC_CMD} "${GPU_SSH_HOST}:${GPU_REMOTE_ROOT}/data/02_preannotation/l2_embedding/" "${PROJECT_ROOT}/data/02_preannotation/l2_embedding/" 2>&1 | tail -2
    fi
    if [ "${l2_type}" = "clap" ] || [ "${l2_type}" = "all" ]; then
        eval ${RSYNC_CMD} "${GPU_SSH_HOST}:${GPU_REMOTE_ROOT}/data/02_preannotation/l2_embedding_clap/" "${PROJECT_ROOT}/data/02_preannotation/l2_embedding_clap/" 2>&1 | tail -2
        eval ${RSYNC_CMD} "${GPU_SSH_HOST}:${GPU_REMOTE_ROOT}/data/02_preannotation/l2_semantic/" "${PROJECT_ROOT}/data/02_preannotation/l2_semantic/" 2>&1 | tail -2
    fi
    echo "✅ ${l2_type} 结果同步完成"
}

run_on_gpu() {
    """在 GPU 实例上执行命令，等待完成并返回退出码"""
    local cmd="$1"
    eval ${SSH_CMD} "cd ${GPU_REMOTE_ROOT} && ${cmd}"
    return $?
}

cd "${PROJECT_ROOT}"

# =============================================================================
# STEP 0: 环境检查
# =============================================================================
log_step "0" "环境检查"
echo "项目根目录: ${PROJECT_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "日志目录: ${LOG_DIR}"

# 检查manifest
if [ ! -f "${MANIFEST}" ]; then
    log_error "Manifest不存在: ${MANIFEST}"
    exit 1
fi
TOTAL_SAMPLES=$(wc -l < "${MANIFEST}" | tr -d ' ')
TOTAL_SAMPLES=$((TOTAL_SAMPLES - 1))  # 减去header
echo "总样本数: ${TOTAL_SAMPLES}"

# 检查标记
GOLDEN_COUNT=$(python3 -c "import pandas as pd; df=pd.read_csv('${MANIFEST}'); print(len(df[df['sample_type']=='golden_seed']))")
CHALLENGE_COUNT=$(python3 -c "import pandas as pd; df=pd.read_csv('${MANIFEST}'); print(len(df[df['sample_type']=='challenge_stress_test']))")
echo "黄金集: ${GOLDEN_COUNT} 首"
echo "Challenge: ${CHALLENGE_COUNT} 首"

if [ "${GOLDEN_COUNT}" -ne 5 ]; then
    log_warn "黄金集数量不是5首（当前${GOLDEN_COUNT}首），请确认"
fi
if [ "${CHALLENGE_COUNT}" -ne 3 ]; then
    log_warn "Challenge数量不是3首（当前${CHALLENGE_COUNT}首），请确认"
fi

# =============================================================================
# STEP 1: QC Gate
# =============================================================================
log_step "1" "QC Gate（85首）"
python3 scripts/00.5_cleaning/qc_gate.py \
    --manifest "${MANIFEST}" \
    --output "${QC_REPORT}" \
    2>&1 | tee "${LOG_DIR}/01_qc_gate.log"

# 验证QC报告
QC_PASS=$(python3 -c "import pandas as pd; df=pd.read_csv('${QC_REPORT}'); print(len(df[df['final_branch']=='pass']))" 2>/dev/null || echo "0")
echo "QC pass: ${QC_PASS} 首"
verify_output "QC pass数量" 80 "${QC_PASS}"

# =============================================================================
# STEP 2: 母版生成
# =============================================================================
log_step "2" "母版生成（85首）"
python3 scripts/01_preprocess/01_generate_master.py \
    --manifest "${MANIFEST}" \
    --qc-report "${QC_REPORT}" \
    --output-dir "${PROJECT_ROOT}/data/01_preprocess/processed_master" \
    2>&1 | tee "${LOG_DIR}/02_master.log"

MASTER_COUNT=$(find "${PROJECT_ROOT}/data/01_preprocess/processed_master" -name "*.flac" | wc -l | tr -d ' ')
echo "母版文件: ${MASTER_COUNT} 首"
verify_output "母版数量" 80 "${MASTER_COUNT}"

# =============================================================================
# STEP 3: L1物理特征
# =============================================================================
log_step "3" "L1物理特征提取（85首）"
python3 scripts/02_preannotation/l1_physical/l1_physical_features.py \
    --manifest "${MANIFEST}" \
    --master-dir "${PROJECT_ROOT}/data/01_preprocess/processed_master" \
    --output-dir "${PROJECT_ROOT}/data/02_preannotation/l1_physical" \
    2>&1 | tee "${LOG_DIR}/03_l1_physical.log"

L1_COUNT=$(find "${PROJECT_ROOT}/data/02_preannotation/l1_physical" -name "*.json" | wc -l | tr -d ' ')
echo "L1物理标签: ${L1_COUNT} 首"
verify_output "L1数量" 80 "${L1_COUNT}"

# =============================================================================
# STEP 4: L2 MERT嵌入（GPU 远程执行 + 自动同步）
# =============================================================================
log_step "4" "L2 MERT嵌入提取（GPU远程执行）"
echo "GPU实例: ${GPU_SSH_HOST}:${GPU_SSH_PORT}"

# 自动同步数据到 GPU（P0 根治：不再手动同步）
sync_to_gpu

# 在 GPU 实例上运行 MERT 嵌入（后台执行，轮询等待）
echo "在 GPU 实例上启动 MERT 嵌入..."
run_on_gpu "source /root/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env && nohup python3 scripts/02_preannotation/extract_mert_embedding.py --input-dir data/01_preprocess/processed_master --output data/02_preannotation/l2_embedding --device cuda > logs/mert_embedding.log 2>&1 & disown && echo 'MERT已启动'"

# 轮询等待完成（最多等30分钟）
echo "等待 MERT 嵌入完成..."
for i in $(seq 1 60); do
    sleep 30
    MERT_DONE=$(run_on_gpu "ps aux | grep extract_mert | grep -v grep | wc -l" 2>/dev/null || echo "0")
    MERT_COUNT=$(run_on_gpu "find data/02_preannotation/l2_embedding -name '*.npy' | wc -l" 2>/dev/null || echo "0")
    echo "  [${i}/60] 进程: ${MERT_DONE}, 已完成: ${MERT_COUNT}/84"
    if [ "${MERT_DONE}" = "0" ] && [ "${MERT_COUNT}" -gt 0 ]; then
        echo "✅ MERT 嵌入完成"
        break
    fi
done

# 同步结果回本地
sync_from_gpu "mert"

MERT_COUNT=$(find "${PROJECT_ROOT}/data/02_preannotation/l2_embedding" -name "*.npy" | wc -l | tr -d ' ')
echo "MERT嵌入: ${MERT_COUNT} 首"
verify_output "MERT数量" 80 "${MERT_COUNT}"

# =============================================================================
# STEP 5: L2 CLAP零样本标注（GPU 远程执行 + 自动同步）
# =============================================================================
log_step "5" "L2 CLAP零样本标注（GPU远程执行）"

# 在 GPU 实例上运行 CLAP（自动检测本地模型，不需要手动传 --model-path）
echo "在 GPU 实例上启动 CLAP 零样本标注..."
run_on_gpu "source /root/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env && nohup python3 scripts/02_preannotation/l2_clap_zero_shot.py --input-dir data/01_preprocess/processed_master --output data/02_preannotation/l2_semantic --embedding-output data/02_preannotation/l2_embedding_clap --device cuda --top-k 5 > logs/clap_zero_shot.log 2>&1 & disown && echo 'CLAP已启动'"

# 轮询等待完成（最多等30分钟）
echo "等待 CLAP 零样本标注完成..."
for i in $(seq 1 60); do
    sleep 30
    CLAP_DONE=$(run_on_gpu "ps aux | grep clap_zero_shot | grep -v grep | wc -l" 2>/dev/null || echo "0")
    CLAP_COUNT=$(run_on_gpu "find data/02_preannotation/l2_semantic -name '*.json' | wc -l" 2>/dev/null || echo "0")
    echo "  [${i}/60] 进程: ${CLAP_DONE}, 已完成: ${CLAP_COUNT}/84"
    if [ "${CLAP_DONE}" = "0" ] && [ "${CLAP_COUNT}" -gt 0 ]; then
        echo "✅ CLAP 零样本标注完成"
        break
    fi
done

# 同步结果回本地
sync_from_gpu "clap"

CLAP_COUNT=$(find "${PROJECT_ROOT}/data/02_preannotation/l2_semantic" -name "*.json" | wc -l | tr -d ' ')
echo "CLAP语义标签: ${CLAP_COUNT} 首"
verify_output "CLAP数量" 80 "${CLAP_COUNT}"

# =============================================================================
# STEP 6: 数据划分（复用现有标记）
# =============================================================================
log_step "6" "数据划分（复用现有5首黄金集+3首Challenge标记）"
python3 scripts/04_dataset/split_dataset.py \
    --input "${MANIFEST}" \
    --output "${PROJECT_ROOT}/data/04_final_dataset/splits" \
    --train 0.80 --val 0.20 --test 0.0 --holdout 0.0 \
    --protect-golden \
    --strict \
    2>&1 | tee "${LOG_DIR}/06_split.log"

TRAIN_COUNT=$(wc -l < "${SPLITS_DIR}/train.csv" | tr -d ' ')
TRAIN_COUNT=$((TRAIN_COUNT - 1))
VAL_COUNT=$(wc -l < "${SPLITS_DIR}/val.csv" | tr -d ' ')
VAL_COUNT=$((VAL_COUNT - 1))
echo "train: ${TRAIN_COUNT} 首, val: ${VAL_COUNT} 首"

# 验证黄金集在train中
GOLDEN_IN_TRAIN=$(python3 -c "
import pandas as pd
m=pd.read_csv('${MANIFEST}')
t=pd.read_csv('${SPLITS_DIR}/train.csv')
golden=set(m[m['sample_type']=='golden_seed']['audio_id'])
print(len(golden & set(t['audio_id'])))
")
echo "黄金集在train中: ${GOLDEN_IN_TRAIN}/5"
if [ "${GOLDEN_IN_TRAIN}" -ne 5 ]; then
    log_error "黄金集未全部在train中"
    exit 1
fi

# =============================================================================
# STEP 7: L3 Qwen-Omni结构标注（黄金集5首）
# =============================================================================
log_step "7" "L3 Qwen-Omni结构标注（黄金集5首）"
echo "脚本自动从 .env 读取 DASHSCOPE_API_KEY 并预检有效性"

python3 scripts/02_preannotation/l3_structural/l3_qwen_audio_structure.py \
    --golden-manifest "${PROJECT_ROOT}/data/02_preannotation/l3_structural/golden_manifest.csv" \
    --master-dir "${PROJECT_ROOT}/data/01_preprocess/processed_master" \
    --output-dir "${PROJECT_ROOT}/data/02_preannotation/l3_structural" \
    --force \
    2>&1 | tee "${LOG_DIR}/07_l3_qwen.log"

L3_COUNT=$(find "${PROJECT_ROOT}/data/02_preannotation/l3_structural" -name "*.json" | wc -l | tr -d ' ')
echo "L3结构标注: ${L3_COUNT} 首"
verify_output "L3数量（黄金集5首）" 5 "${L3_COUNT}"

# =============================================================================
# STEP 8: L4 KNN传播
# =============================================================================
log_step "8" "L4 KNN传播（train拟合，val预测）"
python3 scripts/02_preannotation/l4_propagated/l4_knn_propagation.py \
    --manifest "${MANIFEST}" \
    --embedding-dir "${PROJECT_ROOT}/data/02_preannotation/l2_embedding" \
    --l3-dir "${PROJECT_ROOT}/data/02_preannotation/l3_structural" \
    --splits-dir "${SPLITS_DIR}" \
    --output-dir "${PROJECT_ROOT}/data/02_preannotation/l4_propagated" \
    2>&1 | tee "${LOG_DIR}/08_l4_knn.log"

L4_COUNT=$(find "${PROJECT_ROOT}/data/02_preannotation/l4_propagated" -name "*.json" | wc -l | tr -d ' ')
echo "L4传播结果: ${L4_COUNT} 首"

# =============================================================================
# STEP 9: Stage 5切片（排除黄金集和Challenge）
# =============================================================================
log_step "9" "Stage 5训练切片（排除黄金集和Challenge）"
python3 scripts/05_training_prep/01_audio_chunker.py \
    --manifest "${MANIFEST}" \
    --splits "${SPLITS_DIR}" \
    --only-train-val \
    --master-dir "${PROJECT_ROOT}/data/01_preprocess/processed_master" \
    --output-dir "${PROJECT_ROOT}/data/05_training_segments" \
    --chunk-sec 15 \
    2>&1 | tee "${LOG_DIR}/09_chunker.log"

# 验证切片数量
SEGMENT_COUNT=$(find "${PROJECT_ROOT}/data/05_training_segments" -name "*.wav" | wc -l | tr -d ' ')
echo "训练切片: ${SEGMENT_COUNT} 个"

# =============================================================================
# 完成
# =============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}端到端流水线执行完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo "日志目录: ${LOG_DIR}"
echo "各步骤输出:"
echo "  QC报告: ${QC_REPORT}"
echo "  母版: ${PROJECT_ROOT}/data/01_preprocess/processed_master/"
echo "  L1: ${PROJECT_ROOT}/data/02_preannotation/l1_physical/"
echo "  L2 MERT: ${PROJECT_ROOT}/data/02_preannotation/l2_embedding/"
echo "  L2 CLAP: ${PROJECT_ROOT}/data/02_preannotation/l2_semantic/"
echo "  划分: ${SPLITS_DIR}"
echo "  L3: ${PROJECT_ROOT}/data/02_preannotation/l3_structural/"
echo "  L4: ${PROJECT_ROOT}/data/02_preannotation/l4_propagated/"
echo "  Stage5切片: ${PROJECT_ROOT}/data/05_training_segments/"
