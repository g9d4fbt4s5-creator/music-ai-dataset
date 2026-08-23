"""
run_preannotation_pipeline.py
预标注主流水线（Branch D 增强版）

分层混合式预标注架构：
  L1 物理层（已有）：BPM/调性/SNR/语言
  L2 语义层（GPU）：CLAP zero-shot 分类（流派/情绪）
  L3 结构层（API）：
    - L3a: DeepSeek V4 Flash 文本标签提取（全量）
    - L3b: DeepSeek V4 Pro 疑难样本纠错（10%）
    - L3c: Qwen-Audio 真实音频结构（5%黄金集）
  L4 传播层（Mac）：KNN 传播 + 多源融合
  输出：ls_preannotations.jsonl

用法：
    # 完整流水线（L2→L3→L4）
    python run_preannotation_pipeline.py \
        --config configs/preannotation/preannotation_config.yaml \
        --stages l2,l3,l4

    # 只跑 L3 文本标签提取
    python run_preannotation_pipeline.py \
        --config configs/preannotation/preannotation_config.yaml \
        --stages l3_text

    # 只跑 L4 融合
    python run_preannotation_pipeline.py \
        --config configs/preannotation/preannotation_config.yaml \
        --stages l4

    # 试运行（不调用API）
    python run_preannotation_pipeline.py \
        --config configs/preannotation/preannotation_config.yaml \
        --dry-run
"""
import os
import sys
import json
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ===================== 工具函数 =====================

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_script(script_path: str, args: List[str], description: str,
                dry_run: bool = False) -> bool:
    """
    运行子脚本

    Args:
        script_path: 脚本路径
        args: 参数列表
        description: 描述
        dry_run: 试运行

    Returns:
        是否成功
    """
    cmd = [sys.executable, script_path] + args
    cmd_str = " ".join(cmd)

    if dry_run:
        logger.info(f"[试运行] {description}: {cmd_str}")
        return True

    logger.info(f"运行: {description}")
    logger.info(f"命令: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
        )

        if result.returncode == 0:
            logger.info(f"✅ {description} 完成")
            if result.stdout:
                logger.debug(result.stdout[-500:])
            return True
        else:
            logger.error(f"❌ {description} 失败 (returncode={result.returncode})")
            logger.error(f"stderr: {result.stderr[-1000:]}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"❌ {description} 超时")
        return False
    except Exception as e:
        logger.error(f"❌ {description} 异常: {e}")
        return False


# ===================== 各阶段执行 =====================

def run_l2_semantic(config: Dict, dry_run: bool = False) -> bool:
    """
    L2 语义层：CLAP zero-shot 分类（GPU端）

    注意：此阶段需要在 GPU 上运行，Mac 端只做调度
    """
    logger.info("=" * 60)
    logger.info("L2 语义层：CLAP zero-shot 分类")
    logger.info("=" * 60)

    l2_config = config["l2_semantic"]
    if not l2_config.get("enabled", True):
        logger.info("L2 语义层已禁用，跳过")
        return True

    # 检查是否在 GPU 环境
    device = l2_config.get("device", "cuda")
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("未检测到 CUDA，L2 语义层需要 GPU 环境")
                logger.warning("请在 GPU 服务器上运行此阶段")
                return False
        except ImportError:
            logger.warning("未安装 PyTorch，L2 语义层需要 GPU 环境")
            return False

    # 调用 CLAP zero-shot 脚本
    script_path = str(PROJECT_ROOT / "scripts/02_preannotation/l2_semantic/l2_clap_zero_shot.py")
    if not Path(script_path).exists():
        logger.warning(f"CLAP zero-shot 脚本不存在: {script_path}")
        logger.warning("请先实现 l2_clap_zero_shot.py，或使用 style_consistency_clustering.py 中的 CLAP 嵌入提取")
        return False

    args = [
        "--input-dir", l2_config.get("input", "data/01_preprocess/processed_master"),
        "--output", l2_config.get("output", "data/02_preannotation/l2_semantic"),
        "--embeddings-output", l2_config.get("embeddings_output", "data/02_preannotation/model_output_cache/clap_embeddings"),
        "--model", l2_config.get("model_name", "laion/clap-htsat-unfused"),
        "--batch-size", str(l2_config.get("batch_size", 16)),
        "--top-k", str(l2_config.get("top_k", 5)),
    ]

    return run_script(script_path, args, "L2 CLAP zero-shot 分类", dry_run)


def run_l3_text(config: Dict, dry_run: bool = False) -> bool:
    """L3a: DeepSeek V4 Flash 文本标签提取（全量）"""
    logger.info("=" * 60)
    logger.info("L3a: DeepSeek V4 Flash 文本标签提取")
    logger.info("=" * 60)

    l3_config = config["l3_structural"]["text_label_extraction"]
    if not l3_config.get("enabled", True):
        logger.info("L3a 文本标签提取已禁用，跳过")
        return True

    script_path = str(PROJECT_ROOT / "scripts/02_preannotation/l3_structural/l3_deepseek_label_extraction.py")

    args = [
        "--input-dir", "data/02_preannotation/l1_physical",
        "--l2-dir", "data/02_preannotation/l2_semantic",
        "--output", l3_config.get("output", "data/02_preannotation/l3_structural/text_labels"),
        "--config", str(PROJECT_ROOT / "configs/preannotation/preannotation_config.yaml"),
        "--mode", "extract",
    ]

    return run_script(script_path, args, "L3a DeepSeek 文本标签提取", dry_run)


def run_l3_correction(config: Dict, dry_run: bool = False) -> bool:
    """L3b: DeepSeek V4 Pro 疑难样本纠错（10%抽样）"""
    logger.info("=" * 60)
    logger.info("L3b: DeepSeek V4 Pro 疑难样本纠错")
    logger.info("=" * 60)

    l3_config = config["l3_structural"]["error_correction"]
    if not l3_config.get("enabled", True):
        logger.info("L3b 疑难纠错已禁用，跳过")
        return True

    script_path = str(PROJECT_ROOT / "scripts/02_preannotation/l3_structural/l3_deepseek_label_extraction.py")

    args = [
        "--input-dir", "data/02_preannotation/l3_structural/text_labels",
        "--output", l3_config.get("output", "data/02_preannotation/l3_structural/corrected_labels"),
        "--config", str(PROJECT_ROOT / "configs/preannotation/preannotation_config.yaml"),
        "--mode", "correction",
        "--correction-ratio", str(l3_config.get("sampling", {}).get("ratio", 0.10)),
    ]

    return run_script(script_path, args, "L3b DeepSeek Pro 疑难纠错", dry_run)


def run_l3_audio(config: Dict, dry_run: bool = False) -> bool:
    """L3c: Qwen-Audio 真实音频结构（5%黄金集）"""
    logger.info("=" * 60)
    logger.info("L3c: Qwen-Audio 真实音频结构")
    logger.info("=" * 60)

    l3_config = config["l3_structural"]["audio_structure"]
    if not l3_config.get("enabled", True):
        logger.info("L3c 音频结构已禁用，跳过")
        return True

    script_path = str(PROJECT_ROOT / "scripts/02_preannotation/l3_structural/l3_qwen_audio_structure.py")
    if not Path(script_path).exists():
        logger.warning(f"Qwen-Audio 脚本不存在: {script_path}")
        logger.warning("请先实现 l3_qwen_audio_structure.py")
        return False

    args = [
        "--input-dir", l3_config.get("input", "data/01_preprocess/processed_master"),
        "--output", l3_config.get("output", "data/02_preannotation/l3_structural/audio_structure"),
        "--config", str(PROJECT_ROOT / "configs/preannotation/preannotation_config.yaml"),
        "--sample-ratio", str(l3_config.get("sampling", {}).get("ratio", 0.05)),
    ]

    return run_script(script_path, args, "L3c Qwen-Audio 音频结构", dry_run)


def run_l4_propagation(config: Dict, dry_run: bool = False) -> bool:
    """L4: KNN 传播 + 多源融合"""
    logger.info("=" * 60)
    logger.info("L4: KNN 传播 + 多源融合")
    logger.info("=" * 60)

    l4_config = config["l4_propagated"]
    if not l4_config.get("enabled", True):
        logger.info("L4 传播层已禁用，跳过")
        return True

    script_path = str(PROJECT_ROOT / "scripts/02_preannotation/l4_propagated/l4_knn_propagation.py")

    args = [
        "--embeddings-dir", "data/02_preannotation/model_output_cache/clap_embeddings",
        "--l2-dir", "data/02_preannotation/l2_semantic",
        "--l3-text-dir", "data/02_preannotation/l3_structural/text_labels",
        "--l3-corrected-dir", "data/02_preannotation/l3_structural/corrected_labels",
        "--l3-audio-dir", "data/02_preannotation/l3_structural/audio_structure",
        "--output", l4_config.get("output", "data/02_preannotation/l4_propagated"),
        "--ls-output", l4_config.get("final_labels_output", "data/02_preannotation/ls_preannotations.jsonl"),
        "--config", str(PROJECT_ROOT / "configs/preannotation/preannotation_config.yaml"),
    ]

    return run_script(script_path, args, "L4 KNN 传播 + 多源融合", dry_run)


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(
        description="预标注主流水线（Branch D 增强版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str,
                        default="configs/preannotation/preannotation_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--stages", type=str, default="l2,l3_text,l3_correction,l3_audio,l4",
                        help="要执行的阶段，逗号分隔（l2,l3_text,l3_correction,l3_audio,l4）")
    parser.add_argument("--dry-run", action="store_true",
                        help="试运行（不实际执行，只打印命令）")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",")]

    logger.info("=" * 60)
    logger.info("预标注主流水线（Branch D 增强版）")
    logger.info("=" * 60)
    logger.info(f"执行阶段: {stages}")
    logger.info(f"试运行: {args.dry_run}")
    logger.info(f"开始时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # 阶段执行映射
    stage_map = {
        "l2": run_l2_semantic,
        "l3_text": run_l3_text,
        "l3_correction": run_l3_correction,
        "l3_audio": run_l3_audio,
        "l4": run_l4_propagation,
    }

    # 执行各阶段
    results = {}
    for stage in stages:
        if stage in stage_map:
            success = stage_map[stage](config, args.dry_run)
            results[stage] = success
            if not success and not args.dry_run:
                logger.error(f"阶段 {stage} 失败，是否继续？")
                # 继续执行后续阶段（非致命错误）
        else:
            logger.warning(f"未知阶段: {stage}")

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("流水线执行汇总")
    logger.info("=" * 60)
    for stage, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"  {stage}: {status}")
    logger.info(f"结束时间: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # 成本估算
    cost_config = config.get("cost_estimation", {})
    if cost_config:
        logger.info("\n成本估算（500首）:")
        logger.info(f"  CLAP zero-shot: ¥{cost_config.get('clap_zero_shot', 0)}")
        logger.info(f"  DeepSeek Flash: ¥{cost_config.get('deepseek_flash_500', 0)}")
        logger.info(f"  DeepSeek Pro: ¥{cost_config.get('deepseek_pro_50', 0)}")
        logger.info(f"  Qwen-Audio: ¥{cost_config.get('qwen_audio_25', 0)}")
        logger.info(f"  总计: ¥{cost_config.get('total_estimated', 0)}")
        logger.info(f"  预计耗时: {cost_config.get('total_time_minutes', 0)} 分钟")


if __name__ == "__main__":
    main()
