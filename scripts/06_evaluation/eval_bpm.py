"""
eval_bpm.py
BPM 检测评测脚本（mir_eval 标准评测库）

功能：
- 用 essentia / madmom / librosa 提取 BPM
- 用 mir_eval.beat 计算标准评测指标（F1/Cemgil/Goto/P-score）
- 支持公开数据集真值（JCS/Ballroom/SMC/GTZAN）
- 支持工具间一致性分析（无真值时，找出"工具们打起来"的样本）
- 输出评测报告（CSV/JSON/HTML）

真值来源：
1. 公开数据集自带标注（JCS/Ballroom/SMC 最适合爵士）
2. MIDI 文件导出的时间戳（Lakh MIDI/MAESTRO）
3. 工具分歧仲裁（essentia vs madmom 差异 >5% → 人工仲裁候选）

评测指标（mir_eval.beat）：
- F1-score: 节拍检测的F1分数
- Cemgil: Cemgil et al. 2007 的评测指标
- Goto: Goto and Muraoka 1997 的评测指标
- P-score: McKinney et al. 2007 的评测指标
- BPM 误差: 预测BPM与真值BPM的绝对误差和相对误差

用法：
    # 用公开数据集评测（需要下载数据集）
    python eval_bpm.py --audio-dir data/datasets/ballroom --truth-dir data/datasets/ballroom/truth --tools essentia,madmom

    # 工具间一致性分析（无真值）
    python eval_bpm.py --audio-dir data/00_raw_collect/raw_audio --tools essentia,madmom,librosa --consistency-only

    # 只评测单个工具
    python eval_bpm.py --audio-dir data/datasets/jcs --truth-dir data/datasets/jcs/truth --tools madmom

    # 输出HTML报告
    python eval_bpm.py --audio-dir data/datasets/ballroom --truth-dir data/datasets/ballroom/truth --report-html report.html

    # 高分歧样本导出（人工仲裁候选）
    python eval_bpm.py --audio-dir data/00_raw_collect/raw_audio --tools essentia,madmom --consistency-only --high-disagreement-threshold 5 --export-disagreement disagreement.csv
"""
import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

# mir_eval 标准评测库
try:
    import mir_eval
    MIR_EVAL_AVAILABLE = True
except ImportError:
    MIR_EVAL_AVAILABLE = False
    logging.warning("mir_eval 未安装，部分评测指标不可用。安装: pip install mir_eval")

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 支持的BPM提取工具
SUPPORTED_TOOLS = ["essentia", "madmom", "librosa"]


# ===================== BPM 提取函数 =====================
def extract_bpm_librosa(audio_path: str) -> Optional[float]:
    """用 librosa 提取 BPM"""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception as e:
        logger.warning(f"librosa BPM提取失败: {audio_path} - {e}")
        return None


def extract_bpm_essentia(audio_path: str) -> Optional[float]:
    """用 essentia 提取 BPM（两种算法：RhythmExtractor2013 + PercivalBpmEstimator）"""
    try:
        import essentia.standard as es

        # 加载音频
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()

        # 方法1: RhythmExtractor2013（默认）
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)

        # 方法2: PercivalBpmEstimator（更鲁棒）
        percival = es.PercivalBpmEstimator()
        bpm_percival = percival(audio)

        # 取置信度高的结果
        if beats_confidence > 0.5:
            return float(bpm)
        else:
            return float(bpm_percival)
    except Exception as e:
        logger.warning(f"essentia BPM提取失败: {audio_path} - {e}")
        return None


def extract_bpm_madmom(audio_path: str) -> Optional[float]:
    """用 madmom 提取 BPM（深度学习，最准确）"""
    try:
        from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
        from madmom.features.tempo import TempoEstimationProcessor

        # 方法1: RNN + TempoEstimation
        proc = RNNBeatProcessor()
        act = proc(audio_path)

        tempo_proc = TempoEstimationProcessor(fps=100)
        tempi = tempo_proc(act)

        # 取最可能的BPM
        if len(tempi) > 0:
            bpm = float(tempi[0][0])
            return bpm

        # 方法2: BeatTracking + 从beats计算BPM
        beat_proc = BeatTrackingProcessor(fps=100)
        beats = beat_proc(act)

        if len(beats) > 1:
            # 从beat间隔计算BPM
            intervals = np.diff(beats)
            median_interval = np.median(intervals)
            bpm = 60.0 / median_interval if median_interval > 0 else None
            return float(bpm) if bpm else None

        return None
    except Exception as e:
        logger.warning(f"madmom BPM提取失败: {audio_path} - {e}")
        return None


def extract_bpm(audio_path: str, tool: str) -> Optional[float]:
    """统一BPM提取接口"""
    if tool == "librosa":
        return extract_bpm_librosa(audio_path)
    elif tool == "essentia":
        return extract_bpm_essentia(audio_path)
    elif tool == "madmom":
        return extract_bpm_madmom(audio_path)
    else:
        logger.error(f"不支持的工具: {tool}")
        return None


# ===================== 真值加载 =====================
def load_truth_jcs(truth_dir: Path) -> Dict[str, float]:
    """
    加载 JCS (Jazz-Choro-Samba) 数据集的 BPM 真值

    JCS 数据集格式：
    - 音频文件: *.wav
    - 真值文件: *.bpm（纯文本，每行一个BPM值）
    - 或 *.beats（beat时间戳，每行一个时间戳）
    """
    truth = {}
    for bpm_file in truth_dir.glob("*.bpm"):
        audio_id = bpm_file.stem
        try:
            with open(bpm_file, "r") as f:
                bpm = float(f.read().strip())
                truth[audio_id] = bpm
        except Exception as e:
            logger.warning(f"加载JCS真值失败: {bpm_file} - {e}")

    # 如果没有 .bpm 文件，尝试从 .beats 文件计算
    if not truth:
        for beats_file in truth_dir.glob("*.beats"):
            audio_id = beats_file.stem
            try:
                beats = np.loadtxt(str(beats_file))
                if len(beats) > 1:
                    intervals = np.diff(beats)
                    median_interval = np.median(intervals)
                    bpm = 60.0 / median_interval if median_interval > 0 else None
                    if bpm:
                        truth[audio_id] = float(bpm)
            except Exception as e:
                logger.warning(f"从beats计算BPM失败: {beats_file} - {e}")

    logger.info(f"加载JCS真值: {len(truth)} 首")
    return truth


def load_truth_ballroom(truth_dir: Path) -> Dict[str, float]:
    """
    加载 Ballroom 数据集的 BPM 真值

    Ballroom 数据集格式：
    - 音频文件: *.wav（按流派分子目录）
    - 真值文件: *.bpm（纯文本）
    - 或 evals/ 目录下的 .bpm 文件
    """
    truth = {}

    # 递归查找所有 .bpm 文件
    for bpm_file in truth_dir.rglob("*.bpm"):
        audio_id = bpm_file.stem
        try:
            with open(bpm_file, "r") as f:
                content = f.read().strip()
                # Ballroom 的 .bpm 文件可能包含多个BPM值（主BPM + 备选）
                lines = content.split("\n")
                bpm = float(lines[0].strip())
                truth[audio_id] = bpm
        except Exception as e:
            logger.warning(f"加载Ballroom真值失败: {bpm_file} - {e}")

    logger.info(f"加载Ballroom真值: {len(truth)} 首")
    return truth


def load_truth_smc(truth_dir: Path) -> Dict[str, float]:
    """
    加载 SMC MIREX 数据集的 beat 真值

    SMC 数据集格式：
    - 音频文件: SMC_*.wav
    - 真值文件: SMC_*.beats（每行一个beat时间戳）
    """
    truth = {}
    for beats_file in truth_dir.glob("*.beats"):
        audio_id = beats_file.stem
        try:
            beats = np.loadtxt(str(beats_file))
            if len(beats) > 1:
                intervals = np.diff(beats)
                median_interval = np.median(intervals)
                bpm = 60.0 / median_interval if median_interval > 0 else None
                if bpm:
                    truth[audio_id] = float(bpm)
        except Exception as e:
            logger.warning(f"加载SMC真值失败: {beats_file} - {e}")

    logger.info(f"加载SMC真值: {len(truth)} 首")
    return truth


def load_truth(truth_dir: Path, dataset_type: str = "auto") -> Dict[str, float]:
    """
    统一真值加载接口

    Args:
        truth_dir: 真值目录
        dataset_type: 数据集类型（jcs/ballroom/smc/auto）

    Returns:
        {audio_id: bpm} 字典
    """
    if dataset_type == "auto":
        # 自动检测数据集类型
        if list(truth_dir.glob("*.bpm")):
            return load_truth_ballroom(truth_dir)
        elif list(truth_dir.glob("*.beats")):
            return load_truth_smc(truth_dir)
        else:
            logger.warning(f"无法自动检测数据集类型: {truth_dir}")
            return {}
    elif dataset_type == "jcs":
        return load_truth_jcs(truth_dir)
    elif dataset_type == "ballroom":
        return load_truth_ballroom(truth_dir)
    elif dataset_type == "smc":
        return load_truth_smc(truth_dir)
    else:
        logger.error(f"不支持的数据集类型: {dataset_type}")
        return {}


# ===================== mir_eval 评测 =====================
def evaluate_bpm_mir_eval(predicted_bpm: float, truth_bpm: float,
                           audio_path: Optional[str] = None) -> Dict[str, float]:
    """
    用 mir_eval 计算 BPM 检测评测指标

    注意：mir_eval.beat 需要 beat 时间戳，而不是 BPM 值。
    这里我们从 BPM 值生成等间隔的 beat 时间戳，然后计算指标。

    Args:
        predicted_bpm: 预测的 BPM
        truth_bpm: 真值 BPM
        audio_path: 音频路径（用于获取时长）

    Returns:
        评测指标字典
    """
    results = {
        "predicted_bpm": predicted_bpm,
        "truth_bpm": truth_bpm,
        "bpm_absolute_error": abs(predicted_bpm - truth_bpm),
        "bpm_relative_error": abs(predicted_bpm - truth_bpm) / truth_bpm if truth_bpm > 0 else 0,
        "bpm_correct_1x": abs(predicted_bpm - truth_bpm) < 0.05 * truth_bpm,  # ±5%
        "bpm_correct_2x": abs(predicted_bpm - 2 * truth_bpm) < 0.05 * truth_bpm or
                          abs(predicted_bpm - 0.5 * truth_bpm) < 0.05 * truth_bpm,  # 倍频/半频
    }

    # 如果 mir_eval 可用，且有音频路径，计算 beat 级指标
    if MIR_EVAL_AVAILABLE and audio_path and predicted_bpm > 0 and truth_bpm > 0:
        try:
            import librosa
            # 获取音频时长
            duration = librosa.get_duration(filename=audio_path)

            # 从 BPM 生成等间隔 beat 时间戳
            def bpm_to_beats(bpm: float, duration: float) -> np.ndarray:
                interval = 60.0 / bpm
                beats = np.arange(0, duration, interval)
                return beats

            pred_beats = bpm_to_beats(predicted_bpm, duration)
            ref_beats = bpm_to_beats(truth_bpm, duration)

            # mir_eval.beat 评测
            # 注意：mir_eval.beat 需要参考节拍和估计节拍
            # 这里我们用简化的方式计算
            f1_score = mir_eval.beat.f_measure(ref_beats, pred_beats)
            cemgil_score = mir_eval.beat.cemgil(ref_beats, pred_beats)
            p_score = mir_eval.beat.p_score(ref_beats, pred_beats)
            goto_score = mir_eval.beat.goto(ref_beats, pred_beats)

            results["f1_score"] = float(f1_score)
            results["cemgil_score"] = float(cemgil_score[0]) if isinstance(cemgil_score, tuple) else float(cemgil_score)
            results["p_score"] = float(p_score)
            results["goto_score"] = float(goto_score)

        except Exception as e:
            logger.warning(f"mir_eval评测失败: {audio_path} - {e}")

    return results


# ===================== 工具间一致性分析 =====================
def analyze_consistency(results: Dict[str, List[Dict]],
                        high_disagreement_threshold: float = 5.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    工具间一致性分析（无真值时）

    Args:
        results: {tool: [{"audio_id": ..., "bpm": ...}, ...]}
        high_disagreement_threshold: 高分歧阈值（BPM绝对误差）

    Returns:
        (consistency_df, high_disagreement_df)
    """
    # 构建 {audio_id: {tool: bpm}} 字典
    audio_bpms = defaultdict(dict)
    for tool, tool_results in results.items():
        for r in tool_results:
            audio_id = r["audio_id"]
            bpm = r["bpm"]
            if bpm is not None:
                audio_bpms[audio_id][tool] = bpm

    # 计算一致性指标
    consistency_rows = []
    high_disagreement_rows = []

    for audio_id, tool_bpms in audio_bpms.items():
        tools = list(tool_bpms.keys())
        bpms = list(tool_bpms.values())

        if len(bpms) < 2:
            continue

        row = {
            "audio_id": audio_id,
            "n_tools": len(tools),
            "tools": ",".join(tools),
            "mean_bpm": np.mean(bpms),
            "std_bpm": np.std(bpms),
            "min_bpm": np.min(bpms),
            "max_bpm": np.max(bpms),
            "range_bpm": np.max(bpms) - np.min(bpms),
        }

        # 两两对比
        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                diff = abs(tool_bpms[tools[i]] - tool_bpms[tools[j]])
                row[f"diff_{tools[i]}_{tools[j]}"] = diff

        consistency_rows.append(row)

        # 高分歧样本
        if row["range_bpm"] > high_disagreement_threshold:
            high_disagreement_rows.append(row)

    consistency_df = pd.DataFrame(consistency_rows)
    high_disagreement_df = pd.DataFrame(high_disagreement_rows)

    if not consistency_df.empty:
        logger.info(f"一致性分析: {len(consistency_df)} 首, "
                    f"平均分歧: {consistency_df['range_bpm'].mean():.2f} BPM, "
                    f"高分歧(>{high_disagreement_threshold}): {len(high_disagreement_df)} 首")

    return consistency_df, high_disagreement_df


# ===================== 主评测流程 =====================
def run_evaluation(audio_dir: Path, truth_dir: Optional[Path] = None,
                   tools: List[str] = None, dataset_type: str = "auto",
                   consistency_only: bool = False,
                   limit: Optional[int] = None) -> Tuple[Dict, pd.DataFrame]:
    """
    运行 BPM 评测

    Args:
        audio_dir: 音频目录
        truth_dir: 真值目录（None表示无真值，只做一致性分析）
        tools: 要评测的工具列表
        dataset_type: 数据集类型
        consistency_only: 只做一致性分析
        limit: 限制处理数量

    Returns:
        (results_dict, results_df)
    """
    if tools is None:
        tools = ["essentia", "madmom"]

    # 加载真值
    truth = {}
    if truth_dir and truth_dir.exists() and not consistency_only:
        truth = load_truth(truth_dir, dataset_type)

    # 查找音频文件
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
        audio_files.extend(audio_dir.rglob(ext))

    # 去重
    audio_files = list(set(audio_files))
    if limit:
        audio_files = audio_files[:limit]

    logger.info(f"找到 {len(audio_files)} 个音频文件")
    logger.info(f"评测工具: {tools}")
    if truth:
        logger.info(f"真值: {len(truth)} 首")
    else:
        logger.info("无真值，将做工具间一致性分析")

    # 提取 BPM
    results = {tool: [] for tool in tools}
    eval_results = []

    for i, audio_path in enumerate(audio_files):
        audio_id = audio_path.stem
        logger.info(f"[{i+1}/{len(audio_files)}] 处理: {audio_id}")

        # 用每个工具提取 BPM
        tool_bpms = {}
        for tool in tools:
            bpm = extract_bpm(str(audio_path), tool)
            tool_bpms[tool] = bpm
            results[tool].append({
                "audio_id": audio_id,
                "audio_path": str(audio_path),
                "bpm": bpm,
                "tool": tool,
            })

        # 如果有真值，计算评测指标
        if truth and audio_id in truth:
            truth_bpm = truth[audio_id]
            for tool, pred_bpm in tool_bpms.items():
                if pred_bpm is not None:
                    eval_result = evaluate_bpm_mir_eval(
                        predicted_bpm=pred_bpm,
                        truth_bpm=truth_bpm,
                        audio_path=str(audio_path)
                    )
                    eval_result["audio_id"] = audio_id
                    eval_result["tool"] = tool
                    eval_results.append(eval_result)

    # 汇总结果
    results_df = pd.DataFrame(eval_results) if eval_results else pd.DataFrame()

    summary = {
        "n_audio_files": len(audio_files),
        "tools": tools,
        "n_truth": len(truth),
        "n_evaluated": len(eval_results),
    }

    # 如果有评测结果，计算汇总指标
    if not results_df.empty:
        for tool in tools:
            tool_df = results_df[results_df["tool"] == tool]
            if not tool_df.empty:
                summary[f"{tool}_mean_abs_error"] = float(tool_df["bpm_absolute_error"].mean())
                summary[f"{tool}_mean_rel_error"] = float(tool_df["bpm_relative_error"].mean())
                summary[f"{tool}_accuracy_1x"] = float(tool_df["bpm_correct_1x"].mean())
                summary[f"{tool}_accuracy_2x"] = float(tool_df["bpm_correct_2x"].mean())
                if "f1_score" in tool_df.columns:
                    summary[f"{tool}_f1_score"] = float(tool_df["f1_score"].mean())

    # 如果无真值或只做一致性分析，计算工具间一致性
    if consistency_only or not truth:
        consistency_df, high_disagreement_df = analyze_consistency(results)
        summary["consistency"] = {
            "n_samples": len(consistency_df),
            "mean_range_bpm": float(consistency_df["range_bpm"].mean()) if not consistency_df.empty else 0,
            "n_high_disagreement": len(high_disagreement_df),
        }
        results_df = consistency_df if results_df.empty else results_df

    return summary, results_df


# ===================== 报告生成 =====================
def generate_report(summary: Dict, results_df: pd.DataFrame,
                    output_path: Path, format: str = "json"):
    """生成评测报告"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    elif format == "csv":
        results_df.to_csv(output_path, index=False, encoding="utf-8")
    elif format == "html":
        # 简单HTML报告
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BPM 检测评测报告</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a2e; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #16213e; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .metric {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>BPM 检测评测报告</h1>
    <div class="metric">
        <h2>摘要</h2>
        <p>音频文件数: {summary.get('n_audio_files', 0)}</p>
        <p>评测工具: {', '.join(summary.get('tools', []))}</p>
        <p>真值数量: {summary.get('n_truth', 0)}</p>
        <p>已评测: {summary.get('n_evaluated', 0)}</p>
    </div>
    <h2>详细结果</h2>
    {results_df.to_html(index=False) if not results_df.empty else '<p>无详细结果</p>'}
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

    logger.info(f"报告已保存: {output_path}")


# ===================== CLI 入口 =====================
def main():
    parser = argparse.ArgumentParser(
        description="BPM 检测评测脚本（mir_eval 标准评测库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="音频目录")
    parser.add_argument("--truth-dir", type=str, default=None,
                        help="真值目录（None表示无真值，只做一致性分析）")
    parser.add_argument("--tools", type=str, default="essentia,madmom",
                        help="评测工具（逗号分隔，如 essentia,madmom,librosa）")
    parser.add_argument("--dataset-type", type=str, default="auto",
                        choices=["auto", "jcs", "ballroom", "smc"],
                        help="数据集类型（默认自动检测）")
    parser.add_argument("--consistency-only", action="store_true",
                        help="只做工具间一致性分析（不加载真值）")
    parser.add_argument("--high-disagreement-threshold", type=float, default=5.0,
                        help="高分歧阈值（BPM绝对误差，默认5）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    parser.add_argument("--report-json", type=str, default=None,
                        help="输出JSON报告路径")
    parser.add_argument("--report-csv", type=str, default=None,
                        help="输出CSV报告路径")
    parser.add_argument("--report-html", type=str, default=None,
                        help="输出HTML报告路径")
    parser.add_argument("--export-disagreement", type=str, default=None,
                        help="导出高分歧样本（人工仲裁候选）CSV路径")
    args = parser.parse_args()

    # 解析工具列表
    tools = [t.strip() for t in args.tools.split(",") if t.strip() in SUPPORTED_TOOLS]
    if not tools:
        logger.error(f"没有有效的工具。支持的工具: {SUPPORTED_TOOLS}")
        sys.exit(1)

    # 运行评测
    audio_dir = Path(args.audio_dir)
    truth_dir = Path(args.truth_dir) if args.truth_dir else None

    summary, results_df = run_evaluation(
        audio_dir=audio_dir,
        truth_dir=truth_dir,
        tools=tools,
        dataset_type=args.dataset_type,
        consistency_only=args.consistency_only,
        limit=args.limit,
    )

    # 打印摘要
    logger.info("")
    logger.info("=" * 60)
    logger.info("评测摘要")
    logger.info("=" * 60)
    for k, v in summary.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for k2, v2 in v.items():
                logger.info(f"    {k2}: {v2}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # 生成报告
    if args.report_json:
        generate_report(summary, results_df, Path(args.report_json), format="json")
    if args.report_csv:
        generate_report(summary, results_df, Path(args.report_csv), format="csv")
    if args.report_html:
        generate_report(summary, results_df, Path(args.report_html), format="html")

    # 导出高分歧样本
    if args.export_disagreement and "consistency" in summary:
        # 重新运行一致性分析获取高分歧样本
        # （简化：这里假设results_df包含一致性信息）
        if not results_df.empty and "range_bpm" in results_df.columns:
            high_disagreement = results_df[results_df["range_bpm"] > args.high_disagreement_threshold]
            high_disagreement.to_csv(args.export_disagreement, index=False, encoding="utf-8")
            logger.info(f"高分歧样本已导出: {args.export_disagreement} ({len(high_disagreement)} 首)")


if __name__ == "__main__":
    main()
