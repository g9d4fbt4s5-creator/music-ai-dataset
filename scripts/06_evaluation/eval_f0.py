"""
eval_f0.py
基频（F0）检测评测脚本

用 mir_eval.melody 标准指标评测基频检测工具：
- torchcrepe（深度学习，CREPE模型）
- librosa.pyin（概率性YIN算法）

支持的真值数据集：
- GuitarSet：吉他独奏，有f0标注（6根弦的pitch contour）
- MedleyDB：多轨音乐，有melody标注
- MIR-1K：中文歌曲，有f0标注
- auto：自动检测

无真值时：工具间一致性分析（torchcrepe vs librosa.pyin）

评测指标（mir_eval.melody）：
- RPA (Raw Pitch Accuracy)：原始音高准确率
- RCA (Raw Chroma Accuracy)：原始音高类准确率（八度无关）
- Overall Accuracy：整体准确率（考虑voicing）
- Voicing Recall：浊音召回率
- Voicing False Alarm：浊音误报率

用法：
    # 有真值评测（GuitarSet）
    python eval_f0.py \
        --audio-dir data/datasets/guitarset/audio \
        --truth-dir data/datasets/guitarset/annotations \
        --dataset-type guitarset \
        --tools torchcrepe,librosa \
        --report-json reports/f0_eval_guitarset.json \
        --report-html reports/f0_eval_guitarset.html

    # 工具间一致性分析（无真值）
    python eval_f0.py \
        --audio-dir data/01_preprocess/processed_master \
        --tools torchcrepe,librosa \
        --consistency-only \
        --report-csv reports/f0_consistency.csv
"""
import os
import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

# 抑制警告
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ===================== 工具函数 =====================

def find_audio_files(audio_dir: Path) -> List[Path]:
    """递归查找音频文件"""
    audio_extensions = [".mp3", ".wav", ".flac", ".ogg", ".m4a"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(audio_files)


def load_audio(audio_path: Path, sr: int = 22050, mono: bool = True):
    """加载音频文件"""
    import librosa
    y, sr = librosa.load(str(audio_path), sr=sr, mono=mono)
    return y, sr


# ===================== F0 提取工具 =====================

def extract_f0_torchcrepe(audio_path: Path, sr: int = 22050,
                           model: str = "small",
                           hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 torchcrepe 提取基频

    Args:
        audio_path: 音频文件路径
        sr: 采样率
        model: 模型大小（tiny/small/medium/large/full）
        hop_length: 帧移

    Returns:
        (times, freqs) — 时间数组和频率数组（Hz，0表示无声）
    """
    import torch
    import crepe

    # 加载音频（crepe 需要 16kHz）
    import librosa
    y, sr_orig = librosa.load(str(audio_path), sr=16000, mono=True)

    # 用 torchcrepe 预测
    with torch.no_grad():
        # crepe.predict 返回 (times, freqs, confidence, activation)
        result = crepe.predict(y, sr_orig, model_capacity=model,
                                hop_length=hop_length, center=True,
                                verbose=0)

    times = result[0]  # 时间（秒）
    freqs = result[1]  # 频率（Hz）
    confidence = result[2]  # 置信度

    # 低置信度的帧设为0（无声）
    freqs[confidence < 0.5] = 0

    return times, freqs


def extract_f0_librosa(audio_path: Path, sr: int = 22050,
                        hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    用 librosa.pyin 提取基频

    Args:
        audio_path: 音频文件路径
        sr: 采样率
        hop_length: 帧移

    Returns:
        (times, freqs) — 时间数组和频率数组（Hz，0表示无声）
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # pyin 返回 (f0, voiced_flag, voiced_probs)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr, hop_length=hop_length
    )

    # 生成时间数组
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # 未浊音的帧设为0
    freqs = f0.copy()
    freqs[~voiced_flag] = 0
    freqs = np.nan_to_num(freqs, nan=0.0)

    return times, freqs


def extract_f0(audio_path: Path, tool: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一的 F0 提取接口

    Args:
        audio_path: 音频文件路径
        tool: 工具名称（torchcrepe/librosa）

    Returns:
        (times, freqs)
    """
    if tool == "torchcrepe":
        return extract_f0_torchcrepe(audio_path, **kwargs)
    elif tool == "librosa":
        return extract_f0_librosa(audio_path, **kwargs)
    else:
        raise ValueError(f"不支持的工具: {tool}")


# ===================== 真值加载 =====================

def load_truth_guitarset(truth_dir: Path, audio_stem: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    加载 GuitarSet 的 f0 真值

    GuitarSet 标注格式：
    - 每首歌有6个弦的 f0 标注（CSV格式，列：time, frequency）
    - melody 通常取最高音弦的 f0

    Args:
        truth_dir: 标注目录
        audio_stem: 音频文件名（不含扩展名）

    Returns:
        (times, freqs) 或 None
    """
    # 尝试多种命名方式
    possible_files = [
        truth_dir / f"{audio_stem}_melody.csv",
        truth_dir / f"{audio_stem}_pitch.csv",
        truth_dir / f"{audio_stem}.csv",
    ]

    # 递归查找
    for pattern in [f"*{audio_stem}*melody*.csv", f"*{audio_stem}*pitch*.csv", f"*{audio_stem}*.csv"]:
        possible_files.extend(truth_dir.rglob(pattern))

    for truth_file in possible_files:
        if truth_file.exists():
            try:
                df = pd.read_csv(truth_file)
                # 尝试常见列名
                time_col = None
                freq_col = None
                for col in df.columns:
                    if col.lower() in ["time", "timestamp", "t", "seconds"]:
                        time_col = col
                    elif col.lower() in ["frequency", "freq", "f0", "pitch", "hz"]:
                        freq_col = col

                if time_col and freq_col:
                    times = df[time_col].values
                    freqs = df[freq_col].values
                    freqs = np.nan_to_num(freqs, nan=0.0)
                    return times, freqs
            except Exception as e:
                logger.warning(f"加载真值失败 {truth_file}: {e}")
                continue

    return None


def load_truth(truth_dir: Path, audio_path: Path,
               dataset_type: str = "auto") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    加载 f0 真值（统一接口）

    Args:
        truth_dir: 真值目录
        audio_path: 音频文件路径
        dataset_type: 数据集类型（auto/guitarset/medleydb/mir1k）

    Returns:
        (times, freqs) 或 None
    """
    audio_stem = audio_path.stem

    if dataset_type == "auto":
        # 自动检测：尝试所有格式
        for ds_type in ["guitarset", "medleydb", "mir1k"]:
            result = load_truth(truth_dir, audio_path, ds_type)
            if result is not None:
                return result
        return None

    elif dataset_type == "guitarset":
        return load_truth_guitarset(truth_dir, audio_stem)

    elif dataset_type in ["medleydb", "mir1k"]:
        # 通用 CSV 加载
        return load_truth_guitarset(truth_dir, audio_stem)  # 复用通用加载

    else:
        logger.warning(f"不支持的数据集类型: {dataset_type}")
        return None


# ===================== 评测指标 =====================

def evaluate_f0(ref_times: np.ndarray, ref_freqs: np.ndarray,
                est_times: np.ndarray, est_freqs: np.ndarray) -> Dict:
    """
    用 mir_eval.melody 计算 f0 评测指标

    Args:
        ref_times: 真值时间数组
        ref_freqs: 真值频率数组（0表示无声）
        est_times: 估计时间数组
        est_freqs: 估计频率数组（0表示无声）

    Returns:
        指标字典
    """
    import mir_eval

    # mir_eval.melody.evaluate 需要 (ref_time, ref_freq, est_time, est_freq)
    try:
        metrics = mir_eval.melody.evaluate(
            ref_times, ref_freqs, est_times, est_freqs
        )
        return metrics
    except Exception as e:
        logger.warning(f"mir_eval 评测失败: {e}")
        return {}


def compute_f0_similarity(freqs1: np.ndarray, freqs2: np.ndarray,
                          times1: np.ndarray, times2: np.ndarray) -> Dict:
    """
    计算两个 f0 序列的相似度（无真值时的工具间一致性分析）

    Args:
        freqs1: 工具1的频率数组
        freqs2: 工具2的频率数组
        times1: 工具1的时间数组
        times2: 工具2的时间数组

    Returns:
        相似度指标字典
    """
    # 重采样到相同时间轴
    min_len = min(len(freqs1), len(freqs2))
    freqs1_resampled = freqs1[:min_len]
    freqs2_resampled = freqs2[:min_len]

    # 只比较都有声音的帧
    voiced_mask = (freqs1_resampled > 0) & (freqs2_resampled > 0)
    n_voiced = np.sum(voiced_mask)

    if n_voiced == 0:
        return {
            "n_voiced_frames": 0,
            "mean_abs_error_cents": 0,
            "median_abs_error_cents": 0,
            "chroma_accuracy": 0,
            "voicing_overlap": 0,
        }

    f1_voiced = freqs1_resampled[voiced_mask]
    f2_voiced = freqs2_resampled[voiced_mask]

    # 转换为音分（cents）
    cents_diff = 1200 * np.abs(np.log2(f1_voiced / f2_voiced + 1e-10))

    # 音高类准确率（八度无关，±50音分）
    chroma_diff = cents_diff % 1200
    chroma_diff = np.minimum(chroma_diff, 1200 - chroma_diff)
    chroma_accuracy = np.mean(chroma_diff < 50)

    # 浊音重叠率
    voiced1 = freqs1_resampled > 0
    voiced2 = freqs2_resampled > 0
    voicing_overlap = np.sum(voiced1 & voiced2) / (np.sum(voiced1 | voiced2) + 1e-10)

    return {
        "n_voiced_frames": int(n_voiced),
        "mean_abs_error_cents": float(np.mean(cents_diff)),
        "median_abs_error_cents": float(np.median(cents_diff)),
        "chroma_accuracy": float(chroma_accuracy),
        "voicing_overlap": float(voicing_overlap),
    }


# ===================== 主评测流程 =====================

def run_evaluation(audio_dir: Path, truth_dir: Optional[Path],
                   tools: List[str], dataset_type: str = "auto",
                   consistency_only: bool = False,
                   limit: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    """
    运行 f0 评测

    Args:
        audio_dir: 音频目录
        truth_dir: 真值目录（可选）
        tools: 工具列表
        dataset_type: 数据集类型
        consistency_only: 只做一致性分析
        limit: 限制处理数量

    Returns:
        (详细结果列表, 汇总统计)
    """
    audio_files = find_audio_files(audio_dir)
    if limit:
        audio_files = audio_files[:limit]

    logger.info(f"找到 {len(audio_files)} 个音频文件")
    logger.info(f"评测工具: {tools}")

    if truth_dir and not consistency_only:
        logger.info(f"真值目录: {truth_dir}")
        logger.info(f"数据集类型: {dataset_type}")
    else:
        logger.info("无真值，将做工具间一致性分析")

    results = []
    consistency_results = []

    for i, audio_path in enumerate(audio_files):
        logger.info(f"[{i+1}/{len(audio_files)}] 处理: {audio_path.name}")

        track_result = {
            "audio_id": audio_path.stem,
            "audio_path": str(audio_path),
        }

        # 提取所有工具的 f0
        f0_data = {}
        for tool in tools:
            try:
                times, freqs = extract_f0(audio_path, tool)
                f0_data[tool] = {"times": times, "freqs": freqs}
                track_result[f"f0_{tool}_mean_hz"] = float(np.mean(freqs[freqs > 0])) if np.any(freqs > 0) else 0
                track_result[f"f0_{tool}_voiced_ratio"] = float(np.mean(freqs > 0))
            except Exception as e:
                logger.warning(f"  {tool} 提取失败: {e}")
                f0_data[tool] = None
                track_result[f"f0_{tool}_error"] = str(e)

        # 有真值评测
        if truth_dir and not consistency_only:
            truth = load_truth(truth_dir, audio_path, dataset_type)
            if truth is not None:
                ref_times, ref_freqs = truth
                track_result["has_truth"] = True

                for tool in tools:
                    if f0_data[tool] is not None:
                        est_times = f0_data[tool]["times"]
                        est_freqs = f0_data[tool]["freqs"]
                        metrics = evaluate_f0(ref_times, ref_freqs, est_times, est_freqs)
                        for k, v in metrics.items():
                            track_result[f"{tool}_{k}"] = float(v) if isinstance(v, (int, float, np.floating)) else v
            else:
                track_result["has_truth"] = False
                logger.warning(f"  未找到真值: {audio_path.name}")

        # 工具间一致性分析
        if len(tools) >= 2:
            for i_tool in range(len(tools)):
                for j_tool in range(i_tool + 1, len(tools)):
                    tool1 = tools[i_tool]
                    tool2 = tools[j_tool]
                    if f0_data[tool1] is not None and f0_data[tool2] is not None:
                        sim = compute_f0_similarity(
                            f0_data[tool1]["freqs"], f0_data[tool2]["freqs"],
                            f0_data[tool1]["times"], f0_data[tool2]["times"]
                        )
                        for k, v in sim.items():
                            track_result[f"consistency_{tool1}_{tool2}_{k}"] = v

        results.append(track_result)

    # 汇总统计
    summary = {
        "n_audio_files": len(audio_files),
        "tools": tools,
        "n_truth": sum(1 for r in results if r.get("has_truth", False)),
        "n_evaluated": sum(1 for r in results if r.get("has_truth", False)),
    }

    # 一致性统计
    if len(tools) >= 2:
        consistency_key = f"consistency_{tools[0]}_{tools[1]}_chroma_accuracy"
        consistency_values = [r[consistency_key] for r in results if consistency_key in r]
        if consistency_values:
            summary["consistency"] = {
                "n_samples": len(consistency_values),
                "mean_chroma_accuracy": float(np.mean(consistency_values)),
                "median_chroma_accuracy": float(np.median(consistency_values)),
            }

    return results, summary


# ===================== 报告生成 =====================

def save_reports(results: List[Dict], summary: Dict,
                 report_json: Optional[str] = None,
                 report_csv: Optional[str] = None,
                 report_html: Optional[str] = None):
    """保存评测报告"""

    # JSON 报告
    if report_json:
        json_path = Path(report_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON 报告已保存: {json_path}")

    # CSV 报告
    if report_csv:
        csv_path = Path(report_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV 报告已保存: {csv_path}")

    # HTML 报告
    if report_html:
        html_path = Path(report_html)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        generate_html_report(results, summary, html_path)
        logger.info(f"HTML 报告已保存: {html_path}")


def generate_html_report(results: List[Dict], summary: Dict, output_path: Path):
    """生成 HTML 报告"""
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F0 基频检测评测报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #4f46e5; padding-bottom: 16px; }}
        h2 {{ color: #4f46e5; margin-top: 32px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
        .summary-card {{ background: #f0f4ff; padding: 20px; border-radius: 8px; border-left: 4px solid #4f46e5; }}
        .summary-card .value {{ font-size: 28px; font-weight: bold; color: #4f46e5; }}
        .summary-card .label {{ font-size: 14px; color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f0f4ff; color: #4f46e5; font-weight: 600; }}
        tr:hover {{ background: #f9fafb; }}
        .metric-good {{ color: #059669; font-weight: 600; }}
        .metric-warning {{ color: #d97706; font-weight: 600; }}
        .metric-bad {{ color: #dc2626; font-weight: 600; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>F0 基频检测评测报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>评测摘要</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{summary.get('n_audio_files', 0)}</div>
                <div class="label">音频文件数</div>
            </div>
            <div class="summary-card">
                <div class="value">{', '.join(summary.get('tools', []))}</div>
                <div class="label">评测工具</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('n_truth', 0)}</div>
                <div class="label">有真值样本</div>
            </div>
            <div class="summary-card">
                <div class="value">{summary.get('consistency', {}).get('mean_chroma_accuracy', 0):.2%}</div>
                <div class="label">平均音高类准确率</div>
            </div>
        </div>

        <h2>详细结果</h2>
        <table>
            <tr>
                <th>音频ID</th>
                <th>有真值</th>
"""

    # 动态添加列
    if results:
        first = results[0]
        metric_cols = [k for k in first.keys() if any(
            k.startswith(f"{tool}_") for tool in summary.get('tools', [])
        ) and any(metric in k for metric in ['rpa', 'rca', 'accuracy', 'recall', 'false_alarm'])]
        consistency_cols = [k for k in first.keys() if k.startswith('consistency_')]

        for col in metric_cols + consistency_cols:
            html_content += f"                <th>{col}</th>\n"

    html_content += """            </tr>
"""

    for r in results:
        html_content += f"""            <tr>
                <td>{r.get('audio_id', 'N/A')[:40]}</td>
                <td>{'✅' if r.get('has_truth', False) else '❌'}</td>
"""
        for col in metric_cols + consistency_cols:
            val = r.get(col, 'N/A')
            if isinstance(val, float):
                if 'accuracy' in col or 'recall' in col:
                    color = 'metric-good' if val > 0.8 else ('metric-warning' if val > 0.5 else 'metric-bad')
                    html_content += f'                <td class="{color}">{val:.2%}</td>\n'
                elif 'cents' in col:
                    html_content += f'                <td>{val:.1f}</td>\n'
                else:
                    html_content += f'                <td>{val:.4f}</td>\n'
            else:
                html_content += f'                <td>{val}</td>\n'

        html_content += "            </tr>\n"

    html_content += f"""
        </table>

        <div class="footer">
            <p>本报告由 eval_f0.py 自动生成 | mir_eval.melody 标准指标</p>
            <p>评测指标说明：RPA（原始音高准确率）、RCA（原始音高类准确率）、Overall Accuracy（整体准确率）、Voicing Recall（浊音召回率）</p>
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="F0 基频检测评测脚本（mir_eval.melody 标准指标）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="音频文件目录")
    parser.add_argument("--truth-dir", type=str, default=None,
                        help="真值标注目录（可选）")
    parser.add_argument("--tools", type=str, default="torchcrepe,librosa",
                        help="评测工具，逗号分隔（torchcrepe,librosa）")
    parser.add_argument("--dataset-type", type=str, default="auto",
                        choices=["auto", "guitarset", "medleydb", "mir1k"],
                        help="数据集类型")
    parser.add_argument("--consistency-only", action="store_true",
                        help="只做工具间一致性分析（不需要真值）")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制处理数量")
    parser.add_argument("--report-json", type=str, default=None,
                        help="JSON 报告输出路径")
    parser.add_argument("--report-csv", type=str, default=None,
                        help="CSV 报告输出路径")
    parser.add_argument("--report-html", type=str, default=None,
                        help="HTML 报告输出路径")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    truth_dir = Path(args.truth_dir) if args.truth_dir else None
    tools = [t.strip() for t in args.tools.split(",")]

    # 验证工具
    valid_tools = ["torchcrepe", "librosa"]
    for tool in tools:
        if tool not in valid_tools:
            logger.error(f"不支持的工具: {tool}，支持的工具: {valid_tools}")
            sys.exit(1)

    # 运行评测
    results, summary = run_evaluation(
        audio_dir=audio_dir,
        truth_dir=truth_dir,
        tools=tools,
        dataset_type=args.dataset_type,
        consistency_only=args.consistency_only,
        limit=args.limit,
    )

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("评测摘要")
    logger.info("=" * 60)
    logger.info(f"  音频文件数: {summary['n_audio_files']}")
    logger.info(f"  评测工具: {summary['tools']}")
    logger.info(f"  有真值样本: {summary.get('n_truth', 0)}")
    if "consistency" in summary:
        logger.info(f"  一致性分析:")
        logger.info(f"    样本数: {summary['consistency']['n_samples']}")
        logger.info(f"    平均音高类准确率: {summary['consistency']['mean_chroma_accuracy']:.2%}")
    logger.info("=" * 60)

    # 保存报告
    save_reports(
        results=results,
        summary=summary,
        report_json=args.report_json,
        report_csv=args.report_csv,
        report_html=args.report_html,
    )


if __name__ == "__main__":
    main()
