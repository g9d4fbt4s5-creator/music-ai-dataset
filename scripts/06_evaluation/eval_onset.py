"""
eval_onset.py
音符起始点（Onset）检测评测脚本

用 mir_eval.onset 标准指标评测 onset 检测工具：
- madmom（深度学习，RNN onset detector）
- librosa（能量/频谱通量 onset detection）

支持的真值数据集：
- GuitarSet：吉他独奏，有onset标注
- SMC MIREX：节拍/onset真值
- MAESTRO：钢琴MIDI，有note/onset真值
- auto：自动检测

无真值时：工具间一致性分析（madmom vs librosa）

评测指标（mir_eval.onset）：
- F1-score：F1分数
- Precision：精确率
- Recall：召回率

用法：
    # 有真值评测（GuitarSet）
    python eval_onset.py \
        --audio-dir data/datasets/guitarset/audio \
        --truth-dir data/datasets/guitarset/annotations \
        --dataset-type guitarset \
        --tools madmom,librosa \
        --report-json reports/onset_eval_guitarset.json \
        --report-html reports/onset_eval_guitarset.html

    # 工具间一致性分析（无真值）
    python eval_onset.py \
        --audio-dir data/01_preprocess/processed_master \
        --tools madmom,librosa \
        --consistency-only \
        --report-csv reports/onset_consistency.csv
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


# ===================== Onset 提取工具 =====================

def extract_onset_madmom(audio_path: Path, sr: int = 44100) -> np.ndarray:
    """
    用 madmom 提取 onset

    Args:
        audio_path: 音频文件路径
        sr: 采样率

    Returns:
        onset_times — onset 时间数组（秒）
    """
    from madmom.features.onsets import OnsetPeakPickingProcessor
    from madmom.features import RNNOnsetProcessor

    # 加载音频
    import librosa
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # RNN onset 检测
    proc = RNNOnsetProcessor()
    onset_probs = proc(y)

    # 峰值提取
    peak_proc = OnsetPeakPickingProcessor(
        threshold=0.5, smooth=0.1, pre_avg=0.05, post_avg=0.05,
        pre_max=0.01, post_max=0.01
    )
    onset_times = peak_proc(onset_probs)

    return np.array(onset_times)


def extract_onset_librosa(audio_path: Path, sr: int = 22050,
                           hop_length: int = 512) -> np.ndarray:
    """
    用 librosa 提取 onset

    Args:
        audio_path: 音频文件路径
        sr: 采样率
        hop_length: 帧移

    Returns:
        onset_times — onset 时间数组（秒）
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)

    # 计算 onset 强度
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # 检测 onset
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length,
        backtrack=True
    )

    # 转换为时间
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    return np.array(onset_times)


def extract_onset(audio_path: Path, tool: str, **kwargs) -> np.ndarray:
    """
    统一的 onset 提取接口

    Args:
        audio_path: 音频文件路径
        tool: 工具名称（madmom/librosa）

    Returns:
        onset_times
    """
    if tool == "madmom":
        return extract_onset_madmom(audio_path, **kwargs)
    elif tool == "librosa":
        return extract_onset_librosa(audio_path, **kwargs)
    else:
        raise ValueError(f"不支持的工具: {tool}")


# ===================== 真值加载 =====================

def load_truth_guitarset(truth_dir: Path, audio_stem: str) -> Optional[np.ndarray]:
    """
    加载 GuitarSet 的 onset 真值

    GuitarSet 标注格式：
    - CSV 格式，列：time, note 等
    - onset 时间通常在第一列

    Args:
        truth_dir: 标注目录
        audio_stem: 音频文件名（不含扩展名）

    Returns:
        onset_times 或 None
    """
    possible_files = [
        truth_dir / f"{audio_stem}_onsets.csv",
        truth_dir / f"{audio_stem}_notes.csv",
        truth_dir / f"{audio_stem}.csv",
    ]

    # 递归查找
    for pattern in [f"*{audio_stem}*onset*.csv", f"*{audio_stem}*note*.csv", f"*{audio_stem}*.csv"]:
        possible_files.extend(truth_dir.rglob(pattern))

    for truth_file in possible_files:
        if truth_file.exists():
            try:
                df = pd.read_csv(truth_file)
                # 尝试常见列名
                time_col = None
                for col in df.columns:
                    if col.lower() in ["time", "timestamp", "onset", "t", "seconds", "start_time"]:
                        time_col = col
                        break

                if time_col:
                    onset_times = df[time_col].values
                    onset_times = np.sort(np.unique(onset_times))
                    return onset_times
                else:
                    # 如果没有列名，尝试第一列
                    if len(df.columns) >= 1:
                        onset_times = df.iloc[:, 0].values
                        onset_times = np.sort(np.unique(onset_times))
                        return onset_times
            except Exception as e:
                logger.warning(f"加载真值失败 {truth_file}: {e}")
                continue

    return None


def load_truth_smc(truth_dir: Path, audio_stem: str) -> Optional[np.ndarray]:
    """
    加载 SMC MIREX 的 onset 真值

    SMC 标注格式：
    - .onsets 文件，每行一个时间戳
    - .beats 文件，每行一个时间戳

    Args:
        truth_dir: 标注目录
        audio_stem: 音频文件名（不含扩展名）

    Returns:
        onset_times 或 None
    """
    possible_files = [
        truth_dir / f"{audio_stem}.onsets",
        truth_dir / f"{audio_stem}_onsets.txt",
        truth_dir / f"{audio_stem}.beats",
    ]

    # 递归查找
    for pattern in [f"*{audio_stem}*.onsets", f"*{audio_stem}*onset*.txt", f"*{audio_stem}*.beats"]:
        possible_files.extend(truth_dir.rglob(pattern))

    for truth_file in possible_files:
        if truth_file.exists():
            try:
                onset_times = np.loadtxt(str(truth_file))
                if onset_times.ndim > 1:
                    onset_times = onset_times[:, 0]  # 取第一列
                return np.sort(onset_times)
            except Exception as e:
                logger.warning(f"加载真值失败 {truth_file}: {e}")
                continue

    return None


def load_truth(truth_dir: Path, audio_path: Path,
               dataset_type: str = "auto") -> Optional[np.ndarray]:
    """
    加载 onset 真值（统一接口）

    Args:
        truth_dir: 真值目录
        audio_path: 音频文件路径
        dataset_type: 数据集类型（auto/guitarset/smc/maestro）

    Returns:
        onset_times 或 None
    """
    audio_stem = audio_path.stem

    if dataset_type == "auto":
        # 自动检测：尝试所有格式
        for ds_type in ["guitarset", "smc", "maestro"]:
            result = load_truth(truth_dir, audio_path, ds_type)
            if result is not None:
                return result
        return None

    elif dataset_type == "guitarset":
        return load_truth_guitarset(truth_dir, audio_stem)

    elif dataset_type == "smc":
        return load_truth_smc(truth_dir, audio_stem)

    elif dataset_type == "maestro":
        # MAESTRO 用 MIDI 文件，需要额外解析
        return load_truth_guitarset(truth_dir, audio_stem)  # 复用通用加载

    else:
        logger.warning(f"不支持的数据集类型: {dataset_type}")
        return None


# ===================== 评测指标 =====================

def evaluate_onset(ref_onsets: np.ndarray, est_onsets: np.ndarray,
                   window: float = 0.05) -> Dict:
    """
    用 mir_eval.onset 计算 onset 评测指标

    Args:
        ref_onsets: 真值 onset 时间数组
        est_onsets: 估计 onset 时间数组
        window: 匹配窗口（秒）

    Returns:
        指标字典
    """
    import mir_eval

    try:
        # mir_eval.onset.f_measure 返回 (f_measure, precision, recall)
        f_measure, precision, recall = mir_eval.onset.f_measure(
            ref_onsets, est_onsets, window=window
        )
        return {
            "f1_score": float(f_measure),
            "precision": float(precision),
            "recall": float(recall),
            "n_ref": len(ref_onsets),
            "n_est": len(est_onsets),
        }
    except Exception as e:
        logger.warning(f"mir_eval 评测失败: {e}")
        return {}


def compute_onset_similarity(onsets1: np.ndarray, onsets2: np.ndarray,
                              window: float = 0.05) -> Dict:
    """
    计算两个 onset 序列的相似度（无真值时的工具间一致性分析）

    Args:
        onsets1: 工具1的 onset 时间数组
        onsets2: 工具2的 onset 时间数组
        window: 匹配窗口（秒）

    Returns:
        相似度指标字典
    """
    if len(onsets1) == 0 or len(onsets2) == 0:
        return {
            "n_onsets1": len(onsets1),
            "n_onsets2": len(onsets2),
            "matched_count": 0,
            "match_ratio1": 0,
            "match_ratio2": 0,
            "mean_time_diff": 0,
        }

    # 匹配 onsets1 到 onsets2
    matched1 = 0
    time_diffs = []
    for o1 in onsets1:
        diffs = np.abs(onsets2 - o1)
        min_diff = np.min(diffs)
        if min_diff < window:
            matched1 += 1
            time_diffs.append(min_diff)

    # 匹配 onsets2 到 onsets1
    matched2 = 0
    for o2 in onsets2:
        diffs = np.abs(onsets1 - o2)
        min_diff = np.min(diffs)
        if min_diff < window:
            matched2 += 1

    return {
        "n_onsets1": len(onsets1),
        "n_onsets2": len(onsets2),
        "matched_count": matched1,
        "match_ratio1": matched1 / len(onsets1),
        "match_ratio2": matched2 / len(onsets2),
        "mean_time_diff": float(np.mean(time_diffs)) if time_diffs else 0,
    }


# ===================== 主评测流程 =====================

def run_evaluation(audio_dir: Path, truth_dir: Optional[Path],
                   tools: List[str], dataset_type: str = "auto",
                   consistency_only: bool = False,
                   limit: Optional[int] = None,
                   window: float = 0.05) -> Tuple[List[Dict], Dict]:
    """
    运行 onset 评测

    Args:
        audio_dir: 音频目录
        truth_dir: 真值目录（可选）
        tools: 工具列表
        dataset_type: 数据集类型
        consistency_only: 只做一致性分析
        limit: 限制处理数量
        window: 匹配窗口（秒）

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

    for i, audio_path in enumerate(audio_files):
        logger.info(f"[{i+1}/{len(audio_files)}] 处理: {audio_path.name}")

        track_result = {
            "audio_id": audio_path.stem,
            "audio_path": str(audio_path),
        }

        # 提取所有工具的 onset
        onset_data = {}
        for tool in tools:
            try:
                onset_times = extract_onset(audio_path, tool)
                onset_data[tool] = onset_times
                track_result[f"onset_{tool}_count"] = len(onset_times)
                if len(onset_times) > 0:
                    track_result[f"onset_{tool}_rate"] = len(onset_times) / (onset_times[-1] - onset_times[0] + 1e-10)
                else:
                    track_result[f"onset_{tool}_rate"] = 0
            except Exception as e:
                logger.warning(f"  {tool} 提取失败: {e}")
                onset_data[tool] = None
                track_result[f"onset_{tool}_error"] = str(e)

        # 有真值评测
        if truth_dir and not consistency_only:
            truth = load_truth(truth_dir, audio_path, dataset_type)
            if truth is not None:
                ref_onsets = truth
                track_result["has_truth"] = True
                track_result["n_ref_onsets"] = len(ref_onsets)

                for tool in tools:
                    if onset_data[tool] is not None:
                        metrics = evaluate_onset(ref_onsets, onset_data[tool], window=window)
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
                    if onset_data[tool1] is not None and onset_data[tool2] is not None:
                        sim = compute_onset_similarity(
                            onset_data[tool1], onset_data[tool2], window=window
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
        "window": window,
    }

    # 一致性统计
    if len(tools) >= 2:
        consistency_key = f"consistency_{tools[0]}_{tools[1]}_match_ratio1"
        consistency_values = [r[consistency_key] for r in results if consistency_key in r]
        if consistency_values:
            summary["consistency"] = {
                "n_samples": len(consistency_values),
                "mean_match_ratio1": float(np.mean(consistency_values)),
                "mean_match_ratio2": float(np.mean([
                    r[f"consistency_{tools[0]}_{tools[1]}_match_ratio2"]
                    for r in results if f"consistency_{tools[0]}_{tools[1]}_match_ratio2" in r
                ])),
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
    <title>Onset 音符起始点检测评测报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #059669; padding-bottom: 16px; }}
        h2 {{ color: #059669; margin-top: 32px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 24px 0; }}
        .summary-card {{ background: #ecfdf5; padding: 20px; border-radius: 8px; border-left: 4px solid #059669; }}
        .summary-card .value {{ font-size: 28px; font-weight: bold; color: #059669; }}
        .summary-card .label {{ font-size: 14px; color: #666; margin-top: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #ecfdf5; color: #059669; font-weight: 600; }}
        tr:hover {{ background: #f9fafb; }}
        .metric-good {{ color: #059669; font-weight: 600; }}
        .metric-warning {{ color: #d97706; font-weight: 600; }}
        .metric-bad {{ color: #dc2626; font-weight: 600; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Onset 音符起始点检测评测报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>匹配窗口: {summary.get('window', 0.05)} 秒</p>

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
                <div class="value">{summary.get('consistency', {}).get('mean_match_ratio1', 0):.2%}</div>
                <div class="label">平均匹配率</div>
            </div>
        </div>

        <h2>详细结果</h2>
        <table>
            <tr>
                <th>音频ID</th>
                <th>有真值</th>
"""

    # 动态添加列
    metric_cols = []
    consistency_cols = []
    if results:
        first = results[0]
        metric_cols = [k for k in first.keys() if any(
            k.startswith(f"{tool}_") for tool in summary.get('tools', [])
        ) and any(metric in k for metric in ['f1', 'precision', 'recall'])]
        consistency_cols = [k for k in first.keys() if k.startswith('consistency_') and any(m in k for m in ['match_ratio', 'mean_time_diff'])]

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
                if 'f1' in col or 'precision' in col or 'recall' in col or 'match_ratio' in col:
                    color = 'metric-good' if val > 0.8 else ('metric-warning' if val > 0.5 else 'metric-bad')
                    html_content += f'                <td class="{color}">{val:.2%}</td>\n'
                elif 'time_diff' in col:
                    html_content += f'                <td>{val*1000:.1f}ms</td>\n'
                else:
                    html_content += f'                <td>{val:.4f}</td>\n'
            else:
                html_content += f'                <td>{val}</td>\n'

        html_content += "            </tr>\n"

    html_content += f"""
        </table>

        <div class="footer">
            <p>本报告由 eval_onset.py 自动生成 | mir_eval.onset 标准指标</p>
            <p>评测指标说明：F1-score（F1分数）、Precision（精确率）、Recall（召回率）</p>
            <p>匹配窗口：{summary.get('window', 0.05)} 秒（估计 onset 在真值 onset ±窗口内视为正确）</p>
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="Onset 音符起始点检测评测脚本（mir_eval.onset 标准指标）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="音频文件目录")
    parser.add_argument("--truth-dir", type=str, default=None,
                        help="真值标注目录（可选）")
    parser.add_argument("--tools", type=str, default="madmom,librosa",
                        help="评测工具，逗号分隔（madmom,librosa）")
    parser.add_argument("--dataset-type", type=str, default="auto",
                        choices=["auto", "guitarset", "smc", "maestro"],
                        help="数据集类型")
    parser.add_argument("--consistency-only", action="store_true",
                        help="只做工具间一致性分析（不需要真值）")
    parser.add_argument("--window", type=float, default=0.05,
                        help="匹配窗口（秒），默认 0.05")
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
    valid_tools = ["madmom", "librosa"]
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
        window=args.window,
    )

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("评测摘要")
    logger.info("=" * 60)
    logger.info(f"  音频文件数: {summary['n_audio_files']}")
    logger.info(f"  评测工具: {summary['tools']}")
    logger.info(f"  匹配窗口: {summary['window']} 秒")
    logger.info(f"  有真值样本: {summary.get('n_truth', 0)}")
    if "consistency" in summary:
        logger.info(f"  一致性分析:")
        logger.info(f"    样本数: {summary['consistency']['n_samples']}")
        logger.info(f"    平均匹配率(工具1→工具2): {summary['consistency']['mean_match_ratio1']:.2%}")
        logger.info(f"    平均匹配率(工具2→工具1): {summary['consistency']['mean_match_ratio2']:.2%}")
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
