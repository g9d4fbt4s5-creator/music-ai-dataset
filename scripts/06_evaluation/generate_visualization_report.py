"""
generate_visualization_report.py
MIR 评测结果可视化报告生成器

用 plotly 生成交互式图表，支持：
- BPM/F0/Onset 评测结果可视化
- 工具对比柱状图
- 误差分布直方图
- 一致性散点图
- 高分歧/高误差样本标注

用法：
    # 从 CSV 报告生成可视化
    python generate_visualization_report.py \
        --input reports/bpm_consistency_7tracks.csv \
        --type bpm \
        --output reports/bpm_visualization.html

    # 从 JSON 摘要 + CSV 详细结果生成
    python generate_visualization_report.py \
        --input-csv reports/bpm_eval.csv \
        --input-json reports/bpm_eval.json \
        --type bpm \
        --output reports/bpm_visualization.html

    # 批量生成多个评测的可视化
    python generate_visualization_report.py \
        --input-dir reports/ \
        --output reports/all_visualizations.html
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

# 尝试导入 plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly 未安装，将使用基础 HTML 表格")


# ===================== 数据加载 =====================

def load_data(input_csv: Optional[str], input_json: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    加载评测数据

    Args:
        input_csv: CSV 文件路径
        input_json: JSON 文件路径

    Returns:
        (DataFrame, summary_dict)
    """
    df = None
    summary = None

    if input_csv and Path(input_csv).exists():
        df = pd.read_csv(input_csv)
        logger.info(f"加载 CSV: {input_csv} ({len(df)} 行)")

    if input_json and Path(input_json).exists():
        with open(input_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
        logger.info(f"加载 JSON: {input_json}")

    return df, summary


# ===================== 图表生成 =====================

def create_bpm_charts(df: pd.DataFrame, summary: Optional[Dict] = None) -> List[go.Figure]:
    """
    生成 BPM 评测图表

    Args:
        df: 详细结果 DataFrame
        summary: 摘要字典

    Returns:
        图表列表
    """
    figures = []

    if df is None or len(df) == 0:
        return figures

    # 识别工具列
    tool_cols = [col for col in df.columns if col.startswith("diff_")]
    tools = list(set([col.split("_")[1] for col in tool_cols] + [col.split("_")[2] for col in tool_cols]))

    # 1. BPM 对比柱状图
    if "mean_bpm" in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["audio_id"].str[:30],
            y=df["mean_bpm"],
            name="平均 BPM",
            marker_color="#4f46e5",
        ))
        if "min_bpm" in df.columns and "max_bpm" in df.columns:
            fig.add_trace(go.Bar(
                x=df["audio_id"].str[:30],
                y=df["max_bpm"] - df["min_bpm"],
                base=df["min_bpm"],
                name="BPM 范围",
                marker_color="rgba(79, 70, 229, 0.3)",
            ))
        fig.update_layout(
            title="各曲目 BPM 检测结果",
            xaxis_title="音频 ID",
            yaxis_title="BPM",
            barmode="overlay",
            height=400,
        )
        figures.append(fig)

    # 2. 工具间分歧分布直方图
    if tool_cols:
        for col in tool_cols:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[col],
                nbinsx=20,
                name=col,
                marker_color="#059669",
            ))
            # 标注高分歧阈值
            if df[col].max() > 5:
                fig.add_vline(x=5, line_dash="dash", line_color="red",
                              annotation_text="高分歧阈值 (5 BPM)")
            fig.update_layout(
                title=f"工具间 BPM 分歧分布: {col}",
                xaxis_title="分歧 (BPM)",
                yaxis_title="样本数",
                height=400,
            )
            figures.append(fig)

    # 3. 高分歧样本散点图
    if "range_bpm" in df.columns and "mean_bpm" in df.columns:
        fig = go.Figure()
        # 低分歧样本
        low_disagreement = df[df["range_bpm"] <= 5]
        high_disagreement = df[df["range_bpm"] > 5]

        fig.add_trace(go.Scatter(
            x=low_disagreement["mean_bpm"],
            y=low_disagreement["range_bpm"],
            mode="markers",
            name="一致样本 (≤5 BPM)",
            marker=dict(color="#059669", size=10),
            text=low_disagreement["audio_id"].str[:30],
        ))
        if len(high_disagreement) > 0:
            fig.add_trace(go.Scatter(
                x=high_disagreement["mean_bpm"],
                y=high_disagreement["range_bpm"],
                mode="markers+text",
                name="高分歧样本 (>5 BPM)",
                marker=dict(color="#dc2626", size=12, symbol="star"),
                text=high_disagreement["audio_id"].str[:20],
                textposition="top center",
            ))
        fig.update_layout(
            title="BPM 检测一致性散点图",
            xaxis_title="平均 BPM",
            yaxis_title="工具间分歧 (BPM)",
            height=500,
        )
        figures.append(fig)

    return figures


def create_f0_charts(df: pd.DataFrame, summary: Optional[Dict] = None) -> List[go.Figure]:
    """
    生成 F0 评测图表

    Args:
        df: 详细结果 DataFrame
        summary: 摘要字典

    Returns:
        图表列表
    """
    figures = []

    if df is None or len(df) == 0:
        return figures

    # 识别指标列
    metric_cols = [col for col in df.columns if any(
        metric in col for metric in ["rpa", "rca", "accuracy", "recall", "false_alarm"]
    )]

    # 1. 工具指标对比柱状图
    if metric_cols:
        tools = list(set([col.split("_")[0] for col in metric_cols]))
        metrics = list(set(["_".join(col.split("_")[1:]) for col in metric_cols]))

        fig = go.Figure()
        for tool in tools:
            tool_metrics = [col for col in metric_cols if col.startswith(f"{tool}_")]
            if tool_metrics:
                fig.add_trace(go.Bar(
                    x=["_".join(col.split("_")[1:]) for col in tool_metrics],
                    y=[df[col].mean() for col in tool_metrics],
                    name=tool,
                ))
        fig.update_layout(
            title="F0 检测工具指标对比",
            xaxis_title="指标",
            yaxis_title="数值",
            barmode="group",
            height=400,
        )
        figures.append(fig)

    # 2. 音高类准确率分布
    consistency_cols = [col for col in df.columns if "chroma_accuracy" in col]
    for col in consistency_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[col],
            nbinsx=20,
            name=col,
            marker_color="#8b5cf6",
        ))
        fig.update_layout(
            title=f"音高类准确率分布: {col}",
            xaxis_title="准确率",
            yaxis_title="样本数",
            height=400,
        )
        figures.append(fig)

    return figures


def create_onset_charts(df: pd.DataFrame, summary: Optional[Dict] = None) -> List[go.Figure]:
    """
    生成 Onset 评测图表

    Args:
        df: 详细结果 DataFrame
        summary: 摘要字典

    Returns:
        图表列表
    """
    figures = []

    if df is None or len(df) == 0:
        return figures

    # 识别指标列
    metric_cols = [col for col in df.columns if any(
        metric in col for metric in ["f1_score", "precision", "recall"]
    )]

    # 1. 工具指标对比柱状图
    if metric_cols:
        tools = list(set([col.split("_")[0] for col in metric_cols]))

        fig = go.Figure()
        for tool in tools:
            tool_metrics = [col for col in metric_cols if col.startswith(f"{tool}_")]
            if tool_metrics:
                fig.add_trace(go.Bar(
                    x=["_".join(col.split("_")[1:]) for col in tool_metrics],
                    y=[df[col].mean() for col in tool_metrics],
                    name=tool,
                ))
        fig.update_layout(
            title="Onset 检测工具指标对比",
            xaxis_title="指标",
            yaxis_title="数值",
            barmode="group",
            height=400,
        )
        figures.append(fig)

    # 2. F1 分数分布
    f1_cols = [col for col in df.columns if "f1_score" in col]
    for col in f1_cols:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[col],
            nbinsx=20,
            name=col,
            marker_color="#f59e0b",
        ))
        fig.update_layout(
            title=f"F1 分数分布: {col}",
            xaxis_title="F1 分数",
            yaxis_title="样本数",
            height=400,
        )
        figures.append(fig)

    # 3. onset 数量对比
    count_cols = [col for col in df.columns if "onset_" in col and "_count" in col]
    if count_cols:
        fig = go.Figure()
        for col in count_cols:
            fig.add_trace(go.Bar(
                x=df["audio_id"].str[:30] if "audio_id" in df.columns else range(len(df)),
                y=df[col],
                name=col,
            ))
        fig.update_layout(
            title="各曲目 Onset 数量对比",
            xaxis_title="音频 ID",
            yaxis_title="Onset 数量",
            barmode="group",
            height=400,
        )
        figures.append(fig)

    return figures


# ===================== HTML 报告生成 =====================

def generate_html_report(figures: List[go.Figure], title: str,
                         output_path: Path,
                         summary: Optional[Dict] = None):
    """
    生成包含多个图表的 HTML 报告

    Args:
        figures: 图表列表
        title: 报告标题
        summary: 摘要字典
        output_path: 输出路径
    """
    if not PLOTLY_AVAILABLE or len(figures) == 0:
        # 基础 HTML 表格
        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>{title}</title></head>
<body><h1>{title}</h1><p>plotly 未安装或无图表数据</p></body></html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return

    # 生成每个图表的 HTML
    charts_html = ""
    for i, fig in enumerate(figures):
        chart_html = fig.to_html(full_html=False, include_plotlyjs=(i == 0))
        charts_html += f"""
        <div style="margin: 32px 0; padding: 24px; background: white; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1);">
            {chart_html}
        </div>
        """

    # 摘要卡片
    summary_html = ""
    if summary:
        summary_items = ""
        for k, v in summary.items():
            if isinstance(v, (int, float, str)):
                summary_items += f"""
                <div style="background: #f0f4ff; padding: 16px; border-radius: 8px; border-left: 4px solid #4f46e5;">
                    <div style="font-size: 24px; font-weight: bold; color: #4f46e5;">{v}</div>
                    <div style="font-size: 12px; color: #666; margin-top: 4px;">{k}</div>
                </div>
                """
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        summary_items += f"""
                        <div style="background: #f0f4ff; padding: 16px; border-radius: 8px; border-left: 4px solid #4f46e5;">
                            <div style="font-size: 24px; font-weight: bold; color: #4f46e5;">{v2:.4f}</div>
                            <div style="font-size: 12px; color: #666; margin-top: 4px;">{k}.{k2}</div>
                        </div>
                        """
        summary_html = f"""
        <h2 style="color: #4f46e5; margin-top: 32px;">评测摘要</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 24px 0;">
            {summary_items}
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #4f46e5; padding-bottom: 16px; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #999; font-size: 12px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        {summary_html}
        <h2 style="color: #4f46e5; margin-top: 32px;">可视化图表</h2>
        {charts_html}
        <div class="footer">
            <p>本报告由 generate_visualization_report.py 自动生成 | plotly 交互式图表</p>
        </div>
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"可视化报告已保存: {output_path}")


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(
        description="MIR 评测结果可视化报告生成器（plotly 交互式图表）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None,
                        help="输入 CSV 文件路径")
    parser.add_argument("--input-csv", type=str, default=None,
                        help="输入 CSV 文件路径（详细结果）")
    parser.add_argument("--input-json", type=str, default=None,
                        help="输入 JSON 文件路径（摘要）")
    parser.add_argument("--type", type=str, default="auto",
                        choices=["auto", "bpm", "f0", "onset"],
                        help="评测类型（自动检测/bpm/f0/onset）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 HTML 文件路径")
    parser.add_argument("--title", type=str, default=None,
                        help="报告标题")
    args = parser.parse_args()

    # 确定输入文件
    input_csv = args.input_csv or args.input
    input_json = args.input_json

    if not input_csv and not input_json:
        logger.error("请指定输入文件（--input 或 --input-csv/--input-json）")
        sys.exit(1)

    # 加载数据
    df, summary = load_data(input_csv, input_json)

    if df is None and summary is None:
        logger.error("无法加载数据")
        sys.exit(1)

    # 自动检测类型
    eval_type = args.type
    if eval_type == "auto" and df is not None:
        if any("bpm" in col.lower() for col in df.columns):
            eval_type = "bpm"
        elif any("f0" in col.lower() or "chroma" in col.lower() for col in df.columns):
            eval_type = "f0"
        elif any("onset" in col.lower() for col in df.columns):
            eval_type = "onset"
        else:
            eval_type = "bpm"  # 默认

    logger.info(f"评测类型: {eval_type}")

    # 生成图表
    if eval_type == "bpm":
        figures = create_bpm_charts(df, summary)
    elif eval_type == "f0":
        figures = create_f0_charts(df, summary)
    elif eval_type == "onset":
        figures = create_onset_charts(df, summary)
    else:
        figures = []

    logger.info(f"生成 {len(figures)} 个图表")

    # 生成报告
    title = args.title or f"MIR {eval_type.upper()} 评测可视化报告"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_html_report(figures, title, output_path, summary)


if __name__ == "__main__":
    main()
