"""
lineage_report.py
血缘报告生成工具（Lineage v2.0）

功能：
- 生成血缘摘要报告（JSON/Markdown/HTML）
- 算子耗时分析（瓶颈识别）
- 失败率统计（按算子/按原因）
- 数据集划分统计
- 可视化图表（plotly）
- 导出为可分享的HTML报告

用法：
    # 生成Markdown报告
    python lineage_report.py --lineage data/lineage/lineage_v20260824_100000.json --format markdown

    # 生成HTML报告（含图表）
    python lineage_report.py --lineage data/lineage/lineage_v20260824_100000.json --format html --output report.html

    # 生成JSON报告
    python lineage_report.py --lineage data/lineage/lineage_v20260824_100000.json --format json

    # 只生成图表
    python lineage_report.py --lineage data/lineage/lineage_v20260824_100000.json --charts-only --output-dir charts/
"""
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "lineage" / "reports"


def load_lineage(lineage_path: str) -> Dict:
    """加载血缘文件"""
    path = Path(lineage_path)
    if not path.exists():
        raise FileNotFoundError(f"血缘文件不存在: {lineage_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(lineage: Dict) -> Dict:
    """计算血缘指标"""
    operators = lineage.get("operators", [])
    splits = lineage.get("splits", {})

    # 基本统计
    total_input = sum(op.get("input_count", 0) or 0 for op in operators)
    total_output = sum(op.get("output_count", 0) or 0 for op in operators)
    total_failed = sum(op.get("failed_count", 0) or 0 for op in operators)
    total_duration = sum(op.get("duration_sec", 0) or 0 for op in operators)

    # 状态统计
    status_counts = {}
    for op in operators:
        s = op.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    # 失败原因统计
    failure_reasons = {}
    for op in operators:
        for reason, count in op.get("failure_reasons", {}).items():
            failure_reasons[reason] = failure_reasons.get(reason, 0) + count

    # 算子级指标
    operator_metrics = []
    for op in operators:
        input_count = op.get("input_count", 0) or 0
        failed_count = op.get("failed_count", 0) or 0
        duration = op.get("duration_sec", 0) or 0

        operator_metrics.append({
            "operator_name": op.get("operator_name"),
            "operator_version": op.get("operator_version"),
            "model_version": op.get("model_version"),
            "status": op.get("status"),
            "input_count": input_count,
            "output_count": op.get("output_count", 0),
            "failed_count": failed_count,
            "failure_rate": round(failed_count / input_count, 4) if input_count > 0 else 0,
            "duration_sec": duration,
            "avg_time_per_sample": round(duration / input_count, 4) if input_count > 0 else 0,
            "timestamp": op.get("timestamp"),
        })

    # 划分统计
    split_metrics = {}
    for name, info in splits.items():
        split_metrics[name] = {
            "count": info.get("count", 0),
            "source_manifest": info.get("source_manifest"),
            "source_batch": info.get("source_batch"),
        }

    return {
        "dataset_version": lineage.get("dataset_version"),
        "lineage_version": lineage.get("lineage_version"),
        "created_at": lineage.get("created_at"),
        "updated_at": lineage.get("updated_at"),
        "upstream_lineage": lineage.get("upstream_lineage"),
        "operator_count": len(operators),
        "total_input": total_input,
        "total_output": total_output,
        "total_failed": total_failed,
        "failure_rate": round(total_failed / total_input, 4) if total_input > 0 else 0,
        "total_duration_sec": round(total_duration, 2),
        "status_counts": status_counts,
        "failure_reasons": failure_reasons,
        "operators": operator_metrics,
        "splits": split_metrics,
    }


def generate_charts(metrics: Dict, output_dir: Path) -> List[Path]:
    """生成可视化图表"""
    if not PLOTLY_AVAILABLE:
        print("⚠️  plotly未安装，跳过图表生成。安装: pip install plotly")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_paths = []

    operators_df = pd.DataFrame(metrics["operators"])

    # 图表1：算子耗时柱状图
    if not operators_df.empty and operators_df["duration_sec"].sum() > 0:
        fig = px.bar(
            operators_df.sort_values("duration_sec", ascending=True),
            x="duration_sec",
            y="operator_name",
            color="status",
            title="算子耗时分析",
            labels={"duration_sec": "耗时 (秒)", "operator_name": "算子"},
            orientation="h",
        )
        path = output_dir / "operator_duration.html"
        fig.write_html(str(path))
        chart_paths.append(path)

    # 图表2：算子失败率柱状图
    if not operators_df.empty:
        fig = px.bar(
            operators_df.sort_values("failure_rate", ascending=True),
            x="failure_rate",
            y="operator_name",
            color="failure_rate",
            color_continuous_scale="RdYlGn_r",
            title="算子失败率分析",
            labels={"failure_rate": "失败率", "operator_name": "算子"},
            orientation="h",
        )
        path = output_dir / "operator_failure_rate.html"
        fig.write_html(str(path))
        chart_paths.append(path)

    # 图表3：失败原因饼图
    if metrics["failure_reasons"]:
        reasons_df = pd.DataFrame([
            {"reason": k, "count": v}
            for k, v in metrics["failure_reasons"].items()
        ])
        fig = px.pie(
            reasons_df,
            values="count",
            names="reason",
            title="失败原因分布",
        )
        path = output_dir / "failure_reasons.html"
        fig.write_html(str(path))
        chart_paths.append(path)

    # 图表4：数据集划分饼图
    if metrics["splits"]:
        splits_df = pd.DataFrame([
            {"split": k, "count": v["count"]}
            for k, v in metrics["splits"].items()
        ])
        fig = px.pie(
            splits_df,
            values="count",
            names="split",
            title="数据集划分",
        )
        path = output_dir / "dataset_splits.html"
        fig.write_html(str(path))
        chart_paths.append(path)

    # 图表5：算子状态分布
    if metrics["status_counts"]:
        status_df = pd.DataFrame([
            {"status": k, "count": v}
            for k, v in metrics["status_counts"].items()
        ])
        fig = px.bar(
            status_df,
            x="status",
            y="count",
            color="status",
            title="算子状态分布",
        )
        path = output_dir / "operator_status.html"
        fig.write_html(str(path))
        chart_paths.append(path)

    print(f"✅ 已生成 {len(chart_paths)} 个图表到: {output_dir}")
    return chart_paths


def generate_markdown_report(metrics: Dict) -> str:
    """生成Markdown报告"""
    lines = []

    lines.append(f"# 血缘报告: {metrics['dataset_version']}")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**血缘版本**: {metrics['lineage_version']}")
    lines.append(f"**创建时间**: {metrics['created_at']}")
    lines.append(f"**更新时间**: {metrics['updated_at']}")
    if metrics.get("upstream_lineage"):
        lines.append(f"**上游血缘**: {metrics['upstream_lineage']}")
    lines.append("")

    # 摘要
    lines.append("## 一、摘要")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 算子数量 | {metrics['operator_count']} |")
    lines.append(f"| 总输入样本 | {metrics['total_input']} |")
    lines.append(f"| 总输出样本 | {metrics['total_output']} |")
    lines.append(f"| 总失败样本 | {metrics['total_failed']} |")
    lines.append(f"| 失败率 | {metrics['failure_rate']*100:.2f}% |")
    lines.append(f"| 总耗时 | {metrics['total_duration_sec']:.1f}s |")
    lines.append("")

    # 状态分布
    lines.append("## 二、算子状态分布")
    lines.append("")
    lines.append("| 状态 | 数量 |")
    lines.append("|------|------|")
    for status, count in metrics["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.append("")

    # 算子详情
    lines.append("## 三、算子详情")
    lines.append("")
    lines.append("| 算子 | 版本 | 状态 | 输入 | 输出 | 失败 | 失败率 | 耗时(s) | 平均耗时/样本 |")
    lines.append("|------|------|------|------|------|------|--------|---------|--------------|")
    for op in metrics["operators"]:
        lines.append(
            f"| {op['operator_name']} | {op['operator_version']} | {op['status']} | "
            f"{op['input_count']} | {op['output_count']} | {op['failed_count']} | "
            f"{op['failure_rate']*100:.2f}% | {op['duration_sec']} | {op['avg_time_per_sample']} |"
        )
    lines.append("")

    # 失败原因
    if metrics["failure_reasons"]:
        lines.append("## 四、失败原因统计")
        lines.append("")
        lines.append("| 失败原因 | 数量 | 占比 |")
        lines.append("|----------|------|------|")
        total = sum(metrics["failure_reasons"].values())
        for reason, count in sorted(metrics["failure_reasons"].items(), key=lambda x: -x[1]):
            lines.append(f"| {reason} | {count} | {count/total*100:.1f}% |")
        lines.append("")

    # 数据集划分
    if metrics["splits"]:
        lines.append("## 五、数据集划分")
        lines.append("")
        lines.append("| 划分 | 数量 | 来源 |")
        lines.append("|------|------|------|")
        for name, info in metrics["splits"].items():
            source = info.get("source_manifest") or info.get("source_batch") or "N/A"
            lines.append(f"| {name} | {info['count']} | {source} |")
        lines.append("")

    # 瓶颈分析
    lines.append("## 六、瓶颈分析")
    lines.append("")
    if metrics["operators"]:
        sorted_ops = sorted(metrics["operators"], key=lambda x: x["duration_sec"] or 0, reverse=True)
        top_op = sorted_ops[0]
        lines.append(f"**最耗时算子**: {top_op['operator_name']} ({top_op['duration_sec']}s, "
                     f"占总耗时 {top_op['duration_sec']/metrics['total_duration_sec']*100:.1f}%)")
        lines.append("")
        lines.append("Top 3 耗时算子:")
        lines.append("")
        for i, op in enumerate(sorted_ops[:3], 1):
            lines.append(f"{i}. {op['operator_name']}: {op['duration_sec']}s")
        lines.append("")

    return "\n".join(lines)


def generate_html_report(metrics: Dict, chart_paths: List[Path] = None) -> str:
    """生成HTML报告（含图表嵌入）"""
    md_report = generate_markdown_report(metrics)

    # 简单的HTML包装（Markdown转HTML需要额外库，这里直接用pre标签）
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>血缘报告: {metrics['dataset_version']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #16213e; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #16213e; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .chart-container {{ margin: 20px 0; padding: 15px; border: 1px solid #e0e0e0; border-radius: 8px; }}
        .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .summary h2 {{ color: white; margin-top: 0; }}
    </style>
</head>
<body>
    <h1>血缘报告: {metrics['dataset_version']}</h1>
    <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h2>摘要</h2>
        <p><strong>算子数量:</strong> {metrics['operator_count']} |
        <strong>总输入:</strong> {metrics['total_input']} |
        <strong>总输出:</strong> {metrics['total_output']} |
        <strong>失败率:</strong> {metrics['failure_rate']*100:.2f}% |
        <strong>总耗时:</strong> {metrics['total_duration_sec']:.1f}s</p>
    </div>

    <pre style="white-space: pre-wrap; font-family: inherit; line-height: 1.6;">{md_report}</pre>
</body>
</html>"""
    return html


def generate_report(lineage_path: str, output_path: Optional[str] = None,
                    format: str = "markdown", charts: bool = True,
                    charts_only: bool = False) -> Path:
    """
    生成血缘报告

    Args:
        lineage_path: 血缘文件路径
        output_path: 输出路径
        format: 报告格式（markdown/html/json）
        charts: 是否生成图表
        charts_only: 只生成图表

    Returns:
        输出文件路径
    """
    lineage = load_lineage(lineage_path)
    metrics = compute_metrics(lineage)

    # 输出目录
    if output_path:
        output_path = Path(output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"lineage_report_{metrics['dataset_version']}_{timestamp}.{format}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 只生成图表
    if charts_only:
        chart_dir = output_path.parent / "charts"
        generate_charts(metrics, chart_dir)
        return chart_dir

    # 生成图表
    chart_paths = []
    if charts and PLOTLY_AVAILABLE:
        chart_dir = output_path.parent / "charts"
        chart_paths = generate_charts(metrics, chart_dir)

    # 生成报告
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    elif format == "html":
        html = generate_html_report(metrics, chart_paths)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
    else:  # markdown
        md = generate_markdown_report(metrics)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

    print(f"✅ 报告已生成: {output_path}")
    if chart_paths:
        print(f"   图表目录: {output_path.parent / 'charts'}")

    return output_path


# ===================== 命令行入口 =====================
def main():
    parser = argparse.ArgumentParser(description="血缘报告生成工具（Lineage v2.0）")
    parser.add_argument("--lineage", type=str, required=True,
                        help="血缘文件路径（JSON）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "html", "json"],
                        help="报告格式（默认 markdown）")
    parser.add_argument("--no-charts", action="store_true",
                        help="不生成图表")
    parser.add_argument("--charts-only", action="store_true",
                        help="只生成图表")
    args = parser.parse_args()

    generate_report(
        lineage_path=args.lineage,
        output_path=args.output,
        format=args.format,
        charts=not args.no_charts,
        charts_only=args.charts_only,
    )


if __name__ == "__main__":
    main()
