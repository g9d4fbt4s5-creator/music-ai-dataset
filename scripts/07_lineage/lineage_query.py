"""
lineage_query.py
血缘查询工具（Lineage v2.0）

功能：
- 按算子名称查询执行记录
- 按样本ID查询处理路径（溯源）
- 按状态查询（成功/失败/部分失败）
- 查询失败样本和失败原因
- 查询算子耗时和瓶颈分析
- 输出查询结果为表格/JSON/CSV

用法：
    # 查看所有算子
    python lineage_query.py --lineage data/lineage/lineage_v20260824_100000.json --list-operators

    # 按名称查询
    python lineage_query.py --lineage data/lineage/lineage_v20260824_100000.json --operator yamnet_infer

    # 查询失败样本
    python lineage_query.py --lineage data/lineage/lineage_v20260824_100000.json --failed-samples

    # 查询失败原因统计
    python lineage_query.py --lineage data/lineage/lineage_v20260824_100000.json --failure-reasons

    # 溯源：查某个样本经过了哪些算子
    python lineage_query.py --lineage data/lineage/lineage_v20260824_100000.json --trace-sample 01M0E9X...

    # 导出为CSV
    python lineage_query.py --lineage data/lineage/lineage_v20260824_100000.json --export-csv output.csv
"""
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LINEAGE_DIR = PROJECT_ROOT / "data" / "lineage"


def load_lineage(lineage_path: str) -> Dict:
    """加载血缘文件"""
    path = Path(lineage_path)
    if not path.exists():
        # 尝试在默认目录找
        path = DEFAULT_LINEAGE_DIR / lineage_path
        if not path.exists():
            raise FileNotFoundError(f"血缘文件不存在: {lineage_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def operators_to_df(operators: List[Dict]) -> pd.DataFrame:
    """算子记录转DataFrame"""
    rows = []
    for op in operators:
        rows.append({
            "operator_name": op.get("operator_name"),
            "operator_version": op.get("operator_version"),
            "model_version": op.get("model_version"),
            "timestamp": op.get("timestamp"),
            "input_count": op.get("input_count"),
            "output_count": op.get("output_count"),
            "failed_count": op.get("failed_count", 0),
            "failure_rate": round(op.get("failed_count", 0) / op.get("input_count", 1), 4) if op.get("input_count") else 0,
            "duration_sec": op.get("duration_sec"),
            "status": op.get("status"),
            "input_manifest": op.get("input_manifest"),
            "output_path": op.get("output_path"),
            "input_filter": op.get("input_filter"),
        })
    return pd.DataFrame(rows)


def list_operators(lineage: Dict):
    """列出所有算子"""
    operators = lineage.get("operators", [])
    df = operators_to_df(operators)

    print("\n" + "=" * 80)
    print(f"  算子列表: {lineage.get('dataset_version', 'N/A')}")
    print("=" * 80)

    if df.empty:
        print("  (无算子记录)")
    else:
        # 显示关键列
        display_cols = ["operator_name", "operator_version", "status",
                        "input_count", "output_count", "failed_count",
                        "failure_rate", "duration_sec"]
        print(df[display_cols].to_string(index=False))

        # 统计
        print(f"\n  总计: {len(df)} 个算子")
        print(f"  成功: {(df['status'] == 'success').sum()}")
        print(f"  部分失败: {(df['status'] == 'partial').sum()}")
        print(f"  失败: {(df['status'] == 'failed').sum()}")
        print(f"  跳过: {(df['status'] == 'skipped').sum()}")
        print(f"  总耗时: {df['duration_sec'].sum():.1f}s")

    print("=" * 80 + "\n")
    return df


def query_operator(lineage: Dict, operator_name: str):
    """按名称查询算子"""
    operators = lineage.get("operators", [])
    matches = [op for op in operators if operator_name.lower() in op.get("operator_name", "").lower()]

    print(f"\n查询算子: '{operator_name}'，找到 {len(matches)} 条记录\n")

    for i, op in enumerate(matches, 1):
        print(f"--- 记录 {i} ---")
        print(f"  算子名称:     {op.get('operator_name')}")
        print(f"  算子版本:     {op.get('operator_version')}")
        print(f"  模型版本:     {op.get('model_version', 'N/A')}")
        print(f"  时间戳:       {op.get('timestamp')}")
        print(f"  状态:         {op.get('status')}")
        print(f"  输入清单:     {op.get('input_manifest', 'N/A')}")
        print(f"  输入过滤:     {op.get('input_filter', 'N/A')}")
        print(f"  输入数量:     {op.get('input_count', 'N/A')}")
        print(f"  输出路径:     {op.get('output_path', 'N/A')}")
        print(f"  输出数量:     {op.get('output_count', 'N/A')}")
        print(f"  失败数量:     {op.get('failed_count', 0)}")
        print(f"  失败样本:     {op.get('failed_samples', [])[:10]}{'...' if len(op.get('failed_samples', [])) > 10 else ''}")
        print(f"  失败原因:     {op.get('failure_reasons', {})}")
        print(f"  耗时:         {op.get('duration_sec', 'N/A')}s")
        print(f"  配置:         {json.dumps(op.get('config', {}), ensure_ascii=False, indent=2)}")
        if op.get("error_message"):
            print(f"  错误信息:     {op.get('error_message')}")
        print()

    return matches


def query_failed_samples(lineage: Dict):
    """查询所有失败样本"""
    operators = lineage.get("operators", [])
    all_failed = {}

    for op in operators:
        for sample_id in op.get("failed_samples", []):
            if sample_id not in all_failed:
                all_failed[sample_id] = []
            all_failed[sample_id].append({
                "operator": op.get("operator_name"),
                "timestamp": op.get("timestamp"),
            })

    print(f"\n失败样本汇总: {len(all_failed)} 个唯一样本\n")

    if all_failed:
        df = pd.DataFrame([
            {"sample_id": sid, "failed_operators": ", ".join(f["operator"] for f in failures), "failure_count": len(failures)}
            for sid, failures in all_failed.items()
        ])
        print(df.to_string(index=False))
    else:
        print("  (无失败样本)")

    print()
    return all_failed


def query_failure_reasons(lineage: Dict):
    """查询失败原因统计"""
    operators = lineage.get("operators", [])
    all_reasons = {}

    for op in operators:
        for reason, count in op.get("failure_reasons", {}).items():
            all_reasons[reason] = all_reasons.get(reason, 0) + count

    print(f"\n失败原因统计:\n")

    if all_reasons:
        df = pd.DataFrame([
            {"failure_reason": reason, "count": count}
            for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1])
        ])
        df["percentage"] = (df["count"] / df["count"].sum() * 100).round(2)
        print(df.to_string(index=False))
    else:
        print("  (无失败记录)")

    print()
    return all_reasons


def trace_sample(lineage: Dict, sample_id: str):
    """溯源：查某个样本经过了哪些算子"""
    operators = lineage.get("operators", [])

    print(f"\n样本溯源: {sample_id}\n")
    print("处理路径:")
    print("-" * 60)

    found = False
    for i, op in enumerate(operators, 1):
        # 检查样本是否在失败列表中
        failed = sample_id in op.get("failed_samples", [])

        # 检查样本是否可能经过这个算子（基于输入/输出数量）
        # 注意：算子级血缘不记录每条样本，只能推断
        status = "❌ 失败" if failed else "✅ 经过"
        print(f"  {i}. {op.get('operator_name')} v{op.get('operator_version')} "
              f"[{op.get('timestamp', 'N/A')}] → {status}")

        if failed:
            found = True
            # 找失败原因
            for reason, count in op.get("failure_reasons", {}).items():
                print(f"       失败原因: {reason}")

    print("-" * 60)

    if not found:
        print(f"\n  样本 {sample_id} 未在任何算子的失败列表中找到。")
        print(f"  注意：算子级血缘只记录失败样本，成功样本不单独记录。")
        print(f"  如需完整样本级溯源，请升级到样本级血缘（Lineage v3.0）。")

    print()


def query_bottleneck(lineage: Dict):
    """查询算子耗时瓶颈"""
    operators = lineage.get("operators", [])
    df = operators_to_df(operators)

    if df.empty:
        print("\n无算子记录\n")
        return

    df = df.dropna(subset=["duration_sec"])
    df = df.sort_values("duration_sec", ascending=False)

    print(f"\n算子耗时分析（瓶颈分析）:\n")
    print(df[["operator_name", "duration_sec", "input_count", "output_count", "status"]].to_string(index=False))

    total = df["duration_sec"].sum()
    print(f"\n  总耗时: {total:.1f}s")
    print(f"  最耗时算子: {df.iloc[0]['operator_name']} ({df.iloc[0]['duration_sec']:.1f}s, {df.iloc[0]['duration_sec']/total*100:.1f}%)")
    print()


def export_csv(lineage: Dict, output_path: str):
    """导出算子记录为CSV"""
    operators = lineage.get("operators", [])
    df = operators_to_df(operators)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

    print(f"\n✅ 已导出 {len(df)} 条算子记录到: {path}\n")


def show_summary(lineage: Dict):
    """显示血缘摘要"""
    operators = lineage.get("operators", [])
    splits = lineage.get("splits", {})

    total_input = sum(op.get("input_count", 0) or 0 for op in operators)
    total_output = sum(op.get("output_count", 0) or 0 for op in operators)
    total_failed = sum(op.get("failed_count", 0) or 0 for op in operators)
    total_duration = sum(op.get("duration_sec", 0) or 0 for op in operators)

    status_counts = {}
    for op in operators:
        s = op.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print("\n" + "=" * 60)
    print(f"  血缘摘要: {lineage.get('dataset_version', 'N/A')}")
    print(f"  血缘版本: {lineage.get('lineage_version', 'N/A')}")
    print(f"  创建时间: {lineage.get('created_at', 'N/A')}")
    print(f"  更新时间: {lineage.get('updated_at', 'N/A')}")
    print("=" * 60)
    print(f"  算子数量:     {len(operators)}")
    print(f"  总输入样本:   {total_input}")
    print(f"  总输出样本:   {total_output}")
    print(f"  总失败样本:   {total_failed}")
    print(f"  失败率:       {total_failed/total_input*100:.2f}%" if total_input > 0 else "  失败率:       N/A")
    print(f"  总耗时:       {total_duration:.1f}s")
    print(f"  状态分布:     {status_counts}")
    print(f"  上游血缘:     {lineage.get('upstream_lineage', 'N/A')}")
    splits_str = ", ".join(f"{k}={v.get('count', 0)}" for k, v in splits.items())
    print(f"  数据集划分:   {splits_str}")
    print("=" * 60 + "\n")


# ===================== 命令行入口 =====================
def main():
    parser = argparse.ArgumentParser(description="血缘查询工具（Lineage v2.0）")
    parser.add_argument("--lineage", type=str, required=True,
                        help="血缘文件路径（JSON）")
    parser.add_argument("--list-operators", action="store_true",
                        help="列出所有算子")
    parser.add_argument("--operator", type=str, default=None,
                        help="按名称查询算子")
    parser.add_argument("--failed-samples", action="store_true",
                        help="查询所有失败样本")
    parser.add_argument("--failure-reasons", action="store_true",
                        help="查询失败原因统计")
    parser.add_argument("--trace-sample", type=str, default=None,
                        help="溯源：查某个样本经过了哪些算子")
    parser.add_argument("--bottleneck", action="store_true",
                        help="查询算子耗时瓶颈")
    parser.add_argument("--summary", action="store_true",
                        help="显示血缘摘要")
    parser.add_argument("--export-csv", type=str, default=None,
                        help="导出算子记录为CSV")
    args = parser.parse_args()

    # 加载血缘
    lineage = load_lineage(args.lineage)

    # 执行查询
    if args.summary:
        show_summary(lineage)

    if args.list_operators:
        list_operators(lineage)

    if args.operator:
        query_operator(lineage, args.operator)

    if args.failed_samples:
        query_failed_samples(lineage)

    if args.failure_reasons:
        query_failure_reasons(lineage)

    if args.trace_sample:
        trace_sample(lineage, args.trace_sample)

    if args.bottleneck:
        query_bottleneck(lineage)

    if args.export_csv:
        export_csv(lineage, args.export_csv)

    # 如果没有指定任何查询，默认显示摘要
    if not any([args.list_operators, args.operator, args.failed_samples,
                args.failure_reasons, args.trace_sample, args.bottleneck,
                args.summary, args.export_csv]):
        show_summary(lineage)
        list_operators(lineage)


if __name__ == "__main__":
    main()
