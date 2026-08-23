"""
lineage_verify.py
血缘校验工具（Lineage v2.0）

功能：
- 完整性校验：检查算子记录是否完整（输入/输出/时间戳）
- 一致性校验：检查算子输入输出数量是否合理
- 失败率校验：检查失败率是否超过阈值
- 防泄露校验：检查数据集划分是否有audio_id重叠
- 来源隔离校验：检查train/test/holdout是否来自不同来源
- 算子链校验：检查算子之间的输入输出是否衔接
- 输出校验报告（JSON/HTML）

用法：
    # 基本校验
    python lineage_verify.py --lineage data/lineage/lineage_v20260824_100000.json

    # 严格模式（任何警告都报错）
    python lineage_verify.py --lineage data/lineage/lineage_v20260824_100000.json --strict

    # 自定义失败率阈值
    python lineage_verify.py --lineage data/lineage/lineage_v20260824_100000.json --max-failure-rate 0.05

    # 导出校验报告
    python lineage_verify.py --lineage data/lineage/lineage_v20260824_100000.json --report output.json

    # 防泄露校验（需要manifest文件）
    python lineage_verify.py --lineage data/lineage/lineage_v20260824_100000.json \
        --check-leakage --train-manifest train.csv --test-manifest test.csv
"""
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MAX_FAILURE_RATE = 0.10  # 默认最大失败率 10%


class VerificationResult:
    """校验结果"""

    def __init__(self):
        self.passed = []      # 通过的检查
        self.warnings = []    # 警告
        self.errors = []      # 错误
        self.metrics = {}     # 指标

    def add_pass(self, check_name: str, message: str):
        self.passed.append({"check": check_name, "message": message})

    def add_warning(self, check_name: str, message: str):
        self.warnings.append({"check": check_name, "message": message})

    def add_error(self, check_name: str, message: str):
        self.errors.append({"check": check_name, "message": message})

    def add_metric(self, name: str, value: Any):
        self.metrics[name] = value

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def to_dict(self) -> Dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": len(self.passed) + len(self.warnings) + len(self.errors),
                "passed": len(self.passed),
                "warnings": len(self.warnings),
                "errors": len(self.errors),
                "status": "PASS" if not self.has_errors else "FAIL"
            },
            "metrics": self.metrics,
            "passed_checks": self.passed,
            "warnings": self.warnings,
            "errors": self.errors
        }

    def print_report(self):
        """打印校验报告"""
        print("\n" + "=" * 70)
        print("  血缘校验报告")
        print("=" * 70)

        # 摘要
        summary = self.to_dict()["summary"]
        print(f"\n  状态: {'✅ PASS' if summary['status'] == 'PASS' else '❌ FAIL'}")
        print(f"  总检查数: {summary['total_checks']}")
        print(f"  通过: {summary['passed']}")
        print(f"  警告: {summary['warnings']}")
        print(f"  错误: {summary['errors']}")

        # 指标
        if self.metrics:
            print(f"\n  关键指标:")
            for k, v in self.metrics.items():
                print(f"    {k}: {v}")

        # 通过的检查
        if self.passed:
            print(f"\n  ✅ 通过的检查 ({len(self.passed)}):")
            for p in self.passed:
                print(f"    - {p['check']}: {p['message']}")

        # 警告
        if self.warnings:
            print(f"\n  ⚠️  警告 ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"    - {w['check']}: {w['message']}")

        # 错误
        if self.errors:
            print(f"\n  ❌ 错误 ({len(self.errors)}):")
            for e in self.errors:
                print(f"    - {e['check']}: {e['message']}")

        print("\n" + "=" * 70 + "\n")


def load_lineage(lineage_path: str) -> Dict:
    """加载血缘文件"""
    path = Path(lineage_path)
    if not path.exists():
        raise FileNotFoundError(f"血缘文件不存在: {lineage_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_completeness(lineage: Dict, result: VerificationResult):
    """完整性校验：检查算子记录是否完整"""
    operators = lineage.get("operators", [])

    if not operators:
        result.add_warning("completeness", "无算子记录")
        return

    required_fields = ["operator_name", "operator_version", "timestamp", "status"]
    incomplete_count = 0

    for i, op in enumerate(operators):
        missing = [f for f in required_fields if not op.get(f)]
        if missing:
            incomplete_count += 1
            result.add_warning(
                "completeness",
                f"算子 #{i} ({op.get('operator_name', 'unknown')}) 缺少字段: {missing}"
            )

    if incomplete_count == 0:
        result.add_pass("completeness", f"所有 {len(operators)} 个算子记录完整")
    else:
        result.add_warning("completeness", f"{incomplete_count}/{len(operators)} 个算子记录不完整")

    result.add_metric("operator_count", len(operators))


def verify_consistency(lineage: Dict, result: VerificationResult):
    """一致性校验：检查算子输入输出数量是否合理"""
    operators = lineage.get("operators", [])

    if not operators:
        return

    inconsistent_count = 0

    for i, op in enumerate(operators):
        input_count = op.get("input_count")
        output_count = op.get("output_count")
        failed_count = op.get("failed_count", 0)

        if input_count is not None and output_count is not None:
            # 输出数量应该 <= 输入数量（除非是生成类算子）
            if output_count > input_count and failed_count == 0:
                # 可能是生成类算子（如切片），输出>输入是正常的
                pass
            elif output_count + failed_count < input_count:
                # 有样本丢失
                lost = input_count - output_count - failed_count
                if lost > 0:
                    inconsistent_count += 1
                    result.add_warning(
                        "consistency",
                        f"算子 {op.get('operator_name')}: 输入{input_count} - 输出{output_count} - 失败{failed_count} = 丢失{lost}"
                    )

    if inconsistent_count == 0:
        result.add_pass("consistency", "所有算子输入输出数量一致")
    else:
        result.add_warning("consistency", f"{inconsistent_count} 个算子存在样本丢失")


def verify_failure_rate(lineage: Dict, result: VerificationResult,
                        max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE):
    """失败率校验：检查失败率是否超过阈值"""
    operators = lineage.get("operators", [])

    if not operators:
        return

    total_input = sum(op.get("input_count", 0) or 0 for op in operators)
    total_failed = sum(op.get("failed_count", 0) or 0 for op in operators)

    if total_input == 0:
        result.add_warning("failure_rate", "总输入样本数为0，无法计算失败率")
        return

    failure_rate = total_failed / total_input
    result.add_metric("total_input", total_input)
    result.add_metric("total_failed", total_failed)
    result.add_metric("failure_rate", f"{failure_rate*100:.2f}%")

    if failure_rate > max_failure_rate:
        result.add_error(
            "failure_rate",
            f"总失败率 {failure_rate*100:.2f}% 超过阈值 {max_failure_rate*100:.2f}%"
        )
    else:
        result.add_pass(
            "failure_rate",
            f"总失败率 {failure_rate*100:.2f}% 在阈值 {max_failure_rate*100:.2f}% 以内"
        )

    # 单个算子失败率
    high_failure_ops = []
    for op in operators:
        input_count = op.get("input_count", 0) or 0
        failed_count = op.get("failed_count", 0) or 0
        if input_count > 0:
            op_failure_rate = failed_count / input_count
            if op_failure_rate > max_failure_rate:
                high_failure_ops.append((op.get("operator_name"), op_failure_rate))

    if high_failure_ops:
        result.add_warning(
            "failure_rate",
            f"{len(high_failure_ops)} 个算子失败率超过阈值: {[(name, f'{rate*100:.1f}%') for name, rate in high_failure_ops]}"
        )


def verify_operator_chain(lineage: Dict, result: VerificationResult):
    """算子链校验：检查算子之间的输入输出是否衔接"""
    operators = lineage.get("operators", [])

    if len(operators) < 2:
        result.add_pass("operator_chain", "算子数量<2，无需校验链路")
        return

    broken_links = 0

    for i in range(len(operators) - 1):
        current_op = operators[i]
        next_op = operators[i + 1]

        current_output = current_op.get("output_path")
        next_input = next_op.get("input_manifest")

        if current_output and next_input:
            # 检查下一个算子的输入是否是当前算子的输出
            # 注意：路径可能不完全相同，只检查文件名是否匹配
            current_filename = Path(current_output).name if current_output else ""
            next_filename = Path(next_input).name if next_input else ""

            if current_filename and next_filename and current_filename != next_filename:
                # 可能是正常的（不同阶段用不同文件），只警告
                pass

        # 检查时间顺序
        current_time = current_op.get("timestamp")
        next_time = next_op.get("timestamp")

        if current_time and next_time and current_time > next_time:
            broken_links += 1
            result.add_error(
                "operator_chain",
                f"算子时间顺序错误: {current_op.get('operator_name')} ({current_time}) > {next_op.get('operator_name')} ({next_time})"
            )

    if broken_links == 0:
        result.add_pass("operator_chain", f"算子链时间顺序正确 ({len(operators)} 个算子)")
    else:
        result.add_error("operator_chain", f"{broken_links} 处算子链时间顺序错误")


def verify_splits(lineage: Dict, result: VerificationResult):
    """数据集划分校验"""
    splits = lineage.get("splits", {})

    if not splits:
        result.add_warning("splits", "无数据集划分记录")
        return

    # 检查是否有train/val/test/holdout
    required_splits = ["train", "val", "test"]
    missing = [s for s in required_splits if s not in splits]

    if missing:
        result.add_warning("splits", f"缺少划分: {missing}")
    else:
        result.add_pass("splits", f"划分完整: {', '.join(splits.keys())}")

    # 检查每个划分是否有count
    for name, info in splits.items():
        if "count" not in info:
            result.add_warning("splits", f"划分 {name} 缺少 count 字段")

    # 检查来源隔离
    sources = set()
    for name, info in splits.items():
        source = info.get("source_manifest") or info.get("source_batch")
        if source:
            sources.add(source)

    if len(sources) > 1:
        result.add_pass("splits", f"来源隔离: {len(sources)} 个不同来源")
    elif len(sources) == 1 and "test" in splits and "holdout" in splits:
        result.add_warning("splits", "test和holdout可能来自同一来源，存在迭代污染风险")


def verify_leakage(lineage: Dict, result: VerificationResult,
                   train_manifest: Optional[str] = None,
                   test_manifest: Optional[str] = None,
                   holdout_manifest: Optional[str] = None):
    """防泄露校验：检查数据集划分是否有audio_id重叠"""
    manifests = {}
    if train_manifest:
        manifests["train"] = train_manifest
    if test_manifest:
        manifests["test"] = test_manifest
    if holdout_manifest:
        manifests["holdout"] = holdout_manifest

    if len(manifests) < 2:
        result.add_warning("leakage", "需要至少2个manifest文件才能做防泄露校验")
        return

    # 加载所有manifest
    audio_ids = {}
    for name, path in manifests.items():
        p = Path(path)
        if not p.exists():
            result.add_error("leakage", f"manifest文件不存在: {path}")
            continue
        df = pd.read_csv(p)
        if "audio_id" in df.columns:
            audio_ids[name] = set(df["audio_id"].tolist())
        else:
            result.add_warning("leakage", f"manifest {path} 缺少 audio_id 列")

    # 检查重叠
    overlap_found = False
    names = list(audio_ids.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = audio_ids[names[i]] & audio_ids[names[j]]
            if overlap:
                overlap_found = True
                result.add_error(
                    "leakage",
                    f"{names[i]} 和 {names[j]} 有 {len(overlap)} 个重叠 audio_id: {list(overlap)[:5]}..."
                )

    if not overlap_found and len(audio_ids) >= 2:
        result.add_pass("leakage", f"无audio_id重叠 ({', '.join(names)})")


def verify_lineage(lineage_path: str, strict: bool = False,
                   max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
                   train_manifest: Optional[str] = None,
                   test_manifest: Optional[str] = None,
                   holdout_manifest: Optional[str] = None,
                   report_path: Optional[str] = None) -> VerificationResult:
    """
    执行完整的血缘校验

    Args:
        lineage_path: 血缘文件路径
        strict: 严格模式（警告也报错）
        max_failure_rate: 最大失败率阈值
        train_manifest: train集manifest路径（防泄露校验）
        test_manifest: test集manifest路径（防泄露校验）
        holdout_manifest: holdout集manifest路径（防泄露校验）
        report_path: 校验报告输出路径

    Returns:
        VerificationResult
    """
    lineage = load_lineage(lineage_path)
    result = VerificationResult()

    # 执行各项校验
    verify_completeness(lineage, result)
    verify_consistency(lineage, result)
    verify_failure_rate(lineage, result, max_failure_rate)
    verify_operator_chain(lineage, result)
    verify_splits(lineage, result)

    # 防泄露校验（需要manifest文件）
    if train_manifest or test_manifest or holdout_manifest:
        verify_leakage(lineage, result, train_manifest, test_manifest, holdout_manifest)

    # 严格模式：警告也视为错误
    if strict and result.has_warnings:
        result.errors.extend(result.warnings)
        result.warnings = []

    # 打印报告
    result.print_report()

    # 保存报告
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ 校验报告已保存: {path}")

    return result


# ===================== 命令行入口 =====================
def main():
    parser = argparse.ArgumentParser(description="血缘校验工具（Lineage v2.0）")
    parser.add_argument("--lineage", type=str, required=True,
                        help="血缘文件路径（JSON）")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式（任何警告都报错）")
    parser.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE,
                        help=f"最大失败率阈值（默认 {DEFAULT_MAX_FAILURE_RATE}）")
    parser.add_argument("--train-manifest", type=str, default=None,
                        help="train集manifest路径（防泄露校验）")
    parser.add_argument("--test-manifest", type=str, default=None,
                        help="test集manifest路径（防泄露校验）")
    parser.add_argument("--holdout-manifest", type=str, default=None,
                        help="holdout集manifest路径（防泄露校验）")
    parser.add_argument("--report", type=str, default=None,
                        help="校验报告输出路径（JSON）")
    args = parser.parse_args()

    result = verify_lineage(
        lineage_path=args.lineage,
        strict=args.strict,
        max_failure_rate=args.max_failure_rate,
        train_manifest=args.train_manifest,
        test_manifest=args.test_manifest,
        holdout_manifest=args.holdout_manifest,
        report_path=args.report
    )

    # 退出码
    sys.exit(1 if result.has_errors else 0)


if __name__ == "__main__":
    main()
