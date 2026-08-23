"""
lineage__log.py
算子级血缘记录器（Lineage v2.0）

功能：
- 记录每个算子批次的执行日志（输入/输出/失败/配置/耗时）
- 支持算子嵌套（算子A的输出是算子B的输入）
- 自动记录时间戳、算子版本、模型版本
- 支持失败样本记录和失败原因统计
- 输出标准 lineage.json（v2.0格式）
- 支持增量追加（多次运行合并到同一个lineage文件）

用法：
    from lineage_logger import LineageLogger

    # 初始化
    logger = LineageLogger(
        dataset_version="v20260824_100000",
        output_path="data/lineage/lineage_v20260824_100000.json"
    )

    # 记录算子执行
    _log.log_operator(
        operator_name="yamnet_infer",
        operator_version="1.0",
        model_version="google/yamnet/1",
        input_manifest="data/00_raw_collect/audio_manifest.csv",
        input_count=500,
        output_path="data/00.5_cleaned/reports/v20260824_100000/yamnet_output.csv",
        output_count=499,
        failed_count=1,
        failed_samples=["01M0E9X..."],
        failure_reasons={"audio_corrupted": 1},
        config={"threshold": 0.3, "sample_rate": 16000},
        duration_sec=327.5
    )

    # 记录数据集划分
    _log.log_splits({
        "train": {"count": 399, "source_manifest": "main_pool.csv"},
        "val": {"count": 50, "source_manifest": "main_pool.csv"},
        "test": {"count": 40, "source_manifest": "test_pool.csv"},
        "holdout": {"count": 10, "source_manifest": "holdout_pool.csv"}
    })

    # 保存
    _log.save()

    # 上下文管理器（自动记录耗时）
    with _log.operator("demucs_separate", version="4.0.1", model_version="mdx_extra_q") as op:
        op.set_input("data/00.5_cleaned/reports/v20260824_100000/yamnet_output.csv", count=12)
        op.set_filter("has_vocals == True")
        # ... 执行 demucs ...
        op.set_output("data/01_preprocess/demucs_stems/", count=12, failed_count=0)
        op.set_config({"two_stems": False, "device": "cuda"})
"""
import os
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_LINEAGE_DIR = PROJECT_ROOT / "data" / "lineage"
LINEAGE_VERSION = "2.0"

_log = logging.getLogger(__name__)


# ===================== 数据结构 =====================
@dataclass
class OperatorRecord:
    """单个算子的执行记录"""
    operator_name: str
    operator_version: str
    timestamp: str
    input_manifest: Optional[str] = None
    input_filter: Optional[str] = None
    input_count: Optional[int] = None
    output_path: Optional[str] = None
    output_count: Optional[int] = None
    failed_count: int = 0
    failed_samples: List[str] = field(default_factory=list)
    failure_reasons: Dict[str, int] = field(default_factory=dict)
    model_version: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    duration_sec: Optional[float] = None
    status: str = "success"  # success / failed / partial / skipped
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SplitRecord:
    """数据集划分记录"""
    count: int
    source_manifest: Optional[str] = None
    source_batch: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Lineage:
    """完整的血缘记录"""
    lineage_version: str = LINEAGE_VERSION
    dataset_version: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    operators: List[OperatorRecord] = field(default_factory=list)
    splits: Dict[str, SplitRecord] = field(default_factory=dict)
    upstream_lineage: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        # 转换 splits
        d["splits"] = {k: v.to_dict() if isinstance(v, SplitRecord) else v
                       for k, v in self.splits.items()}
        return d


# ===================== 算子上下文管理器 =====================
class OperatorContext:
    """算子执行上下文（自动记录耗时和状态）"""

    def __init__(self, lineage_logger: 'LineageLogger', name: str,
                 version: str = "1.0", model_version: Optional[str] = None):
        self.logger = lineage_logger
        self.name = name
        self.version = version
        self.model_version = model_version
        self.start_time = None
        self.record = OperatorRecord(
            operator_name=name,
            operator_version=version,
            model_version=model_version,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def __enter__(self):
        self.start_time = time.time()
        _log.info(f"▶ 算子开始: {self.name} v{self.version}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else None
        self.record.duration_sec = round(duration, 2) if duration else None

        if exc_type is not None:
            self.record.status = "failed"
            self.record.error_message = str(exc_val)
            _log.error(f"❌ 算子失败: {self.name} - {exc_val}")
        elif self.record.failed_count > 0 and self.record.output_count and self.record.input_count:
            if self.record.failed_count == self.record.input_count:
                self.record.status = "failed"
            else:
                self.record.status = "partial"
        else:
            self.record.status = "success"

        self._log.operators.append(self.record)
        _log.info(f"✅ 算子完成: {self.name} (耗时 {self.record.duration_sec}s, 状态 {self.record.status})")

        # 不抑制异常
        return False

    def set_input(self, manifest: str, count: int, filter_expr: Optional[str] = None):
        """设置输入"""
        self.record.input_manifest = manifest
        self.record.input_count = count
        self.record.input_filter = filter_expr

    def set_output(self, path: str, count: int, failed_count: int = 0,
                   failed_samples: Optional[List[str]] = None,
                   failure_reasons: Optional[Dict[str, int]] = None):
        """设置输出"""
        self.record.output_path = path
        self.record.output_count = count
        self.record.failed_count = failed_count
        if failed_samples:
            self.record.failed_samples = failed_samples
        if failure_reasons:
            self.record.failure_reasons = failure_reasons

    def set_config(self, config: Dict[str, Any]):
        """设置配置"""
        self.record.config = config

    def add_failure(self, sample_id: str, reason: str):
        """添加失败样本"""
        self.record.failed_samples.append(sample_id)
        self.record.failure_reasons[reason] = self.record.failure_reasons.get(reason, 0) + 1
        self.record.failed_count += 1


# ===================== 主记录器 =====================
class LineageLogger:
    """算子级血缘记录器"""

    def __init__(self, dataset_version: Optional[str] = None,
                 output_path: Optional[str] = None,
                 upstream_lineage: Optional[str] = None,
                 auto_save: bool = False):
        """
        初始化血缘记录器

        Args:
            dataset_version: 数据集版本号（如 v20260824_100000）
            output_path: 输出 lineage.json 路径
            upstream_lineage: 上游血缘文件路径（形成血缘链）
            auto_save: 每次记录算子后自动保存
        """
        self.dataset_version = dataset_version or f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_path = Path(output_path) if output_path else \
            DEFAULT_LINEAGE_DIR / f"lineage_{self.dataset_version}.json"
        self.upstream_lineage = upstream_lineage
        self.auto_save = auto_save

        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果文件已存在，加载（增量追加）
        if self.output_path.exists():
            self.lineage = self._load(self.output_path)
            _log.info(f"加载已有血缘: {self.output_path} ({len(self.lineage.operators)} 个算子)")
        else:
            self.lineage = Lineage(
                lineage_version=LINEAGE_VERSION,
                dataset_version=self.dataset_version,
                created_at=datetime.now(timezone.utc).isoformat(),
                upstream_lineage=upstream_lineage
            )
            _log.info(f"创建新血缘: {self.output_path}")

        self.operators = self.lineage.operators

    def log_operator(self, operator_name: str, operator_version: str = "1.0",
                     model_version: Optional[str] = None,
                     input_manifest: Optional[str] = None,
                     input_filter: Optional[str] = None,
                     input_count: Optional[int] = None,
                     output_path: Optional[str] = None,
                     output_count: Optional[int] = None,
                     failed_count: int = 0,
                     failed_samples: Optional[List[str]] = None,
                     failure_reasons: Optional[Dict[str, int]] = None,
                     config: Optional[Dict[str, Any]] = None,
                     duration_sec: Optional[float] = None,
                     status: str = "success",
                     error_message: Optional[str] = None) -> OperatorRecord:
        """
        记录一个算子的执行

        Args:
            operator_name: 算子名称（如 yamnet_infer, demucs_separate）
            operator_version: 算子版本
            model_version: 模型版本（如 google/yamnet/1, mdx_extra_q）
            input_manifest: 输入清单路径
            input_filter: 输入过滤条件（如 has_vocals == True）
            input_count: 输入样本数
            output_path: 输出路径
            output_count: 输出样本数
            failed_count: 失败样本数
            failed_samples: 失败样本ID列表
            failure_reasons: 失败原因统计 {reason: count}
            config: 算子配置参数
            duration_sec: 执行耗时（秒）
            status: 状态（success/failed/partial/skipped）
            error_message: 错误信息

        Returns:
            OperatorRecord
        """
        record = OperatorRecord(
            operator_name=operator_name,
            operator_version=operator_version,
            model_version=model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_manifest=input_manifest,
            input_filter=input_filter,
            input_count=input_count,
            output_path=output_path,
            output_count=output_count,
            failed_count=failed_count,
            failed_samples=failed_samples or [],
            failure_reasons=failure_reasons or {},
            config=config or {},
            duration_sec=duration_sec,
            status=status,
            error_message=error_message
        )

        self.operators.append(record)
        self.lineage.updated_at = datetime.now(timezone.utc).isoformat()

        _log.info(f"📝 记录算子: {operator_name} v{operator_version} "
                    f"(输入 {input_count} → 输出 {output_count}, 失败 {failed_count})")

        if self.auto_save:
            self.save()

        return record

    @contextmanager
    def operator(self, name: str, version: str = "1.0",
                 model_version: Optional[str] = None) -> OperatorContext:
        """
        算子上下文管理器（自动记录耗时和状态）

        用法：
            with _log.operator("demucs_separate", version="4.0.1") as op:
                op.set_input("input.csv", count=100)
                # ... 执行 ...
                op.set_output("output/", count=98, failed_count=2)
        """
        ctx = OperatorContext(self, name, version, model_version)
        try:
            yield ctx
        finally:
            if self.auto_save:
                self.save()

    def log_splits(self, splits: Dict[str, Dict[str, Any]]):
        """
        记录数据集划分

        Args:
            splits: {split_name: {"count": N, "source_manifest": "xxx.csv", "source_batch": "xxx"}}
        """
        for name, info in splits.items():
            self.lineage.splits[name] = SplitRecord(
                count=info.get("count", 0),
                source_manifest=info.get("source_manifest"),
                source_batch=info.get("source_batch")
            )
        self.lineage.updated_at = datetime.now(timezone.utc).isoformat()
        _log.info(f"📝 记录划分: {', '.join(f'{k}={v.count}' for k, v in self.lineage.splits.items())}")

        if self.auto_save:
            self.save()

    def set_notes(self, notes: str):
        """设置备注"""
        self.lineage.notes = notes
        self.lineage.updated_at = datetime.now(timezone.utc).isoformat()

    def save(self, output_path: Optional[str] = None) -> Path:
        """
        保存血缘到JSON文件

        Args:
            output_path: 输出路径（默认用初始化时的路径）

        Returns:
            保存的文件路径
        """
        path = Path(output_path) if output_path else self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)

        self.lineage.updated_at = datetime.now(timezone.utc).isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.lineage.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        _log.info(f"💾 血缘已保存: {path} ({len(self.operators)} 个算子)")
        return path

    def _load(self, path: Path) -> Lineage:
        """加载已有血缘文件"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lineage = Lineage(
            lineage_version=data.get("lineage_version", LINEAGE_VERSION),
            dataset_version=data.get("dataset_version"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            upstream_lineage=data.get("upstream_lineage"),
            notes=data.get("notes")
        )

        # 加载算子记录
        for op_data in data.get("operators", []):
            lineage.operators.append(OperatorRecord(**op_data))

        # 加载划分
        for name, split_data in data.get("splits", {}).items():
            lineage.splits[name] = SplitRecord(**split_data)

        return lineage

    def get_summary(self) -> Dict[str, Any]:
        """获取血缘摘要"""
        total_input = sum(op.input_count or 0 for op in self.operators)
        total_output = sum(op.output_count or 0 for op in self.operators)
        total_failed = sum(op.failed_count or 0 for op in self.operators)
        total_duration = sum(op.duration_sec or 0 for op in self.operators)

        # 按状态统计
        status_counts = {}
        for op in self.operators:
            status_counts[op.status] = status_counts.get(op.status, 0) + 1

        # 失败原因汇总
        all_failure_reasons = {}
        for op in self.operators:
            for reason, count in op.failure_reasons.items():
                all_failure_reasons[reason] = all_failure_reasons.get(reason, 0) + count

        return {
            "dataset_version": self.lineage.dataset_version,
            "operator_count": len(self.operators),
            "total_input": total_input,
            "total_output": total_output,
            "total_failed": total_failed,
            "failure_rate": round(total_failed / total_input, 4) if total_input > 0 else 0,
            "total_duration_sec": round(total_duration, 2),
            "status_counts": status_counts,
            "failure_reasons": all_failure_reasons,
            "splits": {k: v.count for k, v in self.lineage.splits.items()},
            "operator_names": [op.operator_name for op in self.operators]
        }

    def print_summary(self):
        """打印血缘摘要"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print(f"  血缘摘要: {summary['dataset_version']}")
        print("=" * 60)
        print(f"  算子数量:     {summary['operator_count']}")
        print(f"  总输入样本:   {summary['total_input']}")
        print(f"  总输出样本:   {summary['total_output']}")
        print(f"  总失败样本:   {summary['total_failed']}")
        print(f"  失败率:       {summary['failure_rate']*100:.2f}%")
        print(f"  总耗时:       {summary['total_duration_sec']}s")
        print(f"  状态分布:     {summary['status_counts']}")
        print(f"  失败原因:     {summary['failure_reasons']}")
        print(f"  数据集划分:   {summary['splits']}")
        print(f"  算子列表:     {summary['operator_names']}")
        print("=" * 60 + "\n")


# ===================== 命令行入口 =====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="算子级血缘记录器（Lineage v2.0）")
    parser.add_argument("--dataset-version", type=str, default=None,
                        help="数据集版本号")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 lineage.json 路径")
    parser.add_argument("--summary", type=str, default=None,
                        help="查看指定 lineage.json 的摘要")
    args = parser.parse_args()

    if args.summary:
        # 查看摘要模式
        path = Path(args.summary)
        if path.exists():
            lg = LineageLogger(output_path=str(path))
            lg.print_summary()
        else:
            print(f"❌ 文件不存在: {path}")
    else:
        # 创建示例血缘
        lg = LineageLogger(
            dataset_version=args.dataset_version,
            output_path=args.output
        )

        # 示例：记录几个算子
        lg.log_operator(
            operator_name="import_audio",
            operator_version="1.0",
            input_manifest="downloads/",
            input_count=507,
            output_path="data/00_raw_collect/audio_manifest.csv",
            output_count=500,
            failed_count=7,
            failure_reasons={"duplicate": 5, "corrupted": 2},
            config={"source": "youtube", "format": "mp3"}
        )

        lg.log_operator(
            operator_name="yamnet_infer",
            operator_version="1.0",
            model_version="google/yamnet/1",
            input_manifest="data/00_raw_collect/audio_manifest.csv",
            input_count=500,
            output_path="data/00.5_cleaned/reports/v20260824_100000/yamnet_output.csv",
            output_count=499,
            failed_count=1,
            failure_reasons={"audio_too_short": 1},
            config={"threshold": 0.3, "sample_rate": 16000},
            duration_sec=327.5
        )

        lg.log_splits({
            "train": {"count": 399, "source_manifest": "main_pool.csv"},
            "val": {"count": 50, "source_manifest": "main_pool.csv"},
            "test": {"count": 40, "source_manifest": "test_pool.csv"},
            "holdout": {"count": 10, "source_manifest": "holdout_pool.csv"}
        })

        lg.save()
        lg.print_summary()
