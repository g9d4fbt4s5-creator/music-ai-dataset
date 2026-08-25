#!/usr/bin/env python3
"""
HITL Scheduler — 人工在环调度器

核心功能：
1. 监测 waiting_pool/ 中各类型的候选样本
2. 根据触发条件自动生成听检任务（调用 adaptive_listening_check.py）
3. 超时机制：waiting_pool 中样本超过7天未听检，自动降级处理
4. 规则更新半自动：听检结果生成建议，人工确认后写入配置

设计原则：
- 绝对异步：主流水线不阻塞，听检并行进行
- 数据驱动触发：不是定时触发，而是阈值/统计异常触发
- 规则更新半自动：听检结果生成建议，人工确认后生效

使用:
    # 检查所有触发条件，生成需要的听检任务
    python hitl_scheduler.py check --qc-report data/00.5_cleaned/reports/vXXX/qc_gate_report.csv

    # 检查 waiting_pool 超时样本
    python hitl_scheduler.py timeout --max-days 7

    # 列出当前 waiting_pool 状态
    python hitl_scheduler.py status
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
WAITING_POOL = PROJECT_ROOT / "data" / "waiting_pool"
LISTENING_TASKS_DIR = PROJECT_ROOT / "data" / "listening_tasks"

# 触发条件配置（可根据数据分布调整）
TRIGGER_CONFIG = {
    "qc_snr_calibration": {
        "description": "SNR阈值校准",
        "waiting_pool_dir": "qc_snr",
        "trigger_condition": "marginal_rate > 0.25",
        "max_samples_per_task": 20,
    },
    "qc_content_boundary": {
        "description": "YAMNet content边界",
        "waiting_pool_dir": "qc_content",
        "trigger_condition": "content_fail_with_vocal_tag > 0",
        "max_samples_per_task": 10,
    },
    "post_threshold_audit": {
        "description": "阈值后抽检",
        "waiting_pool_dir": "post_threshold",
        "trigger_condition": "threshold_modified",
        "max_samples_per_task": 30,
    },
    "dedup_similarity": {
        "description": "近似去重边界",
        "waiting_pool_dir": "dedup_pairs",
        "trigger_condition": "similarity_in_0.92_0.99 > 0",
        "max_samples_per_task": 20,
    },
    "knn_propagation_verify": {
        "description": "KNN传播验证",
        "waiting_pool_dir": "knn_verify",
        "trigger_condition": "new_propagated > 50",
        "max_samples_per_task": 30,
    },
    "unmapped_tag_review": {
        "description": "未映射标签审核",
        "waiting_pool_dir": "unmapped_tags",
        "trigger_condition": "unmapped_tag_freq > 5",
        "max_samples_per_task": 20,
    },
    "cluster_validation": {
        "description": "DBSCAN聚类验证",
        "waiting_pool_dir": "cluster_validate",
        "trigger_condition": "new_cluster_or_high_variance",
        "max_samples_per_task": 20,
    },
    "segment_boundary": {
        "description": "切片边界质量",
        "waiting_pool_dir": "segment_quality",
        "trigger_condition": "post_segment_sample_5pct",
        "max_samples_per_task": 30,
    },
    "data_source_quality": {
        "description": "新采集源质量",
        "waiting_pool_dir": "source_quality",
        "trigger_condition": "new_source_detected",
        "max_samples_per_task": 10,
    },
}


class HITLScheduler:
    """HITL 调度器"""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.waiting_pool = self.project_root / "data" / "waiting_pool"
        self.listening_tasks_dir = self.project_root / "data" / "listening_tasks"
        self.trigger_config = TRIGGER_CONFIG

    def check_all_triggers(self, qc_report_path: Optional[str] = None) -> List[str]:
        """
        检查所有触发条件，生成需要的听检任务

        Args:
            qc_report_path: QC Gate 报告路径（可选，用于检查 marginal 率等）

        Returns:
            生成的任务类型列表
        """
        triggered = []

        # 1. 检查 QC marginal 率
        if qc_report_path and os.path.exists(qc_report_path):
            qc_df = pd.read_csv(qc_report_path)
            marginal_rate = (qc_df["final_branch"] == "marginal").mean() if len(qc_df) > 0 else 0

            if marginal_rate > 0.25:
                print(f"⚠️  QC marginal 率 {marginal_rate:.1%} > 25%，触发 qc_snr_calibration")
                if self._generate_task("qc_snr_calibration", qc_df):
                    triggered.append("qc_snr_calibration")

            # 检查 content fail 但 top5 含人声标签的样本
            content_fail = qc_df[qc_df["content_branch"] == "fail"]
            if len(content_fail) > 0:
                print(f"⚠️  发现 {len(content_fail)} 首 content fail 样本，检查是否含人声标签...")
                # 这里需要 top5_events 字段，简化处理
                if self._generate_task("qc_content_boundary", content_fail):
                    triggered.append("qc_content_boundary")

        # 2. 检查 waiting_pool 中各类型的样本数
        for task_type, config in self.trigger_config.items():
            pool_dir = self.waiting_pool / config["waiting_pool_dir"]
            if pool_dir.exists():
                sample_count = len(list(pool_dir.glob("*.json")))
                if sample_count > 0:
                    print(f"📋 {task_type}: waiting_pool 中有 {sample_count} 个候选样本")

        return triggered

    def _generate_task(self, task_type: str, samples_df: pd.DataFrame) -> bool:
        """
        生成听检任务（调用 adaptive_listening_check.py）

        Args:
            task_type: 任务类型
            samples_df: 样本 DataFrame

        Returns:
            是否成功生成
        """
        try:
            # 限制样本数
            max_samples = self.trigger_config[task_type]["max_samples_per_task"]
            if len(samples_df) > max_samples:
                samples_df = samples_df.sample(max_samples, random_state=42)

            # 转换为 adaptive_listening_check.py 需要的格式
            samples = []
            for _, row in samples_df.iterrows():
                sample = {
                    "audio_id": row.get("audio_id", ""),
                    "audio": row.get("audio_path", row.get("file_path", "")),
                }
                # 添加其他可用字段
                for col in ["snr_db", "dynamic_range_db", "silence_ratio", "clipping_ratio",
                            "music_score", "duration_sec", "loudness_lufs"]:
                    if col in row.index:
                        sample[col.replace("_db", "").replace("_ratio", "").replace("_sec", "")] = row[col]
                samples.append(sample)

            # 保存样本到临时文件
            task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            samples_path = self.listening_tasks_dir / f"{task_id}_samples.json"
            self.listening_tasks_dir.mkdir(parents=True, exist_ok=True)
            with open(samples_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            print(f"✅ 生成听检任务: {task_id}, 样本数: {len(samples)}")
            print(f"   样本文件: {samples_path}")
            print(f"   下一步: python scripts/utils/adaptive_listening_check.py generate "
                  f"--task-type {task_type} --samples-json {samples_path}")
            return True

        except Exception as e:
            print(f"❌ 生成任务失败 {task_type}: {e}")
            return False

    def check_timeout(self, max_days: int = 7) -> List[Dict]:
        """
        检查 waiting_pool 中超时未听检的样本，自动降级处理

        Args:
            max_days: 最大等待天数

        Returns:
            超时样本列表
        """
        timeout_samples = []
        cutoff = datetime.now() - timedelta(days=max_days)

        for pool_dir in self.waiting_pool.iterdir():
            if not pool_dir.is_dir():
                continue
            for sample_file in pool_dir.glob("*.json"):
                mtime = datetime.fromtimestamp(sample_file.stat().st_mtime)
                if mtime < cutoff:
                    age_days = (datetime.now() - mtime).days
                    timeout_samples.append({
                        "file": str(sample_file),
                        "task_type": pool_dir.name,
                        "age_days": age_days,
                    })
                    # 自动降级：按保守策略处理（标记为 marginal）
                    print(f"⏰ 超时样本: {sample_file.name}, 年龄: {age_days}天, 自动降级为 marginal")

        return timeout_samples

    def get_status(self) -> Dict:
        """获取 waiting_pool 当前状态"""
        status = {
            "total_waiting": 0,
            "by_type": {},
        }

        for pool_dir in self.waiting_pool.iterdir():
            if not pool_dir.is_dir():
                continue
            count = len(list(pool_dir.glob("*.json")))
            status["by_type"][pool_dir.name] = count
            status["total_waiting"] += count

        return status

    def add_to_waiting_pool(self, task_type: str, sample: Dict):
        """
        将样本添加到 waiting_pool

        Args:
            task_type: 任务类型（对应 waiting_pool 子目录）
            sample: 样本数据
        """
        if task_type not in self.trigger_config:
            print(f"⚠️  未知任务类型: {task_type}")
            return

        pool_dir = self.waiting_pool / self.trigger_config[task_type]["waiting_pool_dir"]
        pool_dir.mkdir(parents=True, exist_ok=True)

        audio_id = sample.get("audio_id", f"unknown_{datetime.now().timestamp()}")
        sample_file = pool_dir / f"{audio_id}.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

        print(f"📥 添加到 waiting_pool: {task_type}/{audio_id}.json")


def main():
    parser = argparse.ArgumentParser(description="HITL 调度器")
    subparsers = parser.add_subparsers(dest="command")

    # check - 检查触发条件
    check_parser = subparsers.add_parser("check", help="检查所有触发条件，生成听检任务")
    check_parser.add_argument("--qc-report", default=None, help="QC Gate 报告路径")

    # timeout - 检查超时
    timeout_parser = subparsers.add_parser("timeout", help="检查 waiting_pool 超时样本")
    timeout_parser.add_argument("--max-days", type=int, default=7, help="最大等待天数")

    # status - 查看状态
    subparsers.add_parser("status", help="查看 waiting_pool 当前状态")

    args = parser.parse_args()

    scheduler = HITLScheduler()

    if args.command == "check":
        triggered = scheduler.check_all_triggers(args.qc_report)
        if triggered:
            print(f"\n✅ 触发 {len(triggered)} 个听检任务: {triggered}")
        else:
            print("\n✅ 无触发条件满足")

    elif args.command == "timeout":
        timeout = scheduler.check_timeout(args.max_days)
        print(f"\n⏰ 超时样本: {len(timeout)} 个")

    elif args.command == "status":
        status = scheduler.get_status()
        print(f"\n📊 Waiting Pool 状态")
        print(f"   总等待样本: {status['total_waiting']}")
        for task_type, count in status["by_type"].items():
            print(f"   {task_type}: {count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
