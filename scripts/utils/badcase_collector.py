#!/usr/bin/env python3
"""
Badcase Collector — 从各环节自动收集 badcase 到 pool

核心功能：
1. 从 QC Gate 结果中自动识别 YAMNet 误杀等 badcase
2. 从 Label Studio 听检结果中收集人工判定的 badcase
3. 从标注结果中收集分歧/返工样本
4. badcase 生命周期管理：过程态 → 双确认 → 终态

设计原则：
- 只收集"预期正确但实际错误"的样本，不收集普通质量 fail
- 过程态自动收集，终态需双确认
- 所有 badcase 记录来源、类型、原因、时间戳

使用:
    # 从 QC Gate 报告中收集 badcase
    python badcase_collector.py collect-qc --qc-report data/00.5_cleaned/reports/vXXX/qc_gate_report.csv

    # 从 Label Studio 导出结果中收集 badcase
    python badcase_collector.py collect-ls --ls-export data/listening_tasks/xxx.json --task-type qc_snr_calibration

    # 查看当前 badcase 状态
    python badcase_collector.py status

    # 将过程态 badcase 迁移到终态（双确认后）
    python badcase_collector.py promote --audio-id xxx --confirmer "reviewer_name"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
BADCASE_PROCESS_DIR = PROJECT_ROOT / "data" / "03_human_annotation" / "badcase"
BADCASE_FINAL_DIR = PROJECT_ROOT / "data" / "04_final_dataset" / "badcase_pool"
AUTO_COLLECTED_FILE = BADCASE_PROCESS_DIR / "auto_collected.jsonl"

# 人声音乐标签（用于 YAMNet 误杀检测）
VOCAL_MUSIC_TAGS = ["Singing", "Male singing", "Female singing", "Choir",
                     "Vocalization", "Hum", "Beatboxing", "Rapping",
                     "Chant", "Mantra"]

# badcase 类型定义
BADCASE_TYPES = {
    "yamnet_misclassify": "YAMNet误杀正常音乐",
    "snr_threshold_dispute": "SNR阈值边界争议",
    "dedup_false_positive": "去重误判（高度相似但非重复）",
    "knn_propagation_error": "KNN传播标签错误",
    "mapping_error": "标签映射错误",
    "cluster_invalid": "聚类簇语义混乱",
    "segment_boundary_error": "切片切断乐句",
    "annotation_disagreement": "标注分歧",
    "source_contamination": "数据源污染（合成样本混入）",
    "has_vocals_mismatch": "has_vocals与top5_events不一致",
}


class BadcaseCollector:
    """Badcase 收集器"""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.process_dir = self.project_root / "data" / "03_human_annotation" / "badcase"
        self.final_dir = self.project_root / "data" / "04_final_dataset" / "badcase_pool"
        self.auto_file = self.process_dir / "auto_collected.jsonl"

        # 确保目录存在
        self.process_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def add_badcase(self, audio_id: str, badcase_type: str, reason: str,
                     source_stage: str, metadata: Optional[Dict] = None) -> bool:
        """
        添加 badcase 到过程态

        Args:
            audio_id: 音频ID
            badcase_type: badcase类型（见 BADCASE_TYPES）
            reason: 原因描述
            source_stage: 来源阶段
            metadata: 附加元数据

        Returns:
            是否成功添加（重复则跳过）
        """
        # 检查是否已存在
        existing = self._load_auto_collected()
        for item in existing:
            if item["audio_id"] == audio_id and item["badcase_type"] == badcase_type:
                print(f"⏭️  已存在: {audio_id} / {badcase_type}，跳过")
                return False

        badcase = {
            "audio_id": audio_id,
            "badcase_type": badcase_type,
            "badcase_type_desc": BADCASE_TYPES.get(badcase_type, badcase_type),
            "reason": reason,
            "source_stage": source_stage,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "status": "pending_confirmation",
            "confirmer": None,
            "promoted_at": None,
        }

        # 追加到 jsonl 文件
        with open(self.auto_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(badcase, ensure_ascii=False) + "\n")

        print(f"📥 添加 badcase: {audio_id} / {badcase_type}")
        return True

    def collect_from_qc(self, qc_report_path: str) -> List[Dict]:
        """
        从 QC Gate 报告中收集 badcase

        只收集"预期应该 pass 但实际 fail"的样本：
        - YAMNet 误杀：top5 含人声标签，但 music_score < 0.3 判 fail
        - has_vocals 不一致：top5 含 Singing，但 has_vocals=False

        普通质量 fail（SNR<10dB, 静音>80%）不收集。
        """
        if not os.path.exists(qc_report_path):
            print(f"❌ QC 报告不存在: {qc_report_path}")
            return []

        qc_df = pd.read_csv(qc_report_path)
        collected = []

        for _, row in qc_df.iterrows():
            audio_id = row.get("audio_id", "")
            content_branch = row.get("content_branch", "pass")
            final_branch = row.get("final_branch", "pass")

            # 只检查 fail 样本
            if final_branch != "fail":
                continue

            # 解析 reasons 字段（JSON 格式）
            reasons = {}
            reasons_str = row.get("reasons", "{}")
            try:
                reasons = json.loads(reasons_str) if isinstance(reasons_str, str) else reasons_str
            except (json.JSONDecodeError, TypeError):
                pass

            # 检查 YAMNet 误杀：content fail 但可能含人声标签
            # 注意：qc_gate_report 中没有 top5_events 字段，需要从 yamnet_output.csv 关联
            # 这里简化处理：content fail 的样本都标记为候选，后续人工确认
            if content_branch == "fail":
                music_score = reasons.get("content", "")
                collected.append(self.add_badcase(
                    audio_id=audio_id,
                    badcase_type="yamnet_misclassify",
                    reason=f"YAMNet content fail: {music_score}",
                    source_stage="stage3_qc",
                    metadata={"final_branch": final_branch, "reasons": reasons}
                ))

        # 过滤掉 None（重复跳过的）
        collected = [c for c in collected if c]
        print(f"\n✅ 从 QC 报告收集 {len(collected)} 个 badcase")
        return collected

    def collect_from_listening_check(self, ls_export_path: str, task_type: str) -> List[Dict]:
        """
        从 Label Studio 听检结果中收集 badcase

        人工判定为"噪声过大"/"不确定"/"退回marginal"/"判fail"的样本
        """
        if not os.path.exists(ls_export_path):
            print(f"❌ Label Studio 导出文件不存在: {ls_export_path}")
            return []

        with open(ls_export_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        collected = []
        for ann in annotations:
            audio_id = ann.get("id", ann.get("audio_id", ""))
            # 解析标注结果
            result = ann.get("annotations", [{}])[0].get("result", []) if ann.get("annotations") else []

            decision = None
            reason = ""
            for r in result:
                if r.get("from_name") in ["snr_decision", "audit_decision", "is_music"]:
                    decision = r.get("value", {}).get("choices", [None])[0]
                if r.get("from_name") in ["reject_reason", "note", "comment"]:
                    reason = r.get("value", {}).get("text", [""])[0]

            # 判定为负面的样本进 badcase
            if decision in ["noise_too_high", "uncertain", "reject", "marginal", "not_music"]:
                collected.append(self.add_badcase(
                    audio_id=audio_id,
                    badcase_type=f"{task_type}_negative",
                    reason=f"听检判定: {decision}, 备注: {reason}",
                    source_stage="listening_check",
                    metadata={"task_type": task_type, "decision": decision}
                ))

        collected = [c for c in collected if c]
        print(f"\n✅ 从听检结果收集 {len(collected)} 个 badcase")
        return collected

    def collect_from_yamnet_output(self, yamnet_output_path: str) -> List[Dict]:
        """
        从 YAMNet 输出中收集 has_vocals 不一致的 badcase

        top5_events 含 Singing 等标签，但 has_vocals=False
        """
        if not os.path.exists(yamnet_output_path):
            print(f"❌ YAMNet 输出文件不存在: {yamnet_output_path}")
            return []

        yamnet_df = pd.read_csv(yamnet_output_path)
        collected = []

        for _, row in yamnet_df.iterrows():
            audio_id = row.get("audio_id", "")
            has_vocals = row.get("has_vocals", False)
            top5_str = row.get("top5_events", "")

            # 检查 top5 是否含人声标签
            has_vocal_in_top5 = any(tag in str(top5_str) for tag in VOCAL_MUSIC_TAGS)

            if has_vocal_in_top5 and not has_vocals:
                collected.append(self.add_badcase(
                    audio_id=audio_id,
                    badcase_type="has_vocals_mismatch",
                    reason=f"top5含人声标签但has_vocals=False, top5={top5_str}",
                    source_stage="stage3_yamnet",
                    metadata={"has_vocals": has_vocals, "top5_events": top5_str}
                ))

        collected = [c for c in collected if c]
        print(f"\n✅ 从 YAMNet 输出收集 {len(collected)} 个 has_vocals 不一致 badcase")
        return collected

    def promote_to_final(self, audio_id: str, badcase_type: str, confirmer: str) -> bool:
        """
        将过程态 badcase 迁移到终态（双确认后）

        Args:
            audio_id: 音频ID
            badcase_type: badcase类型
            confirmer: 确认人

        Returns:
            是否成功迁移
        """
        existing = self._load_auto_collected()
        found = None
        remaining = []

        for item in existing:
            if item["audio_id"] == audio_id and item["badcase_type"] == badcase_type:
                found = item
            else:
                remaining.append(item)

        if not found:
            print(f"❌ 未找到 badcase: {audio_id} / {badcase_type}")
            return False

        # 更新状态
        found["status"] = "confirmed"
        found["confirmer"] = confirmer
        found["promoted_at"] = datetime.now().isoformat()

        # 写入终态文件
        final_file = self.final_dir / f"{audio_id}_{badcase_type}.json"
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(found, f, ensure_ascii=False, indent=2)

        # 从过程态文件中移除
        with open(self.auto_file, "w", encoding="utf-8") as f:
            for item in remaining:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ 迁移到终态: {audio_id} / {badcase_type} (确认人: {confirmer})")
        return True

    def get_status(self) -> Dict:
        """获取 badcase 当前状态"""
        process_items = self._load_auto_collected()
        final_items = list(self.final_dir.glob("*.json"))

        # 按类型统计
        by_type = {}
        for item in process_items:
            btype = item["badcase_type"]
            by_type[btype] = by_type.get(btype, 0) + 1

        return {
            "process_pending": len(process_items),
            "final_confirmed": len(final_items),
            "by_type": by_type,
        }

    def _load_auto_collected(self) -> List[Dict]:
        """加载过程态 badcase 列表"""
        if not self.auto_file.exists():
            return []
        items = []
        with open(self.auto_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return items


def main():
    parser = argparse.ArgumentParser(description="Badcase 收集器")
    subparsers = parser.add_subparsers(dest="command")

    # collect-qc
    qc_parser = subparsers.add_parser("collect-qc", help="从 QC Gate 报告收集 badcase")
    qc_parser.add_argument("--qc-report", required=True, help="QC Gate 报告路径")

    # collect-ls
    ls_parser = subparsers.add_parser("collect-ls", help="从 Label Studio 导出收集 badcase")
    ls_parser.add_argument("--ls-export", required=True, help="Label Studio 导出文件路径")
    ls_parser.add_argument("--task-type", required=True, help="听检任务类型")

    # collect-yamnet
    yamnet_parser = subparsers.add_parser("collect-yamnet", help="从 YAMNet 输出收集 has_vocals 不一致")
    yamnet_parser.add_argument("--yamnet-output", required=True, help="YAMNet 输出文件路径")

    # status
    subparsers.add_parser("status", help="查看 badcase 当前状态")

    # promote
    promote_parser = subparsers.add_parser("promote", help="将 badcase 迁移到终态")
    promote_parser.add_argument("--audio-id", required=True, help="音频ID")
    promote_parser.add_argument("--badcase-type", required=True, help="badcase类型")
    promote_parser.add_argument("--confirmer", required=True, help="确认人")

    args = parser.parse_args()
    collector = BadcaseCollector()

    if args.command == "collect-qc":
        collector.collect_from_qc(args.qc_report)

    elif args.command == "collect-ls":
        collector.collect_from_listening_check(args.ls_export, args.task_type)

    elif args.command == "collect-yamnet":
        collector.collect_from_yamnet_output(args.yamnet_output)

    elif args.command == "status":
        status = collector.get_status()
        print(f"\n📊 Badcase 状态")
        print(f"   过程态（待确认）: {status['process_pending']}")
        print(f"   终态（已确认）: {status['final_confirmed']}")
        print(f"\n   按类型分布:")
        for btype, count in status["by_type"].items():
            desc = BADCASE_TYPES.get(btype, btype)
            print(f"     {btype}: {count} ({desc})")

    elif args.command == "promote":
        collector.promote_to_final(args.audio_id, args.badcase_type, args.confirmer)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
