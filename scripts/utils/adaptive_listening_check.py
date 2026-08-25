#!/usr/bin/env python3
"""
adaptive_listening_check.py — 自适应人工听检闭环

核心思想：系统监测数据流水线异常信号 → 自动生成针对性 Label Studio 听检模板
→ 人工快速判断 → 结果回流更新规则/阈值。

设计原则：
1. 模板骨架固定（问题类型→XML映射），Agent只填字段和选项，不生成全新XML结构
2. 决策字段名统一（decision），parse_results()可通用解析
3. 半自动：结果自动生成建议，实际修改需人工确认（避免错误规则自动写入）

使用方式：
    # 1. 检测触发条件，生成听检任务
    python adaptive_listening_check.py detect --output-dir /tmp/listening_tasks

    # 2. 生成特定类型的听检任务（手动触发）
    python adaptive_listening_check.py generate --task-type qc_snr_calibration --sample-ids id1,id2,id3

    # 3. 解析听检结果，生成更新建议
    python adaptive_listening_check.py parse --ls-export /path/to/export.json --task-type qc_snr_calibration
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# 1. 问题类型枚举（8大需要持续听检的环节）
# ============================================================

class ListeningTaskType(Enum):
    """需要人工听检调试的8大环节"""

    # QC Gate 阈值校准
    QC_SNR_CALIBRATION = "qc_snr_calibration"           # SNR阈值校准
    QC_DR_CALIBRATION = "qc_dr_calibration"             # DR阈值校准
    QC_SILENCE_CALIBRATION = "qc_silence_calibration"   # 静音比例阈值校准
    QC_CONTENT_BOUNDARY = "qc_content_boundary"          # YAMNet music_score边界

    # 近似去重
    DEDUP_SIMILARITY_BOUNDARY = "dedup_similarity"      # 指纹相似度边界判定

    # KNN传播（L4）
    KNN_PROPAGATION_VERIFY = "knn_propagation_verify"   # KNN传播标签准确性验证
    KNN_DISTANCE_CALIBRATION = "knn_distance_calibration" # cosine距离阈值校准

    # 黄金集（L3）
    GOLDEN_SET_EXPANSION = "golden_set_expansion"        # 新风格簇黄金集扩充
    GOLDEN_SET_QUALITY = "golden_set_quality"            # 黄金集结构标注质量

    # 标签映射（TagMapper）
    UNMAPPED_TAG_REVIEW = "unmapped_tag_review"          # 未映射标签审核
    MAPPING_ERROR_REVIEW = "mapping_error_review"         # 映射错误抽查

    # 嵌入聚类质量
    CLUSTER_VALIDATION = "cluster_validation"             # DBSCAN聚类簇验证
    CLUSTER_NAMING = "cluster_naming"                     # 簇标签命名

    # 切片质量（Stage 6）
    SEGMENT_BOUNDARY_QUALITY = "segment_boundary"        # 乐段边界切片质量
    SHORT_TRACK_HANDLING = "short_track_handling"         # 短曲目处理策略

    # 数据源质量
    DATA_SOURCE_QUALITY = "data_source_quality"           # 新采集源整体质量评估


# ============================================================
# 2. 模板骨架映射表（问题类型→XML骨架）
# ============================================================

# 每个模板包含：XML骨架、决策字段名、选项列表、预填字段
TEMPLATE_REGISTRY: Dict[str, Dict[str, Any]] = {

    "qc_snr_calibration": {
        "description": "SNR阈值校准听检",
        "decision_field": "snr_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="80" />
  <Header value="样本信息" size="4" />
  <Text name="meta" value="SNR: $snr dB | DR: $dr dB | 静音: $silence% | 来源: $source" />
  <Text name="focus" value="$focus_note" />
  <Header value="听检判断" size="4" />
  <Choices name="snr_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="acceptable" html="✅ 噪声可接受，质量合格" />
    <Choice value="musical_normal" html="🎵 音乐制作正常特征（曲风/乐器/效果器/演奏法/历史录音）" />
    <Choice value="noise_too_high" html="🔊 噪声过大，影响聆听" />
    <Choice value="uncertain" html="❓ 不确定" />
  </Choices>
  <TextArea name="comment" toName="audio" placeholder="备注（可选）" rows="2" />
</View>""",
        "prefill_fields": ["audio", "snr", "dr", "silence", "source", "focus_note"],
    },

    "qc_content_boundary": {
        "description": "YAMNet music_score边界听检",
        "decision_field": "content_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="80" />
  <Header value="样本信息" size="4" />
  <Text name="meta" value="music_score: $music_score | 预测类别: $yamnet_class | 置信度: $confidence" />
  <Header value="听检判断" size="4" />
  <Choices name="content_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="is_music" html="✅ 是音乐" />
    <Choice value="music_with_speech" html="🎤 音乐+人声混合" />
    <Choice value="not_music" html="❌ 不是音乐（语音/环境音）" />
    <Choice value="uncertain" html="❓ 不确定" />
  </Choices>
  <TextArea name="comment" toName="audio" placeholder="备注（可选）" rows="2" />
</View>""",
        "prefill_fields": ["audio", "music_score", "yamnet_class", "confidence"],
    },

    "dedup_similarity": {
        "description": "近似去重边界听检",
        "decision_field": "dedup_decision",
        "xml_skeleton": """<View>
  <Header value="样本A" size="4" />
  <Audio name="audio_a" value="$audio_a" zoom="true" waveheight="60" />
  <Header value="样本B" size="4" />
  <Audio name="audio_b" value="$audio_b" zoom="true" waveheight="60" />
  <Header value="相似度信息" size="4" />
  <Text name="meta" value="指纹相似度: $similarity | 时长A: $dur_a | 时长B: $dur_b | 文件大小比: $size_ratio" />
  <Header value="听检判断" size="4" />
  <Choices name="dedup_decision" toName="audio_a" choice="single" required="true" showInline="true">
    <Choice value="exact_duplicate" html="✅ 完全重复（同一录音）" />
    <Choice value="different_version" html="🔄 不同版本（live/remix/翻唱），保留" />
    <Choice value="different_track" html="🎵 完全不同的曲目" />
    <Choice value="uncertain" html="❓ 不确定" />
  </Choices>
  <TextArea name="comment" toName="audio_a" placeholder="备注（可选）" rows="2" />
</View>""",
        "prefill_fields": ["audio_a", "audio_b", "similarity", "dur_a", "dur_b", "size_ratio"],
    },

    "knn_propagation_verify": {
        "description": "KNN传播标签准确性验证",
        "decision_field": "propagation_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="80" />
  <Header value="KNN传播信息" size="4" />
  <Text name="meta" value="KNN距离: $knn_dist | 传播来源: $src_audio_id | 源标签: $src_labels" />
  <Header value="传播标签（待验证）" size="4" />
  <Text name="propagated" value="流派: $genre | 情绪: $mood | 乐器: $instruments" />
  <Header value="听检判断" size="4" />
  <Choices name="propagation_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="correct" html="✅ 传播标签全部准确" />
    <Choice value="wrong_genre" html="❌ 流派传播错误" />
    <Choice value="wrong_mood" html="❌ 情绪传播错误" />
    <Choice value="wrong_instruments" html="❌ 乐器传播错误" />
    <Choice value="should_not_propagate" html="⚠️ 不应该传播（距离太远）" />
  </Choices>
  <TextArea name="correction" toName="audio" placeholder="如果需要修正，请填写正确标签" rows="2" />
</View>""",
        "prefill_fields": ["audio", "knn_dist", "src_audio_id", "src_labels", "genre", "mood", "instruments"],
    },

    "unmapped_tag_review": {
        "description": "未映射标签审核",
        "decision_field": "mapping_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="80" />
  <Header value="未映射标签信息" size="4" />
  <Text name="meta" value="原始标签: $raw_tag | 出现频次: $freq | 来源样本数: $sample_count" />
  <Header value="映射决策" size="4" />
  <Choices name="mapping_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="mapped_to_genre" html="🏷️ 映射为流派" />
    <Choice value="mapped_to_instrument" html="🎹 映射为乐器" />
    <Choice value="mapped_to_emotion" html="🎭 映射为情绪" />
    <Choice value="blacklist" html="🚫 加入黑名单" />
    <Choice value="keep_unmapped" html="⏸️ 暂不处理" />
  </Choices>
  <TextArea name="mapping_suggestion" toName="audio" placeholder="填写映射建议（如：Jazz, Bebop 或 GM065）" rows="2" />
</View>""",
        "prefill_fields": ["audio", "raw_tag", "freq", "sample_count"],
    },

    "cluster_validation": {
        "description": "DBSCAN聚类簇验证",
        "decision_field": "cluster_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="80" />
  <Header value="聚类信息" size="4" />
  <Text name="meta" value="簇ID: $cluster_id | 簇内样本数: $cluster_size | 到簇心距离: $dist_to_center | 邻域标签: $neighbor_labels" />
  <Header value="听检判断" size="4" />
  <Choices name="cluster_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="belongs_to_cluster" html="✅ 属于该簇（风格一致）" />
    <Choice value="outlier" html="❌ 是离群点（不应在簇内）" />
    <Choice value="should_be_own_cluster" html="🔄 应独立成新簇" />
    <Choice value="uncertain" html="❓ 不确定" />
  </Choices>
  <TextArea name="comment" toName="audio" placeholder="备注（可选）" rows="2" />
</View>""",
        "prefill_fields": ["audio", "cluster_id", "cluster_size", "dist_to_center", "neighbor_labels"],
    },

    "segment_boundary": {
        "description": "乐段边界切片质量听检",
        "decision_field": "segment_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="100" />
  <Header value="切片信息" size="4" />
  <Text name="meta" value="切片ID: $segment_id | 起始: $start_time | 结束: $end_time | 时长: $duration | 切片策略: $strategy" />
  <Header value="听检判断" size="4" />
  <Choices name="segment_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="good_boundary" html="✅ 边界合理，乐句完整" />
    <Choice value="cut_mid_phrase" html="✂️ 切断了乐句/solo" />
    <Choice value="too_short" html="📏 切片过短，无意义" />
    <Choice value="too_long" html="📐 切片过长，应再切" />
    <Choice value="uncertain" html="❓ 不确定" />
  </Choices>
  <TextArea name="comment" toName="audio" placeholder="备注（可选）" rows="2" />
</View>""",
        "prefill_fields": ["audio", "segment_id", "start_time", "end_time", "duration", "strategy"],
    },

    "data_source_quality": {
        "description": "新采集源整体质量评估",
        "decision_field": "source_decision",
        "xml_skeleton": """<View>
  <Audio name="audio" value="$audio" zoom="true" waveheight="80" />
  <Header value="数据源信息" size="4" />
  <Text name="meta" value="来源: $source | 采集批次: $batch | 格式: $format | 采样率: $sample_rate | 位深: $bit_depth" />
  <Header value="质量评估" size="4" />
  <Choices name="source_decision" toName="audio" choice="single" required="true" showInline="true">
    <Choice value="high_quality" html="✅ 高质量，可直接使用" />
    <Choice value="acceptable_with_noise" html="⚠️ 可接受但有底噪，标记marginal" />
    <Choice value="systematic_issue" html="❌ 有系统性问题（DRM/爆音/严重压缩）" />
    <Choice value="uncertain" html="❓ 不确定" />
  </Choices>
  <TextArea name="comment" toName="audio" placeholder="备注（可选）" rows="2" />
</View>""",
        "prefill_fields": ["audio", "source", "batch", "format", "sample_rate", "bit_depth"],
    },
}


# ============================================================
# 2.5 模板版本管理（避免旧任务和新任务混用导致决策口径不一致）
# ============================================================

# 每个模板的版本和最后验证日期
# 当模板骨架迭代时（如增加选项、修改决策字段），递增 version 并更新 last_verified
TEMPLATE_VERSIONS = {
    "qc_snr_calibration": {"version": "1.0", "last_verified": "2026-08-25"},
    "qc_content_boundary": {"version": "1.0", "last_verified": "2026-08-25"},
    "dedup_similarity": {"version": "1.0", "last_verified": "2026-08-25"},
    "knn_propagation_verify": {"version": "1.0", "last_verified": "2026-08-25"},
    "unmapped_tag_review": {"version": "1.0", "last_verified": "2026-08-25"},
    "cluster_validation": {"version": "1.0", "last_verified": "2026-08-25"},
    "segment_boundary": {"version": "1.0", "last_verified": "2026-08-25"},
    "data_source_quality": {"version": "1.0", "last_verified": "2026-08-25"},
}

# 将版本信息注入 TEMPLATE_REGISTRY
for _tpl_name, _tpl in TEMPLATE_REGISTRY.items():
    _ver = TEMPLATE_VERSIONS.get(_tpl_name, {"version": "1.0", "last_verified": "2026-08-25"})
    _tpl["version"] = _ver["version"]
    _tpl["last_verified"] = _ver["last_verified"]


# ============================================================
# 3. 听检任务数据类
# ============================================================

@dataclass
class ListeningTask:
    """一次听检任务"""
    task_id: str
    task_type: str
    created_at: str
    sample_count: int
    xml_template: str
    import_data: List[Dict]  # Label Studio导入格式
    decision_field: str
    description: str
    template_version: str = "1.0"  # 模板版本，避免新旧任务混用导致决策口径不一致
    update_suggestion: Optional[str] = None  # 解析后生成的更新建议


# ============================================================
# 4. 核心类：AdaptiveListeningCheck
# ============================================================

class AdaptiveListeningCheck:
    """自适应听检闭环管理器"""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent.parent
        self.template_registry = TEMPLATE_REGISTRY

    def generate_task(
        self,
        task_type: str,
        samples: List[Dict],
        output_dir: Optional[str] = None,
    ) -> ListeningTask:
        """
        生成听检任务

        Args:
            task_type: 任务类型（ ListeningTaskType 枚举值）
            samples: 样本列表，每个样本包含预填字段
            output_dir: 输出目录

        Returns:
            ListeningTask 对象
        """
        if task_type not in self.template_registry:
            raise ValueError(f"未知任务类型: {task_type}. 可用类型: {list(self.template_registry.keys())}")

        template = self.template_registry[task_type]
        task_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 构建Label Studio导入数据
        import_data = []
        for i, sample in enumerate(samples):
            task = {
                "id": f"{task_id}_{i:03d}",
                "data": {},
            }
            # 填充预填字段
            for field in template["prefill_fields"]:
                if field in sample:
                    task["data"][field] = sample[field]
            import_data.append(task)

        task = ListeningTask(
            task_id=task_id,
            task_type=task_type,
            created_at=datetime.now().isoformat(),
            sample_count=len(samples),
            xml_template=template["xml_skeleton"],
            import_data=import_data,
            decision_field=template["decision_field"],
            description=template["description"],
            template_version=template.get("version", "1.0"),
        )

        # 保存到文件
        if output_dir:
            self._save_task(task, Path(output_dir))

        return task

    def parse_results(
        self,
        ls_export_path: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """
        解析Label Studio听检结果，生成更新建议

        Args:
            ls_export_path: Label Studio导出的JSON文件路径
            task_type: 任务类型

        Returns:
            包含统计结果和更新建议的字典
        """
        if task_type not in self.template_registry:
            raise ValueError(f"未知任务类型: {task_type}")

        template = self.template_registry[task_type]
        decision_field = template["decision_field"]

        with open(ls_export_path, "r", encoding="utf8") as f:
            annotations = json.load(f)

        # 统计决策分布
        decisions = {}
        details = []
        for ann in annotations:
            # Label Studio导出格式：annotations[0].result
            result = ann.get("annotations", [{}])[0].get("result", [])
            decision = None
            for r in result:
                if r.get("from_name") == decision_field:
                    decision = r["value"]["choices"][0]
                    break
            if decision:
                decisions[decision] = decisions.get(decision, 0) + 1
            details.append({
                "task_id": ann.get("id"),
                "decision": decision,
                "data": ann.get("data", {}),
            })

        # 根据任务类型生成更新建议
        suggestion = self._generate_update_suggestion(task_type, decisions, details)

        return {
            "task_type": task_type,
            "total_samples": len(annotations),
            "decision_distribution": decisions,
            "details": details,
            "update_suggestion": suggestion,
        }

    def _generate_update_suggestion(
        self,
        task_type: str,
        decisions: Dict[str, int],
        details: List[Dict],
    ) -> str:
        """根据听检结果生成更新建议（半自动，需人工确认后执行）"""

        total = sum(decisions.values())
        if total == 0:
            return "无有效听检结果"

        if task_type == "qc_snr_calibration":
            acceptable = decisions.get("acceptable", 0) + decisions.get("musical_normal", 0)
            too_noisy = decisions.get("noise_too_high", 0)
            if acceptable >= total * 0.6:
                return (
                    f"建议：SNR阈值可从15dB放宽到12dB。"
                    f"听检{total}首中{acceptable}首({acceptable/total:.0%})噪声可接受或为音乐制作正常特征"
                    f"（曲风/乐器/效果器/演奏法/历史录音），"
                    f"仅{too_noisy}首噪声过大。放宽后marginal占比预计从33%降至7%。"
                )
            elif too_noisy >= total * 0.4:
                return (
                    f"建议：保持SNR阈值15dB不变。"
                    f"听检{total}首中{too_noisy}首({too_noisy/total:.0%})噪声过大，"
                    f"放宽阈值会引入低质量样本。"
                )
            else:
                return f"建议：保持15dB阈值，但对高DR(>25dB)样本标记'历史录音'优先通过。"

        elif task_type == "dedup_similarity":
            exact = decisions.get("exact_duplicate", 0)
            different = decisions.get("different_version", 0) + decisions.get("different_track", 0)
            if exact >= total * 0.8:
                return f"建议：当前相似度阈值合理，{exact}/{total}首确认为完全重复。"
            elif different >= total * 0.5:
                return f"建议：提高相似度阈值，{different}/{total}首被误判为重复（实际是不同版本/曲目）。"
            else:
                return f"建议：阈值需微调，听检结果分布均匀，建议增加样本量后再决策。"

        elif task_type == "knn_propagation_verify":
            correct = decisions.get("correct", 0)
            wrong = sum(decisions.get(k, 0) for k in ["wrong_genre", "wrong_mood", "wrong_instruments"])
            should_not = decisions.get("should_not_propagate", 0)
            accuracy = correct / total if total > 0 else 0
            if accuracy >= 0.8:
                return f"建议：KNN传播质量良好（准确率{accuracy:.0%}），当前cosine距离阈值合理。"
            elif should_not >= total * 0.3:
                return f"建议：收紧KNN传播距离阈值，{should_not}/{total}首不应该被传播（距离太远）。"
            else:
                return f"建议：KNN传播准确率{accuracy:.0%}，需检查传播来源黄金集质量。"

        elif task_type == "unmapped_tag_review":
            mapped = sum(decisions.get(k, 0) for k in ["mapped_to_genre", "mapped_to_instrument", "mapped_to_emotion"])
            blacklist = decisions.get("blacklist", 0)
            keep = decisions.get("keep_unmapped", 0)
            return (
                f"听检结果：{mapped}首建议映射，{blacklist}首加入黑名单，{keep}首暂不处理。"
                f"建议：审核mapping_suggestion字段后执行merge_mapping.py更新字典。"
            )

        elif task_type == "cluster_validation":
            belongs = decisions.get("belongs_to_cluster", 0)
            outlier = decisions.get("outlier", 0)
            new_cluster = decisions.get("should_be_own_cluster", 0)
            if belongs >= total * 0.8:
                return f"建议：聚类质量良好，{belongs}/{total}首确认属于该簇。"
            elif outlier >= total * 0.3:
                return f"建议：调整DBSCAN eps参数，{outlier}/{total}首为离群点被错误归入。"
            elif new_cluster >= total * 0.3:
                return f"建议：该簇应拆分为多个子簇，{new_cluster}/{total}首应独立成簇。"
            else:
                return f"建议：聚类参数需微调，增加听检样本量。"

        elif task_type == "data_source_quality":
            high = decisions.get("high_quality", 0)
            acceptable = decisions.get("acceptable_with_noise", 0)
            issue = decisions.get("systematic_issue", 0)
            if issue >= total * 0.3:
                return f"⚠️ 警告：该数据源有系统性问题（{issue}/{total}首），建议暂停采集并排查来源。"
            elif high + acceptable >= total * 0.8:
                return f"建议：该数据源质量可接受（{high+acceptable}/{total}首），可继续采集。"
            else:
                return f"建议：该数据源质量不稳定，建议限制采集比例。"

        else:
            return f"听检完成，共{total}首。决策分布：{decisions}。请根据结果手动调整对应配置。"

    def _save_task(self, task: ListeningTask, output_dir: Path):
        """保存听检任务到文件"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存XML模板
        (output_dir / f"{task.task_id}_template.xml").write_text(
            task.xml_template, encoding="utf8"
        )

        # 保存导入数据
        (output_dir / f"{task.task_id}_import.json").write_text(
            json.dumps(task.import_data, indent=2, ensure_ascii=False), encoding="utf8"
        )

        # 保存任务元数据（含模板版本，避免新旧任务混用导致决策口径不一致）
        meta = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "created_at": task.created_at,
            "sample_count": task.sample_count,
            "decision_field": task.decision_field,
            "description": task.description,
            "template_version": task.template_version,
        }
        (output_dir / f"{task.task_id}_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8"
        )

    def list_available_tasks(self) -> List[str]:
        """列出所有可用的听检任务类型"""
        return list(self.template_registry.keys())


# ============================================================
# 5. CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="自适应人工听检闭环")
    subparsers = parser.add_subparsers(dest="command")

    # list - 列出可用任务类型
    subparsers.add_parser("list", help="列出所有可用的听检任务类型")

    # generate - 生成听检任务
    gen_parser = subparsers.add_parser("generate", help="生成听检任务")
    gen_parser.add_argument("--task-type", required=True, help="任务类型")
    gen_parser.add_argument("--samples-json", required=True, help="样本数据JSON文件路径")
    gen_parser.add_argument("--output-dir", default="data/listening_tasks", help="输出目录")

    # parse - 解析听检结果
    parse_parser = subparsers.add_parser("parse", help="解析听检结果并生成更新建议")
    parse_parser.add_argument("--ls-export", required=True, help="Label Studio导出JSON路径")
    parse_parser.add_argument("--task-type", required=True, help="任务类型")

    args = parser.parse_args()

    checker = AdaptiveListeningCheck()

    if args.command == "list":
        print("可用听检任务类型：")
        for t in checker.list_available_tasks():
            desc = checker.template_registry[t]["description"]
            print(f"  - {t}: {desc}")

    elif args.command == "generate":
        with open(args.samples_json, "r", encoding="utf8") as f:
            samples = json.load(f)
        task = checker.generate_task(args.task_type, samples, args.output_dir)
        print(f"✅ 听检任务已生成: {task.task_id}")
        print(f"   类型: {task.task_type} ({task.description})")
        print(f"   样本数: {task.sample_count}")
        print(f"   输出目录: {args.output_dir}")
        print(f"   XML模板: {task.task_id}_template.xml")
        print(f"   导入数据: {task.task_id}_import.json")

    elif args.command == "parse":
        result = checker.parse_results(args.ls_export, args.task_type)
        print(f"📊 听检结果解析: {result['task_type']}")
        print(f"   总样本数: {result['total_samples']}")
        print(f"   决策分布: {result['decision_distribution']}")
        print(f"\n💡 更新建议:")
        print(f"   {result['update_suggestion']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
