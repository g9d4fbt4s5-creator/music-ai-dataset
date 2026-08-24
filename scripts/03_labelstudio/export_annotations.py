#!/usr/bin/env python3
"""
export_annotations.py — Label Studio 标注结果导出 + 半自动映射更新

功能:
1. 解析 Label Studio 导出的 JSON/JSONL 标注结果
2. 提取审核决策、修正标签、annotation_note
3. 半自动提取映射更新建议（生成 diff，不直接改字典）
4. 输出标准化标注结果 JSON（可用于训练）
5. 统计 IAA（标注者间一致性）、审核通过率、修正率

安全原则:
- 映射字典更新用"半自动 diff + 人工确认"，不直接自动写入
- 原始 unmapped_original_tags 永远保留，不得删除
- 每次修改字典必须升级 version + 记录 changelog

使用:
    # 导出标注结果 + 提取映射更新建议
    python export_annotations.py \
        --ls-export labelstudio_export.json \
        --mapping-dict configs/label_mapping_dict.json \
        --output-dir data/03_human_review/exported/

    # 人工确认后，合并映射更新
    python export_annotations.py \
        --merge-pending data/03_human_review/exported/mapping_updates_pending.json \
        --mapping-dict configs/label_mapping_dict.json \
        --apply-merge
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


def parse_ls_export(ls_export_path: str) -> List[Dict]:
    """解析 Label Studio 导出文件（JSON 或 JSONL）"""
    path = Path(ls_export_path)
    if not path.exists():
        raise FileNotFoundError(f"标注导出文件不存在: {ls_export_path}")

    annotations = []

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            annotations = data
        elif isinstance(data, dict) and "tasks" in data:
            annotations = data["tasks"]
        else:
            annotations = [data]

    return annotations


def extract_annotation_result(ann: Dict) -> Dict:
    """从单条标注中提取结构化结果"""
    audio_id = ann.get("id", ann.get("audio_id", "unknown"))

    # 提取标注结果（可能在 annotations 或 result 中）
    results = []
    if "annotations" in ann and ann["annotations"]:
        for a in ann["annotations"]:
            results.extend(a.get("result", []))
    elif "result" in ann:
        results = ann["result"]

    # 解析各字段
    parsed = {
        "audio_id": audio_id,
        "genre_primary": None,
        "genre_secondary": [],
        "mood": [],
        "instruments": [],
        "vocal_presence": None,
        "quality_grade": None,
        "caption": "",
        "review_decision": None,
        "golden_set": None,
        "review_flag": None,
        "annotation_note": "",
        "structure_segments": [],
        "key_tonic": None,
        "key_mode": None,
        "key_modulation": "",
        "mood_vad": "",
        "completed_at": ann.get("completed_at", ann.get("updated_at", "")),
        "annotator": ann.get("annotator", ann.get("created_by", "")),
    }

    for r in results:
        from_name = r.get("from_name", "")
        value = r.get("value", {})

        if from_name == "genre_primary":
            parsed["genre_primary"] = value.get("choices", [None])[0]
        elif from_name == "genre_secondary":
            parsed["genre_secondary"] = value.get("choices", [])
        elif from_name == "mood":
            parsed["mood"] = value.get("choices", [])
        elif from_name == "instruments":
            if "labels" in value:
                parsed["instruments"].extend(value["labels"])
        elif from_name == "vocal_presence":
            parsed["vocal_presence"] = value.get("choices", [None])[0]
        elif from_name == "quality_grade":
            parsed["quality_grade"] = value.get("choices", [None])[0]
        elif from_name == "caption":
            parsed["caption"] = value.get("text", [""])[0]
        elif from_name == "review_decision":
            parsed["review_decision"] = value.get("choices", [None])[0]
        elif from_name == "golden_set":
            parsed["golden_set"] = value.get("choices", [None])[0]
        elif from_name == "review_flag":
            parsed["review_flag"] = value.get("choices", [None])[0]
        elif from_name == "annotation_note":
            parsed["annotation_note"] = value.get("text", [""])[0]
        elif from_name == "structure":
            if "start" in value and "end" in value:
                parsed["structure_segments"].append({
                    "start": value["start"],
                    "end": value["end"],
                    "label": value.get("labels", [""])[0],
                })
        elif from_name == "key_tonic":
            parsed["key_tonic"] = value.get("choices", [None])[0]
        elif from_name == "key_mode":
            parsed["key_mode"] = value.get("choices", [None])[0]
        elif from_name == "key_modulation":
            parsed["key_modulation"] = value.get("text", [""])[0]
        elif from_name == "mood_vad":
            parsed["mood_vad"] = value.get("text", [""])[0]

    # 去重乐器
    parsed["instruments"] = list(set(parsed["instruments"]))
    # 排序结构段落
    parsed["structure_segments"].sort(key=lambda x: x["start"])

    return parsed


import re

# 结构化映射建议正则：【映射建议】原始标签 "xxx" 应映射为 "yyy" (类型: zzz)
MAPPING_PATTERN = re.compile(
    r'【映射建议】原始标签\s*"([^"]+)"\s*应映射为\s*"([^"]+)"\s*(?:\(类型:\s*(\w+)\))?'
)
VALID_MAPPING_TYPES = {"instrument", "genre", "emotion", "blacklist"}


def extract_mapping_updates(annotations: List[Dict], mapping_dict: Dict) -> Dict:
    """
    从标注结果中提取映射更新建议（结构化解析 + unparsed 兜底）。

    输出格式: {"parsed": [...], "unparsed": [...]}
    - parsed: 成功匹配结构化格式的建议，含 mapping_type 字段
    - unparsed: 有映射意图但格式不规范，需人工解析

    安全原则: 只生成 diff 待人工确认，不直接修改字典。
    """
    parsed_updates = []
    unparsed = []

    for ann in annotations:
        parsed = extract_annotation_result(ann)
        note = parsed.get("annotation_note", "")
        audio_id = parsed["audio_id"]
        annotator = parsed.get("annotator", "")

        if not note:
            continue

        # 结构化解析
        matches = MAPPING_PATTERN.findall(note)

        if matches:
            for raw_tag, proposed, type_hint in matches:
                type_hint = type_hint.lower().strip() if type_hint else "unknown"

                # 类型前置校验
                if type_hint not in VALID_MAPPING_TYPES:
                    unparsed.append({
                        "audio_id": audio_id,
                        "raw_note": note,
                        "reason": f"非法 mapping_type: '{type_hint}'，允许值: {', '.join(sorted(VALID_MAPPING_TYPES))}",
                        "status": "needs_manual_parsing",
                        "created_at": datetime.now().isoformat(),
                    })
                    continue

                parsed_updates.append({
                    "audio_id": audio_id,
                    "original_tag": raw_tag,
                    "proposed_mapping": proposed,
                    "mapping_type": type_hint,  # 下游 merge_mapping.py 唯一依据
                    "annotator": annotator,
                    "review_decision": parsed.get("review_decision", ""),
                    "status": "pending",  # 人工审核前
                    "created_at": datetime.now().isoformat(),
                })
        elif any(kw in note for kw in ["映射", "unmapped", "应映射", "标签错误"]):
            # 兜底：有映射意图但没匹配到结构化格式
            unparsed.append({
                "audio_id": audio_id,
                "annotator": annotator,
                "raw_note": note,
                "reason": "未匹配到结构化格式【映射建议】原始标签 \"xxx\" 应映射为 \"yyy\" (类型: zzz)",
                "status": "needs_manual_parsing",
                "created_at": datetime.now().isoformat(),
            })

    return {"parsed": parsed_updates, "unparsed": unparsed}


def apply_mapping_updates(pending_path: str, mapping_dict_path: str,
                          dry_run: bool = True) -> Dict:
    """
    人工确认后，合并映射更新到字典。

    安全: 默认 dry_run，需 --apply-merge 才真正写入。
    每次合并自动升级 version + 记录 changelog。
    """
    with open(pending_path, "r", encoding="utf-8") as f:
        pending = json.load(f)

    with open(mapping_dict_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # 只合并 status=approved 的更新
    approved = [u for u in pending if u.get("status") == "approved"]
    changes = {"genre_added": [], "instrument_added": [], "emotion_added": [],
               "blacklist_added": []}

    for update in approved:
        proposal = update.get("proposed_mapping", {})
        if not proposal:
            continue

        original = proposal.get("original_tag", "").lower()
        target = proposal.get("target_mapping", "")
        field = proposal.get("field", "unknown")

        if field == "genre" and original and original not in mapping["genre_3level_map"]:
            mapping["genre_3level_map"][original] = [target, target, target]
            changes["genre_added"].append(original)
        elif field == "instrument" and original and original not in mapping["instrument_gm128_map"]:
            mapping["instrument_gm128_map"][original] = 0  # 占位，需人工填GM编号
            changes["instrument_added"].append(original)
        elif field == "emotion" and original and original not in mapping["emotion_vad_map"]:
            mapping["emotion_vad_map"][original] = {"valence": 0.5, "arousal": 0.5, "dominance": 0.5}
            changes["emotion_added"].append(original)

    # 升级版本
    old_version = mapping.get("version", "v1.0")
    new_version = bump_version(old_version)
    mapping["version"] = new_version
    mapping["updated_at"] = datetime.now().strftime("%Y-%m-%d")

    # 记录 changelog
    if "changelog" not in mapping:
        mapping["changelog"] = []
    mapping["changelog"].append({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "version": new_version,
        "action": "merge_from_human_review",
        "changes": changes,
        "note": f"从 {len(approved)} 条人工审核中合并映射更新",
    })

    if dry_run:
        print(f"[DRY RUN] 将合并 {len(approved)} 条更新:")
        print(f"  流派新增: {changes['genre_added']}")
        print(f"  乐器新增: {changes['instrument_added']}")
        print(f"  情绪新增: {changes['emotion_added']}")
        print(f"  新版本: {old_version} → {new_version}")
        print(f"  使用 --apply-merge 真正写入")
        return mapping

    # 写入
    with open(mapping_dict_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print(f"已合并 {len(approved)} 条更新到 {mapping_dict_path}")
    print(f"版本: {old_version} → {new_version}")
    return mapping


def bump_version(version: str) -> str:
    """简单版本号递增: v2.0 → v2.1"""
    try:
        parts = version.lstrip("v").split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return f"v{major}.{minor + 1}"
    except Exception:
        return "v2.1"


def compute_stats(annotations: List[Dict]) -> Dict:
    """统计标注结果: 审核通过率、修正率、质量分布"""
    total = len(annotations)
    decisions = {}
    quality_dist = {}
    golden_count = 0
    edit_count = 0

    for ann in annotations:
        parsed = extract_annotation_result(ann)
        decision = parsed.get("review_decision", "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1

        quality = parsed.get("quality_grade", "unknown")
        quality_dist[quality] = quality_dist.get(quality, 0) + 1

        if "golden" in str(parsed.get("golden_set", "")).lower():
            golden_count += 1
        if "edits" in str(decision).lower():
            edit_count += 1

    approve_rate = sum(v for k, v in decisions.items() if "approve" in k.lower()) / max(total, 1)

    return {
        "total": total,
        "decisions": decisions,
        "quality_distribution": quality_dist,
        "golden_set_count": golden_count,
        "edit_count": edit_count,
        "approve_rate": round(approve_rate, 3),
        "edit_rate": round(edit_count / max(total, 1), 3),
    }


def run_export(ls_export_path: str, mapping_dict_path: str, output_dir: str):
    """主流程: 导出标注 + 提取映射更新 + 统计"""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Label Studio 标注结果导出")
    print("=" * 60)

    # 1. 解析标注
    annotations = parse_ls_export(ls_export_path)
    print(f"\n解析到 {len(annotations)} 条标注")

    # 2. 加载映射字典
    with open(mapping_dict_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"映射字典版本: {mapping.get('version', 'unknown')}")

    # 3. 提取结构化结果
    parsed_results = [extract_annotation_result(a) for a in annotations]
    parsed_path = os.path.join(output_dir, "annotations_parsed.json")
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, indent=2, ensure_ascii=False)
    print(f"结构化结果: {parsed_path}")

    # 4. 提取映射更新建议（结构化解析 + unparsed 兜底）
    mapping_result = extract_mapping_updates(annotations, mapping)
    parsed_count = len(mapping_result.get("parsed", []))
    unparsed_count = len(mapping_result.get("unparsed", []))
    pending_path = os.path.join(output_dir, "mapping_updates_pending.json")
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(mapping_result, f, indent=2, ensure_ascii=False)
    print(f"映射更新建议: {pending_path} (parsed={parsed_count}, unparsed={unparsed_count})")

    # 5. 统计
    stats = compute_stats(annotations)
    stats_path = os.path.join(output_dir, "annotation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"导出完成")
    print(f"{'='*60}")
    print(f"  总标注数: {stats['total']}")
    print(f"  审核通过率: {stats['approve_rate']*100:.1f}%")
    print(f"  修正率: {stats['edit_rate']*100:.1f}%")
    print(f"  黄金集数: {stats['golden_set_count']}")
    print(f"  审核决策分布: {stats['decisions']}")
    print(f"  质量分布: {stats['quality_distribution']}")
    print(f"\n  ⚠️ 映射更新需人工确认:")
    print(f"     1. 编辑 {pending_path}，将确认的条目标记 status='approved'")
    print(f"     2. 运行: python export_annotations.py --merge-pending {pending_path} --mapping-dict {mapping_dict_path} --apply-merge")

    return parsed_results, mapping_updates, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Studio 标注导出 + 半自动映射更新")
    parser.add_argument("--ls-export", help="Label Studio 导出文件路径")
    parser.add_argument("--mapping-dict", default="configs/label_mapping_dict.json",
                        help="映射字典路径")
    parser.add_argument("--output-dir", default="data/03_human_review/exported/",
                        help="输出目录")
    parser.add_argument("--merge-pending", help="待合并的映射更新文件路径")
    parser.add_argument("--apply-merge", action="store_true",
                        help="真正写入映射字典（默认 dry_run）")
    args = parser.parse_args()

    if args.merge_pending:
        apply_mapping_updates(args.merge_pending, args.mapping_dict,
                              dry_run=not args.apply_merge)
    elif args.ls_export:
        run_export(args.ls_export, args.mapping_dict, args.output_dir)
    else:
        parser.print_help()
