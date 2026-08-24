#!/usr/bin/env python3
"""
merge_mapping.py — 人工确认后合并映射更新（原子性保证）

功能:
1. 读取 mapping_updates_pending.json（export_annotations.py 产出）
2. 筛选 status=approved 的条目
3. 预校验全部条目（任何错误都整体取消，保证原子性）
4. 全部通过后执行写入，自动升级版本号 + 记录 changelog
5. 支持 --major 升级大版本

安全原则:
- 完全依赖 mapping_type 字段，不做字符串猜测
- 未知类型直接抛 ValueError，阻断而非猜测
- 预校验 + 原子写入：任何错误都不产生半更新状态

使用:
    # 预校验（不写入）
    python merge_mapping.py --pending mapping_updates_pending.json --mapping configs/label_mapping_dict.json

    # 执行合并（小版本 +1）
    python merge_mapping.py --pending mapping_updates_pending.json --mapping configs/label_mapping_dict.json --apply

    # 执行合并（大版本升级）
    python merge_mapping.py --pending mapping_updates_pending.json --mapping configs/label_mapping_dict.json --apply --major
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


VALID_MAPPING_TYPES = {"instrument", "genre", "emotion", "blacklist"}


def load_pending(pending_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """加载 pending 文件，返回 (parsed列表, unparsed列表)"""
    with open(pending_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容两种格式：{"parsed": [...], "unparsed": [...]} 或直接列表
    if isinstance(data, dict):
        parsed = data.get("parsed", [])
        unparsed = data.get("unparsed", [])
    elif isinstance(data, list):
        parsed = data
        unparsed = []
    else:
        raise ValueError(f"无法解析 pending 文件格式: {type(data)}")

    return parsed, unparsed


def validate_all(approved: List[Dict], mapping: Dict) -> List[str]:
    """
    预校验：全部通过才允许写入，保证原子性。

    返回错误列表，空列表表示全部合法。
    """
    errors = []

    for idx, u in enumerate(approved, 1):
        tag = u.get("original_tag", "")
        prop = u.get("proposed_mapping", "")
        mtype = u.get("mapping_type", "unknown").lower().strip()

        # 基本字段检查
        if not tag:
            errors.append(f"[{idx}] 缺少 original_tag 字段")
            continue
        if not prop:
            errors.append(f"[{idx}] '{tag}': 缺少 proposed_mapping 字段")
            continue

        # 类型检查
        if mtype not in VALID_MAPPING_TYPES:
            errors.append(
                f"[{idx}] '{tag}': 非法 mapping_type '{mtype}'，"
                f"允许值: {', '.join(sorted(VALID_MAPPING_TYPES))}"
            )
            continue

        # instrument: 值必须以 GM 开头
        if mtype == "instrument" and not prop.startswith("GM"):
            errors.append(f"[{idx}] '{tag}': instrument 类型值必须以 GM 开头，got '{prop}'")

        # genre: 值不能为空或含空段
        elif mtype == "genre":
            parts = [x.strip() for x in prop.split(",")]
            if len(parts) < 1 or any(not p for p in parts):
                errors.append(f"[{idx}] '{tag}': genre 类型值不能为空或含空段，got '{prop}'")

        # emotion: 必须是 3 个浮点数
        elif mtype == "emotion":
            try:
                vad = [float(x.strip()) for x in prop.split(",")]
                if len(vad) != 3:
                    raise ValueError(f"需要3个值，got {len(vad)}")
                for v in vad:
                    if v < 0 or v > 1:
                        raise ValueError(f"VAD值应在[0,1]，got {v}")
            except (ValueError, TypeError) as e:
                errors.append(f"[{idx}] '{tag}': emotion 类型必须是 3 个[0,1]浮点数，got '{prop}' ({e})")

        # blacklist: 值应为 "未映射"
        elif mtype == "blacklist" and prop != "未映射":
            errors.append(f"[{idx}] '{tag}': blacklist 类型值应为 '未映射'，got '{prop}'")

        # 重复 key 检查
        if mtype == "instrument" and tag in mapping.get("instrument_gm128_map", {}):
            errors.append(f"[{idx}] '{tag}': 已存在于 instrument_gm128_map，将覆盖旧值 '{mapping['instrument_gm128_map'][tag]}'")
        elif mtype == "genre" and tag in mapping.get("genre_3level_map", {}):
            errors.append(f"[{idx}] '{tag}': 已存在于 genre_3level_map，将覆盖旧值")
        elif mtype == "emotion" and tag in mapping.get("emotion_vad_map", {}):
            errors.append(f"[{idx}] '{tag}': 已存在于 emotion_vad_map，将覆盖旧值")

    return errors


def apply_merge(approved: List[Dict], mapping: Dict, major: bool = False) -> Tuple[Dict, List[Dict]]:
    """
    执行合并（假设已通过预校验）。

    返回 (更新后的mapping, changelog_entries)
    """
    changelog_entries = []

    for u in approved:
        tag = u["original_tag"]
        prop = u["proposed_mapping"]
        mtype = u["mapping_type"].lower().strip()

        if mtype == "instrument":
            mapping.setdefault("instrument_gm128_map", {})[tag] = prop
            entry = {"action": "add_inst", "tag": tag, "to": prop}

        elif mtype == "genre":
            parts = [x.strip() for x in prop.split(",")]
            mapping.setdefault("genre_3level_map", {})[tag] = parts
            entry = {"action": "add_genre", "tag": tag, "to": parts}

        elif mtype == "emotion":
            vad = [float(x.strip()) for x in prop.split(",")]
            mapping.setdefault("emotion_vad_map", {})[tag] = vad
            entry = {"action": "add_emotion", "tag": tag, "to": vad}

        elif mtype == "blacklist":
            mapping.setdefault("hard_blacklist", []).append(tag)
            entry = {"action": "blacklist", "tag": tag, "severity": "hard"}

        else:
            # 理论上预校验已拦截，这里是防御性编程
            raise ValueError(f"未知 mapping_type: '{mtype}' for tag '{tag}'")

        changelog_entries.append(entry)

    # 版本升级
    old_ver = mapping.get("version", "v1.0")
    try:
        major_v, minor_v = old_ver.lstrip("v").split(".")
        major_v, minor_v = int(major_v), int(minor_v)
    except (ValueError, IndexError):
        major_v, minor_v = 2, 0

    if major:
        mapping["version"] = f"v{major_v + 1}.0"
        vtype = "major"
    else:
        mapping["version"] = f"v{major_v}.{minor_v + 1}"
        vtype = "minor"

    mapping["updated_at"] = datetime.now().strftime("%Y-%m-%d")

    # 记录 changelog
    if "changelog" not in mapping:
        mapping["changelog"] = []
    mapping["changelog"].append({
        "date": datetime.now().isoformat(),
        "version": mapping["version"],
        "type": vtype,
        "entries": changelog_entries,
        "note": f"从 {len(approved)} 条人工审核中合并映射更新",
    })

    return mapping, changelog_entries


def merge_mapping_updates(pending_path: str, mapping_path: str,
                          apply: bool = False, major: bool = False) -> Dict:
    """
    主流程：加载 → 筛选approved → 预校验 → (apply时)写入。

    Args:
        pending_path: mapping_updates_pending.json 路径
        mapping_path: label_mapping_dict.json 路径
        apply: 是否真正写入（False=仅预校验）
        major: 是否升级大版本

    Returns:
        结果字典
    """
    pending_path = Path(pending_path)
    mapping_path = Path(mapping_path)

    if not pending_path.exists():
        raise FileNotFoundError(f"pending 文件不存在: {pending_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"映射字典不存在: {mapping_path}")

    # 1. 加载 pending
    parsed, unparsed = load_pending(pending_path)
    print(f"加载 pending: {len(parsed)} 条已解析, {len(unparsed)} 条未解析")

    # 2. 筛选 approved
    approved = [p for p in parsed if p.get("status") == "approved"]
    pending_count = len([p for p in parsed if p.get("status") == "pending"])
    rejected_count = len([p for p in parsed if p.get("status") == "rejected"])
    print(f"状态统计: approved={len(approved)}, pending={pending_count}, "
          f"rejected={rejected_count}, unparsed={len(unparsed)}")

    if not approved:
        print("无 approved 更新，跳过")
        return {"status": "skipped", "reason": "no_approved_updates"}

    # 3. 加载当前字典
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    old_version = mapping.get("version", "unknown")
    print(f"当前字典版本: {old_version}")

    # 4. 预校验
    print(f"\n预校验 {len(approved)} 条 approved 更新...")
    errors = validate_all(approved, mapping)

    if errors:
        print(f"\n❌ 预校验失败，发现 {len(errors)} 处错误，整体合并已取消：")
        for e in errors:
            print(f"   - {e}")
        print("\n请修正 pending 文件后重新运行。")
        return {"status": "validation_failed", "errors": errors}

    print("✅ 预校验全部通过")

    if not apply:
        print(f"\n[DRY RUN] 将合并 {len(approved)} 条更新（使用 --apply 真正写入）")
        for u in approved:
            print(f"  - [{u['mapping_type']}] {u['original_tag']} → {u['proposed_mapping']}")
        return {"status": "dry_run", "approved_count": len(approved)}

    # 5. 执行合并
    mapping, changelog_entries = apply_merge(approved, mapping, major)

    # 6. 原子性写入
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    new_version = mapping["version"]
    print(f"\n✅ 已原子性更新 {mapping_path}")
    print(f"   版本: {old_version} → {new_version} ({'major' if major else 'minor'})")
    print(f"   变更: {len(changelog_entries)} 条")
    for entry in changelog_entries:
        print(f"   - {entry['action']}: {entry['tag']}")

    return {
        "status": "success",
        "old_version": old_version,
        "new_version": new_version,
        "changes": len(changelog_entries),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="合并映射更新（预校验+原子写入）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅预校验（不写入）
  python merge_mapping.py --pending pending.json --mapping dict.json

  # 执行合并（小版本+1）
  python merge_mapping.py --pending pending.json --mapping dict.json --apply

  # 执行合并（大版本升级）
  python merge_mapping.py --pending pending.json --mapping dict.json --apply --major
        """
    )
    parser.add_argument("--pending", required=True, help="mapping_updates_pending.json 路径")
    parser.add_argument("--mapping", default="configs/label_mapping_dict.json",
                        help="label_mapping_dict.json 路径")
    parser.add_argument("--apply", action="store_true", help="真正写入（默认仅预校验）")
    parser.add_argument("--major", action="store_true", help="升级大版本（默认小版本+1）")
    args = parser.parse_args()

    result = merge_mapping_updates(args.pending, args.mapping, args.apply, args.major)

    if result["status"] == "validation_failed":
        sys.exit(1)
    elif result["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(0)
