#!/usr/bin/env python3
"""
genre_annotation_pipeline.py — Genre 多标签分层标注正式流水线（L4）

=====================================================================
架构（unified_v3_multi_label_no_primary）
=====================================================================
Genre 是社会文化标签，不是纯声学属性。本流水线用「分层来源裁决 + 多标签并存」：

  P0  文本 LLM + 搜索（有曲目名+艺术家，社会共识标签，主标签 weight=1.0）
  P1  Qwen-Omni 听音频（unknown 无曲目名，主标签 weight=0.8）
  EXCLUDED  ACE Studio 生成曲（demucs_vocals），不参与标注
  用户裁决  user_rulings.json（人工 HITL 最终裁定，最高优先级）

规则：
  - 多标签并存，不设 primary_genre（不变相单选）；每个标签带 source/weight/confidence
  - 标签之间不裁决、不选一，全保留；冲突交 Label Studio 人工审核
  - P0 subgenre weight=0.7；P1 subgenre weight=0.6；rejected 标签 weight=0.0
  - label_selection：有用户裁决/分层裁决 resolution/单来源 → locked；
    多来源且无任何裁决 → human_review_pending
  - KNN 已退役（一致率 0%），knn_propagated 恒为 False，详见 archive/l4_knn_legacy/

=====================================================================
输入（只读）
=====================================================================
  data/02_preannotation/genre_annotation_plan.json       84首分类（named/unknown/ace）
  data/02_preannotation/genre_text_llm_annotations.json  58首文本LLM标注（P0）
  data/02_preannotation/l4_deepseek/*_text_labels.json   Qwen 标注（P1，目录名为历史遗留）
  data/02_preannotation/l3_structural/*_l3_qwen.json     5首黄金集 Qwen 精标（优先覆盖）
  data/02_preannotation/user_rulings.json                用户人工裁决配置

=====================================================================
输出
=====================================================================
  data/02_preannotation/genre_unified_final.json         84首汇总
  data/02_preannotation/l4_unified/{audio_id}_unified_tags.json  逐首产物

用法：
  python genre_annotation_pipeline.py                 # 跑完整流程
  python genre_annotation_pipeline.py --verify        # 只验证当前产物 schema/一致性
  python genre_annotation_pipeline.py --check-repro   # 重跑到临时目录，与现产物比对（复现性测试）
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------
# 常量：权重与版本（唯一事实来源，禁止散落硬编码）
# ---------------------------------------------------------------------
L4_VERSION = "unified_v3_multi_label_no_primary"
WEIGHT = {
    "P0_primary": 1.0,
    "P0_subgenre": 0.7,
    "P1_primary": 0.8,
    "P1_subgenre": 0.6,
    "rejected": 0.0,
}
SOURCE_NAME_TEXT_LLM = "text_llm_search"
SOURCE_NAME_QWEN = "qwen_omni"


# =====================================================================
# 加载层
# =====================================================================
def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_annotation_plan(preann_dir: str) -> dict[str, list[str]]:
    """返回 {priority: [audio_id,...]}，priority ∈ named/unknown/ace"""
    plan = load_json(os.path.join(preann_dir, "genre_annotation_plan.json"))
    return {
        "named": [x["audio_id"] for x in plan.get("named_samples", [])],
        "unknown": [x["audio_id"] for x in plan.get("unknown_samples", [])],
        "ace": [x["audio_id"] for x in plan.get("ace_samples", [])],
    }


def load_text_llm_labels(preann_dir: str) -> dict:
    return load_json(os.path.join(preann_dir, "genre_text_llm_annotations.json"))


def load_qwen_labels(preann_dir: str) -> dict:
    """
    加载 Qwen 标注。l3_structural 的黄金集精标优先于 l4_deepseek（历史目录名）里的旧标签。
    统一输出字段：genre/subgenre/confidence/source。
    """
    qwen: dict[str, dict] = {}
    # 先加载 l4_deepseek（批量 Qwen 补标）
    for fp in glob.glob(os.path.join(preann_dir, "l4_deepseek", "*_text_labels.json")):
        d = load_json(fp)
        aid = os.path.basename(fp).replace("_text_labels.json", "")
        qwen[aid] = d
    # 黄金集精标覆盖（annotation 可能嵌一层）
    for fp in glob.glob(os.path.join(preann_dir, "l3_structural", "*_l3_qwen.json")):
        d = load_json(fp)
        aid = os.path.basename(fp).replace("_l3_qwen.json", "")
        ann = d.get("annotation", d)
        qwen[aid] = ann
    return qwen


def load_user_rulings(preann_dir: str) -> dict:
    path = os.path.join(preann_dir, "user_rulings.json")
    if not os.path.exists(path):
        return {}
    raw = load_json(path)
    return raw.get("rulings", {})


def load_layered_resolutions(preann_dir: str) -> dict:
    """12首文本LLM vs Qwen冲突的分层裁决（category1/category2/pending）。"""
    path = os.path.join(preann_dir, "layered_conflict_resolutions.json")
    if not os.path.exists(path):
        return {}
    raw = load_json(path)
    return raw.get("resolutions", {})


def load_reannotation_manifest(preann_dir: str) -> tuple[dict, float]:
    """Qwen 重标清单（哪些样本从 deepseek 旧标签重标、是否分段）+ 原补标默认 confidence。"""
    path = os.path.join(preann_dir, "qwen_reannotation_manifest.json")
    if not os.path.exists(path):
        return {}, 0.8
    raw = load_json(path)
    return raw.get("reannotated", {}), raw.get("default_supplement_confidence", 0.8)


# =====================================================================
# 多标签构建
# =====================================================================
def _qwen_subgenre_list(q_meta: dict) -> list[str]:
    """从 Qwen 产物的 subgenre 字段（可能是 'A / B' 字符串）拆成列表。"""
    sub = q_meta.get("subgenre")
    if not sub:
        return []
    if isinstance(sub, list):
        return [s.strip() for s in sub if s.strip()]
    return [s.strip() for s in str(sub).split("/") if s.strip()]


def _dedup_labels(labels: list[dict]) -> list[dict]:
    """同 (source,label) 去重，保留首次出现顺序。"""
    seen, out = set(), []
    for g in labels:
        key = (g.get("source"), g.get("label"))
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def build_labels_for_sample(aid: str, priority: str, tl: dict, qwen: dict,
                            is_conflict: bool = False) -> list[dict]:
    """
    为单个样本构建多标签列表（不应用用户裁决；用户裁决在 apply_user_rulings 单独覆盖）。
    priority: named(P0) / unknown(P1) / ace(EXCLUDED)
    is_conflict: 是否为文本LLM vs Qwen冲突样本（仅冲突样本才在 P0 基础上额外保留 Qwen 主标签）

    标签纳入规则（与已验收产物对齐）：
      - 普通 named(P0)：只放文本LLM（主 1.0 + subgenre 0.7），不放 Qwen
      - 冲突 named(P0)：文本LLM（主+sub）+ Qwen【仅主标签 0.8】，Qwen subgenre 合并进顶层 sub_genres
      - unknown(P1)：只放 Qwen（主 0.8 + subgenre 0.6）
    """
    labels: list[dict] = []
    is_named = priority == "P0"
    is_unknown = priority == "P1"

    # P0 文本 LLM（named 样本）
    if is_named and aid in tl:
        t = tl[aid]
        primary = t.get("primary_genre", "")
        conf = t.get("confidence", 0.9)
        if primary:
            labels.append({
                "label": primary, "source": SOURCE_NAME_TEXT_LLM, "source_priority": "P0",
                "weight": WEIGHT["P0_primary"], "confidence": conf,
            })
        for sg in t.get("sub_genres", []):
            if sg and sg != primary:
                labels.append({
                    "label": sg, "source": SOURCE_NAME_TEXT_LLM, "source_priority": "P0",
                    "weight": WEIGHT["P0_subgenre"], "confidence": conf * 0.8,
                    "tag_type": "subgenre",
                })

    # P1 Qwen：unknown 样本放主+subgenre；named 冲突样本只放主标签
    if (is_unknown or is_conflict) and aid in qwen:
        q = qwen[aid]
        qgenre = q.get("genre", "")
        qconf = q.get("confidence") or 0.8
        if isinstance(qconf, str):
            qconf = 0.8
        if qgenre:
            labels.append({
                "label": qgenre, "source": SOURCE_NAME_QWEN, "source_priority": "P1",
                "weight": WEIGHT["P1_primary"], "confidence": qconf,
            })
        # 仅 unknown 样本把 Qwen subgenre 纳入 genres；冲突样本的 Qwen subgenre 走顶层 sub_genres 合并
        if is_unknown:
            qsub = q.get("subgenre", "")
            if qsub:
                for s in str(qsub).split("/"):
                    s = s.strip()
                    if s and s != qgenre:
                        labels.append({
                            "label": s, "source": SOURCE_NAME_QWEN, "source_priority": "P1",
                            "weight": WEIGHT["P1_subgenre"], "confidence": qconf * 0.8,
                            "tag_type": "subgenre",
                        })

    return _dedup_labels(labels)


def determine_selection(aid: str, labels: list[dict], has_resolution: bool,
                        has_user_ruling: bool, is_excluded: bool) -> str:
    """locked / human_review_pending 判定。"""
    if is_excluded or has_user_ruling or has_resolution:
        return "locked"
    accepted = [g for g in labels if not g.get("rejected", False)]
    sources = {g.get("source") for g in accepted}
    if len(sources) <= 1:
        return "locked"
    return "human_review_pending"


# =====================================================================
# 主流水线
# =====================================================================
def run_pipeline(project_root: str, out_dir: str | None = None) -> dict:
    preann = os.path.join(project_root, "data", "02_preannotation")
    out_preann = out_dir if out_dir else preann
    unified_dir = os.path.join(out_preann, "l4_unified")
    os.makedirs(unified_dir, exist_ok=True)

    plan = load_annotation_plan(preann)
    tl = load_text_llm_labels(preann)
    qwen = load_qwen_labels(preann)
    rulings = load_user_rulings(preann)
    layered = load_layered_resolutions(preann)
    reannot, default_supp_conf = load_reannotation_manifest(preann)

    named, unknown, ace = set(plan["named"]), set(plan["unknown"]), set(plan["ace"])
    all_ids = sorted(named | unknown | ace)

    final: dict[str, dict] = {}
    stats = Counter()

    for aid in all_ids:
        if aid in ace:
            entry = {
                "sub_genres": [], "source": "ace_studio_generated", "source_priority": "EXCLUDED",
                "confidence": 0.0, "title": "ACE generated", "artist": "ACE Studio",
                "reference": "ace_studio_exclude_list",
                "genres": [], "multi_label": True, "label_selection": "locked",
            }
            stats["EXCLUDED"] += 1
        else:
            priority = "P0" if aid in named else "P1"
            is_conflict = aid in layered
            labels = build_labels_for_sample(aid, priority, tl, qwen, is_conflict=is_conflict)

            # 基础元信息。
            # 顶层 source = 来源通道名（P0 固定 text_llm_search；P1 取 Qwen 产物的 source）
            # 顶层 reference = 具体出处（文本LLM 的 source 字段：Discogs/Wikipedia/文件名推断）
            tl_meta = tl.get(aid, {})
            q_meta = qwen.get(aid, {})
            title = tl_meta.get("title") or q_meta.get("title") or ("unknown" if aid in unknown else "")
            artist = tl_meta.get("artist") or q_meta.get("artist") or ("unknown" if aid in unknown else "")

            if aid in named:
                top_source = SOURCE_NAME_TEXT_LLM
                reference = tl_meta.get("source", "")  # Discogs/Wikipedia/文件名推断
                confidence = tl_meta.get("confidence", 0)
                sub_genres = list(tl_meta.get("sub_genres", []))
            else:
                top_source = q_meta.get("source", "qwen_omni")
                # 重标样本 reference 标记待重标来源；原补标标记为 Qwen 听音频
                reference = "needs_qwen_reannotate" if aid in reannot else "qwen_omni_audio"
                # 重标样本用 Qwen 实际 confidence；其余原补标统一默认 0.8
                if aid in reannot:
                    confidence = q_meta.get("confidence") or default_supp_conf
                    sub_genres = _qwen_subgenre_list(q_meta)
                else:
                    confidence = default_supp_conf
                    sub_genres = []  # 原补标样本顶层 sub_genres 留空（细分标签只在 genres 列表内）

            entry = {
                "sub_genres": sub_genres,
                "source": top_source,
                "source_priority": priority,
                "confidence": confidence,
                "title": title, "artist": artist,
                "reference": reference,
                "genres": labels, "multi_label": True,
            }
            # 重标/分段标记从 reannotation manifest 读取（历史操作固化，Qwen源文件本身不带）
            if aid in reannot:
                entry["reannotated"] = True
                entry["reannotated_from"] = reannot[aid].get("reannotated_from", "deepseek_v4_flash")
                if reannot[aid].get("segmented"):
                    entry["segmented"] = True
            stats[priority] += 1

        # 应用分层裁决（12首文本LLM vs Qwen冲突：category1/category2/pending）
        if aid in layered:
            lr = layered[aid]
            entry["resolution"] = lr["resolution"]
            entry["resolution_note"] = lr.get("resolution_note", "")
            if "candidate_labels" in lr:
                entry["candidate_labels"] = lr["candidate_labels"]
            if lr.get("merged_sub_genres"):
                entry["sub_genres"] = lr["merged_sub_genres"]

        # 应用用户裁决（最高优先级，整体覆盖 genres + 顶层 ruling；保留分层裁决历史字段）
        if aid in rulings:
            r = rulings[aid]
            entry["genres"] = [dict(g) for g in r["labels"]]
            entry["user_ruling"] = r["ruling"]
            entry["ruled_by"] = "user"
            entry["label_selection"] = "locked"
            stats["user_ruled"] += 1
        else:
            has_resolution = aid in layered
            entry["label_selection"] = determine_selection(
                aid, entry["genres"], has_resolution, False, aid in ace)

        final[aid] = entry

    # 写汇总
    with open(os.path.join(out_preann, "genre_unified_final.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 写逐首产物
    for aid, entry in final.items():
        tag = dict(entry)
        tag["audio_id"] = aid
        tag["knn_propagated"] = False
        tag["l4_version"] = L4_VERSION
        tag["deprecated_knn_note"] = (
            "KNN传播已从L4移除，一致率0%，详见archive/l4_knn_legacy/DEPRECATED.md")
        if entry.get("source_priority") == "EXCLUDED":
            tag["excluded"] = True
            tag["exclude_reason"] = "ACE Studio generated (demucs_vocals)"
        with open(os.path.join(unified_dir, f"{aid}_unified_tags.json"), "w", encoding="utf-8") as f:
            json.dump(tag, f, ensure_ascii=False, indent=2)

    return {"final": final, "stats": dict(stats), "n": len(final)}


def _has_layered_resolution(aid: str, entry: dict) -> bool:
    """判断是否已有分层裁决（category1 伪冲突 / category2 社会vs听觉）。
    判据：同时含 P0 文本LLM 与 P1 Qwen 两个来源标签（这些在 unified_v2 已分层裁决过）。"""
    sources = {g.get("source") for g in entry.get("genres", []) if not g.get("rejected")}
    return SOURCE_NAME_TEXT_LLM in sources and SOURCE_NAME_QWEN in sources


# =====================================================================
# 验证层
# =====================================================================
def verify_outputs(project_root: str, target_dir: str | None = None) -> list[str]:
    """对产物做 schema/规则校验，返回错误列表（空=通过）。"""
    preann = target_dir or os.path.join(project_root, "data", "02_preannotation")
    errors: list[str] = []
    final_path = os.path.join(preann, "genre_unified_final.json")
    if not os.path.exists(final_path):
        return [f"缺少 {final_path}"]
    final = load_json(final_path)

    if len(final) != 84:
        errors.append(f"样本数={len(final)}，期望84")

    pri = Counter(v.get("source_priority") for v in final.values())
    if pri.get("P0") != 58:
        errors.append(f"P0={pri.get('P0')}，期望58")
    if pri.get("P1") != 21:
        errors.append(f"P1={pri.get('P1')}，期望21")
    if pri.get("EXCLUDED") != 5:
        errors.append(f"EXCLUDED={pri.get('EXCLUDED')}，期望5")

    for aid, v in final.items():
        # 不得有 primary_genre
        if "primary_genre" in v:
            errors.append(f"{aid} 残留 primary_genre 字段")
        # 每个标签字段完整
        for g in v.get("genres", []):
            for k in ("label", "source", "source_priority", "weight", "confidence"):
                if k not in g:
                    errors.append(f"{aid} 标签 {g.get('label','?')} 缺字段 {k}")
        # ACE genres 必须空
        if v.get("source_priority") == "EXCLUDED" and v.get("genres"):
            errors.append(f"{aid} ACE 样本不应有 genres")
        # KNN 必须退役
        if v.get("knn_propagated"):
            errors.append(f"{aid} knn_propagated 应为 False")

    # 用户裁决 6 首必须 locked 且带 ruling
    rulings = load_user_rulings(os.path.join(project_root, "data", "02_preannotation"))
    for aid in rulings:
        if aid not in final:
            errors.append(f"用户裁决样本 {aid} 不在产物中")
            continue
        if not final[aid].get("user_ruling"):
            errors.append(f"{aid} 缺 user_ruling")
        if final[aid].get("label_selection") != "locked":
            errors.append(f"{aid} 用户裁决样本应为 locked")

    # 逐首文件齐全
    unified_dir = os.path.join(preann, "l4_unified")
    n_files = len(glob.glob(os.path.join(unified_dir, "*_unified_tags.json")))
    if n_files != len(final):
        errors.append(f"l4_unified 文件数={n_files}，期望{len(final)}")

    return errors


def check_reproducibility(project_root: str) -> tuple[bool, list[str]]:
    """重跑到临时目录，与当前已提交产物逐字段比对。"""
    preann = os.path.join(project_root, "data", "02_preannotation")
    current = load_json(os.path.join(preann, "genre_unified_final.json"))

    with tempfile.TemporaryDirectory() as tmp:
        # 在临时目录建 data/02_preannotation 结构，输出指向它；输入仍读正式目录
        # run_pipeline 的输入固定读 preann，输出读 out_preann，因此直接传 out_dir=tmp
        res = run_pipeline(project_root, out_dir=tmp)
        regenerated = res["final"]

        diffs: list[str] = []
        if set(current.keys()) != set(regenerated.keys()):
            diffs.append("样本ID集合不一致")
        for aid in current:
            a, b = current.get(aid, {}), regenerated.get(aid, {})
            if json.dumps(a, sort_keys=True, ensure_ascii=False) != json.dumps(b, sort_keys=True, ensure_ascii=False):
                # 找出具体差异字段
                for k in set(a) | set(b):
                    if a.get(k) != b.get(k):
                        diffs.append(f"{aid} 字段 {k} 不一致: 现={str(a.get(k))[:60]} | 重跑={str(b.get(k))[:60]}")
                        break
        return (len(diffs) == 0, diffs[:20])


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Genre 多标签分层标注正式流水线（L4）")
    ap.add_argument("--project-root", default=os.path.expanduser("~/music_corpus_project"))
    ap.add_argument("--verify", action="store_true", help="只验证当前产物")
    ap.add_argument("--check-repro", action="store_true", help="重跑并与现产物比对（复现性测试）")
    args = ap.parse_args()

    if args.verify:
        errs = verify_outputs(args.project_root)
        if errs:
            print("❌ 验证失败:")
            for e in errs:
                print(f"  - {e}")
            sys.exit(1)
        print("✅ 产物验证通过：84首、无primary_genre、字段完整、用户裁决locked、KNN退役")
        return

    if args.check_repro:
        ok, diffs = check_reproducibility(args.project_root)
        if ok:
            print("✅ 复现性测试通过：正式脚本重跑结果与当前产物完全一致")
        else:
            print(f"❌ 复现性测试失败，{len(diffs)} 处差异:")
            for d in diffs:
                print(f"  - {d}")
            sys.exit(1)
        return

    # 完整运行
    res = run_pipeline(args.project_root)
    print(f"✅ 流水线完成：{res['n']}首")
    print(f"   分布: {res['stats']}")
    sel = Counter(v["label_selection"] for v in res["final"].values())
    print(f"   label_selection: {dict(sel)}")
    errs = verify_outputs(args.project_root)
    if errs:
        print("⚠️ 验证发现问题:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print("✅ 产物自检通过")


if __name__ == "__main__":
    main()
