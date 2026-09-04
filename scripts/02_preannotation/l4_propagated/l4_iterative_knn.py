#!/usr/bin/env python3
"""
L4 动态迭代 KNN 传播（Active Learning 架构）

架构：
  第一轮：初始黄金集 5 首 → KNN 传播 → 未覆盖的用 Qwen 补标
  评估节点：Qwen 补标样本质量评估（置信度高、字段完整、非泛类）
  第二轮：高质量 Qwen 标签 + 原黄金集 = 扩大种子池 → 再 KNN
  对比两轮覆盖率，输出评估报告

关键铁律：
  - Qwen 补标数据不是最终产物，是迭代输入
  - 每一轮覆盖率变化必须写成评估报告
  - 种子池可以动态扩大，不再固定在 5 首
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

# 复用 l4_knn_propagation.py 的核心函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from l4_knn_propagation import (
    load_embeddings, load_deepseek_labels, load_golden_labels,
    compute_similarity_matrix, find_nearest_golden, should_propagate,
    is_generic_genre, GENERIC_GENRES, VOIDED_GENRE_IDS, ACE_EXCLUDE_IDS,
    FUSION_CONFIG,
)


def evaluate_qwen_label_quality(label: dict) -> Tuple[bool, str]:
    """
    评估 Qwen 补标样本是否可以升级为 KNN 种子。

    升级标准：
    1. genre 非空且非泛类（Pop/Jazz/Rock/Electronic/Folk/Classical）
    2. subgenre 非空
    3. mood 非空
    4. instrumentation 非空（至少1个乐器）
    5. confidence 为 high 或 medium
    6. 不在 22首 voided 清单
    7. 不在 ACE exclude 清单

    Returns:
        (is_qualified, reason)
    """
    audio_id = label.get("audio_id", "")

    # 排除 voided 和 ACE
    if audio_id in VOIDED_GENRE_IDS:
        return False, "voided_genre"
    if audio_id in ACE_EXCLUDE_IDS:
        return False, "ace_excluded"

    # genre 检查
    genre = label.get("genre", "")
    if not genre:
        return False, "empty_genre"
    if is_generic_genre(genre):
        return False, f"generic_genre:{genre}"

    # 字段完整性
    if not label.get("subgenre"):
        return False, "empty_subgenre"
    if not label.get("mood"):
        return False, "empty_mood"
    if not label.get("instrumentation"):
        return False, "empty_instruments"

    # confidence 检查
    conf = label.get("confidence", "medium")
    if isinstance(conf, (int, float)):
        conf_level = "high" if conf >= 0.8 else ("medium" if conf >= 0.5 else "low")
    else:
        conf_level = str(conf).lower()
    if conf_level not in ("high", "medium"):
        return False, f"low_confidence:{conf_level}"

    return True, "qualified"


def run_knn_round(
    embeddings_dir: str,
    deepseek_dir: str,
    golden_labels: Dict[str, dict],
    golden_ids: Set[str],
    output_dir: str,
    round_name: str,
    exclude_splits: str = None,
    splits_dir: str = None,
) -> Tuple[Dict[str, dict], dict]:
    """
    跑一轮 KNN 传播。

    Args:
        golden_labels: 种子池标签（audio_id -> label dict）
        golden_ids: 种子池 ID 集合

    Returns:
        (results, stats)
    """
    audio_ids, vectors = load_embeddings(embeddings_dir)
    deepseek_labels = load_deepseek_labels(deepseek_dir)

    # 计算相似度矩阵
    sim_matrix = compute_similarity_matrix(vectors)

    # 种子索引
    golden_indices = [i for i, aid in enumerate(audio_ids) if aid in golden_ids]

    # 防泄漏：加载排除子集
    excluded_ids = set()
    if exclude_splits and splits_dir:
        splits_path = Path(splits_dir)
        for sname in [s.strip() for s in exclude_splits.split(",") if s.strip()]:
            for cand in [sname + ".csv", sname + "_gold.csv"]:
                fpath = splits_path / cand
                if fpath.exists():
                    import pandas as pd
                    df = pd.read_csv(fpath)
                    if "audio_id" in df.columns:
                        excluded_ids.update(set(df["audio_id"].tolist()))
                    break

    results = {}
    stats = {
        "round": round_name,
        "total": 0,
        "golden": 0,
        "knn_propagated": 0,
        "qwen_supplement": 0,
        "unlabeled": 0,
        "excluded": 0,
        "genre_dist": Counter(),
        "source_dist": Counter(),
    }

    for i, audio_id in enumerate(audio_ids):
        stats["total"] += 1

        # 防泄漏：排除子集
        if audio_id in excluded_ids:
            stats["excluded"] += 1
            continue

        is_golden = audio_id in golden_ids
        deepseek_label = deepseek_labels.get(audio_id, {})

        # 找最近黄金集
        nearest_idx, cosine_dist, nearest_sim = find_nearest_golden(i, golden_indices, sim_matrix)
        nearest_golden_id = audio_ids[nearest_idx]
        golden_label = golden_labels.get(nearest_golden_id, {})

        # 融合
        result = fuse_single_sample_iterative(
            audio_id, is_golden, deepseek_label, golden_label,
            nearest_golden_id, cosine_dist, nearest_sim,
        )
        results[audio_id] = result

        # 统计
        src = result.get("fusion", {}).get("genre_source", "")
        genre = result.get("genre", "")
        if is_golden:
            stats["golden"] += 1
            stats["source_dist"]["golden"] += 1
        elif src.startswith("knn"):
            stats["knn_propagated"] += 1
            stats["source_dist"]["knn"] += 1
        elif "qwen_supplement" in src:
            stats["qwen_supplement"] += 1
            stats["source_dist"]["qwen_supplement"] += 1
        elif "unlabeled" in src or not genre:
            stats["unlabeled"] += 1
            stats["source_dist"]["unlabeled"] += 1

        if genre:
            stats["genre_dist"][genre] += 1

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for audio_id, result in results.items():
        out_file = output_path / f"{audio_id}_full_tags.json"
        with open(out_file, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return results, stats


def fuse_single_sample_iterative(
    audio_id: str, is_golden: bool,
    deepseek_label: dict, golden_label: dict,
    nearest_golden_id: str, cosine_dist: float,
    nearest_sim: float,
) -> dict:
    """
    迭代版融合函数（与 l4_knn_propagation.py 的 fuse_single_sample 逻辑一致）
    """
    result = {
        "audio_id": audio_id,
        "fusion": {},
        "propagated_from": None,
        "propagation_similarity": None,
        "propagation_cosine_dist": None,
    }

    # 基础物理特征
    for key in ["bpm", "key", "duration_sec", "snr_db", "loudness_db",
                "quality_assessment", "vocal_presence", "subgenre"]:
        if key in deepseek_label:
            result[key] = deepseek_label[key]

    has_ds_genre = bool(deepseek_label.get("genre"))
    has_ds_mood = bool(deepseek_label.get("mood"))
    has_ds_instr = bool(deepseek_label.get("instrumentation"))

    is_voided = audio_id in VOIDED_GENRE_IDS
    is_ace = audio_id in ACE_EXCLUDE_IDS

    if is_golden:
        golden_genre = golden_label.get("genre", "")
        result["genre"] = golden_genre or deepseek_label.get("genre", "")
        genre_source = "qwen_omni_golden" if golden_label.get("genre") else ("qwen_supplement" if has_ds_genre else "unlabeled")
        result["subgenre"] = golden_label.get("subgenre", deepseek_label.get("subgenre", ""))
        result["mood"] = golden_label.get("mood", deepseek_label.get("mood", []))
        result["instrumentation"] = golden_label.get("instruments", deepseek_label.get("instrumentation", []))
        result["caption"] = golden_label.get("caption", deepseek_label.get("caption", ""))
        result["segments"] = golden_label.get("segments", [])
        result["propagated_from"] = "golden_set"
        result["fusion"] = {
            "genre_source": genre_source,
            "mood_source": "qwen_omni_golden" if golden_label.get("mood") else ("qwen_supplement" if has_ds_mood else "unlabeled"),
            "instrumentation_source": "qwen_omni_golden" if golden_label.get("instruments") else ("qwen_supplement" if has_ds_instr else "unlabeled"),
            "caption_source": "qwen_omni_golden",
        }
    else:
        result["genre"] = deepseek_label.get("genre", "")
        result["subgenre"] = deepseek_label.get("subgenre", "")
        result["mood"] = deepseek_label.get("mood", [])
        result["instrumentation"] = deepseek_label.get("instrumentation", [])
        result["caption"] = deepseek_label.get("caption", "")
        result["segments"] = []

        fusion = {
            "genre_source": "qwen_supplement" if has_ds_genre else "unlabeled",
            "mood_source": "qwen_supplement" if has_ds_mood else "unlabeled",
            "instrumentation_source": "qwen_supplement" if has_ds_instr else "unlabeled",
            "caption_source": "qwen_supplement (not_propagated)" if deepseek_label.get("caption") else "unlabeled",
        }

        propagated_any = False

        if is_voided:
            result["genre"] = ""
            fusion["genre_source"] = "unlabeled (voided_by_user_decision)"

        if not is_ace and not is_voided:
            if golden_label and should_propagate("genre", cosine_dist, golden_label.get("confidence", "high")):
                golden_genre = golden_label.get("genre", "")
                if not is_generic_genre(golden_genre):
                    result["genre"] = golden_genre
                    result["subgenre"] = golden_label.get("subgenre", result.get("subgenre", ""))
                    propagated_any = True
                    fusion["genre_source"] = f"knn(from {nearest_golden_id}, dist={cosine_dist:.3f})"

            if golden_label and should_propagate("mood", cosine_dist, golden_label.get("confidence", "high")):
                result["mood"] = golden_label.get("mood", result["mood"])
                fusion["mood_source"] = f"knn(from {nearest_golden_id}, dist={cosine_dist:.3f})"
                propagated_any = True

            if golden_label and should_propagate("instruments", cosine_dist, golden_label.get("confidence", "high")):
                result["instrumentation"] = golden_label.get("instruments", result["instrumentation"])
                fusion["instrumentation_source"] = f"knn(from {nearest_golden_id}, dist={cosine_dist:.3f})"
                propagated_any = True
        elif is_ace:
            fusion["genre_source"] = fusion.get("genre_source", "unlabeled") + " (ace_excluded_from_knn)"

        if propagated_any:
            result["propagated_from"] = nearest_golden_id
            result["propagation_similarity"] = round(float(nearest_sim), 4)
            result["propagation_cosine_dist"] = round(float(cosine_dist), 4)

        result["fusion"] = fusion

    return result


def build_seed_pool_from_qwen(
    deepseek_labels: Dict[str, dict],
    first_round_results: Dict[str, dict],
    original_golden_ids: Set[str],
) -> Tuple[Dict[str, dict], List[dict]]:
    """
    从 Qwen 补标样本中筛选高质量标签，构建扩大的种子池。

    Returns:
        (expanded_golden_labels, upgrade_report)
    """
    expanded = {}
    upgrade_report = []

    # 原始黄金集全部保留
    for aid in original_golden_ids:
        if aid in deepseek_labels:
            expanded[aid] = deepseek_labels[aid]

    # 从 Qwen 补标中筛选高质量
    for audio_id, result in first_round_results.items():
        if audio_id in original_golden_ids:
            continue
        src = result.get("fusion", {}).get("genre_source", "")
        if "qwen_supplement" not in src:
            continue

        # 用 deepseek_label（即 Qwen 补标标签）评估质量
        label = deepseek_labels.get(audio_id, {})
        is_qualified, reason = evaluate_qwen_label_quality(label)

        upgrade_report.append({
            "audio_id": audio_id,
            "genre": label.get("genre", ""),
            "subgenre": label.get("subgenre", ""),
            "qualified": is_qualified,
            "reason": reason,
        })

        if is_qualified:
            expanded[audio_id] = label

    return expanded, upgrade_report


def main():
    parser = argparse.ArgumentParser(description="L4 动态迭代 KNN 传播")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--l4-deepseek-dir", required=True, help="Qwen 补标标签目录")
    parser.add_argument("--l3-golden-dir", required=True, help="初始黄金集目录")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-splits", default="test,holdout,ood")
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--max-rounds", type=int, default=3, help="最大迭代轮数")
    parser.add_argument("--convergence-threshold", type=float, default=0.02,
                        help="覆盖率收敛阈值（两轮覆盖率差小于此值则停止）")
    args = parser.parse_args()

    print("=" * 70)
    print("L4 动态迭代 KNN 传播（Active Learning）")
    print("=" * 70)

    # 加载初始黄金集
    golden_labels = load_golden_labels(args.l3_golden_dir)
    original_golden_ids = set(golden_labels.keys())
    print(f"\n初始黄金集: {len(original_golden_ids)} 首")
    for aid in sorted(original_golden_ids):
        g = golden_labels[aid].get("genre", "?")
        print(f"  {aid[:20]}: {g}")

    # 加载 Qwen 补标标签
    deepseek_labels = load_deepseek_labels(args.l4_deepseek_dir)
    print(f"\nQwen 补标标签: {len(deepseek_labels)} 首")

    all_stats = []
    current_golden_labels = dict(golden_labels)
    current_golden_ids = set(original_golden_ids)

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"第 {round_num} 轮 KNN 传播（种子池: {len(current_golden_ids)} 首）")
        print(f"{'=' * 70}")

        round_output = Path(args.output_dir) / f"round_{round_num}"
        results, stats = run_knn_round(
            args.embeddings_dir, args.l4_deepseek_dir,
            current_golden_labels, current_golden_ids,
            str(round_output), f"round_{round_num}",
            args.exclude_splits, args.splits_dir,
        )

        coverage = (stats["golden"] + stats["knn_propagated"]) / max(stats["total"], 1)
        stats["coverage"] = coverage
        all_stats.append(stats)

        print(f"\n第 {round_num} 轮结果:")
        print(f"  总计: {stats['total']}")
        print(f"  黄金集: {stats['golden']}")
        print(f"  KNN 传播: {stats['knn_propagated']}")
        print(f"  Qwen 补充: {stats['qwen_supplement']}")
        print(f"  unlabeled: {stats['unlabeled']}")
        print(f"  覆盖率（golden+knn）: {coverage:.1%}")
        print(f"  流派分布（前5）:")
        for g, c in stats["genre_dist"].most_common(5):
            print(f"    {g}: {c}")

        # 收敛检查
        if round_num >= 2:
            prev_coverage = all_stats[-2]["coverage"]
            delta = abs(coverage - prev_coverage)
            print(f"\n  覆盖率变化: {prev_coverage:.1%} → {coverage:.1%} (Δ={delta:.1%})")
            if delta < args.convergence_threshold:
                print(f"  已收敛（Δ < {args.convergence_threshold:.0%}），停止迭代")
                break

        # 构建扩大种子池（最后一轮不构建）
        if round_num < args.max_rounds:
            print(f"\n  评估 Qwen 补标样本质量，构建扩大种子池...")
            expanded_labels, upgrade_report = build_seed_pool_from_qwen(
                deepseek_labels, results, current_golden_ids
            )

            qualified = [r for r in upgrade_report if r["qualified"]]
            rejected = [r for r in upgrade_report if not r["qualified"]]

            print(f"  Qwen 补标样本评估: {len(upgrade_report)} 首")
            print(f"    合格（升级为种子）: {len(qualified)} 首")
            print(f"    不合格: {len(rejected)} 首")

            # 不合格原因分布
            reject_reasons = Counter(r["reason"] for r in rejected)
            print(f"    不合格原因分布:")
            for reason, count in reject_reasons.most_common():
                print(f"      {reason}: {count}")

            print(f"\n  升级为种子的样本:")
            for r in qualified[:10]:
                print(f"    {r['audio_id'][:20]}: {r['genre']} ({r['subgenre']})")
            if len(qualified) > 10:
                print(f"    ... 共 {len(qualified)} 首")

            current_golden_labels = expanded_labels
            current_golden_ids = set(expanded_labels.keys())

    # 输出最终评估报告
    print(f"\n{'=' * 70}")
    print("迭代评估报告")
    print(f"{'=' * 70}")

    report = {
        "initial_golden_count": len(original_golden_ids),
        "final_seed_pool_count": len(current_golden_ids),
        "rounds": [],
        "conclusion": "",
    }

    for i, stats in enumerate(all_stats):
        round_info = {
            "round": i + 1,
            "seed_pool_size": len(original_golden_ids) if i == 0 else len(current_golden_ids),
            "total": stats["total"],
            "golden": stats["golden"],
            "knn_propagated": stats["knn_propagated"],
            "qwen_supplement": stats["qwen_supplement"],
            "unlabeled": stats["unlabeled"],
            "coverage": round(stats["coverage"], 4),
            "genre_dist": dict(stats["genre_dist"].most_common(10)),
        }
        report["rounds"].append(round_info)

        print(f"\n  第 {i+1} 轮（种子池: {round_info['seed_pool_size']} 首）:")
        print(f"    KNN 传播: {stats['knn_propagated']}")
        print(f"    Qwen 补充: {stats['qwen_supplement']}")
        print(f"    unlabeled: {stats['unlabeled']}")
        print(f"    覆盖率: {stats['coverage']:.1%}")

    if len(all_stats) >= 2:
        first_cov = all_stats[0]["coverage"]
        final_cov = all_stats[-1]["coverage"]
        delta = final_cov - first_cov
        report["conclusion"] = (
            f"迭代 {len(all_stats)} 轮，覆盖率从 {first_cov:.1%} → {final_cov:.1%} "
            f"(提升 {delta:+.1%})，种子池从 {len(original_golden_ids)} → {len(current_golden_ids)} 首"
        )
        print(f"\n  结论: {report['conclusion']}")

    # 保存报告
    report_path = Path(args.output_dir) / "iteration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已保存: {report_path}")

    # 保存最终结果（最后一轮）到 output_dir 根目录
    final_round = len(all_stats)
    final_round_dir = Path(args.output_dir) / f"round_{final_round}"
    if final_round_dir.exists():
        import shutil
        for f in final_round_dir.glob("*.json"):
            shutil.copy2(f, Path(args.output_dir) / f.name)
        print(f"最终结果（第 {final_round} 轮）已复制到: {args.output_dir}")


if __name__ == "__main__":
    main()
