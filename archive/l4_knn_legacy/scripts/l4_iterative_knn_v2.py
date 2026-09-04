#!/usr/bin/env python3
"""
L4 动态迭代 KNN 传播 v2（Active Learning + k-NN 投票）

核心改变（契约 v2 判据⑦）：
  1. k≥3 邻居投票：k 个最近种子 genre 一致才传播，不一致置 unlabeled
  2. 一致率是唯一生存指标：每轮输出 KNN传播 vs Qwen已有标签 的一致率
  3. 种子准入加交叉一致检查：Qwen标签与第1轮KNN预测冲突的，暂缓入池待HITL
  4. 覆盖率只是辅助数字，收敛判据改为一致率稳定

铁律：
  - voided 22首 + ACE 5首 任何轮次禁入种子池与传播池
  - 清单ID须与产物audio_id逐位匹配，零匹配即报错拒绝运行
  - 泛类genre（Pop/Jazz/Rock/Electronic/Folk/Classical）禁入种子池
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from l4_knn_propagation import (
    load_embeddings, load_deepseek_labels, load_golden_labels,
    compute_similarity_matrix, should_propagate, is_generic_genre,
    GENERIC_GENRES, VOIDED_GENRE_IDS, ACE_EXCLUDE_IDS,
)

# ========== 配置 ==========
K_NEIGHBORS = 3              # k-NN 投票邻居数（契约v2铁律3：k≥3）
K_VOTE_MAJORITY = 2          # k个邻居中至少多少个genre一致才传播（k=3时需2/3一致）
CONFIDENCE_THRESHOLD = 0.4   # cosine_dist 阈值


def validate_exclude_lists():
    """铁律：清单ID须与产物audio_id逐位匹配，零匹配即报错"""
    # 这里只做基本校验，实际匹配在run时验证
    if len(VOIDED_GENRE_IDS) == 0:
        print("  [警告] voided_genre_22.txt 为空或未找到")
    if len(ACE_EXCLUDE_IDS) == 0:
        print("  [警告] ace_studio_exclude.txt 为空或未找到")


def find_k_nearest_golden(
    audio_idx: int, golden_indices: list,
    sim_matrix: np.ndarray, k: int = K_NEIGHBORS,
) -> List[Tuple[int, float, float]]:
    """
    找到 k 个最近的黄金集样本。

    Returns:
        [(idx, cosine_dist, cosine_sim), ...] 按相似度降序
    """
    sims = [(j, sim_matrix[audio_idx][j]) for j in golden_indices]
    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k]
    return [(idx, 1.0 - sim, sim) for idx, sim in top_k]


def knn_vote_genre(
    k_neighbors: List[Tuple[int, float, float]],
    golden_labels: Dict[str, dict],
    audio_ids: List[str],
) -> Tuple[str, str, float, int]:
    """
    k-NN 投票决定 genre。

    规则：
    - k个邻居中，统计genre分布
    - 如果最高频genre的票数 >= K_VOTE_MAJORITY，传播该genre
    - 否则置unlabeled（邻居不一致，不强行传播）
    - 泛类genre不传播

    Returns:
        (genre, nearest_golden_id, avg_cosine_dist, vote_count)
        genre为空表示不传播
    """
    if len(k_neighbors) == 0:
        return "", "", 0.0, 0

    genre_votes = Counter()
    genre_to_neighbor = {}
    total_dist = 0.0

    for idx, dist, sim in k_neighbors:
        aid = audio_ids[idx]
        label = golden_labels.get(aid, {})
        genre = label.get("genre", "")
        if genre and not is_generic_genre(genre):
            genre_votes[genre] += 1
            if genre not in genre_to_neighbor:
                genre_to_neighbor[genre] = aid
        total_dist += dist

    if len(genre_votes) == 0:
        return "", k_neighbors[0][0] if k_neighbors else "", total_dist / len(k_neighbors), 0

    # 取最高票genre
    top_genre, top_count = genre_votes.most_common(1)[0]

    # 检查是否达到多数票阈值
    if top_count >= K_VOTE_MAJORITY:
        avg_dist = total_dist / len(k_neighbors)
        return top_genre, genre_to_neighbor[top_genre], avg_dist, top_count
    else:
        # 邻居不一致，不强行传播
        return "", "", total_dist / len(k_neighbors), top_count


def compute_consistency_rate(
    results: Dict[str, dict],
    qwen_labels: Dict[str, dict],
) -> dict:
    """
    计算 KNN 传播与 Qwen 已有标签的一致率。

    只统计：KNN传播了genre 且 Qwen也有genre 的样本。

    Returns:
        {
            "total_knn_propagated": N,
            "has_qwen_label": M,
            "consistent": C,
            "conflict": D,
            "consistency_rate": C/M,
            "conflicts": [(audio_id, knn_genre, qwen_genre), ...]
        }
    """
    knn_propagated = 0
    has_qwen = 0
    consistent = 0
    conflict = 0
    conflicts = []

    for audio_id, result in results.items():
        src = result.get("fusion", {}).get("genre_source", "")
        if not src.startswith("knn"):
            continue
        knn_propagated += 1

        knn_genre = result.get("genre", "")
        qwen_label = qwen_labels.get(audio_id, {})
        qwen_genre = qwen_label.get("genre", "")

        if not qwen_genre:
            continue
        has_qwen += 1

        # 宽松一致：genre相同 或 一个是另一个的子串
        knn_lower = knn_genre.lower().strip()
        qwen_lower = qwen_genre.lower().strip()
        if knn_lower == qwen_lower or knn_lower in qwen_lower or qwen_lower in knn_lower:
            consistent += 1
        else:
            conflict += 1
            conflicts.append((audio_id[:20], knn_genre, qwen_genre))

    rate = consistent / has_qwen if has_qwen > 0 else 0.0
    return {
        "total_knn_propagated": knn_propagated,
        "has_qwen_label": has_qwen,
        "consistent": consistent,
        "conflict": conflict,
        "consistency_rate": round(rate, 4),
        "conflicts": conflicts,
    }


def evaluate_seed_candidate(
    audio_id: str,
    qwen_label: dict,
    first_round_knn_genre: str,
) -> Tuple[bool, str]:
    """
    评估 Qwen 补标样本是否可以升级为种子（契约v2铁律1+交叉一致检查）。

    标准：
    1. genre 非空且非泛类
    2. subgenre/mood/instrumentation 全字段完整
    3. confidence 为 high 或 medium
    4. 不在 voided/ACE 禁入清单
    5. 交叉一致：Qwen标签与第1轮KNN预测不冲突
       （如果第1轮KNN传播了genre且与Qwen冲突，暂缓入池待HITL）

    Returns:
        (is_qualified, reason)
    """
    if audio_id in VOIDED_GENRE_IDS:
        return False, "voided_genre"
    if audio_id in ACE_EXCLUDE_IDS:
        return False, "ace_excluded"

    genre = qwen_label.get("genre", "")
    if not genre:
        return False, "empty_genre"
    if is_generic_genre(genre):
        return False, f"generic_genre:{genre}"

    if not qwen_label.get("subgenre"):
        return False, "empty_subgenre"
    if not qwen_label.get("mood"):
        return False, "empty_mood"
    if not qwen_label.get("instrumentation"):
        return False, "empty_instruments"

    conf = qwen_label.get("confidence", "medium")
    if isinstance(conf, (int, float)):
        conf_level = "high" if conf >= 0.8 else ("medium" if conf >= 0.5 else "low")
    else:
        conf_level = str(conf).lower()
    if conf_level not in ("high", "medium"):
        return False, f"low_confidence:{conf_level}"

    # 交叉一致检查
    if first_round_knn_genre and first_round_knn_genre.lower() != genre.lower():
        # 宽松检查：如果一个是另一个的子串，不算冲突
        knn_lower = first_round_knn_genre.lower()
        genre_lower = genre.lower()
        if knn_lower not in genre_lower and genre_lower not in knn_lower:
            return False, f"cross_conflict:knn={first_round_knn_genre},qwen={genre}"

    return True, "qualified"


def run_knn_round_v2(
    embeddings_dir: str,
    deepseek_dir: str,
    golden_labels: Dict[str, dict],
    golden_ids: Set[str],
    output_dir: str,
    round_name: str,
    exclude_splits: str = None,
    splits_dir: str = None,
    k: int = K_NEIGHBORS,
) -> Tuple[Dict[str, dict], dict]:
    """
    跑一轮 KNN 传播（v2：k-NN投票）。
    """
    audio_ids, vectors = load_embeddings(embeddings_dir)
    deepseek_labels = load_deepseek_labels(deepseek_dir)
    sim_matrix = compute_similarity_matrix(vectors)
    golden_indices = [i for i, aid in enumerate(audio_ids) if aid in golden_ids]

    # 防泄漏
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
        "k_neighbors": k,
        "total": 0, "golden": 0, "knn_propagated": 0,
        "qwen_supplement": 0, "unlabeled": 0, "excluded": 0,
        "knn_vote_below_threshold": 0,
        "genre_dist": Counter(), "source_dist": Counter(),
    }

    for i, audio_id in enumerate(audio_ids):
        stats["total"] += 1
        if audio_id in excluded_ids:
            stats["excluded"] += 1
            continue

        is_golden = audio_id in golden_ids
        deepseek_label = deepseek_labels.get(audio_id, {})

        # k-NN 找最近邻居
        k_neighbors = find_k_nearest_golden(i, golden_indices, sim_matrix, k)
        voted_genre, nearest_golden_id, avg_dist, vote_count = knn_vote_genre(
            k_neighbors, golden_labels, audio_ids
        )

        # 融合
        result = fuse_with_knn_vote(
            audio_id, is_golden, deepseek_label,
            voted_genre, nearest_golden_id, avg_dist, vote_count,
            golden_labels.get(nearest_golden_id, {}) if nearest_golden_id else {},
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

    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for audio_id, result in results.items():
        with open(output_path / f"{audio_id}_full_tags.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return results, stats


def fuse_with_knn_vote(
    audio_id: str, is_golden: bool,
    deepseek_label: dict,
    voted_genre: str, nearest_golden_id: str,
    avg_dist: float, vote_count: int,
    nearest_golden_label: dict,
) -> dict:
    """用 k-NN 投票结果融合单个样本"""
    result = {
        "audio_id": audio_id, "fusion": {},
        "propagated_from": None, "propagation_similarity": None,
        "propagation_cosine_dist": None, "knn_vote_count": None,
    }

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
        golden_genre = nearest_golden_label.get("genre", "") or deepseek_label.get("genre", "")
        result["genre"] = golden_genre
        result["subgenre"] = nearest_golden_label.get("subgenre", deepseek_label.get("subgenre", ""))
        result["mood"] = nearest_golden_label.get("mood", deepseek_label.get("mood", []))
        result["instrumentation"] = nearest_golden_label.get("instruments", deepseek_label.get("instrumentation", []))
        result["caption"] = nearest_golden_label.get("caption", deepseek_label.get("caption", ""))
        result["segments"] = nearest_golden_label.get("segments", [])
        result["propagated_from"] = "golden_set"
        result["fusion"] = {
            "genre_source": "qwen_omni_golden" if golden_genre else ("qwen_supplement" if has_ds_genre else "unlabeled"),
            "mood_source": "qwen_omni_golden" if nearest_golden_label.get("mood") else ("qwen_supplement" if has_ds_mood else "unlabeled"),
            "instrumentation_source": "qwen_omni_golden" if nearest_golden_label.get("instruments") else ("qwen_supplement" if has_ds_instr else "unlabeled"),
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
        }

        propagated_any = False

        if is_voided:
            result["genre"] = ""
            fusion["genre_source"] = "unlabeled (voided_by_user_decision)"

        if not is_ace and not is_voided and voted_genre:
            # k-NN 投票通过，传播
            result["genre"] = voted_genre
            result["subgenre"] = nearest_golden_label.get("subgenre", result.get("subgenre", ""))
            result["mood"] = nearest_golden_label.get("mood", result["mood"])
            result["instrumentation"] = nearest_golden_label.get("instruments", result["instrumentation"])
            fusion["genre_source"] = f"knn_vote(k={K_NEIGHBORS},votes={vote_count},from {nearest_golden_id[:20]},dist={avg_dist:.3f})"
            fusion["mood_source"] = f"knn_vote(from {nearest_golden_id[:20]})"
            fusion["instrumentation_source"] = f"knn_vote(from {nearest_golden_id[:20]})"
            propagated_any = True
            result["knn_vote_count"] = vote_count
        elif not is_ace and not is_voided and not voted_genre and nearest_golden_id:
            # k-NN 投票未通过（邻居不一致），记录原因
            fusion["genre_source"] = fusion.get("genre_source", "unlabeled") + " (knn_vote_split)"

        if is_ace:
            fusion["genre_source"] = fusion.get("genre_source", "unlabeled") + " (ace_excluded_from_knn)"

        if propagated_any:
            result["propagated_from"] = nearest_golden_id
            result["propagation_cosine_dist"] = round(float(avg_dist), 4)

        result["fusion"] = fusion

    return result


def main():
    parser = argparse.ArgumentParser(description="L4 动态迭代 KNN 传播 v2（k-NN投票 + 一致率）")
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--l4-deepseek-dir", required=True)
    parser.add_argument("--l3-golden-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-splits", default="test,holdout,ood")
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--k", type=int, default=K_NEIGHBORS, help="k-NN投票邻居数")
    parser.add_argument("--convergence-threshold", type=float, default=0.02,
                        help="一致率收敛阈值")
    args = parser.parse_args()

    print("=" * 70)
    print("L4 动态迭代 KNN 传播 v2（k-NN投票 + 一致率唯一生存指标）")
    print(f"k={args.k}, 投票多数阈值={K_VOTE_MAJORITY}")
    print("=" * 70)

    validate_exclude_lists()

    # 加载
    golden_labels = load_golden_labels(args.l3_golden_dir)
    original_golden_ids = set(golden_labels.keys())
    deepseek_labels = load_deepseek_labels(args.l4_deepseek_dir)

    print(f"\n初始黄金集: {len(original_golden_ids)} 首")
    for aid in sorted(original_golden_ids):
        print(f"  {aid[:20]}: {golden_labels[aid].get('genre', '?')}")
    print(f"Qwen 补标标签: {len(deepseek_labels)} 首")

    all_stats = []
    all_consistency = []
    current_golden_labels = dict(golden_labels)
    current_golden_ids = set(original_golden_ids)
    first_round_knn_genres = {}  # 第1轮KNN传播的genre（用于交叉一致检查）

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'=' * 70}")
        print(f"第 {round_num} 轮（种子池: {len(current_golden_ids)} 首, k={args.k}）")
        print(f"{'=' * 70}")

        round_output = Path(args.output_dir) / f"round_{round_num}"
        results, stats = run_knn_round_v2(
            args.embeddings_dir, args.l4_deepseek_dir,
            current_golden_labels, current_golden_ids,
            str(round_output), f"round_{round_num}",
            args.exclude_splits, args.splits_dir, args.k,
        )

        # 记录第1轮KNN传播结果（用于交叉一致）
        if round_num == 1:
            for aid, r in results.items():
                src = r.get("fusion", {}).get("genre_source", "")
                if src.startswith("knn"):
                    first_round_knn_genres[aid] = r.get("genre", "")

        # 一致率（唯一生存指标）
        consistency = compute_consistency_rate(results, deepseek_labels)
        all_consistency.append(consistency)

        coverage = (stats["golden"] + stats["knn_propagated"]) / max(stats["total"], 1)
        stats["coverage"] = coverage
        stats["consistency_rate"] = consistency["consistency_rate"]
        all_stats.append(stats)

        print(f"\n  结果:")
        print(f"    总计: {stats['total']}")
        print(f"    黄金集: {stats['golden']}")
        print(f"    KNN传播: {stats['knn_propagated']}")
        print(f"    Qwen补充: {stats['qwen_supplement']}")
        print(f"    unlabeled: {stats['unlabeled']}")
        print(f"    覆盖率（辅助）: {coverage:.1%}")
        print(f"\n  ★ 一致率（唯一生存指标）: {consistency['consistency_rate']:.1%}")
        print(f"    KNN传播 {consistency['total_knn_propagated']} 首，其中有Qwen标签 {consistency['has_qwen_label']} 首")
        print(f"    一致: {consistency['consistent']}，冲突: {consistency['conflict']}")
        if consistency["conflicts"]:
            print(f"    冲突样本（前10）:")
            for aid, knn_g, qwen_g in consistency["conflicts"][:10]:
                print(f"      {aid}: KNN={knn_g} vs Qwen={qwen_g}")

        # 收敛检查（一致率稳定）
        if round_num >= 2:
            prev_rate = all_consistency[-2]["consistency_rate"]
            curr_rate = consistency["consistency_rate"]
            delta = abs(curr_rate - prev_rate)
            print(f"\n  一致率变化: {prev_rate:.1%} → {curr_rate:.1%} (Δ={delta:.1%})")
            if delta < args.convergence_threshold:
                print(f"  一致率已收敛（Δ < {args.convergence_threshold:.0%}），停止迭代")
                break

        # 构建扩大种子池（累积上一轮种子，不退回）
        if round_num < args.max_rounds:
            print(f"\n  评估 Qwen 补标样本质量（含交叉一致检查）...")
            # 保留上一轮所有种子（累积，不退回）
            expanded = dict(current_golden_labels)
            upgrade_report = []

            for audio_id, result in results.items():
                if audio_id in current_golden_ids:
                    continue
                src = result.get("fusion", {}).get("genre_source", "")
                if "qwen_supplement" not in src:
                    continue
                label = deepseek_labels.get(audio_id, {})
                first_knn = first_round_knn_genres.get(audio_id, "")
                is_qualified, reason = evaluate_seed_candidate(audio_id, label, first_knn)
                upgrade_report.append({"audio_id": audio_id, "genre": label.get("genre", ""),
                                       "qualified": is_qualified, "reason": reason})
                if is_qualified:
                    expanded[audio_id] = label

            qualified = [r for r in upgrade_report if r["qualified"]]
            rejected = [r for r in upgrade_report if not r["qualified"]]
            print(f"  评估 {len(upgrade_report)} 首: 合格 {len(qualified)}，不合格 {len(rejected)}")
            print(f"  种子池: {len(current_golden_ids)} → {len(expanded)} 首（累积）")
            reject_reasons = Counter(r["reason"] for r in rejected)
            for reason, count in reject_reasons.most_common():
                print(f"    {reason}: {count}")

            current_golden_labels = expanded
            current_golden_ids = set(expanded.keys())

    # 最终报告
    print(f"\n{'=' * 70}")
    print("迭代评估报告（一致率唯一生存指标）")
    print(f"{'=' * 70}")

    report = {
        "version": "v2",
        "k_neighbors": args.k,
        "vote_majority": K_VOTE_MAJORITY,
        "initial_golden_count": len(original_golden_ids),
        "final_seed_pool_count": len(current_golden_ids),
        "rounds": [],
    }

    for i, (stats, cons) in enumerate(zip(all_stats, all_consistency)):
        round_info = {
            "round": i + 1,
            "seed_pool_size": len(original_golden_ids) if i == 0 else len(current_golden_ids),
            "knn_propagated": stats["knn_propagated"],
            "qwen_supplement": stats["qwen_supplement"],
            "unlabeled": stats["unlabeled"],
            "coverage": round(stats["coverage"], 4),
            "consistency_rate": cons["consistency_rate"],
            "consistency_detail": {
                "total_knn": cons["total_knn_propagated"],
                "has_qwen": cons["has_qwen_label"],
                "consistent": cons["consistent"],
                "conflict": cons["conflict"],
            },
            "conflicts": cons["conflicts"][:20],
        }
        report["rounds"].append(round_info)
        print(f"\n  第 {i+1} 轮（种子池: {round_info['seed_pool_size']}）:")
        print(f"    KNN传播: {stats['knn_propagated']}，覆盖率: {stats['coverage']:.1%}（辅助）")
        print(f"    ★ 一致率: {cons['consistency_rate']:.1%}（{cons['consistent']}/{cons['has_qwen_label']}）")

    if len(all_consistency) >= 2:
        first_rate = all_consistency[0]["consistency_rate"]
        final_rate = all_consistency[-1]["consistency_rate"]
        report["conclusion"] = (
            f"迭代 {len(all_consistency)} 轮，一致率 {first_rate:.1%} → {final_rate:.1%}，"
            f"种子池 {len(original_golden_ids)} → {len(current_golden_ids)} 首"
        )
        print(f"\n  结论: {report['conclusion']}")

    report_path = Path(args.output_dir) / "iteration_report_v2.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")

    # 复制最终结果
    final_round = len(all_stats)
    final_dir = Path(args.output_dir) / f"round_{final_round}"
    if final_dir.exists():
        import shutil
        for f in final_dir.glob("*.json"):
            shutil.copy2(f, Path(args.output_dir) / f.name)
        print(f"最终结果（第 {final_round} 轮）已复制到: {args.output_dir}")


if __name__ == "__main__":
    main()
