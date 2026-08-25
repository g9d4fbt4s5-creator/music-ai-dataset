#!/usr/bin/env python3
"""
50首试点采集分布缺口统计脚本

ADR-003 50首试点工具：统计当前采集的分布与目标分布的差距，输出缺口，指导补采。

用法：
  # 从 manifest 统计当前分布，对比目标，输出缺口
  python scripts/utils/check_pilot_gaps.py --manifest data/00_raw_collect/audio_manifest.csv

  # 从检查表 CSV 统计（采集阶段使用，manifest 还没更新）
  python scripts/utils/check_pilot_gaps.py --checklist reports/pilot_50/pilot_50_checklist.csv

  # 输出 JSON 格式报告
  python scripts/utils/check_pilot_gaps.py --manifest data/00_raw_collect/audio_manifest.csv --json reports/pilot_50/gap_report.json

  # 指定目标数量（默认50）
  python scripts/utils/check_pilot_gaps.py --manifest data/00_raw_collect/audio_manifest.csv --target 50
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ========== 目标分布（50首）==========
TARGET_TOTAL = 50

TARGET_DISTRIBUTION = {
    "genre_major": {
        "Jazz": 10,
        "Rock": 10,
        "Classical": 8,
        "Pop": 10,
        "Electronic": 7,
        "Other": 5,
    },
    "vocal_presence": {
        "vocal": 16,
        "instrumental": 17,
        "mixed": 17,
    },
    "source_type": {
        "normal": 40,
        "ace_studio_generated": 5,
        "demucs_vocals": 5,
    },
    "duration_category": {
        "standard (3-5min)": 35,
        "long (10-15min)": 10,
        "short (<2min)": 5,
    },
    "decade": {
        "1950s-1970s": 10,
        "1980s-2020s": 40,
    },
}

# 艺术家目标（特殊处理）
TARGET_ARTIST = {
    "min_unique": 15,
    "artists_with_3plus": 3,
}


def categorize_duration(duration_sec: float) -> str:
    """将时长（秒）分类"""
    if pd.isna(duration_sec):
        return "unknown"
    if duration_sec < 120:  # <2min
        return "short (<2min)"
    elif duration_sec <= 300:  # 3-5min
        return "standard (3-5min)"
    elif duration_sec <= 900:  # 5-15min
        return "medium (5-10min)"
    else:  # >15min
        return "long (10-15min)"


def categorize_decade(decade: str) -> str:
    """将年代分类为老录音/现代"""
    if pd.isna(decade) or decade == "":
        return "unknown"
    decade_str = str(decade)
    # 提取年代数字
    import re
    match = re.search(r"(\d{4})s?", decade_str)
    if match:
        year = int(match.group(1))
        if year <= 1979:
            return "1950s-1970s"
        else:
            return "1980s-2020s"
    # 直接匹配
    if any(x in decade_str for x in ["1950", "1960", "1970"]):
        return "1950s-1970s"
    return "1980s-2020s"


def load_data(
    manifest_path: Optional[Path] = None,
    checklist_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    加载数据（优先 manifest，其次 checklist）。

    返回标准化的 DataFrame，包含六个维度列。
    """
    if manifest_path and manifest_path.exists():
        logger.info(f"从 manifest 加载: {manifest_path}")
        df = pd.read_csv(manifest_path)
        # 标准化列名
        if "duration_sec" not in df.columns and "duration" in df.columns:
            df["duration_sec"] = df["duration"]
        return df

    if checklist_path and checklist_path.exists():
        logger.info(f"从检查表加载: {checklist_path}")
        df = pd.read_csv(checklist_path)
        # 只保留已采集的
        if "collected" in df.columns:
            df = df[df["collected"].astype(str).str.lower().isin(["yes", "true", "1"])]
            logger.info(f"已采集: {len(df)} 首")
        return df

    raise FileNotFoundError("请提供 --manifest 或 --checklist")


def compute_current_distribution(df: pd.DataFrame) -> Dict:
    """计算当前分布"""
    dist = {}

    # genre_major
    if "genre_major" in df.columns:
        # 合并小类为 Other
        genre_counts = df["genre_major"].fillna("unknown").value_counts()
        target_genres = set(TARGET_DISTRIBUTION["genre_major"].keys())
        normalized = {}
        other_count = 0
        for genre, count in genre_counts.items():
            if genre in target_genres:
                normalized[genre] = count
            else:
                other_count += count
        if other_count > 0:
            normalized["Other"] = normalized.get("Other", 0) + other_count
        dist["genre_major"] = normalized

    # vocal_presence
    if "vocal_presence" in df.columns:
        dist["vocal_presence"] = df["vocal_presence"].fillna("unknown").value_counts().to_dict()

    # source_type
    if "source_type" in df.columns:
        # 合并细分类型
        source_counts = df["source_type"].fillna("normal").value_counts()
        normalized = {"normal": 0, "ace_studio_generated": 0, "demucs_vocals": 0}
        for stype, count in source_counts.items():
            if "ace_studio" in str(stype) or "ai_generated" in str(stype):
                normalized["ace_studio_generated"] += count
            elif "demucs" in str(stype) or "vocals_only" in str(stype):
                normalized["demucs_vocals"] += count
            else:
                normalized["normal"] += count
        dist["source_type"] = normalized

    # duration_category
    if "duration_sec" in df.columns:
        df["_duration_cat"] = df["duration_sec"].apply(categorize_duration)
        duration_counts = df["_duration_cat"].value_counts()
        # 合并 medium 到 standard
        normalized = {"standard (3-5min)": 0, "long (10-15min)": 0, "short (<2min)": 0}
        for cat, count in duration_counts.items():
            if "short" in cat:
                normalized["short (<2min)"] += count
            elif "long" in cat or "medium" in cat:
                normalized["long (10-15min)"] += count
            else:
                normalized["standard (3-5min)"] += count
        dist["duration_category"] = normalized

    # decade
    if "decade" in df.columns:
        df["_decade_cat"] = df["decade"].apply(categorize_decade)
        dist["decade"] = df["_decade_cat"].value_counts().to_dict()

    # artist（特殊统计）
    if "artist_id" in df.columns:
        artist_counts = df["artist_id"].fillna("unknown").value_counts()
        # 排除 unknown
        known_artists = artist_counts[artist_counts.index != "unknown"]
        dist["artist"] = {
            "total_unique": len(known_artists),
            "artists_with_3plus": len(known_artists[known_artists >= 3]),
            "top_artists": known_artists.head(5).to_dict(),
        }

    return dist


def compute_gaps(current: Dict, target: Dict, total: int) -> Dict:
    """
    计算当前分布与目标分布的缺口。

    返回每个维度的缺口：正数=不足，负数=超额。
    """
    gaps = {}

    for dim, target_dist in target.items():
        if dim not in current:
            gaps[dim] = {k: {"target": v, "current": 0, "gap": v} for k, v in target_dist.items()}
            continue

        current_dist = current[dim]
        dim_gaps = {}
        for category, target_count in target_dist.items():
            current_count = current_dist.get(category, 0)
            gap = target_count - current_count
            dim_gaps[category] = {
                "target": target_count,
                "current": current_count,
                "gap": gap,  # 正数=不足，负数=超额
            }
        gaps[dim] = dim_gaps

    return gaps


def print_gap_report(current: Dict, gaps: Dict, total: int, target_total: int):
    """打印缺口报告"""
    print("\n" + "=" * 70)
    print(f"50首试点采集分布缺口报告")
    print(f"当前: {total} 首 | 目标: {target_total} 首 | 缺口: {target_total - total} 首")
    print("=" * 70)

    dimension_names = {
        "genre_major": "🎵 风格（genre_major）",
        "vocal_presence": "🎤 人声/器乐（vocal_presence）",
        "source_type": "📦 来源类型（source_type）",
        "duration_category": "⏱️  时长分类（duration_category）",
        "decade": "📅 年代（decade）",
    }

    for dim, dim_name in dimension_names.items():
        if dim not in gaps:
            continue

        print(f"\n{dim_name}")
        print("-" * 50)
        print(f"  {'类别':<25} {'目标':>6} {'当前':>6} {'缺口':>6}  状态")
        print("  " + "-" * 60)

        for category, info in gaps[dim].items():
            target = info["target"]
            current = info["current"]
            gap = info["gap"]
            if gap > 0:
                status = f"❌ 缺 {gap}"
            elif gap < 0:
                status = f"⚠️  超 {-gap}"
            else:
                status = "✅ 达标"
            print(f"  {category:<25} {target:>6} {current:>6} {gap:>+6}  {status}")

    # 艺术家统计
    artist_info = current.get("artist") if isinstance(current, dict) else None
    if artist_info and isinstance(artist_info, dict):
        print(f"\n👤 艺术家（artist_id）")
        print("-" * 50)
        print(f"  不同艺术家数: {artist_info['total_unique']} (目标 ≥{TARGET_ARTIST['min_unique']})")
        print(f"  有3首以上的艺术家: {artist_info['artists_with_3plus']} (目标 {TARGET_ARTIST['artists_with_3plus']})")
        if artist_info.get("top_artists"):
            print(f"  Top 艺术家:")
            for artist, count in artist_info["top_artists"].items():
                print(f"    {artist}: {count} 首")

    # 总结建议
    print("\n" + "=" * 70)
    print("补采建议")
    print("=" * 70)

    suggestions = []
    for dim, dim_gaps in gaps.items():
        for category, info in dim_gaps.items():
            if info["gap"] > 0:
                suggestions.append(f"  - 补采 {dim}/{category}: {info['gap']} 首")

    if total < target_total:
        print(f"  📊 总量缺口: {target_total - total} 首")
    if suggestions:
        print("\n".join(suggestions))
    else:
        print("  ✅ 所有维度均已达标或超额！")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="50首试点采集分布缺口统计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=None,
                        help="audio_manifest.csv 路径")
    parser.add_argument("--checklist", type=Path, default=None,
                        help="pilot_50_checklist.csv 路径（采集阶段使用）")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL,
                        help=f"目标总数（默认 {TARGET_TOTAL}）")
    parser.add_argument("--json", type=Path, default=None,
                        help="输出 JSON 报告路径")
    args = parser.parse_args()

    # 加载数据
    df = load_data(args.manifest, args.checklist)
    total = len(df)
    logger.info(f"当前采集: {total} 首")

    if total == 0:
        logger.warning("没有已采集的数据，请先采集或填写检查表")
        return

    # 计算当前分布
    current = compute_current_distribution(df)

    # 计算缺口
    gaps = compute_gaps(current, TARGET_DISTRIBUTION, total)

    # 打印报告
    print_gap_report(current, gaps, total, args.target)

    # 输出 JSON
    if args.json:
        report = {
            "total_current": total,
            "total_target": args.target,
            "total_gap": args.target - total,
            "current_distribution": current,
            "target_distribution": TARGET_DISTRIBUTION,
            "gaps": gaps,
            "target_artist": TARGET_ARTIST,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"JSON 报告已保存: {args.json}")


if __name__ == "__main__":
    main()
