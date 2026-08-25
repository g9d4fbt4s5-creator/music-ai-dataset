#!/usr/bin/env python3
"""
生成 50 首试点采集检查表 CSV（ADR-003 试点验证）

六个维度覆盖：
1. 风格（genre_major）：至少5个流派
2. 人声/器乐（vocal_presence）：人声/纯器乐/混合
3. 来源（source_type）：normal/AI生成/分轨人声
4. 时长（duration_sec）：标准/长曲/短曲
5. 年代（decade）：老录音/现代
6. 艺术家（artist_id）：至少15个不同artist，3个各有3-5首

用法：
    python scripts/utils/generate_pilot_checklist.py
    # 生成空模板，手动填写

    python scripts/utils/generate_pilot_checklist.py --from-manifest data/00_raw_collect/audio_manifest.csv
    # 从现有 manifest 生成已采集部分的检查表
"""
import argparse
from pathlib import Path

import pandas as pd

# 检查表列定义
CHECKLIST_COLUMNS = [
    # 基本信息
    "audio_id",              # 入库后分配的ID，采集阶段可留空
    "original_filename",     # 原始文件名
    "source_url",            # 来源链接（如有）
    # 六个维度
    "genre_major",           # 风格大类：Jazz/Rock/Classical/Pop/Electronic/...
    "vocal_presence",        # instrumental/vocal/mixed
    "source_type",           # normal/ace_studio_generated/demucs_vocals/...
    "duration_sec",          # 时长（秒）
    "decade",                # 年代：1950s/1960s/1970s/1980s/1990s/2000s/2010s/2020s
    "artist_id",             # 艺术家标识，未知填 unknown_<hash前8位>
    # 采集状态
    "collected",             # 是否已采集：yes/no
    "qc_status",             # QC状态：pass/marginal/fail/pending
    "notes",                 # 备注
]

# 目标分布（50首）
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
    "artist": {
        "total_unique": ">=15",
        "artists_with_3plus": "3个artist各有3-5首",
    },
}


def generate_empty_template(output_path: Path):
    """生成空的检查表模板"""
    df = pd.DataFrame(columns=CHECKLIST_COLUMNS)
    # 预填50行空行，方便填写
    for i in range(50):
        df.loc[i] = ["" for _ in CHECKLIST_COLUMNS]
    df.to_csv(output_path, index=False)
    print(f"✅ 已生成空检查表模板: {output_path}")
    print(f"   共 {len(df)} 行，请按六个维度填写采集计划")


def generate_from_manifest(manifest_path: Path, output_path: Path):
    """从现有 manifest 生成已采集部分的检查表"""
    manifest_df = pd.read_csv(manifest_path)
    print(f"加载 manifest: {len(manifest_df)} 条")

    # 映射 manifest 列到检查表列
    checklist_data = []
    for _, row in manifest_df.iterrows():
        item = {
            "audio_id": row.get("audio_id", ""),
            "original_filename": row.get("original_filename", ""),
            "source_url": "",
            "genre_major": row.get("genre_major", ""),
            "vocal_presence": row.get("vocal_presence", ""),
            "source_type": row.get("source_type", "normal"),
            "duration_sec": row.get("duration_sec", row.get("duration", "")),
            "decade": row.get("decade", ""),
            "artist_id": row.get("artist_id", ""),
            "collected": "yes",
            "qc_status": row.get("final_branch", "pending"),
            "notes": "",
        }
        checklist_data.append(item)

    df = pd.DataFrame(checklist_data, columns=CHECKLIST_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"✅ 已从 manifest 生成检查表: {output_path}")
    print(f"   共 {len(df)} 条已采集记录")

    # 打印当前分布统计
    print("\n=== 当前分布统计 ===")
    for col in ["genre_major", "vocal_presence", "source_type", "decade"]:
        if col in df.columns and df[col].notna().any():
            print(f"\n{col}:")
            print(df[col].value_counts().to_string())


def print_target_distribution():
    """打印目标分布参考"""
    print("\n" + "=" * 60)
    print("50 首试点目标分布参考")
    print("=" * 60)
    for dim, dist in TARGET_DISTRIBUTION.items():
        print(f"\n【{dim}】")
        if isinstance(dist, dict):
            for key, val in dist.items():
                print(f"  {key}: {val}")
        else:
            print(f"  {dist}")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="生成 50 首试点采集检查表")
    parser.add_argument("--output", default="reports/pilot_50/pilot_50_checklist.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--from-manifest", default=None,
                        help="从现有 manifest 生成已采集部分的检查表")
    parser.add_argument("--show-target", action="store_true",
                        help="只打印目标分布参考，不生成文件")
    args = parser.parse_args()

    if args.show_target:
        print_target_distribution()
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.from_manifest:
        manifest_path = Path(args.from_manifest)
        if not manifest_path.exists():
            print(f"❌ manifest 不存在: {manifest_path}")
            return
        generate_from_manifest(manifest_path, output_path)
    else:
        generate_empty_template(output_path)

    print_target_distribution()


if __name__ == "__main__":
    main()
