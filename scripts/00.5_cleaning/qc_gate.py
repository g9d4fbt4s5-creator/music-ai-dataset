#!/usr/bin/env python3
"""
QC Gate — 统一质量检查三分支决策 (pass/marginal/fail)

整合 YAMNet 内容分类 + librosa 音质检查 + 元数据检查，
输出统一的三分支决策，fail 不进下游 Stage。

使用:
    python qc_gate.py --manifest data/00_raw_collect/audio_manifest.csv \
        --audio-dir data/00_raw_collect/raw_audio \
        --output data/00.5_cleaned/reports/vXXX/qc_gate_report.csv

阈值可在 configs/cleaning_config.yaml 中配置。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ========== 阈值配置 ==========
THRESHOLDS = {
    # 内容分类 (YAMNet)
    "music_score_pass": 0.7,      # > 0.7 直接通过
    "music_score_fail": 0.3,      # < 0.3 直接拒绝
    "vocal_score_threshold": 0.1, # > 0.1 判定有人声

    # 音质 (librosa, 使用原始指标 orig_*)
    "snr_db_fail": 10.0,          # < 10dB fail
    "snr_db_marginal": 12.0,      # < 12dB marginal (9首marginal人工听检100%可接受, 从15dB放宽到12dB; 老爵士/黑胶/风格乐器效果器正常底噪范围10-12dB)
    "clip_ratio_fail": 0.05,      # > 5% 削波 fail
    "clip_ratio_marginal": 0.02,  # > 2% 削波 marginal
    "silence_ratio_fail": 0.80,   # > 80% 静音 fail
    "silence_ratio_marginal": 0.60, # > 60% 静音 marginal (爵士/古典长前奏/间奏/尾奏正常, 从50%放宽到60%)
    "dr_low_fail": 3.0,            # DR < 3 fail (过度压缩)
    "dr_marginal": 5.0,            # 3 ≤ DR < 5 marginal (压缩偏重)
    "dr_high_info": 20.0,          # DR > 20 仅记录 info (高动态范围是优点, 古典/爵士/原声常见, 不是缺陷)

    # v6新增: LUFS 集成响度
    "lufs_fail_low": -36.0,        # < -36 LUFS fail (太轻)
    "lufs_fail_high": -4.0,         # > -4 LUFS fail (爆响; 金属/电子 -8~-4为正常母带风格, 从-6放宽到-4)
    "lufs_marginal_low": -28.0,     # -36 ~ -28 LUFS marginal
    "lufs_marginal_high": -8.0,     # -8 ~ -4 LUFS marginal (金属/电子正常母带风格, 从-11调整到-8)

    # v6新增: DC offset 直流偏移 (quality报告暂无, 预留接口)
    "dc_offset_fail": 0.15,         # > 0.15 fail (直流偏移带来爆音)
    "dc_offset_marginal": 0.05,     # 0.05 ~ 0.15 marginal

    # v6新增: 可解码性
    "decode_fail_on_corrupted": True,  # corrupted=True 直接 fail

    # 时长
    "duration_min_fail": 5.0,      # < 5s fail
    "duration_long_form": 900.0,   # > 15min tag=long_form
    "duration_dj_mix": 1800.0,     # > 30min tag=dj_mix

    # 源质量
    "sample_rate_min": 22050,      # < 22050Hz marginal
    # 注意: 比特率检查只对 orig_bitrate(原始有损格式)有意义, FLAC无损压缩比特率随内容变化, 不检查
}


def classify_content(yamnet_row):
    """YAMNet 内容分类，返回 (is_music, has_vocals, content_branch, reason)"""
    music_score = float(yamnet_row.get("music_score", 0))
    vocal_score = float(yamnet_row.get("vocal_score", 0))

    is_music = music_score > THRESHOLDS["music_score_fail"]
    has_vocals = vocal_score > THRESHOLDS["vocal_score_threshold"]

    if music_score >= THRESHOLDS["music_score_pass"]:
        return is_music, has_vocals, "pass", f"music_score={music_score:.3f}>=0.7"
    elif music_score <= THRESHOLDS["music_score_fail"]:
        return is_music, has_vocals, "fail", f"music_score={music_score:.3f}<=0.3"
    else:
        return is_music, has_vocals, "marginal", f"music_score={music_score:.3f} in [0.3,0.7]"


def check_quality(quality_row):
    """librosa 音质检查，返回 (quality_branch, flags, reason, quality_warn)

    v6新增: LUFS集成响度检查、DC offset直流偏移检查(预留接口)
    quality_warn: 触发marginal的全部原因列表, 人工审核时一目了然
    """
    flags = []
    quality_warn = []  # v6新增: 只记录marginal/fail级别的警告
    branch = "pass"

    snr = float(quality_row.get("snr_db", 999))
    clip = float(quality_row.get("clipping_ratio", quality_row.get("clip_ratio", 0)))
    silence = float(quality_row.get("silence_ratio", 0))
    dr = float(quality_row.get("dynamic_range_db", quality_row.get("dynamic_range", 10)))  # 兼容两种列名
    # v6新增: LUFS集成响度
    lufs = quality_row.get("loudness_lufs", None)
    if lufs is not None and str(lufs).lower() not in ("nan", "none", ""):
        lufs = float(lufs)
    else:
        lufs = None
    # v6新增: DC offset (quality报告暂无, 预留接口)
    dc_offset = quality_row.get("dc_offset", None)
    if dc_offset is not None and str(dc_offset).lower() not in ("nan", "none", ""):
        dc_offset = float(dc_offset)
    else:
        dc_offset = None

    if snr < THRESHOLDS["snr_db_fail"]:
        flags.append(f"low_snr({snr:.1f}dB)")
        quality_warn.append(f"low_snr:{snr:.1f}dB")
        branch = "fail"
    elif snr < THRESHOLDS["snr_db_marginal"]:
        flags.append(f"marginal_snr({snr:.1f}dB)")
        quality_warn.append(f"marginal_snr:{snr:.1f}dB")
        if branch == "pass":
            branch = "marginal"

    if clip > THRESHOLDS["clip_ratio_fail"]:
        flags.append(f"high_clipping({clip:.1%})")
        quality_warn.append(f"high_clipping:{clip:.1%}")
        branch = "fail"
    elif clip > THRESHOLDS["clip_ratio_marginal"]:
        flags.append(f"marginal_clipping({clip:.1%})")
        quality_warn.append(f"marginal_clipping:{clip:.1%}")
        if branch == "pass":
            branch = "marginal"

    if silence > THRESHOLDS["silence_ratio_fail"]:
        flags.append(f"high_silence({silence:.1%})")
        quality_warn.append(f"high_silence:{silence:.1%}")
        branch = "fail"
    elif silence > THRESHOLDS["silence_ratio_marginal"]:
        flags.append(f"marginal_silence({silence:.1%})")
        quality_warn.append(f"marginal_silence:{silence:.1%}")
        if branch == "pass":
            branch = "marginal"

    # DR (动态范围): 高DR是优点, 低DR才是问题
    if dr < THRESHOLDS["dr_low_fail"]:
        flags.append(f"low_dr({dr:.1f})")
        quality_warn.append(f"low_dr:{dr:.1f}")
        branch = "fail"
    elif dr < THRESHOLDS["dr_marginal"]:
        flags.append(f"marginal_dr({dr:.1f})")
        quality_warn.append(f"marginal_dr:{dr:.1f}")
        if branch == "pass":
            branch = "marginal"
    elif dr > THRESHOLDS["dr_high_info"]:
        # DR > 20 是高动态范围(古典/爵士/原声常见), 仅记录info, 不影响分支
        flags.append(f"high_dr({dr:.1f})")

    # v6新增: LUFS集成响度检查
    if lufs is not None:
        if lufs < THRESHOLDS["lufs_fail_low"] or lufs > THRESHOLDS["lufs_fail_high"]:
            flags.append(f"lufs_extreme({lufs:.1f} LUFS)")
            quality_warn.append(f"lufs_extreme:{lufs:.1f}LUFS")
            branch = "fail"
        elif (THRESHOLDS["lufs_fail_low"] <= lufs <= THRESHOLDS["lufs_marginal_low"] or
              THRESHOLDS["lufs_marginal_high"] <= lufs <= THRESHOLDS["lufs_fail_high"]):
            flags.append(f"marginal_lufs({lufs:.1f} LUFS)")
            quality_warn.append(f"marginal_lufs:{lufs:.1f}LUFS")
            if branch == "pass":
                branch = "marginal"

    # v6新增: DC offset直流偏移检查 (预留接口, 当前quality报告暂无此字段)
    if dc_offset is not None:
        if dc_offset > THRESHOLDS["dc_offset_fail"]:
            flags.append(f"high_dc_offset({dc_offset:.3f})")
            quality_warn.append(f"high_dc_offset:{dc_offset:.3f}")
            branch = "fail"
        elif dc_offset > THRESHOLDS["dc_offset_marginal"]:
            flags.append(f"marginal_dc_offset({dc_offset:.3f})")
            quality_warn.append(f"marginal_dc_offset:{dc_offset:.3f}")
            if branch == "pass":
                branch = "marginal"

    reason = "; ".join(flags) if flags else "all_checks_pass"
    return branch, flags, reason, quality_warn


def check_duration(duration_sec):
    """时长检查，返回 (duration_branch, tags, reason)"""
    tags = []
    branch = "pass"

    if duration_sec < THRESHOLDS["duration_min_fail"]:
        branch = "fail"
        tags.append("too_short")
        reason = f"duration={duration_sec:.1f}s<5s"
    elif duration_sec > THRESHOLDS["duration_dj_mix"]:
        tags.append("dj_mix")
        reason = f"duration={duration_sec:.1f}s>30min, needs_segmentation"
    elif duration_sec > THRESHOLDS["duration_long_form"]:
        tags.append("long_form")
        reason = f"duration={duration_sec:.1f}s>15min, tagged_long_form"
        branch = "marginal"
    else:
        reason = f"duration={duration_sec:.1f}s normal"

    return branch, tags, reason


def check_source_quality(meta_row):
    """源质量检查（采样率/声道/原始比特率），返回 (source_branch, flags, reason)

    注意: 比特率只检查 orig_bitrate(原始有损格式如mp3/aac的比特率)。
    FLAC是无损压缩, 比特率随内容动态变化, 不反映源质量, 因此不检查。
    """
    flags = []
    branch = "pass"

    sr = int(meta_row.get("sample_rate", 44100))
    channels = int(meta_row.get("channels", 2))
    # 只读取原始比特率, FLAC转码后的比特率不检查
    orig_bitrate = meta_row.get("orig_bitrate", None)

    if sr < THRESHOLDS["sample_rate_min"]:
        flags.append(f"low_sr({sr}Hz)")
        branch = "marginal"
    if channels == 1:
        # mono 只作为 info 标记，不影响分支：很多爵士/老录音本身就是单声道，不是质量问题
        flags.append("mono")
    # 比特率检查: 仅当 orig_bitrate 存在且为有损格式时检查
    if orig_bitrate is not None:
        orig_bitrate = int(orig_bitrate)
        if orig_bitrate > 0 and orig_bitrate < 128000:
            flags.append(f"low_orig_bitrate({orig_bitrate//1000}kbps)")
            if branch == "pass":
                branch = "marginal"

    reason = "; ".join(flags) if flags else "source_quality_ok"
    return branch, flags, reason


def check_decodeability(quality_row):
    """v6新增: 可解码性检查，返回 (decode_branch, flags, reason)

    corrupted=True 或 全零/全NaN音频 直接 fail。
    """
    flags = []
    branch = "pass"

    corrupted = quality_row.get("corrupted", False)
    # 兼容字符串格式
    if isinstance(corrupted, str):
        corrupted = corrupted.lower() in ("true", "1", "yes")

    if corrupted and THRESHOLDS["decode_fail_on_corrupted"]:
        flags.append("corrupted_audio")
        branch = "fail"
        reason = "corrupted_audio_unable_to_decode"
    else:
        reason = "decode_ok"

    return branch, flags, reason


def check_mapping_blacklist(raw_tags, mapping_dict_path=None):
    """
    映射字典黑名单检查：若原始标签命中黑名单，直接 fail。

    黑名单标签示例: noise, speech, silence, low quality, distorted 等。
    这些标签表示音频不是音乐或质量极差，应在映射阶段提前过滤，
    不浪费下游 GPU 算力。

    Args:
        raw_tags: 原始标签列表（从文本描述/模型输出提取）
        mapping_dict_path: label_mapping_dict.json 路径

    Returns:
        (branch, hit_tags, reason)
    """
    if not raw_tags:
        return "pass", [], "no_raw_tags"

    # 加载黑名单
    blacklist = set()
    if mapping_dict_path and os.path.exists(mapping_dict_path):
        try:
            with open(mapping_dict_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            blacklist = set(mapping.get("blacklist_tags", []))
        except Exception:
            pass

    if not blacklist:
        return "pass", [], "no_blacklist_configured"

    hit_tags = [tag for tag in raw_tags if tag.lower().strip() in blacklist]

    if hit_tags:
        return "fail", hit_tags, f"blacklist_tags_hit: {hit_tags}"
    return "pass", [], "no_blacklist_hit"


def merge_branches(branches):
    """合并多个分支决策: fail > marginal > pass"""
    if "fail" in branches:
        return "fail"
    if "marginal" in branches:
        return "marginal"
    return "pass"


def run_qc_gate(manifest_path, yamnet_path, quality_path, output_path,
                mapping_dict_path=None):
    """主流程: 读取各检查结果，合并为统一三分支决策"""
    manifest = pd.read_csv(manifest_path)
    yamnet = pd.read_csv(yamnet_path) if os.path.exists(yamnet_path) else pd.DataFrame()
    quality = pd.read_csv(quality_path) if os.path.exists(quality_path) else pd.DataFrame()

    results = []
    for _, row in manifest.iterrows():
        audio_id = row["audio_id"]
        duration = float(row.get("duration_sec", 0))

        # 1. 内容分类
        content_branch = "pass"
        is_music = True
        has_vocals = False
        content_reason = "no_yamnet_data"
        if not yamnet.empty and audio_id in yamnet["audio_id"].values:
            yrow = yamnet[yamnet["audio_id"] == audio_id].iloc[0]
            is_music, has_vocals, content_branch, content_reason = classify_content(yrow)

        # 2. 音质检查
        quality_branch = "pass"
        quality_flags = []
        quality_reason = "no_quality_data"
        quality_warn = []  # v6新增: 触发marginal/fail的全部原因
        if not quality.empty and audio_id in quality["audio_id"].values:
            qrow = quality[quality["audio_id"] == audio_id].iloc[0]
            quality_branch, quality_flags, quality_reason, quality_warn = check_quality(qrow)

        # v6新增: 2.5 可解码性检查
        decode_branch = "pass"
        decode_flags = []
        decode_reason = "no_quality_data"
        if not quality.empty and audio_id in quality["audio_id"].values:
            qrow = quality[quality["audio_id"] == audio_id].iloc[0]
            decode_branch, decode_flags, decode_reason = check_decodeability(qrow)
            if decode_branch == "fail":
                quality_warn.append("corrupted_audio")

        # 3. 时长检查
        duration_branch, duration_tags, duration_reason = check_duration(duration)

        # 4. 源质量检查
        source_branch, source_flags, source_reason = check_source_quality(row)

        # 5. 映射字典黑名单检查（从元数据的原始标签提取）
        blacklist_branch = "pass"
        blacklist_hits = []
        blacklist_reason = "no_raw_tags"
        raw_tags_str = row.get("raw_tags", row.get("aspect_list", ""))
        if raw_tags_str:
            raw_tags = [t.strip() for t in str(raw_tags_str).split(",") if t.strip()]
            blacklist_branch, blacklist_hits, blacklist_reason = check_mapping_blacklist(
                raw_tags, mapping_dict_path
            )

        # 6. 合并决策
        final_branch = merge_branches([
            content_branch, quality_branch, duration_branch,
            source_branch, blacklist_branch, decode_branch
        ])

        all_flags = quality_flags + duration_tags + source_flags + blacklist_hits + decode_flags
        all_reasons = {
            "content": content_reason,
            "quality": quality_reason,
            "duration": duration_reason,
            "source": source_reason,
            "blacklist": blacklist_reason,
            "decode": decode_reason,
        }

        results.append({
            "audio_id": audio_id,
            "duration_sec": duration,
            "is_music": is_music,
            "has_vocals": has_vocals,
            "content_branch": content_branch,
            "quality_branch": quality_branch,
            "duration_branch": duration_branch,
            "source_branch": source_branch,
            "decode_branch": decode_branch,  # v6新增
            "final_branch": final_branch,
            "flags": json.dumps(all_flags, ensure_ascii=False),
            "quality_warn": json.dumps(quality_warn, ensure_ascii=False),  # v6新增: 触发marginal/fail的全部原因
            "flag_for_review": final_branch == "marginal",
            "reasons": json.dumps(all_reasons, ensure_ascii=False),
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # 统计摘要
    summary = {
        "total": len(df),
        "pass": int((df["final_branch"] == "pass").sum()),
        "marginal": int((df["final_branch"] == "marginal").sum()),
        "fail": int((df["final_branch"] == "fail").sum()),
        "flag_for_review": int(df["flag_for_review"].sum()),
        "has_vocals": int(df["has_vocals"].sum()),
        "is_music": int(df["is_music"].sum()),
    }
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"QC Gate 完成: {output_path}")
    print(f"{'='*60}")
    print(f"  总计: {summary['total']}")
    print(f"  ✅ pass: {summary['pass']} ({summary['pass']/summary['total']:.1%})")
    print(f"  ⚠️  marginal: {summary['marginal']} ({summary['marginal']/summary['total']:.1%})")
    print(f"  ❌ fail: {summary['fail']} ({summary['fail']/summary['total']:.1%})")
    print(f"  🏷️  flag_for_review: {summary['flag_for_review']}")
    print(f"  🎤 has_vocals: {summary['has_vocals']}")
    print(f"  🎵 is_music: {summary['is_music']}")
    print(f"\n  fail 样本将不进入下游 Stage")
    print(f"  marginal 样本进入下游但标记 flag_for_review=true")

    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QC Gate — 统一质量检查三分支决策")
    parser.add_argument("--manifest", required=True, help="audio_manifest.csv 路径")
    parser.add_argument("--yamnet", default="", help="YAMNet 结果 CSV 路径")
    parser.add_argument("--quality", default="", help="音质检查结果 CSV 路径")
    parser.add_argument("--output", required=True, help="输出 qc_gate_report.csv 路径")
    args = parser.parse_args()

    run_qc_gate(args.manifest, args.yamnet, args.quality, args.output)
