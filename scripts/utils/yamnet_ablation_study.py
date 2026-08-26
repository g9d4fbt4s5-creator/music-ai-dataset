#!/usr/bin/env python3
"""
YAMNet 误杀原因归因实验：2x2 控制变量对比

四组样本：
  A组: Ace Studio 生成 + 未分轨（原版）
  B组: 非 Ace Studio + Demucs 分轨后人声单轨
  C组: Ace Studio 生成 + Demucs 分轨后人声单轨（已有，5首误杀）
  D组: 非 Ace Studio + 未分轨（已有，19首jazz正常pass）

通过对比四组的 YAMNet music_score 和 top5_events，精确定位误杀原因。

用法：
    python scripts/utils/yamnet_ablation_study.py \\
        --group-a data/ablation/ace_original/*.wav \\
        --group-b data/ablation/real_demucs_vocals/*.wav \\
        --group-c data/00_raw_collect/raw_audio/.../ace_demucs_*.wav \\
        --group-d data/00_raw_collect/raw_audio/.../jazz_*.mp3 \\
        --output data/ablation/yamnet_ablation_results.csv
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# 尝试导入 YAMNet 相关库
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    YAMNET_AVAILABLE = True
except ImportError:
    YAMNET_AVAILABLE = False
    print("⚠️ TensorFlow/Hab not available, will use pre-computed YAMNet results if provided")


# ============================================================
# YAMNet 推理
# ============================================================
class YamnetInferencer:
    """YAMNet 模型推理封装"""

    def __init__(self, model_url: str = "https://tfhub.dev/google/yamnet/1"):
        if not YAMNET_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        self.model = hub.load(model_url)
        self.class_names = self.model.class_names

    def infer(self, audio_path: str, sr: int = 16000) -> Dict[str, Any]:
        """
        对单个音频文件运行 YAMNet 推理

        Returns:
            包含 music_score, speech_score, top5_events 等的字典
        """
        import librosa

        # 加载音频
        wav_data, _ = librosa.load(audio_path, sr=sr, mono=True)
        wav_data = wav_data.astype(np.float32)

        # 运行推理
        scores, embeddings, spectrogram = self.model(wav_data)
        scores_np = scores.numpy()

        # 计算各类别平均分数
        mean_scores = np.mean(scores_np, axis=0)

        # 获取 top5 事件
        top5_indices = np.argsort(mean_scores)[::-1][:5]
        top5_events = [
            {"class": self.class_names[i], "score": float(mean_scores[i])}
            for i in top5_indices
        ]

        # 查找特定类别的分数
        def get_class_score(class_name: str) -> float:
            try:
                idx = self.class_names.index(class_name)
                return float(mean_scores[idx])
            except ValueError:
                return 0.0

        music_score = get_class_score("Music")
        speech_score = get_class_score("Speech")
        singing_score = get_class_score("Singing")
        chant_score = get_class_score("Chant")
        mantra_score = get_class_score("Mantra")

        # 人声音乐总分（Singing + Chant + Mantra）
        vocal_music_score = singing_score + chant_score + mantra_score

        return {
            "music_score": round(music_score, 4),
            "speech_score": round(speech_score, 4),
            "singing_score": round(singing_score, 4),
            "chant_score": round(chant_score, 4),
            "mantra_score": round(mantra_score, 4),
            "vocal_music_score": round(vocal_music_score, 4),
            "top5_events": json.dumps(top5_events, ensure_ascii=False),
            "top1_class": top5_events[0]["class"],
            "top1_score": round(top5_events[0]["score"], 4),
        }


# ============================================================
# 实验逻辑
# ============================================================
def load_audio_paths(patterns: List[str]) -> List[str]:
    """从 glob 模式加载音频文件路径"""
    import glob
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(paths)


def run_ablation_study(
    group_a_paths: List[str],
    group_b_paths: List[str],
    group_c_paths: List[str],
    group_d_paths: List[str],
    output_path: Path,
    use_precomputed: Optional[str] = None
) -> pd.DataFrame:
    """
    运行 2x2 消融实验

    Args:
        group_a_paths: A组（Ace Studio + 未分轨）
        group_b_paths: B组（真实录音 + 分轨人声）
        group_c_paths: C组（Ace Studio + 分轨人声，已有）
        group_d_paths: D组（真实录音 + 未分轨，已有）
        output_path: 输出 CSV 路径
        use_precomputed: 预计算的 YAMNet 结果 CSV 路径（可选）

    Returns:
        结果 DataFrame
    """
    results = []

    # 如果有预计算结果，直接加载
    precomputed_df = None
    if use_precomputed and Path(use_precomputed).exists():
        precomputed_df = pd.read_csv(use_precomputed)
        print(f"✅ 加载预计算 YAMNet 结果: {len(precomputed_df)} 条")

    # 初始化 YAMNet 推理器（如果需要）
    inferencer = None
    if precomputed_df is None and YAMNET_AVAILABLE:
        print("🔄 加载 YAMNet 模型...")
        inferencer = YamnetInferencer()
        print("✅ YAMNet 模型加载完成")

    groups = {
        "A_ace_original": group_a_paths,
        "B_real_demucs": group_b_paths,
        "C_ace_demucs": group_c_paths,
        "D_real_original": group_d_paths,
    }

    for group_name, paths in groups.items():
        print(f"\n=== 处理组 {group_name}: {len(paths)} 首 ===")

        for audio_path in paths:
            audio_name = Path(audio_path).name

            # 尝试从预计算结果中获取
            result = None
            if precomputed_df is not None:
                # 按文件名匹配
                match = precomputed_df[precomputed_df["audio_id"].str.contains(audio_name[:10], na=False)]
                if len(match) > 0:
                    row = match.iloc[0]
                    result = {
                        "music_score": float(row.get("music_score", 0)),
                        "speech_score": float(row.get("speech_score", 0)),
                        "vocal_score": float(row.get("vocal_score", 0)),
                        "top5_events": row.get("top5_events", "[]"),
                        "top1_class": "unknown",
                        "top1_score": 0,
                        "singing_score": 0,
                        "chant_score": 0,
                        "mantra_score": 0,
                        "vocal_music_score": 0,
                    }

            # 如果没有预计算结果，运行推理
            if result is None and inferencer is not None:
                try:
                    result = inferencer.infer(audio_path)
                    print(f"  ✅ {audio_name[:30]}: music={result['music_score']}, top1={result['top1_class']}")
                except Exception as e:
                    print(f"  ❌ {audio_name[:30]}: {e}")
                    continue
            elif result is None:
                print(f"  ⚠️ {audio_name[:30]}: 无预计算结果且无法运行推理，跳过")
                continue

            # 添加分组信息
            result["group"] = group_name
            result["audio_path"] = audio_path
            result["audio_name"] = audio_name

            # 解析分组标签
            if group_name.startswith("A"):
                result["source"] = "ace_studio"
                result["processing"] = "original"
            elif group_name.startswith("B"):
                result["source"] = "real_recording"
                result["processing"] = "demucs_vocals"
            elif group_name.startswith("C"):
                result["source"] = "ace_studio"
                result["processing"] = "demucs_vocals"
            elif group_name.startswith("D"):
                result["source"] = "real_recording"
                result["processing"] = "original"

            results.append(result)

    # 转换为 DataFrame
    df = pd.DataFrame(results)

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ 结果已保存到: {output_path}")

    return df


def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析实验结果，给出归因结论
    """
    print("\n" + "=" * 60)
    print("📊 实验结果分析")
    print("=" * 60)

    # 按组统计
    group_stats = df.groupby("group").agg({
        "music_score": ["mean", "std", "min", "max", "count"],
        "speech_score": ["mean"],
        "vocal_music_score": ["mean"],
    }).round(4)

    print("\n=== 各组统计 ===")
    print(group_stats.to_string())

    # 2x2 表格
    print("\n=== 2x2 对比表 (music_score 均值) ===")
    pivot = df.pivot_table(
        values="music_score",
        index="source",
        columns="processing",
        aggfunc="mean"
    ).round(4)
    print(pivot.to_string())

    # 归因分析
    print("\n=== 归因分析 ===")

    # 获取各组均值
    def get_group_mean(group_prefix: str) -> float:
        group_df = df[df["group"].str.startswith(group_prefix)]
        return group_df["music_score"].mean() if len(group_df) > 0 else 0

    a_mean = get_group_mean("A")  # Ace + 原版
    b_mean = get_group_mean("B")  # 真实 + 分轨
    c_mean = get_group_mean("C")  # Ace + 分轨（误杀）
    d_mean = get_group_mean("D")  # 真实 + 原版（正常）

    print(f"  A组 (Ace+原版):   music_score = {a_mean:.4f}")
    print(f"  B组 (真实+分轨):  music_score = {b_mean:.4f}")
    print(f"  C组 (Ace+分轨):   music_score = {c_mean:.4f} (已知误杀)")
    print(f"  D组 (真实+原版):   music_score = {d_mean:.4f} (已知正常)")

    # 判断阈值（YAMNet fail 阈值为 0.3）
    threshold = 0.3
    a_fail = a_mean < threshold
    b_fail = b_mean < threshold
    c_fail = c_mean < threshold
    d_fail = d_mean < threshold

    print(f"\n  阈值: music_score < {threshold} 判为非音乐")
    print(f"  A组误杀: {'是' if a_fail else '否'}")
    print(f"  B组误杀: {'是' if b_fail else '否'}")
    print(f"  C组误杀: {'是' if c_fail else '否'} (已知)")
    print(f"  D组误杀: {'是' if d_fail else '否'} (已知)")

    # 归因结论
    print("\n=== 归因结论 ===")
    if not a_fail and b_fail and c_fail:
        conclusion = "原因是 Demucs 分轨（人声单轨频谱特征导致 YAMNet 误判）"
    elif a_fail and not b_fail and c_fail:
        conclusion = "原因是 Ace Studio 生成（AI合成人声声学特征导致 YAMNet 误判）"
    elif not a_fail and not b_fail and c_fail:
        conclusion = "原因是两者共同作用（单独一个因素不够，Ace Studio + Demucs 叠加才误杀）"
    elif a_fail and b_fail and c_fail:
        conclusion = "原因是人声单轨本身（任何纯人声都会被 YAMNet 误判）"
    else:
        conclusion = "结果模式不明确，需要更多样本或进一步分析"

    print(f"  🎯 {conclusion}")

    # 检查 top5_events 中的 Singing 出现频率
    print("\n=== top5 中 Singing/Chant/Mantra 出现频率 ===")
    for group in ["A", "B", "C", "D"]:
        group_df = df[df["group"].str.startswith(group)]
        if len(group_df) == 0:
            continue
        singing_count = 0
        for _, row in group_df.iterrows():
            top5 = row.get("top5_events", "[]")
            if isinstance(top5, str):
                try:
                    top5_list = json.loads(top5)
                    classes = [item.get("class", "") for item in top5_list]
                    if any(c in ["Singing", "Chant", "Mantra"] for c in classes):
                        singing_count += 1
                except:
                    pass
        print(f"  {group}组: {singing_count}/{len(group_df)} 首 top5 含人声音乐标签")

    return {
        "group_stats": group_stats.to_dict(),
        "pivot_table": pivot.to_dict(),
        "conclusion": conclusion,
    }


def main():
    parser = argparse.ArgumentParser(
        description="YAMNet 误杀原因归因实验：2x2 控制变量对比"
    )
    parser.add_argument("--group-a", nargs="+", default=[],
                        help="A组：Ace Studio 生成 + 未分轨（原版）音频路径或 glob 模式")
    parser.add_argument("--group-b", nargs="+", default=[],
                        help="B组：非 Ace Studio + Demucs 分轨后人声单轨")
    parser.add_argument("--group-c", nargs="+", default=[],
                        help="C组：Ace Studio 生成 + Demucs 分轨后人声单轨（已有误杀样本）")
    parser.add_argument("--group-d", nargs="+", default=[],
                        help="D组：非 Ace Studio + 未分轨（已有正常样本）")
    parser.add_argument("--output", type=Path,
                        default=Path("data/ablation/yamnet_ablation_results.csv"),
                        help="输出结果 CSV 路径")
    parser.add_argument("--precomputed", type=str, default=None,
                        help="预计算的 YAMNet 结果 CSV 路径（可选）")
    parser.add_argument("--analyze-only", action="store_true",
                        help="仅分析已有结果，不运行推理")

    args = parser.parse_args()

    # 加载音频路径
    group_a = load_audio_paths(args.group_a) if args.group_a else []
    group_b = load_audio_paths(args.group_b) if args.group_b else []
    group_c = load_audio_paths(args.group_c) if args.group_c else []
    group_d = load_audio_paths(args.group_d) if args.group_d else []

    print(f"=== 样本数量 ===")
    print(f"  A组 (Ace+原版):   {len(group_a)} 首")
    print(f"  B组 (真实+分轨):  {len(group_b)} 首")
    print(f"  C组 (Ace+分轨):   {len(group_c)} 首")
    print(f"  D组 (真实+原版):   {len(group_d)} 首")

    if args.analyze_only and args.output.exists():
        df = pd.read_csv(args.output)
        analyze_results(df)
        return

    # 运行实验
    df = run_ablation_study(
        group_a, group_b, group_c, group_d,
        args.output, args.precomputed
    )

    # 分析结果
    if len(df) > 0:
        analyze_results(df)


if __name__ == "__main__":
    main()
