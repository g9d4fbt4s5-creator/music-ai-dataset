"""
fingerprint_dedup.py
音频指纹去重脚本（Stage 4.2 - 感知指纹层）

功能：
- 使用 Chromaprint (fpcalc) 提取音频感知指纹（工业标准，Shazam同款原理）
- 比对指纹相似度，检测"同一首歌不同格式/码率/音量"的重复
- 支持降级：如果 fpcalc 未安装，自动降级到 librosa chroma 特征
- 输出去重映射表（保留音质最高的版本）
- 支持批量处理和增量更新

技术方案：
- Chromaprint：基于音频内容的感知哈希，对格式/码率/音量变化鲁棒
- 指纹比对：Chromaprint 内置的相似度算法，>0.95 视为同一首歌
- 性能：~50-100首/秒（CPU），指纹大小~2-5KB/首，10万首指纹库~500MB

与现有去重的关系：
- Stage 4.1 精确去重（MD5/SHA-256）：文件级，完全相同的文件
- Stage 4.2 音频指纹去重（Chromaprint）：感知级，同一首歌不同格式 ✅ 本脚本
- Stage 4.3 片段级去重（滑动窗口）：片段级，同一首歌的不同节选
- Stage 4.4 跨集去重：全局指纹库+子集标记，训练/验证/测试集泄露

用法：
    # 全部去重（自动检测 fpcalc，未安装则降级到 chroma）
    python fingerprint_dedup.py

    # 强制使用 Chromaprint
    python fingerprint_dedup.py --method chromaprint

    # 强制使用 librosa chroma（降级方案）
    python fingerprint_dedup.py --method chroma

    # 自定义相似度阈值
    python fingerprint_dedup.py --threshold 0.95

    # 只处理指定数量
    python fingerprint_dedup.py --limit 100

    # 预览模式（不输出结果）
    python fingerprint_dedup.py --dry-run

    # 输出去重映射表
    python fingerprint_dedup.py --output ./data/00.5_cleaned/fingerprint_dedup_mapping.csv

    # 增量更新（只处理新增音频）
    python fingerprint_dedup.py --incremental
"""
import os
import sys
import subprocess
import hashlib
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
from collections import defaultdict

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 添加 utils 目录到路径
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from get_audio_physical_path import get_audio_absolute_path

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"fingerprint_dedup_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认路径
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "00_raw_collect" / "audio_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "00.5_cleaned" / "dedup_results"
DEFAULT_FINGERPRINT_DB = PROJECT_ROOT / "data" / "00.5_cleaned" / "dedup_results" / "fingerprint_db.csv"

# 默认阈值
DEFAULT_THRESHOLD = 0.95  # Chromaprint 相似度 >0.95 视为同一首歌


# ===================== 工具函数 =====================
def check_fpcalc_available() -> bool:
    """检查 fpcalc (Chromaprint) 是否可用"""
    try:
        result = subprocess.run(["fpcalc", "-version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_audio_path_from_row(row: pd.Series) -> Optional[str]:
    """从 manifest 行获取音频的绝对路径"""
    # 优先从 file_relative_path 读取完整路径
    rel_path = row.get("file_relative_path", "")
    if rel_path and isinstance(rel_path, str):
        abs_path = PROJECT_ROOT / "data" / "00_raw_collect" / rel_path
        if abs_path.exists():
            return str(abs_path)

    # 尝试用 audio_id 查找
    audio_id = row.get("audio_id", "")
    if audio_id:
        abs_path = get_audio_absolute_path(audio_id)
        if abs_path and Path(abs_path).exists():
            return abs_path

    return None


# ===================== Chromaprint 指纹提取 =====================
def extract_fingerprint_chromaprint(audio_path: str) -> Optional[Tuple[float, str]]:
    """
    使用 Chromaprint (fpcalc) 提取音频指纹

    Args:
        audio_path: 音频文件路径

    Returns:
        (duration, fingerprint_str) 或 None（提取失败）
    """
    try:
        # fpcalc -raw 输出原始指纹（用于比对）
        result = subprocess.run(
            ["fpcalc", "-raw", audio_path],
            capture_output=True,
            text=True,
            timeout=30  # 单首超时30秒
        )

        if result.returncode != 0:
            logger.warning(f"fpcalc 提取失败: {audio_path}")
            return None

        # 解析输出
        duration = None
        fingerprint = None
        for line in result.stdout.strip().split("\n"):
            if line.startswith("DURATION="):
                duration = float(line.split("=", 1)[1])
            elif line.startswith("FINGERPRINT="):
                fingerprint = line.split("=", 1)[1]

        if duration is None or fingerprint is None:
            logger.warning(f"fpcalc 输出解析失败: {audio_path}")
            return None

        return (duration, fingerprint)

    except subprocess.TimeoutExpired:
        logger.warning(f"fpcalc 提取超时: {audio_path}")
        return None
    except Exception as e:
        logger.warning(f"fpcalc 提取异常: {audio_path}, {e}")
        return None


def compare_fingerprints_chromaprint(fp1: str, fp2: str) -> float:
    """
    比对两个 Chromaprint 指纹的相似度

    使用 fpcalc 的 -compare 功能，或手动计算汉明距离

    Args:
        fp1: 指纹1（逗号分隔的整数）
        fp2: 指纹2

    Returns:
        相似度（0-1）
    """
    try:
        # 解析指纹为整数列表
        ints1 = [int(x) for x in fp1.split(",") if x.strip()]
        ints2 = [int(x) for x in fp2.split(",") if x.strip()]

        # 对齐长度
        min_len = min(len(ints1), len(ints2))
        if min_len == 0:
            return 0.0

        ints1 = ints1[:min_len]
        ints2 = ints2[:min_len]

        # 计算汉明距离（每32位整数的不同位数）
        total_bits = min_len * 32
        diff_bits = 0
        for a, b in zip(ints1, ints2):
            xor = a ^ b
            diff_bits += bin(xor).count("1")

        # 相似度 = 1 - 汉明距离/总位数
        similarity = 1.0 - (diff_bits / total_bits)
        return max(0.0, min(1.0, similarity))

    except Exception as e:
        logger.warning(f"指纹比对失败: {e}")
        return 0.0


# ===================== librosa chroma 降级方案 =====================
def extract_fingerprint_chroma(audio_path: str) -> Optional[Tuple[float, np.ndarray]]:
    """
    使用 librosa chroma 特征作为降级方案

    Args:
        audio_path: 音频文件路径

    Returns:
        (duration, chroma_mean) 或 None
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr

        # 提取 chroma 特征
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)  # 12维均值

        return (duration, chroma_mean)

    except Exception as e:
        logger.warning(f"chroma 提取失败: {audio_path}, {e}")
        return None


def compare_fingerprints_chroma(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """比对两个 chroma 特征的余弦相似度"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(fp1.reshape(1, -1), fp2.reshape(1, -1))[0][0]
        return float(sim)
    except Exception as e:
        logger.warning(f"chroma 比对失败: {e}")
        return 0.0


# ===================== 主流程 =====================
def extract_all_fingerprints(
    manifest_df: pd.DataFrame,
    method: str = "auto",
    limit: Optional[int] = None,
    incremental: bool = False,
    fingerprint_db_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    批量提取音频指纹

    Args:
        manifest_df: 音频清单
        method: 提取方法（auto/chromaprint/chroma）
        limit: 处理数量限制
        incremental: 增量更新（跳过已提取的）
        fingerprint_db_path: 指纹数据库路径（增量更新用）

    Returns:
        (fingerprint_df, method_used)
    """
    # 自动检测方法
    if method == "auto":
        if check_fpcalc_available():
            method = "chromaprint"
            logger.info("自动检测：使用 Chromaprint (fpcalc)")
        else:
            method = "chroma"
            logger.info("自动检测：fpcalc 未安装，降级到 librosa chroma")

    # 加载已有指纹库（增量更新）
    existing_fps = {}
    if incremental and fingerprint_db_path and fingerprint_db_path.exists():
        existing_df = pd.read_csv(fingerprint_db_path)
        for _, row in existing_df.iterrows():
            existing_fps[row["audio_id"]] = row
        logger.info(f"增量更新：加载已有指纹 {len(existing_fps)} 条")

    # 限制处理数量
    if limit:
        manifest_df = manifest_df.head(limit)

    # 批量提取
    fingerprints = []
    skipped = 0
    failed = 0

    for idx, row in manifest_df.iterrows():
        audio_id = row.get("audio_id", f"unknown_{idx}")

        # 增量更新：跳过已提取的
        if incremental and audio_id in existing_fps:
            fingerprints.append(existing_fps[audio_id].to_dict())
            skipped += 1
            continue

        audio_path = get_audio_path_from_row(row)
        if not audio_path or not Path(audio_path).exists():
            logger.warning(f"音频文件不存在: {audio_id}")
            failed += 1
            continue

        # 提取指纹
        if method == "chromaprint":
            result = extract_fingerprint_chromaprint(audio_path)
            if result:
                duration, fp = result
                fingerprints.append({
                    "audio_id": audio_id,
                    "duration": duration,
                    "fingerprint": fp,
                    "method": "chromaprint",
                })
            else:
                failed += 1
        else:  # chroma
            result = extract_fingerprint_chroma(audio_path)
            if result:
                duration, fp = result
                fingerprints.append({
                    "audio_id": audio_id,
                    "duration": duration,
                    "fingerprint": ",".join(map(str, fp)),
                    "method": "chroma",
                })
            else:
                failed += 1

        # 进度日志
        if (len(fingerprints) % 50) == 0:
            logger.info(f"已提取 {len(fingerprints)} 首指纹（跳过 {skipped}，失败 {failed}）")

    logger.info(f"指纹提取完成：成功 {len(fingerprints)}，跳过 {skipped}，失败 {failed}")

    return pd.DataFrame(fingerprints), method


def find_duplicates(
    fingerprint_df: pd.DataFrame,
    method: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Dict]:
    """
    比对指纹，找出重复对

    Args:
        fingerprint_df: 指纹数据框
        method: 指纹方法（chromaprint/chroma）
        threshold: 相似度阈值

    Returns:
        重复对列表 [{audio_id_a, audio_id_b, similarity, duration_a, duration_b}]
    """
    duplicates = []
    n = len(fingerprint_df)

    if n < 2:
        return duplicates

    logger.info(f"开始比对 {n} 首指纹（阈值 {threshold}）...")

    # 预解析指纹
    fp_cache = {}
    for _, row in fingerprint_df.iterrows():
        audio_id = row["audio_id"]
        if method == "chromaprint":
            fp_cache[audio_id] = row["fingerprint"]
        else:  # chroma
            fp_cache[audio_id] = np.array([float(x) for x in row["fingerprint"].split(",")])

    # 两两比对（O(n^2)，大数据集需要LSH优化）
    comparisons = 0
    for i in range(n):
        row_i = fingerprint_df.iloc[i]
        id_i = row_i["audio_id"]
        fp_i = fp_cache[id_i]

        for j in range(i + 1, n):
            row_j = fingerprint_df.iloc[j]
            id_j = row_j["audio_id"]
            fp_j = fp_cache[id_j]

            # 比对
            if method == "chromaprint":
                similarity = compare_fingerprints_chromaprint(fp_i, fp_j)
            else:
                similarity = compare_fingerprints_chroma(fp_i, fp_j)

            comparisons += 1

            if similarity >= threshold:
                duplicates.append({
                    "audio_id_a": id_i,
                    "audio_id_b": id_j,
                    "similarity": round(similarity, 4),
                    "duration_a": row_i.get("duration", 0),
                    "duration_b": row_j.get("duration", 0),
                })
                logger.info(f"  发现重复: {id_i} ↔ {id_j} (相似度 {similarity:.4f})")

        # 进度日志
        if (i % 50) == 0 and i > 0:
            logger.info(f"已比对 {comparisons} 对，发现 {len(duplicates)} 组重复")

    logger.info(f"指纹比对完成：共 {comparisons} 对，发现 {len(duplicates)} 组重复")

    return duplicates


def generate_dedup_mapping(
    duplicates: List[Dict],
    manifest_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    生成去重映射表（保留音质最高的版本）

    Args:
        duplicates: 重复对列表
        manifest_df: 音频清单

    Returns:
        去重映射表 [audio_id, duplicate_of, action, reason]
    """
    # 构建重复组（并查集）
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for dup in duplicates:
        union(dup["audio_id_a"], dup["audio_id_b"])

    # 按组聚合
    groups = defaultdict(list)
    for audio_id in parent:
        groups[find(audio_id)].append(audio_id)

    # 生成映射表
    mapping_rows = []
    for group_id, members in groups.items():
        if len(members) < 2:
            continue

        # 选择保留的版本（音质最高：采样率/位深/文件大小）
        # 简化：选择文件最大的（通常质量最高）
        best_member = None
        best_size = -1

        for member in members:
            row = manifest_df[manifest_df["audio_id"] == member]
            if len(row) > 0:
                # 尝试获取文件大小
                audio_path = get_audio_path_from_row(row.iloc[0])
                if audio_path and Path(audio_path).exists():
                    size = Path(audio_path).stat().st_size
                    if size > best_size:
                        best_size = size
                        best_member = member

        if best_member is None:
            best_member = members[0]  # 兜底

        # 生成映射
        for member in members:
            if member == best_member:
                mapping_rows.append({
                    "audio_id": member,
                    "duplicate_of": "",
                    "action": "keep",
                    "reason": f"组内音质最高（{len(members)}首重复）",
                    "group_id": group_id,
                })
            else:
                mapping_rows.append({
                    "audio_id": member,
                    "duplicate_of": best_member,
                    "action": "reject",
                    "reason": f"与 {best_member} 重复",
                    "group_id": group_id,
                })

    return pd.DataFrame(mapping_rows)


def main():
    parser = argparse.ArgumentParser(
        description="音频指纹去重脚本（Stage 4.2 - 感知指纹层，Chromaprint）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_MANIFEST),
                        help="输入音频清单 CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="输出去重映射表路径")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "chromaprint", "chroma"],
                        help="指纹提取方法（auto自动检测，chromaprint工业标准，chroma降级方案）")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"相似度阈值（默认 {DEFAULT_THRESHOLD}，>阈值视为重复）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理指定数量的音频")
    parser.add_argument("--incremental", action="store_true",
                        help="增量更新（跳过已提取的指纹）")
    parser.add_argument("--save-fingerprint-db", action="store_true",
                        help="保存指纹数据库（用于增量更新）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不输出结果")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("音频指纹去重（Stage 4.2 - 感知指纹层）")
    logger.info(f"输入: {args.input}")
    logger.info(f"方法: {args.method}")
    logger.info(f"阈值: {args.threshold}")
    logger.info("=" * 60)

    # 检查 fpcalc
    fpcalc_available = check_fpcalc_available()
    logger.info(f"fpcalc (Chromaprint): {'✅ 可用' if fpcalc_available else '❌ 未安装'}")

    # 加载清单
    manifest_path = Path(args.input)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"加载音频清单: {len(manifest_df)} 首")

    # 提取指纹
    fingerprint_db_path = DEFAULT_FINGERPRINT_DB if args.incremental else None
    fingerprint_df, method_used = extract_all_fingerprints(
        manifest_df,
        method=args.method,
        limit=args.limit,
        incremental=args.incremental,
        fingerprint_db_path=fingerprint_db_path,
    )

    if len(fingerprint_df) == 0:
        logger.error("没有成功提取任何指纹，退出")
        return

    # 保存指纹数据库
    if args.save_fingerprint_db and not args.dry_run:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fingerprint_df.to_csv(DEFAULT_FINGERPRINT_DB, index=False)
        logger.info(f"指纹数据库已保存: {DEFAULT_FINGERPRINT_DB}")

    # 比对指纹
    duplicates = find_duplicates(fingerprint_df, method=method_used, threshold=args.threshold)

    # 生成去重映射表
    if duplicates:
        mapping_df = generate_dedup_mapping(duplicates, manifest_df)
        logger.info(f"生成去重映射表: {len(mapping_df)} 条（保留 {len(mapping_df[mapping_df['action']=='keep'])}，剔除 {len(mapping_df[mapping_df['action']=='reject'])}）")
    else:
        mapping_df = pd.DataFrame(columns=["audio_id", "duplicate_of", "action", "reason", "group_id"])
        logger.info("未发现重复音频")

    # 输出结果
    if not args.dry_run:
        if args.output:
            output_path = Path(args.output)
        else:
            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = DEFAULT_OUTPUT_DIR / f"fingerprint_dedup_mapping_{time_str}.csv"

        mapping_df.to_csv(output_path, index=False)
        logger.info(f"去重映射表已保存: {output_path}")

        # 保存重复对详情
        if duplicates:
            dup_details_path = DEFAULT_OUTPUT_DIR / f"fingerprint_duplicates_detail_{time_str}.csv"
            pd.DataFrame(duplicates).to_csv(dup_details_path, index=False)
            logger.info(f"重复对详情已保存: {dup_details_path}")

    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("去重完成")
    logger.info(f"  方法: {method_used}")
    logger.info(f"  处理音频: {len(fingerprint_df)} 首")
    logger.info(f"  发现重复组: {len(duplicates)} 对")
    logger.info(f"  阈值: {args.threshold}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
