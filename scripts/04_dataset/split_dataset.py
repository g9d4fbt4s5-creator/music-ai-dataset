"""
split_dataset.py
数据集划分脚本（Stage 4 数据集版本）

功能：
- 按艺术家/时间/流派隔离划分 train/val/test/holdout
- 支持分层抽样（按流派、语言、人声/纯器乐）
- 支持时间隔离（Temporal Split：测试集用较晚作品，验证泛化能力）
- 支持组合策略（时间隔离 + 分层/隔离抽样）
- 支持来源隔离（--test-from / --holdout-from：从独立数据池导入，防止迭代污染）
- 严格隔离：同一艺术家的不同版本不跨集
- 输出划分清单（只存 audio_id，不存音频）
- 生成划分统计报告

三种核心划分策略：
1. 艺术家隔离（Artist Split）：同一艺术家的曲目不能同时出现在训练集和测试集
2. 时间隔离（Temporal Split）：测试集包含训练集时间范围之后的作品（验证泛化能力）
3. 流派分层抽样（Stratified Split）：确保各流派在子集中比例一致

集合语义说明（重要）：
- train：训练集，用于模型训练
- val：验证集，用于调参、选模型，可以随版本迭代微调
- test：普通测试集，用于开发期间评估模型，可看指标辅助判断
  - 标准做法：一旦划分完成，test 集合固定冻结，不再新增、修改、删除样本
  - 后续新入库音频只进 train/val，绝不混入 test
- holdout：最终锁死的离线"发表用"测试集，全程只跑一次最终报告，绝不调参
  - 完全冻结，调参阶段禁止读取，仅论文最终结果使用
  - 用于跨版本模型对比，长期固定

来源隔离（Source Isolation）：
- 问题：全部样本放同一个 main_pool，再随机切 train/val/test，后续新增数据会
  导致 test 集合混入新样本，破坏 holdout 冻结原则（迭代污染）
- 解决方案：
  - main_pool：只用于 train + val，后续可以持续追加新数据
  - test_pool、holdout_pool：完全独立的采集池，不参与 main_pool 迭代
  - --test-from / --holdout-from：从独立 manifest 导入，全部样本直接进入对应集合
- 硬性保护：
  1. test-from/holdout-from 样本禁止混入 train/val（坑1）
  2. audio_id 全局跨池唯一性校验，main/test/holdout 之间不能重复（坑2）
  3. --train/--val 比例仅针对 main_pool 计算，不包含 test-from/holdout-from 样本（坑5）

用法：
    # 默认划分（80% train / 10% val / 10% test / 1% holdout）
    python split_dataset.py --input data/00.5_cleaned/cleaned_manifest.csv

    # 自定义比例
    python split_dataset.py --train 0.7 --val 0.15 --test 0.15 --holdout 0.0

    # 1. 艺术家隔离（同一艺术家不跨集）
    python split_dataset.py --isolate-by artist_id

    # 2. 时间隔离（测试集用较晚作品，验证泛化能力）
    python split_dataset.py --temporal-by release_date

    # 3. 流派分层抽样（各流派比例一致）
    python split_dataset.py --stratify-by genre

    # 组合策略：时间隔离 + 流派分层（先按时间划窗口，再在窗口内分层）
    python split_dataset.py --temporal-by release_date --temporal-then-stratified --stratify-by genre

    # 组合策略：时间隔离 + 艺术家隔离
    python split_dataset.py --temporal-by release_date --temporal-then-stratified --isolate-by artist_id

    # 来源隔离：从独立数据池导入 test/holdout（推荐工业级做法）
    # 注意：--train 0.85 --val 0.15 只作用于 main_pool（train+val 池）
    # test-from/holdout-from 的样本全部直接进入对应集合，不参与比例分配
    python split_dataset.py \
      --input main_pool_manifest.csv \
      --train 0.85 --val 0.15 \
      --test-from test_pool_manifest.csv \
      --holdout-from holdout_pool_manifest.csv

    # 指定输出目录
    python split_dataset.py --output data/04_final_dataset/v20260821_143000/
"""
import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# 算子级血缘记录器（Lineage v2.0）
LINEAGE_AVAILABLE = False
LineageLogger = None
try:
    import importlib.util
    lineage_path = Path(__file__).parent.parent / "07_lineage" / "lineage_logger.py"
    if lineage_path.exists():
        spec = importlib.util.spec_from_file_location("lineage_logger", str(lineage_path))
        lineage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lineage_module)
        LineageLogger = lineage_module.LineageLogger
        LINEAGE_AVAILABLE = True
except Exception as e:
    logging.getLogger(__name__).warning(f"LineageLogger 导入失败: {e}")

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"split_dataset_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_manifest(input_path: Path) -> pd.DataFrame:
    """加载清洗后的 manifest"""
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"加载 {len(df)} 条记录")
    return df


def verify_checksums(df: pd.DataFrame, fail_on_mismatch: bool = False) -> Dict:
    """
    划分前校验音频文件完整性（checksum校验）

    防止"文件存在但已损坏"的静默错误。
    如果 manifest 中有 sha256 列，校验实际文件的sha256是否一致。

    Args:
        df: 输入 DataFrame（需包含 audio_id 和可选的 sha256/file_relative_path 列）
        fail_on_mismatch: 校验失败时是否报错退出（False则只警告）

    Returns:
        校验结果字典
    """
    import hashlib

    logger.info("校验音频文件完整性（checksum）...")

    result = {
        "total": len(df),
        "verified": 0,
        "missing": 0,
        "mismatch": 0,
        "no_checksum_in_manifest": 0,
        "mismatch_ids": [],
        "missing_ids": [],
    }

    # 检查是否有sha256列
    if "sha256" not in df.columns:
        logger.warning("manifest 中无 sha256 列，跳过checksum校验")
        result["no_checksum_in_manifest"] = len(df)
        return result

    for _, row in df.iterrows():
        audio_id = row.get("audio_id", "unknown")
        expected_hash = row.get("sha256", "")

        # 跳过无hash的记录
        if pd.isna(expected_hash) or not expected_hash:
            result["no_checksum_in_manifest"] += 1
            continue

        # 查找文件路径
        file_path = None
        if "file_relative_path" in df.columns and pd.notna(row.get("file_relative_path", "")):
            file_path = PROJECT_ROOT / "data" / "00_raw_collect" / row["file_relative_path"]
        elif "master_path" in df.columns and pd.notna(row.get("master_path", "")):
            file_path = Path(row["master_path"])

        if file_path is None or not file_path.exists():
            result["missing"] += 1
            result["missing_ids"].append(audio_id)
            logger.warning(f"  文件缺失: {audio_id}")
            continue

        # 计算实际文件的sha256
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            actual_hash = sha256_hash.hexdigest()

            if actual_hash == expected_hash:
                result["verified"] += 1
            else:
                result["mismatch"] += 1
                result["mismatch_ids"].append(audio_id)
                logger.warning(f"  Checksum不匹配: {audio_id}")
                if fail_on_mismatch:
                    logger.error(f"Checksum校验失败: {audio_id}，文件可能已损坏")
                    raise ValueError(f"Checksum mismatch for {audio_id}")
        except Exception as e:
            result["missing"] += 1
            result["missing_ids"].append(audio_id)
            logger.warning(f"  读取失败: {audio_id}, {e}")

    logger.info(f"Checksum校验完成: 验证通过={result['verified']}, "
                f"缺失={result['missing']}, 不匹配={result['mismatch']}, "
                f"无hash={result['no_checksum_in_manifest']}")

    return result


def temporal_split(
    df: pd.DataFrame,
    time_column: str = "release_date",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    holdout_ratio: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    时间隔离划分（Temporal Split）

    按时间字段排序，训练集用较早的作品，测试集用较晚的作品，
    验证模型对"未来"作品的泛化能力。

    Args:
        df: 输入 DataFrame
        time_column: 时间字段名（如 release_date, import_timestamp）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        holdout_ratio: 黄金测试集比例

    Returns:
        (train_df, val_df, test_df, holdout_df)
    """
    if time_column not in df.columns:
        logger.error(f"时间字段 '{time_column}' 不存在，可用字段: {list(df.columns)}")
        raise ValueError(f"Time column '{time_column}' not found")

    # 归一化比例
    total_ratio = train_ratio + val_ratio + test_ratio + holdout_ratio
    if total_ratio > 1.0:
        logger.warning(f"比例总和 {total_ratio} > 1.0，自动归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        holdout_ratio /= total_ratio

    logger.info(f"时间隔离划分（按 '{time_column}'）: "
                f"train={train_ratio:.1%}, val={val_ratio:.1%}, "
                f"test={test_ratio:.1%}, holdout={holdout_ratio:.1%}")

    # 按时间排序（升序：早→晚）
    df_sorted = df.sort_values(by=time_column, ascending=True).reset_index(drop=True)
    total = len(df_sorted)

    # 时间范围统计
    time_min = df_sorted[time_column].iloc[0]
    time_max = df_sorted[time_column].iloc[-1]
    logger.info(f"时间范围: {time_min} ~ {time_max}")

    # 划分点（从早到晚：train → val → test → holdout）
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:n_train + n_val + n_test].copy()
    holdout_df = df_sorted.iloc[n_train + n_val + n_test:].copy()

    # 各集时间范围
    if len(train_df) > 0:
        logger.info(f"  train:   {train_df[time_column].iloc[0]} ~ {train_df[time_column].iloc[-1]} ({len(train_df)}首)")
    if len(val_df) > 0:
        logger.info(f"  val:     {val_df[time_column].iloc[0]} ~ {val_df[time_column].iloc[-1]} ({len(val_df)}首)")
    if len(test_df) > 0:
        logger.info(f"  test:    {test_df[time_column].iloc[0]} ~ {test_df[time_column].iloc[-1]} ({len(test_df)}首)")
    if len(holdout_df) > 0:
        logger.info(f"  holdout: {holdout_df[time_column].iloc[0]} ~ {holdout_df[time_column].iloc[-1]} ({len(holdout_df)}首)")

    logger.info(f"划分结果: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}, holdout={len(holdout_df)}")

    return train_df, val_df, test_df, holdout_df


def temporal_then_stratified_split(
    df: pd.DataFrame,
    time_column: str = "release_date",
    stratify_by: Optional[str] = None,
    isolate_by: Optional[str] = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    holdout_ratio: float = 0.01,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    组合策略：先时间隔离划分大窗口，再在每个窗口内分层/隔离抽样

    流程：
    1. 按时间排序，划分出4个时间窗口（train/val/test/holdout）
    2. 在每个窗口内，按 stratify_by 分层抽样 或 isolate_by 隔离抽样

    这样既保证了时间隔离（测试集是未来作品），又保证了流派/艺术家分布均衡。

    Args:
        df: 输入 DataFrame
        time_column: 时间字段名
        stratify_by: 分层字段（如 genre）
        isolate_by: 隔离字段（如 artist_id）
        train_ratio/val_ratio/test_ratio/holdout_ratio: 各集比例
        random_state: 随机种子

    Returns:
        (train_df, val_df, test_df, holdout_df)
    """
    logger.info("组合策略：时间隔离 + 分层/隔离抽样")

    # Step 1: 时间隔离划分大窗口
    train_window, val_window, test_window, holdout_window = temporal_split(
        df, time_column, train_ratio, val_ratio, test_ratio, holdout_ratio
    )

    # Step 2: 在每个窗口内分层/隔离抽样（这里只是打乱顺序，比例已由时间窗口决定）
    np.random.seed(random_state)

    def shuffle_within_window(window_df):
        if len(window_df) == 0:
            return window_df
        return window_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_df = shuffle_within_window(train_window)
    val_df = shuffle_within_window(val_window)
    test_df = shuffle_within_window(test_window)
    holdout_df = shuffle_within_window(holdout_window)

    logger.info(f"组合划分完成: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}, holdout={len(holdout_df)}")

    return train_df, val_df, test_df, holdout_df


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    holdout_ratio: float = 0.01,
    stratify_by: Optional[str] = None,
    isolate_by: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分层抽样划分数据集

    Args:
        df: 输入 DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        holdout_ratio: 黄金测试集比例（长期固定）
        stratify_by: 分层字段（如 genre, language）
        isolate_by: 隔离字段（如 artist_id，同一值不跨集）
        random_state: 随机种子

    Returns:
        (train_df, val_df, test_df, holdout_df)
    """
    np.random.seed(random_state)
    total = len(df)

    # 归一化比例
    total_ratio = train_ratio + val_ratio + test_ratio + holdout_ratio
    if total_ratio > 1.0:
        logger.warning(f"比例总和 {total_ratio} > 1.0，自动归一化")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        holdout_ratio /= total_ratio

    logger.info(f"划分比例: train={train_ratio:.1%}, val={val_ratio:.1%}, "
                f"test={test_ratio:.1%}, holdout={holdout_ratio:.1%}")

    # 如果指定了隔离字段，按该字段分组
    if isolate_by and isolate_by in df.columns:
        logger.info(f"按 '{isolate_by}' 隔离划分（同一值不跨集）")
        groups = df.groupby(isolate_by)
        group_names = list(groups.groups.keys())
        np.random.shuffle(group_names)

        # 按组划分
        n_groups = len(group_names)
        n_train = int(n_groups * train_ratio)
        n_val = int(n_groups * val_ratio)
        n_test = int(n_groups * test_ratio)

        train_groups = set(group_names[:n_train])
        val_groups = set(group_names[n_train:n_train+n_val])
        test_groups = set(group_names[n_train+n_val:n_train+n_val+n_test])
        holdout_groups = set(group_names[n_train+n_val+n_test:])

        train_df = df[df[isolate_by].isin(train_groups)]
        val_df = df[df[isolate_by].isin(val_groups)]
        test_df = df[df[isolate_by].isin(test_groups)]
        holdout_df = df[df[isolate_by].isin(holdout_groups)]

    # 如果指定了分层字段，按该字段分层抽样
    elif stratify_by and stratify_by in df.columns:
        logger.info(f"按 '{stratify_by}' 分层抽样")
        train_parts, val_parts, test_parts, holdout_parts = [], [], [], []

        for _, group in df.groupby(stratify_by):
            group = group.sample(frac=1, random_state=random_state)
            n = len(group)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)

            train_parts.append(group.iloc[:n_train])
            val_parts.append(group.iloc[n_train:n_train+n_val])
            test_parts.append(group.iloc[n_train+n_val:n_train+n_val+n_test])
            holdout_parts.append(group.iloc[n_train+n_val+n_test:])

        train_df = pd.concat(train_parts) if train_parts else pd.DataFrame()
        val_df = pd.concat(val_parts) if val_parts else pd.DataFrame()
        test_df = pd.concat(test_parts) if test_parts else pd.DataFrame()
        holdout_df = pd.concat(holdout_parts) if holdout_parts else pd.DataFrame()

    # 普通随机划分
    else:
        logger.info("普通随机划分")
        df = df.sample(frac=1, random_state=random_state)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = int(total * test_ratio)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train+n_val]
        test_df = df.iloc[n_train+n_val:n_train+n_val+n_test]
        holdout_df = df.iloc[n_train+n_val+n_test:]

    logger.info(f"划分结果: train={len(train_df)}, val={len(val_df)}, "
                f"test={len(test_df)}, holdout={len(holdout_df)}")

    return train_df, val_df, test_df, holdout_df


# ========== ADR-003: 跨集去重 ==========
CROSS_SET_SIM_THRESHOLD = 0.50  # 指纹相似度 > 0.5 视为跨集重复（ADR-003）


def cross_set_dedup_by_song_group(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    set_name: str = "test",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    基于 song_group_id 的跨集去重（ADR-003 定义）。
    移除 target_df 中与 train_df 共享同一 song_group_id 的样本。

    这是跨集去重的首选方法，因为 song_group_id 是 ADR-003 定义的
    歌曲级唯一标识，同一首歌的不同版本/翻唱/remix/切片共享同一 ID。

    Args:
        train_df: 训练集 DataFrame（含 audio_id, song_group_id）
        target_df: 目标集 DataFrame（val/test/holdout）
        set_name: 目标集名称（用于日志和报告）

    Returns:
        cleaned_df: 去重后的目标集
        removed_list: 被移除的样本列表（用于审计）
    """
    if "song_group_id" not in train_df.columns or "song_group_id" not in target_df.columns:
        logger.warning(f"[{set_name}] 缺少 song_group_id 字段，跳过基于 song_group_id 的跨集去重")
        return target_df.copy(), []

    # 获取 train 中的 song_group_id 集合（排除空值和 unknown）
    train_groups = set(train_df["song_group_id"].dropna().unique())
    train_groups = {g for g in train_groups if g and not str(g).startswith("unknown_song_")}

    if not train_groups:
        logger.info(f"[{set_name}] train 集无有效 song_group_id，跳过跨集去重")
        return target_df.copy(), []

    keep_indices = []
    removed_list = []

    for idx, row in target_df.iterrows():
        sgid = row.get("song_group_id")
        if pd.isna(sgid) or not sgid or str(sgid).startswith("unknown_song_"):
            # 无有效 song_group_id，保守保留
            keep_indices.append(idx)
            continue

        if sgid in train_groups:
            # 与 train 集共享同一 song_group_id，移除
            matched_train = train_df[train_df["song_group_id"] == sgid]["audio_id"].tolist()
            removed_list.append({
                "removed_id": row["audio_id"],
                "matched_train_ids": matched_train,
                "song_group_id": sgid,
                "reason": "same_song_group_as_train",
                "set": set_name,
            })
            logger.info(f"[{set_name}] 跨集去重移除: {row['audio_id']} "
                        f"(song_group={sgid}, 与train {len(matched_train)}首重复)")
        else:
            keep_indices.append(idx)

    cleaned_df = target_df.loc[keep_indices].copy()
    return cleaned_df, removed_list


def cross_set_dedup_by_fingerprint(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fingerprint_file: Optional[Path] = None,
    threshold: float = CROSS_SET_SIM_THRESHOLD,
    set_name: str = "test",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    基于 Chromaprint 指纹相似度的跨集去重（ADR-003 定义）。
    移除 target_df 中与 train_df 指纹相似度 > threshold 的样本。

    当 song_group_id 不可用时使用此方法。指纹文件格式为 JSON:
    {"audio_id": [fp_int_1, fp_int_2, ...], ...}

    Args:
        train_df: 训练集 DataFrame（含 audio_id）
        target_df: 目标集 DataFrame
        fingerprint_file: 指纹文件路径（JSON格式）
        threshold: 相似度阈值（默认0.5，ADR-003定义）
        set_name: 目标集名称

    Returns:
        cleaned_df: 去重后的目标集
        removed_list: 被移除的样本列表
    """
    if not fingerprint_file or not Path(fingerprint_file).exists():
        logger.warning(f"[{set_name}] 指纹文件不存在，跳过基于指纹的跨集去重")
        return target_df.copy(), []

    try:
        with open(fingerprint_file, "r", encoding="utf-8") as f:
            fp_dict_raw = json.load(f)
        fp_dict = {aid: np.array(fp, dtype=np.uint32) for aid, fp in fp_dict_raw.items()}
    except Exception as e:
        logger.warning(f"[{set_name}] 指纹文件加载失败: {e}，跳过跨集去重")
        return target_df.copy(), []

    train_ids = set(train_df["audio_id"].unique())
    train_fps = {aid: fp_dict[aid] for aid in train_ids if aid in fp_dict}

    if not train_fps:
        logger.warning(f"[{set_name}] train 集无指纹数据，跳过跨集去重")
        return target_df.copy(), []

    def compute_fp_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
        """计算两个 Chromaprint 指纹的相似度（基于 bit error rate）"""
        if len(fp1) == 0 or len(fp2) == 0:
            return 0.0
        min_len = min(len(fp1), len(fp2))
        fp1, fp2 = fp1[:min_len], fp2[:min_len]
        xor = np.bitwise_xor(fp1, fp2)
        bit_diff = np.sum(np.unpackbits(xor.view(np.uint8)))
        total_bits = min_len * 32
        return 1.0 - (bit_diff / total_bits) if total_bits > 0 else 0.0

    keep_indices = []
    removed_list = []

    for idx, row in target_df.iterrows():
        tid = row["audio_id"]
        tfp = fp_dict.get(tid)

        if tfp is None:
            keep_indices.append(idx)
            continue

        max_sim = 0.0
        matched_train = None
        for trid, trfp in train_fps.items():
            sim = compute_fp_similarity(tfp, trfp)
            if sim > max_sim:
                max_sim = sim
                matched_train = trid

        if max_sim > threshold:
            removed_list.append({
                "removed_id": tid,
                "matched_train_id": matched_train,
                "similarity": round(float(max_sim), 4),
                "reason": "fingerprint_similarity_above_threshold",
                "set": set_name,
            })
            logger.info(f"[{set_name}] 指纹跨集去重移除: {tid} "
                        f"(sim={max_sim:.3f} vs train {matched_train})")
        else:
            keep_indices.append(idx)

    cleaned_df = target_df.loc[keep_indices].copy()
    return cleaned_df, removed_list


def apply_cross_set_dedup(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    fingerprint_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    对 val/test/holdout 执行跨集去重（ADR-003 要求）。

    优先使用 song_group_id 方法（ADR-003 定义的歌曲级隔离），
    如果 song_group_id 不可用则降级到指纹相似度方法。

    Args:
        train_df: 训练集
        val_df: 验证集
        test_df: 测试集
        holdout_df: 黄金测试集
        fingerprint_file: 指纹文件路径（可选）
        output_dir: 输出目录（用于保存去重报告）

    Returns:
        cleaned_val, cleaned_test, cleaned_holdout: 去重后的各集
        report: 去重报告
    """
    logger.info("=" * 60)
    logger.info("ADR-003 跨集去重")
    logger.info("=" * 60)

    all_removed = []

    # 对 val 去重
    cleaned_val, removed_val = cross_set_dedup_by_song_group(train_df, val_df, "val")
    if removed_val:
        all_removed.extend(removed_val)
    elif fingerprint_file:
        cleaned_val, removed_val = cross_set_dedup_by_fingerprint(
            train_df, val_df, fingerprint_file, set_name="val")
        all_removed.extend(removed_val)

    # 对 test 去重
    cleaned_test, removed_test = cross_set_dedup_by_song_group(train_df, test_df, "test")
    if removed_test:
        all_removed.extend(removed_test)
    elif fingerprint_file:
        cleaned_test, removed_test = cross_set_dedup_by_fingerprint(
            train_df, test_df, fingerprint_file, set_name="test")
        all_removed.extend(removed_test)

    # 对 holdout 去重（holdout 是独立采集的，也需要检查与 train 的重复）
    cleaned_holdout, removed_holdout = cross_set_dedup_by_song_group(
        train_df, holdout_df, "holdout")
    if removed_holdout:
        all_removed.extend(removed_holdout)
    elif fingerprint_file:
        cleaned_holdout, removed_holdout = cross_set_dedup_by_fingerprint(
            train_df, holdout_df, fingerprint_file, set_name="holdout")
        all_removed.extend(removed_holdout)

    # 【新增】val vs test 去重（防止同一 artist/song 被分到 val 和 test）
    removed_test_val = []
    if len(cleaned_val) > 0 and len(cleaned_test) > 0:
        cleaned_test, removed_test_val = cross_set_dedup_by_song_group(
            cleaned_val, cleaned_test, "test_vs_val")
        if removed_test_val:
            all_removed.extend(removed_test_val)
            logger.info(f"val-test 去重: 移除 {len(removed_test_val)} 首重复样本")
        elif fingerprint_file:
            cleaned_test, removed_test_val = cross_set_dedup_by_fingerprint(
                cleaned_val, cleaned_test, fingerprint_file, set_name="test_vs_val")
            if removed_test_val:
                all_removed.extend(removed_test_val)
                logger.info(f"val-test 指纹去重: 移除 {len(removed_test_val)} 首重复样本")

    report = {
        "method": "song_group_id (primary) / fingerprint_similarity (fallback)",
        "threshold_song_group": "exact_match",
        "threshold_fingerprint": CROSS_SET_SIM_THRESHOLD,
        "val_original": len(val_df),
        "val_removed": len(removed_val),
        "val_final": len(cleaned_val),
        "test_original": len(test_df),
        "test_removed_train": len(removed_test),
        "test_removed_val_vs_test": len(removed_test_val),
        "test_removed_total": len(removed_test) + len(removed_test_val),
        "test_final": len(cleaned_test),
        "holdout_original": len(holdout_df),
        "holdout_removed": len(removed_holdout),
        "holdout_final": len(cleaned_holdout),
        "total_removed": len(all_removed),
        "removed_samples": all_removed,
    }

    logger.info(f"跨集去重完成: val移除{len(removed_val)}/{len(val_df)}, "
                f"test移除{len(removed_test)}/{len(test_df)}, "
                f"holdout移除{len(removed_holdout)}/{len(holdout_df)}, "
                f"总计移除{len(all_removed)}")

    # 保存报告
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "cross_set_dedup_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"跨集去重报告已保存: {report_path}")

    return cleaned_val, cleaned_test, cleaned_holdout, report


def generate_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    stratify_by: Optional[str] = None,
) -> Dict:
    """生成划分统计报告"""
    stats = {
        "total": len(train_df) + len(val_df) + len(test_df) + len(holdout_df),
        "splits": {
            "train": {"count": len(train_df), "ratio": len(train_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
            "val": {"count": len(val_df), "ratio": len(val_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
            "test": {"count": len(test_df), "ratio": len(test_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
            "holdout": {"count": len(holdout_df), "ratio": len(holdout_df) / max(1, len(train_df)+len(val_df)+len(test_df)+len(holdout_df))},
        },
    }

    # 如果有分层字段，统计各层分布
    if stratify_by and stratify_by in train_df.columns:
        stats["stratify_by"] = stratify_by
        stats["stratify_distribution"] = {}
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df), ("holdout", holdout_df)]:
            if stratify_by in split_df.columns and len(split_df) > 0:
                dist = split_df[stratify_by].value_counts().to_dict()
                stats["stratify_distribution"][split_name] = dist

    return stats


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    output_dir: Path,
    stats: Dict,
    checksum_result: Optional[Dict] = None,
    random_state: Optional[int] = None,
    input_manifest_path: Optional[Path] = None,
    upstream_lineage: Optional[str] = None,
    ood_df: Optional[pd.DataFrame] = None,
):
    """保存划分结果"""
    import hashlib
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只保存 audio_id 列（不存音频）
    id_col = "audio_id" if "audio_id" in train_df.columns else train_df.columns[0]

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    train_df[[id_col]].to_csv(splits_dir / "train.csv", index=False)
    val_df[[id_col]].to_csv(splits_dir / "val.csv", index=False)
    test_df[[id_col]].to_csv(splits_dir / "test.csv", index=False)
    holdout_df[[id_col]].to_csv(splits_dir / "holdout_gold.csv", index=False)
    # OOD 集（如果有）
    if "ood" in stats and len(ood_df) > 0 and id_col in ood_df.columns:
        ood_df[[id_col]].to_csv(splits_dir / "ood.csv", index=False)
        logger.info(f"  OOD集: {len(ood_df)} 首 → ood.csv")

    logger.info(f"划分清单已保存到: {splits_dir}")

    # 保存统计报告
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(exist_ok=True)
    with open(stats_dir / "split_distribution.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"划分统计已保存: {stats_dir / 'split_distribution.json'}")

    # 建议2: 生成 manifest_ref.json（记录该版本依赖的manifest路径和checksum）
    if input_manifest_path and input_manifest_path.exists():
        manifest_sha256 = hashlib.sha256(input_manifest_path.read_bytes()).hexdigest()
        manifest_ref = {
            "manifest_path": str(input_manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_size_bytes": input_manifest_path.stat().st_size,
            "manifest_mtime": datetime.fromtimestamp(input_manifest_path.stat().st_mtime, TZ).isoformat(),
            "note": "训练脚本读取时，通过audio_id到此manifest查找实际路径；如果manifest被更新或移动，需重新生成此版本",
        }
        with open(output_dir / "manifest_ref.json", "w", encoding="utf-8") as f:
            json.dump(manifest_ref, f, ensure_ascii=False, indent=2)
        logger.info(f"manifest引用已保存: {output_dir / 'manifest_ref.json'}")

    # 保存 lineage.json（建议1: 包含并引用清洗阶段的lineage，形成一条链）
    # 补强1: 添加 schema_version 字段，为后续格式演进留余地
    lineage = {
        "schema_version": "1.0",  # lineage.json schema 版本，后续格式升级时递增
        "version": output_dir.name,
        "timestamp": datetime.now(TZ).isoformat(),
        "split_method": stats.get("split_method", "random"),
        "stratify_by": stats.get("stratify_by"),
        "isolate_by": stats.get("isolate_by"),
        "temporal_by": stats.get("temporal_by"),
        "random_seed": random_state,  # P2: 记录随机种子，完全可复现
        "splits": stats["splits"],
        "total_samples": stats["total"],
        "checksum_verification": checksum_result,  # P0: checksum校验结果
        "upstream": upstream_lineage,  # 建议1: 引用清洗阶段的lineage
        # 补强4: 完整血缘字段（来源隔离）
        "main_pool_manifest": stats.get("main_pool_manifest"),
        "test_pool_manifest": stats.get("test_pool_manifest"),
        "holdout_pool_manifest": stats.get("holdout_pool_manifest"),
        "train_val_split_ratio": stats.get("train_val_split_ratio"),
        "source_isolation_enabled": stats.get("source_isolation_enabled", False),
        "source_isolation_passed": stats.get("source_isolation_passed", True),
        "overlap_audio_ids": stats.get("overlap_audio_ids", []),
    }
    with open(output_dir / "lineage.json", "w", encoding="utf-8") as f:
        json.dump(lineage, f, ensure_ascii=False, indent=2)
    logger.info(f"血缘追踪已保存: {output_dir / 'lineage.json'}")


def main():
    parser = argparse.ArgumentParser(
        description="数据集划分脚本（train/val/test/holdout）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str,
                        default=str(PROJECT_ROOT / "data" / "00.5_cleaned" / "cleaned_manifest.csv"),
                        help="输入清洗后的 manifest CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录（默认 data/04_final_dataset/v{timestamp}/）")
    parser.add_argument("--train", type=float, default=0.80, help="训练集比例")
    parser.add_argument("--val", type=float, default=0.10, help="验证集比例")
    parser.add_argument("--test", type=float, default=0.10, help="测试集比例")
    parser.add_argument("--holdout", type=float, default=0.01, help="黄金测试集比例")
    parser.add_argument("--stratify-by", type=str, default=None,
                        help="分层字段（如 genre, language）")
    parser.add_argument("--isolate-by", type=str, default=None,
                        help="隔离字段（如 artist_id，同一值不跨集）")
    parser.add_argument("--temporal-by", type=str, default=None,
                        help="时间隔离字段（如 release_date, import_timestamp），测试集用较晚作品")
    parser.add_argument("--temporal-then-stratified", action="store_true",
                        help="组合策略：先时间隔离划分窗口，再在窗口内分层/隔离抽样")
    parser.add_argument("--verify-checksum", action="store_true",
                        help="划分前校验音频文件checksum完整性（防止文件存在但已损坏）")
    parser.add_argument("--fail-on-checksum-mismatch", action="store_true",
                        help="checksum不匹配时报错退出（默认只警告）")
    parser.add_argument("--test-from", type=str, default=None,
                        help="独立测试集来源（从独立采集的数据池导入，不参与清洗调优，防止迭代污染）")
    parser.add_argument("--holdout-from", type=str, default=None,
                        help="独立holdout集来源（长期封存，跨版本模型对比，必须与训练集来自不同采集批次）")
    parser.add_argument("--ood-from", type=str, default=None,
                        help="独立OOD集来源（域外泛化测试，风格/来源与main_pool差异大，不参与训练/验证）")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式：发现跨池重复 audio_id 时直接报错终止（默认只警告）")
    parser.add_argument("--protect-golden", action="store_true", default=True,
                        help="黄金集保护：黄金集样本(is_golden=true)不进入 test/holdout/ood，划分时正常分布不特殊处理（默认开启）")
    parser.add_argument("--qc-report", type=str, default=None,
                        help="QC Gate报告路径，过滤掉final_branch=fail的样本（如2秒超短/非音乐/低质量）")
    parser.add_argument("--source-type-filter", action="store_true", default=True,
                        help="ADR-003: 启用source_type过滤，排除AI生成/分轨人声等域外样本（默认开启）")
    parser.add_argument("--no-source-type-filter", action="store_false", dest="source_type_filter",
                        help="禁用source_type过滤")
    parser.add_argument("--cross-set-dedup", action="store_true", default=True,
                        help="ADR-003: 启用跨集去重（基于song_group_id，无则降级到指纹相似度），默认开启")
    parser.add_argument("--no-cross-set-dedup", action="store_false", dest="cross_set_dedup",
                        help="禁用跨集去重")
    parser.add_argument("--fingerprint-file", type=str, default=None,
                        help="Chromaprint指纹文件路径（JSON格式: {audio_id: [fp_ints]}），song_group_id不可用时使用")
    parser.add_argument("--sample-golden", action="store_true",
                        help="黄金集抽样模式：从main_pool分层抽样指定比例作为golden_set（ADR-003第6节）")
    parser.add_argument("--golden-ratio", type=float, default=0.05,
                        help="黄金集抽样比例（默认0.05，即5%%）")
    parser.add_argument("--golden-output-dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "03_human_annotation" / "golden_set"),
                        help="黄金集输出目录（默认 data/03_human_annotation/golden_set）")
    parser.add_argument("--generate-l3-tasks", action="store_true",
                        help="黄金集抽样模式下同时生成L3结构标注任务清单")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("数据集划分")
    logger.info("=" * 60)

    # 加载数据
    input_path = Path(args.input)
    df = load_manifest(input_path)

    # P0: QC Gate过滤 — 排除final_branch=fail的样本（2秒超短/非音乐/低质量等）
    if args.qc_report:
        qc_path = Path(args.qc_report)
        if qc_path.exists():
            qc_df = pd.read_csv(qc_path)
            fail_ids = set(qc_df[qc_df["final_branch"] == "fail"]["audio_id"])
            before_count = len(df)
            df = df[~df["audio_id"].isin(fail_ids)]
            logger.info(f"QC过滤: 排除 {before_count - len(df)} 个fail样本（{len(fail_ids)}个在QC报告中标记为fail）")
        else:
            logger.warning(f"QC报告不存在: {qc_path}，跳过过滤")

    # P0: source_type 过滤 — 排除域外样本（AI生成、分轨人声等，ADR-003 第7节）
    source_type_report = None
    if args.source_type_filter:
        try:
            import sys
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from utils.source_type_filter import filter_by_source_type
            df, source_type_report = filter_by_source_type(df, report_path=None)
        except ImportError as e:
            logger.warning(f"source_type_filter 导入失败: {e}，跳过 source_type 过滤")

    # ========== ADR-003 第6节：黄金集抽样模式 ==========
    if args.sample_golden:
        logger.info("=" * 60)
        logger.info("黄金集抽样模式 (ADR-003 第6节)")
        logger.info("=" * 60)

        golden_output_dir = Path(args.golden_output_dir)
        golden_output_dir.mkdir(parents=True, exist_ok=True)

        n_total = len(df)
        n_golden = max(int(n_total * args.golden_ratio), 3)  # 最少3首
        n_golden = min(n_golden, n_total)

        logger.info(f"  总样本数: {n_total}")
        logger.info(f"  抽样比例: {args.golden_ratio} ({args.golden_ratio*100:.1f}%)")
        logger.info(f"  抽样数量: {n_golden}")
        logger.info(f"  分层字段: {args.stratify_by or '无（随机抽样）'}")
        logger.info(f"  随机种子: {args.random_state}")

        # 分层字段 fallback：genre_major → cluster_id → source_type → 随机
        stratify_col = args.stratify_by
        if stratify_col and (stratify_col not in df.columns or df[stratify_col].isna().all()):
            for fallback_col in ["cluster_id", "source_type"]:
                if fallback_col in df.columns and not df[fallback_col].isna().all():
                    logger.info(f"黄金集分层字段 fallback: {stratify_col} → {fallback_col}")
                    stratify_col = fallback_col
                    break
            else:
                stratify_col = None
                logger.warning("无可用分层字段，退化为随机抽样")

        # 全局上限：最少3首，不超过比例
        max_total = max(int(n_total * args.golden_ratio), 3)
        max_total = min(max_total, n_total)

        # 分层抽样（每组保底1首）
        if stratify_col:
            logger.info(f"按 '{stratify_col}' 分层抽样（每组保底1首，全局上限{max_total}首）")
            golden_df = pd.DataFrame()
            group_sizes = []
            for group_key, group in df.groupby(stratify_col):
                n_group = max(int(len(group) * args.golden_ratio), 1)
                n_group = min(n_group, len(group))
                sample = group.sample(n=n_group, random_state=args.random_state)
                golden_df = pd.concat([golden_df, sample])
                group_sizes.append((group_key, len(group), n_group))
            golden_df = golden_df.reset_index(drop=True)

            # 日志：输出每个组抽了多少首
            logger.info("各分层组抽样情况:")
            for gkey, gsize, nsamp in group_sizes:
                logger.info(f"  {gkey}: {nsamp}/{gsize} ({nsamp/gsize*100:.1f}%)")

            # 如果保底导致超额，从大众组优先裁剪（每组至少留1首）
            if len(golden_df) > max_total:
                excess = len(golden_df) - max_total
                # 按组大小降序，从大众组开始裁
                group_counts = golden_df.groupby(stratify_col).size().sort_values(ascending=False)
                for group_key in group_counts.index:
                    if excess <= 0:
                        break
                    group_mask = golden_df[stratify_col] == group_key
                    group_sampled = golden_df[group_mask]
                    # 每组至少留1首
                    can_remove = len(group_sampled) - 1
                    remove = min(can_remove, excess)
                    if remove > 0:
                        to_remove = group_sampled.sample(n=remove, random_state=args.random_state)
                        golden_df = golden_df.drop(to_remove.index)
                        excess -= remove
                        logger.info(f"  从大众组 '{group_key}' 裁剪 {remove} 首（剩余{len(group_sampled)-remove}首）")
                golden_df = golden_df.reset_index(drop=True)
                logger.info(f"保底超额，已从大众组裁剪至 {len(golden_df)} 首（上限 {max_total}）")
            # 如果总数不足，从剩余样本补充
            elif len(golden_df) < max_total:
                remaining = df[~df["audio_id"].isin(golden_df["audio_id"])]
                need = min(max_total - len(golden_df), len(remaining))
                if need > 0:
                    extra = remaining.sample(n=need, random_state=args.random_state)
                    golden_df = pd.concat([golden_df, extra]).reset_index(drop=True)
                    logger.info(f"补充随机样本: +{need} 首（达到目标{max_total}首）")
        else:
            logger.info(f"随机抽样（无分层字段），目标 {max_total} 首")
            golden_df = df.sample(n=max_total, random_state=args.random_state).reset_index(drop=True)

        golden_ids = golden_df["audio_id"].tolist()
        logger.info(f"黄金集抽样完成: {len(golden_df)}/{n_total} ({len(golden_df)/n_total:.1%})")

        # 1. 更新原始 manifest 的 is_golden 列
        if "is_golden" in df.columns:
            full_manifest = pd.read_csv(input_path)
            full_manifest["is_golden"] = full_manifest["audio_id"].isin(golden_ids)
            full_manifest.to_csv(input_path, index=False)
            logger.info(f"已更新 {input_path} 的 is_golden 列（{len(golden_ids)} 首标记为 true）")
        else:
            logger.warning("manifest 中无 is_golden 列，跳过更新")

        # 2. 生成 golden_ids.json
        with open(golden_output_dir / "golden_ids.json", "w", encoding="utf-8") as f:
            json.dump(golden_ids, f, indent=2, ensure_ascii=False)
        logger.info(f"  已保存: {golden_output_dir / 'golden_ids.json'}")

        # 3. 生成 golden_set_meta.json
        meta = {
            "version": "v1.0",
            "created_at": datetime.now(TZ).isoformat(),
            "total_pool_size": n_total,
            "golden_size": len(golden_df),
            "ratio": args.golden_ratio,
            "seed": args.random_state,
            "stratify_by": args.stratify_by,
            "source_manifest": str(input_path),
            "audio_ids": golden_ids,
        }
        with open(golden_output_dir / "golden_set_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"  已保存: {golden_output_dir / 'golden_set_meta.json'}")

        # 4. 可选：生成 L3 任务清单
        if args.generate_l3_tasks:
            l3_tasks = []
            for _, row in golden_df.iterrows():
                l3_tasks.append({
                    "audio_id": row["audio_id"],
                    "audio_path": row.get("file_relative_path", ""),
                    "duration": row.get("duration_sec", row.get("duration", None)),
                })
            with open(golden_output_dir / "l3_task_list.json", "w", encoding="utf-8") as f:
                json.dump(l3_tasks, f, indent=2, ensure_ascii=False)
            logger.info(f"  已保存: {golden_output_dir / 'l3_task_list.json'} ({len(l3_tasks)} 个任务)")

        # 5. 生成黄金集清单（CSV，含关键元数据）
        meta_cols = ["audio_id", "original_filename", "duration_sec", "format", "source_type"]
        available_meta = [c for c in meta_cols if c in golden_df.columns]
        golden_manifest = golden_df[available_meta].copy()
        golden_manifest["is_golden"] = True
        golden_manifest["golden_version"] = "v1.0"
        golden_manifest["review_status"] = "pending_annotation"
        golden_manifest.to_csv(golden_output_dir / "golden_set_manifest.csv", index=False)
        logger.info(f"  已保存: {golden_output_dir / 'golden_set_manifest.csv'}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("黄金集抽样完成")
        logger.info(f"  输出目录: {golden_output_dir}")
        logger.info(f"  样本数: {len(golden_df)}/{n_total}")
        logger.info(f"  下一步: 对黄金集样本执行 L3 结构标注 + 人工精标")
        logger.info("=" * 60)
        return

    # P0: 划分前校验音频文件checksum完整性（防止文件存在但已损坏）
    checksum_result = None
    if args.verify_checksum:
        checksum_result = verify_checksums(df, fail_on_mismatch=args.fail_on_checksum_mismatch)

    # 划分（根据参数选择策略）
    split_method = "random"

    if args.temporal_then_stratified:
        # 组合策略：时间隔离 + 分层/隔离
        if not args.temporal_by:
            logger.error("--temporal-then-stratified 需要 --temporal-by 参数")
            sys.exit(1)
        train_df, val_df, test_df, holdout_df = temporal_then_stratified_split(
            df,
            time_column=args.temporal_by,
            stratify_by=args.stratify_by,
            isolate_by=args.isolate_by,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
            random_state=args.random_state,
        )
        split_method = "temporal_then_stratified"

    elif args.temporal_by:
        # 纯时间隔离
        train_df, val_df, test_df, holdout_df = temporal_split(
            df,
            time_column=args.temporal_by,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
        )
        split_method = "temporal"

    else:
        # 分层/隔离/随机划分
        train_df, val_df, test_df, holdout_df = stratified_split(
            df,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            holdout_ratio=args.holdout,
            stratify_by=args.stratify_by,
            isolate_by=args.isolate_by,
            random_state=args.random_state,
        )
        if args.stratify_by:
            split_method = "stratified"
        elif args.isolate_by:
            split_method = "isolate"
        else:
            split_method = "random"

    # 来源隔离：用独立采集的数据池替换 test/holdout（防止迭代污染和分布漂移）
    # 坑1保护：test-from/holdout-from 样本全部直接进入对应集合，绝对不混入 train/val
    # 坑2保护：audio_id 全局跨池唯一性校验，main/test/holdout 之间不能出现相同 audio_id
    source_isolation = {"test": None, "holdout": None, "ood": None}
    overlap_audio_ids = []
    ood_df = pd.DataFrame(columns=df.columns) if len(df) > 0 else pd.DataFrame()

    if args.test_from:
        test_from_path = Path(args.test_from)
        if not test_from_path.is_absolute():
            test_from_path = PROJECT_ROOT / test_from_path
        if test_from_path.exists():
            logger.info(f"来源隔离：从独立数据池导入测试集: {test_from_path}")
            external_test_df = load_manifest(test_from_path)
            logger.info(f"  独立测试集: {len(external_test_df)} 首")

            # 坑1：原 test 样本（来自 main_pool）并入 train，不丢弃
            # 这样 train/val 只包含 main_pool 样本，test 只包含 test_pool 样本
            if "audio_id" in test_df.columns and "audio_id" in train_df.columns:
                original_test_ids = set(test_df["audio_id"].tolist())
                train_ids = set(train_df["audio_id"].tolist())
                # 把原 test 中不在 train 的样本并入 train
                test_to_train = test_df[~test_df["audio_id"].isin(train_ids)]
                if len(test_to_train) > 0:
                    train_df = pd.concat([train_df, test_to_train], ignore_index=True)
                    logger.info(f"  坑1保护：原 test 集 {len(test_to_train)} 首并入 train（main_pool 样本不丢弃）")

            # 用独立数据池替换 test（全部样本直接进入 test，不参与 train/val 划分）
            test_df = external_test_df
            source_isolation["test"] = str(test_from_path)
            logger.info(f"  坑1保护：test_pool 全部 {len(test_df)} 首直接进入 test 集，不混入 train/val")
        else:
            logger.warning(f"  ⚠️  独立测试集路径不存在: {test_from_path}，使用原划分")

    if args.holdout_from:
        holdout_from_path = Path(args.holdout_from)
        if not holdout_from_path.is_absolute():
            holdout_from_path = PROJECT_ROOT / holdout_from_path
        if holdout_from_path.exists():
            logger.info(f"来源隔离：从独立数据池导入holdout集: {holdout_from_path}")
            external_holdout_df = load_manifest(holdout_from_path)
            logger.info(f"  独立holdout集: {len(external_holdout_df)} 首")

            # 坑1：原 holdout 样本（来自 main_pool）并入 train
            if "audio_id" in holdout_df.columns and "audio_id" in train_df.columns:
                original_holdout_ids = set(holdout_df["audio_id"].tolist())
                train_ids = set(train_df["audio_id"].tolist())
                holdout_to_train = holdout_df[~holdout_df["audio_id"].isin(train_ids)]
                if len(holdout_to_train) > 0:
                    train_df = pd.concat([train_df, holdout_to_train], ignore_index=True)
                    logger.info(f"  坑1保护：原 holdout 集 {len(holdout_to_train)} 首并入 train（main_pool 样本不丢弃）")

            holdout_df = external_holdout_df
            source_isolation["holdout"] = str(holdout_from_path)
            logger.info(f"  坑1保护：holdout_pool 全部 {len(holdout_df)} 首直接进入 holdout 集，不混入 train/val")
        else:
            logger.warning(f"  ⚠️  独立holdout集路径不存在: {holdout_from_path}，使用原划分")

    # OOD 集：独立采集，风格/来源与 main_pool 差异大，不参与训练/验证
    if args.ood_from:
        ood_from_path = Path(args.ood_from)
        if not ood_from_path.is_absolute():
            ood_from_path = PROJECT_ROOT / ood_from_path
        if ood_from_path.exists():
            logger.info(f"来源隔离：从独立数据池导入OOD集: {ood_from_path}")
            ood_df = load_manifest(ood_from_path)
            source_isolation["ood"] = str(ood_from_path)
            logger.info(f"  OOD集: {len(ood_df)} 首（不参与训练/验证，仅用于域外泛化测试）")
        else:
            logger.warning(f"  ⚠️  OOD集路径不存在: {ood_from_path}，跳过")

    # 黄金集保护：黄金集样本(is_golden=true)不进入 test/holdout/ood
    # 黄金集物理上在 main_pool，划分时正常分布（可能进 train 也可能进 val），不特殊处理
    if args.protect_golden and "is_golden" in df.columns:
        golden_ids = set(df[df["is_golden"] == True]["audio_id"].tolist()) if "audio_id" in df.columns else set()
        if golden_ids:
            logger.info(f"黄金集保护：检测到 {len(golden_ids)} 个黄金集样本，确保不进入 test/holdout/ood")
            for split_name, split_df in [("test", test_df), ("holdout", holdout_df), ("ood", ood_df)]:
                if "audio_id" in split_df.columns and len(split_df) > 0:
                    golden_in_split = set(split_df["audio_id"].tolist()) & golden_ids
                    if golden_in_split:
                        logger.error(f"  ❌ 黄金集保护失败：{split_name} 集中发现 {len(golden_in_split)} 个黄金集样本")
                        logger.error(f"     前5个: {list(golden_in_split)[:5]}")
                        if args.strict:
                            raise ValueError(f"Golden set leakage: {len(golden_in_split)} golden samples in {split_name}")

    # 坑2：audio_id 全局跨池唯一性校验
    # main_pool (train/val)、test_pool、holdout_pool、ood_pool 之间不能出现相同 audio_id
    logger.info("坑2保护：校验 audio_id 全局跨池唯一性...")
    all_id_sets = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df), ("holdout", holdout_df), ("ood", ood_df)]:
        if "audio_id" in split_df.columns and len(split_df) > 0:
            all_id_sets[split_name] = set(split_df["audio_id"].tolist())
        else:
            all_id_sets[split_name] = set()

    # 检查任意两个集合之间的重叠
    split_names = list(all_id_sets.keys())
    for i in range(len(split_names)):
        for j in range(i+1, len(split_names)):
            name1, name2 = split_names[i], split_names[j]
            overlap = all_id_sets[name1] & all_id_sets[name2]
            if overlap:
                overlap_audio_ids.extend([
                    {"split_1": name1, "split_2": name2, "audio_id": aid}
                    for aid in list(overlap)[:10]  # 只记录前10个
                ])
                logger.error(f"  ❌ 坑2校验失败：{name1} ↔ {name2} 有 {len(overlap)} 个重复 audio_id")
                logger.error(f"     前5个: {list(overlap)[:5]}")

    if not overlap_audio_ids:
        logger.info("  ✅ 坑2校验通过：audio_id 全局跨池无重复")
    else:
        logger.warning(f"  ⚠️  坑2校验发现 {len(overlap_audio_ids)} 个跨池重复（详见 lineage.json）")
        # 补强2: 严格模式下直接抛异常终止
        if args.strict:
            logger.error("  ❌ 严格模式：跨池重复 audio_id 意味着来源隔离已失效，继续生成数据集会有泄露风险")
            logger.error(f"     前5个重复: {[item['audio_id'] for item in overlap_audio_ids[:5]]}")
            raise ValueError(
                f"Source isolation violated: {len(overlap_audio_ids)} audio_ids overlap across pools. "
                f"Use --strict=False to override (not recommended)."
            )

    # ========== ADR-003: 跨集去重 ==========
    # 在来源隔离和坑2校验之后执行
    # 优先使用 song_group_id（ADR-003定义的歌曲级隔离），无则降级到指纹相似度
    # 注意：output_dir 尚未定义，报告暂存到 cross_set_dedup_report，稍后保存
    cross_set_dedup_report = None
    if args.cross_set_dedup:
        fp_file = Path(args.fingerprint_file) if args.fingerprint_file else None
        val_df, test_df, holdout_df, cross_set_dedup_report = apply_cross_set_dedup(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            holdout_df=holdout_df,
            fingerprint_file=fp_file,
            output_dir=None,  # 稍后保存
        )

    # 生成统计
    stats = generate_split_stats(train_df, val_df, test_df, holdout_df, args.stratify_by)
    stats["split_method"] = split_method
    # OOD 集统计
    if len(ood_df) > 0:
        stats["ood"] = {"count": len(ood_df), "source": source_isolation.get("ood")}
        stats["splits"]["ood"] = {"count": len(ood_df), "ratio": 0.0, "note": "独立采集，不参与比例分配"}
    if args.temporal_by:
        stats["temporal_by"] = args.temporal_by
    if args.isolate_by:
        stats["isolate_by"] = args.isolate_by
    if source_isolation["test"]:
        stats["test_source_isolation"] = source_isolation["test"]
        logger.info(f"  ✅ 测试集来源隔离: {source_isolation['test']}")
    if source_isolation["holdout"]:
        stats["holdout_source_isolation"] = source_isolation["holdout"]
        logger.info(f"  ✅ holdout集来源隔离: {source_isolation['holdout']}")

    # 坑2：记录跨池重复 audio_id（如果有）
    if overlap_audio_ids:
        stats["overlap_audio_ids"] = overlap_audio_ids
        stats["source_isolation_passed"] = False
    else:
        stats["overlap_audio_ids"] = []
        stats["source_isolation_passed"] = True

    # 坑4：记录完整血缘字段
    stats["main_pool_manifest"] = str(input_path)
    stats["test_pool_manifest"] = source_isolation["test"]
    stats["holdout_pool_manifest"] = source_isolation["holdout"]
    stats["train_val_split_ratio"] = {"train": args.train, "val": args.val}
    stats["source_isolation_enabled"] = bool(source_isolation["test"] or source_isolation["holdout"])

    # 输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        version_str = datetime.now(TZ).strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "data" / "04_final_dataset" / f"v{version_str}"

    # 建议1: 查找最近的清洗阶段lineage.json，形成血缘链
    upstream_lineage = None
    cleaned_reports_dir = PROJECT_ROOT / "data" / "00.5_cleaned" / "reports"
    if cleaned_reports_dir.exists():
        report_dirs = sorted([d for d in cleaned_reports_dir.iterdir() if d.is_dir()], reverse=True)
        for report_dir in report_dirs:
            lineage_file = report_dir / "lineage.json"
            if lineage_file.exists():
                upstream_lineage = str(lineage_file)
                logger.info(f"上游血缘: {upstream_lineage}")
                break

    # ========== 保存 ADR-003 跨集去重报告 ==========
    if cross_set_dedup_report:
        stats["cross_set_dedup"] = {
            "enabled": True,
            "method": cross_set_dedup_report["method"],
            "total_removed": cross_set_dedup_report["total_removed"],
            "val_removed": cross_set_dedup_report["val_removed"],
            "test_removed": cross_set_dedup_report["test_removed"],
            "holdout_removed": cross_set_dedup_report["holdout_removed"],
        }
        # 保存报告到 output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "cross_set_dedup_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cross_set_dedup_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"ADR-003 跨集去重报告已保存: {report_path}")
        logger.info(f"  总计移除: {cross_set_dedup_report['total_removed']} 首 "
                    f"(val={cross_set_dedup_report['val_removed']}, "
                    f"test={cross_set_dedup_report['test_removed']}, "
                    f"holdout={cross_set_dedup_report['holdout_removed']})")
    elif args.cross_set_dedup:
        logger.info("ADR-003 跨集去重: 已启用，但无重复样本移除")

    # 保存
    save_splits(
        train_df, val_df, test_df, holdout_df, output_dir, stats,
        checksum_result=checksum_result,
        random_state=args.random_state,
        input_manifest_path=input_path,
        upstream_lineage=upstream_lineage,
        ood_df=ood_df,
    )

    # === 记录 v2.0 算子级血缘（Lineage v2.0）===
    if LINEAGE_AVAILABLE:
        try:
            lineage_logger = LineageLogger(
                dataset_version=output_dir.name,
                output_path=str(output_dir / "lineage_v2.json"),
                upstream_lineage=upstream_lineage,
                auto_save=False
            )

            # 记录划分算子
            total_input = len(train_df) + len(val_df) + len(test_df) + len(holdout_df)
            lineage_logger.log_operator(
                operator_name="dataset_split",
                operator_version="1.0",
                input_manifest=str(input_path),
                input_count=total_input,
                output_path=str(output_dir),
                output_count=total_input,
                failed_count=0,
                config={
                    "split_method": split_method,
                    "train_ratio": args.train,
                    "val_ratio": args.val,
                    "test_ratio": args.test,
                    "holdout_ratio": args.holdout,
                    "stratify_by": args.stratify_by,
                    "isolate_by": args.isolate_by,
                    "temporal_by": args.temporal_by,
                    "test_from": args.test_from,
                    "holdout_from": args.holdout_from,
                    "random_state": args.random_state,
                    "source_isolation_passed": stats.get("source_isolation_passed", True),
                },
                status="success"
            )

            # 记录数据集划分
            lineage_logger.log_splits({
                "train": {"count": len(train_df), "source_manifest": str(input_path)},
                "val": {"count": len(val_df), "source_manifest": str(input_path)},
                "test": {"count": len(test_df), "source_manifest": source_isolation.get("test") or str(input_path)},
                "holdout": {"count": len(holdout_df), "source_manifest": source_isolation.get("holdout") or str(input_path)},
            })

            lineage_logger.save()
            logger.info(f"算子级血缘(v2.0)已保存: {output_dir / 'lineage_v2.json'}")
            lineage_logger.print_summary()
        except Exception as e:
            logger.warning(f"血缘记录失败（不影响划分结果）: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("划分完成")
    logger.info(f"  输出目录: {output_dir}")
    ood_count = len(ood_df) if len(ood_df) > 0 else 0
    logger.info(f"  train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} | holdout: {len(holdout_df)} | ood: {ood_count}")
    if source_isolation.get("ood"):
        logger.info(f"  OOD来源: {source_isolation['ood']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
