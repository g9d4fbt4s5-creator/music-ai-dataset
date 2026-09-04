# 历史遗留归档（historical_leftovers）

> 归档时间：2026-09-04 ｜ 依据：WorkBuddy T3 终审计 §3.2 + 用户拍板（63 文件全量 + splits/ 内 2 个藏匿 backup 一并归档）
> 方式：**仅 `mv` 移动，未物理删除**，镜像原始相对路径；需要时整目录 `mv` 回原位即可恢复。
> 这些都是**历史过程版/手动备份/已退役实验输出**，正本均在正式目录且被 git 跟踪，归档后不进入任何生产流程。

## 归档清单与正本指向

| 归档位置（本目录下） | 原路径 | 文件数 | 性质/归档原因 | 正式正本（未动） |
|---------------------|--------|-------|--------------|----------------|
| `scripts/03_human_annotation/` | `scripts/03_human_annotation/` | 2 | 08-27 候选池采样/定稿早期脚本（finalize_candidates / sample_candidates），黄金集流程已改道，无他脚本引用 | 当前黄金集走 `genre_annotation_pipeline.py` |
| `scripts/04_dataset/split_dataset.py.backup_before_ad003_fix` | `scripts/04_dataset/` | 1 | ADR-003 修复前的 split_dataset 旧版备份 | `scripts/04_dataset/split_dataset.py`（git 跟踪） |
| `data/00_raw_collect/` | `data/00_raw_collect/` | 2 | audio_manifest 两次操作（challenge 标记前、archive 前）的手动备份 csv | `data/00_raw_collect/audio_manifest.csv`（git 跟踪） |
| `data/04_final_dataset/splits_v2_ad003_fixed` … `splits_v7_all_paths_fixed` | `data/04_final_dataset/` | 6 目录 | 数据集划分 v2→v7 共 6 代迭代过程版 | **无后缀 `data/04_final_dataset/splits/`**（08-27 11:29 最后定型，git 跟踪） |
| `data/04_final_dataset/stress_test/` | `data/04_final_dataset/` | 1 目录 | 划分压力测试残留（代码中的 `challenge_stress_test` 是 sample_type 字段值，与此目录无关） | — |
| `data/04_final_dataset/splits/splits_backup_before_force_golden/` | `data/04_final_dataset/splits/` | 4 | 强制黄金集前的 train/val/test/holdout 备份 | `data/04_final_dataset/splits/splits/{train,val,test,holdout_gold}.csv` |
| `data/04_final_dataset/splits/splits_backup_v1_old/` | `data/04_final_dataset/splits/` | 4 | v1 旧划分备份（同上 4 文件） | 同上 |
| `data/03_human_annotation/golden_set/golden_index_v2.csv` | `data/03_human_annotation/golden_set/` | 1 | 旧黄金索引 v2（08-27） | `golden_manifest_v2.csv`（09-03，git 跟踪） |
| `data/05_auxiliary/style_clustering_hdbscan_85/` | `data/05_auxiliary/` | 2 | 早期 HDBSCAN 风格聚类实验输出（clustering_results / outliers）；聚类/KNN 已退役，style_consistency_clustering 仅把 05_auxiliary 当 `--output-dir` 默认值且无调用方 | — |

合计 **71 个文件 / 约 488K**。

## 两点特别说明

1. **git 跟踪状态不同**：上表绝大多数为**未跟踪**残留（移动后工作区 `??` 消失）；唯独 `splits/` 内的两个 backup（`splits_backup_before_force_golden`、`splits_backup_v1_old`，共 8 文件）原本**已被 git 跟踪**——移动需经一次 commit 才在版本库层面完成归档（commit 前 `git status` 会显示这 8 个为删除，属预期）。
2. **零代码依赖已实测**：归档前全仓 grep 反查，正式脚本无对这些具体路径的真实 import/读取；命中的 `stress_test` 是 `sample_type` 字段值、`05_auxiliary` 是输出目录默认值，均非依赖。

## 与其他归档目录的区别

- `archive/one_off_scripts/`：一次性**脚本**（T1/T3 归档，有正式 pipeline 取代关系对照）
- `archive/l4_knn_legacy/`：退役的 KNN 传播代码与数据
- `archive/historical_leftovers/`（本目录）：**数据/备份/过程版**历史残留，非脚本、非 KNN
