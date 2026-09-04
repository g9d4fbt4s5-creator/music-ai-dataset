# 一次性脚本归档（one_off_scripts）

这些脚本已被正式流水线/配置取代，或在架构转向（DeepSeek 撤销、KNN 退役、多标签分层）后被架空。
分两批归档：**T1**（L4 多标签改造一次性救火脚本）、**T3**（终审计的死入口/时代残留/架空脚本）。
归档保留以备追溯，**禁止在生产流程中再次调用**。

## 正式入口（取代下列全部脚本）

- **`scripts/02_preannotation/genre_annotation_pipeline.py`** — Genre 多标签分层标注正式流水线
  - `python genre_annotation_pipeline.py` 跑完整流程
  - `--verify` 校验产物 schema/规则
  - `--check-repro` 重跑到临时目录并与现产物逐字段比对（复现性测试）
- 人工裁决与历史操作已固化为配置（改数据不改代码）：
  - `data/02_preannotation/user_rulings.json` — 用户人工裁决（6 首）
  - `data/02_preannotation/layered_conflict_resolutions.json` — 12 首文本LLM vs Qwen 分层裁决
  - `data/02_preannotation/qwen_reannotation_manifest.json` — Qwen 重标/分段清单（10 首）

## 归档清单与取代关系

### root_scripts/（原 scripts/ 根目录）

| 归档脚本 | 一次性用途 | 被什么取代 |
|---------|-----------|-----------|
| `convert_to_multi_label.py` | 单选 genre 转多标签列表 | `genre_annotation_pipeline.py::build_labels_for_sample` |
| `fix_multi_label_three_issues.py` | 写入用户裁决/删 primary_genre/修 locked | pipeline + `user_rulings.json` |
| `fix_source_url.py` | 给 58 首文本 LLM 标注补 source_url | 已落盘到 `genre_text_llm_annotations.json`，无需重跑 |

### l3_structural/（原 scripts/02_preannotation/l3_structural/）

**T1 多标签改造一次性脚本：**

| 归档脚本 | 一次性用途 | 被什么取代 |
|---------|-----------|-----------|
| `qwen_supplement_30.py` | 最初 30 首 Qwen 补标（硬编码 ID） | 补标结果已落盘；重标逻辑统一走 `l3_qwen_audio_structure.py` |
| `qwen_reannotate_9.py` | 9 首 deepseek_v4_flash 残留重标 | 结果落盘 + 登记进 `qwen_reannotation_manifest.json` |
| `qwen_segmented_14.py` | 17 首长曲分段标注 | 统一走 `l3_qwen_audio_structure.py::annotate_long_audio` |
| `qwen_segmented_retry_4.py` | 4 首长曲分段重试 | 同上 |

**T3 终审计归档（DeepSeek 文本标签路 09-03 已撤销 / 被正式库收敛）：**

| 归档脚本 | 归档原因 | 取代/去向 |
|---------|---------|----------|
| `l3_deepseek_real.py` | DeepSeek 文本标签路已撤（5 首 4 错，用户裁定），当前产物 source 无 deepseek | 保留供溯源；genre 现走文本LLM(P0)+Qwen(P1) |
| `l3_deepseek_label_extraction.py` | 同上，另一 DeepSeek 变体；无 API key 时会静默产模拟假数据 | 同上；正式标签走 `genre_annotation_pipeline.py` |
| `prepare_qwen_omni.py` | 早期 Qwen 准备脚本 | 功能已并入 `l3_qwen_audio_structure.py` |

### 02_preannotation/（原 scripts/02_preannotation/ 根）

**T3 终审计归档（架构转向被架空的死入口 + LabelStudio 时代一次性脚本）：**

| 归档脚本 | 归档原因（B=归档候选） | 取代/去向 |
|---------|----------------------|----------|
| `run_preannotation_pipeline.py` | **死入口**：内部调已撤 deepseek + clap + knn_propagate，整链属 KNN/DeepSeek 时代 | `genre_annotation_pipeline.py`（多标签分层，无 KNN） |
| `run_end_to_end_preannotation.py` | **死入口**：自带 knn_propagate + l2_mock_golden_seed，输出已 ignore 的 l4_propagated | 同上 |
| `run_preann_infer.py` | 早期推理入口，被上面的 run_preannotation_pipeline 取代（后者本身也已归档） | — |
| `stat_unmapped.py` | LabelStudio 时代一次性统计 | — |
| `custom_audio_preprocess.py` | 早期自有音频预处理，已被 TagMapper 收敛 | TagMapper |
| `custom_audio_mapped.py` | docstring 自述"收敛到 TagMapper" | TagMapper |
| `merge_mapping.py` | docstring 自述"人工确认后合并映射"，LabelStudio 时代 | — |
| `tag_mapping_musiccaps.py` | docstring 自述"收敛到 TagMapper" | TagMapper |

### 03_labelstudio/（原 scripts/03_labelstudio/）

| 归档脚本 | 级别 | 取代链 |
|---------|------|--------|
| `ls_import_converter.py`（v1） | **C 深度归档** | v1 → v4（已完整取代 v1）→ **未来 v5**（multi-label 新产物导入器，待写） |
| `ls_import_converter_v4.py` | B | 绑定已撤的 `deepseek+knn` 融合结构（fusion_strategy/model_version 写死），无法导入 multi-label 新产物 → v4 退役，待 v5 |

> 注：multi-label 产物进 Label Studio 的 v5 导入器尚未编写，是已知待办，不在本次归档范围。

### utils/（原 scripts/utils/）

| 归档脚本 | 归档原因 |
|---------|---------|
| `download_bilibili.py` | bilibili_urls.txt 一次性外源采集 |
| `check_pilot_gaps.py` | 50 首试点（ADR-003）专用 |
| `generate_pilot_checklist.py` | 同上，50 首试点清单 |
| `yamnet_ablation_study.py` | YAMNet 2x2 归因诊断实验（一次性） |
| `fill_artist_from_filename.py` | 早期文件名解析填充 artist_id 的数据修复 |

## 保留在正式目录的相关文件

- `l3_qwen_audio_structure.py` — Qwen-Omni 调用核心库（prepare/segment/merge/convert_to_l4_format），**正式保留**
- `test_qwen_supplement_regression.py` — 回归测试（29 用例），**正式保留**

### 实验特征链：保留原地、标注"暂停"（T3 用户拍板，不 git mv）

下列脚本属 L1/L2 特征提取与嵌入可视化，KNN 虽退役，但特征是历史资产，**保留在原位置**，仅在文件头标注"实验链暂停"：

- `scripts/02_preannotation/l1_physical/l1_physical_features.py`
- `scripts/02_preannotation/l2_embedding/extract_mert_embedding.py`
- `scripts/02_preannotation/l2_semantic/l2_clap_zero_shot.py`
- `scripts/05_visualization/visualize_mert_clustering.py`
- `scripts/05_analysis/visualize_embeddings.py`
