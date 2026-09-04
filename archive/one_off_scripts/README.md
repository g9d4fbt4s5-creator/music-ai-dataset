# 一次性脚本归档（one_off_scripts）

这些脚本是 L4 多标签改造过程中的**一次性救火/补数脚本**，已被正式流水线和配置文件取代。
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

| 归档脚本 | 一次性用途 | 被什么取代 |
|---------|-----------|-----------|
| `qwen_supplement_30.py` | 最初 30 首 Qwen 补标（硬编码 ID） | 补标结果已落盘；重标逻辑统一走 `l3_qwen_audio_structure.py` |
| `qwen_reannotate_9.py` | 9 首 deepseek_v4_flash 残留重标 | 结果落盘 + 登记进 `qwen_reannotation_manifest.json` |
| `qwen_segmented_14.py` | 17 首长曲分段标注 | 统一走 `l3_qwen_audio_structure.py::annotate_long_audio` |
| `qwen_segmented_retry_4.py` | 4 首长曲分段重试 | 同上 |

## 保留在正式目录的相关文件

- `l3_qwen_audio_structure.py` — Qwen-Omni 调用核心库（prepare/segment/merge），**正式保留**
- `test_qwen_supplement_regression.py` — 回归测试，**正式保留**
