# 算子级血缘模块（Lineage v2.0）

> 记录每一条样本从哪里来，经过哪些算子，产出什么结果。支持溯源、复现、防泄露。

## 一、模块结构

```
scripts/07_lineage/
├── __init__.py              # 模块入口
├── lineage_logger.py        # 血缘记录器（核心）
├── lineage_query.py         # 血缘查询工具
├── lineage_verify.py        # 血缘校验工具
└── lineage_report.py        # 血缘报告生成
```

## 二、核心概念

### 什么是算子级血缘？

不记录每条样本的完整路径，而是记录**每个算子批次的执行日志**：

```
原始音频 → 算子: yamnet_infer (v1.0) → yamnet_output.csv
         → 算子: demucs_separate (v4.0.1) → stems/
         → 算子: llm_tagging (gpt-4o) → tags.json
         → 划分: holdout-test
```

### 三大作用

| 作用 | 说明 |
|------|------|
| **溯源** | 某个坏标签，回查是哪一步算子输出错误 |
| **复现** | 完全复现数据集构建（按算子顺序+配置重跑） |
| **防泄露** | 检查 holdout 样本有没有意外流入 train |

## 三、lineage.json 格式（v2.0）

```json
{
  "lineage_version": "2.0",
  "dataset_version": "v20260824_100000",
  "created_at": "2026-08-24T10:00:00Z",
  "updated_at": "2026-08-24T10:30:00Z",
  "upstream_lineage": "data/00.5_cleaned/reports/v20260824_100000/lineage.json",
  "operators": [
    {
      "operator_name": "yamnet_infer",
      "operator_version": "1.0",
      "model_version": "google/yamnet/1",
      "timestamp": "2026-08-24T10:05:00Z",
      "input_manifest": "data/00_raw_collect/audio_manifest.csv",
      "input_filter": null,
      "input_count": 500,
      "output_path": "data/00.5_cleaned/reports/v20260824_100000/yamnet_output.csv",
      "output_count": 499,
      "failed_count": 1,
      "failed_samples": ["01M0E9X..."],
      "failure_reasons": {"audio_too_short": 1},
      "config": {"threshold": 0.3, "sample_rate": 16000},
      "duration_sec": 327.5,
      "status": "success"
    }
  ],
  "splits": {
    "train": {"count": 399, "source_manifest": "main_pool.csv"},
    "val": {"count": 50, "source_manifest": "main_pool.csv"},
    "test": {"count": 40, "source_manifest": "test_pool.csv"},
    "holdout": {"count": 10, "source_manifest": "holdout_pool.csv"}
  }
}
```

## 四、使用方法

### 4.1 记录算子执行

```python
from scripts.07_lineage.lineage_logger import LineageLogger

# 初始化
logger = LineageLogger(
    dataset_version="v20260824_100000",
    output_path="data/lineage/lineage_v20260824_100000.json"
)

# 方式1：手动记录
logger.log_operator(
    operator_name="yamnet_infer",
    operator_version="1.0",
    model_version="google/yamnet/1",
    input_manifest="data/00_raw_collect/audio_manifest.csv",
    input_count=500,
    output_path="data/00.5_cleaned/reports/v20260824_100000/yamnet_output.csv",
    output_count=499,
    failed_count=1,
    failure_reasons={"audio_too_short": 1},
    config={"threshold": 0.3, "sample_rate": 16000},
    duration_sec=327.5
)

# 方式2：上下文管理器（自动记录耗时和状态）
with logger.operator("demucs_separate", version="4.0.1", model_version="mdx_extra_q") as op:
    op.set_input("data/00.5_cleaned/reports/v20260824_100000/yamnet_output.csv", count=12)
    op.set_filter("has_vocals == True")
    # ... 执行 demucs ...
    op.set_output("data/01_preprocess/demucs_stems/", count=12, failed_count=0)
    op.set_config({"two_stems": False, "device": "cuda"})

# 记录数据集划分
logger.log_splits({
    "train": {"count": 399, "source_manifest": "main_pool.csv"},
    "val": {"count": 50, "source_manifest": "main_pool.csv"},
    "test": {"count": 40, "source_manifest": "test_pool.csv"},
    "holdout": {"count": 10, "source_manifest": "holdout_pool.csv"}
})

# 保存
logger.save()
```

### 4.2 查询血缘

```bash
# 列出所有算子
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --list-operators

# 按名称查询
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --operator yamnet_infer

# 查询失败样本
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --failed-samples

# 查询失败原因统计
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --failure-reasons

# 溯源：查某个样本经过了哪些算子
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --trace-sample 01M0E9X...

# 瓶颈分析
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --bottleneck

# 导出为CSV
python scripts/07_lineage/lineage_query.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --export-csv output.csv
```

### 4.3 校验血缘

```bash
# 基本校验
python scripts/07_lineage/lineage_verify.py \
    --lineage data/lineage/lineage_v20260824_100000.json

# 严格模式（任何警告都报错）
python scripts/07_lineage/lineage_verify.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --strict

# 自定义失败率阈值
python scripts/07_lineage/lineage_verify.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --max-failure-rate 0.05

# 防泄露校验（需要manifest文件）
python scripts/07_lineage/lineage_verify.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --train-manifest train.csv \
    --test-manifest test.csv \
    --holdout-manifest holdout.csv

# 导出校验报告
python scripts/07_lineage/lineage_verify.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --report verification_report.json
```

#### 校验项

| 校验项 | 说明 |
|--------|------|
| **完整性** | 检查算子记录是否完整（名称/版本/时间戳/状态） |
| **一致性** | 检查算子输入输出数量是否合理（有无样本丢失） |
| **失败率** | 检查总失败率是否超过阈值（默认10%） |
| **算子链** | 检查算子时间顺序是否正确 |
| **划分** | 检查数据集划分是否完整（train/val/test/holdout） |
| **来源隔离** | 检查train/test/holdout是否来自不同来源 |
| **防泄露** | 检查划分是否有audio_id重叠（需要manifest文件） |

### 4.4 生成报告

```bash
# Markdown报告
python scripts/07_lineage/lineage_report.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --format markdown

# HTML报告（含图表）
python scripts/07_lineage/lineage_report.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --format html \
    --output report.html

# JSON报告
python scripts/07_lineage/lineage_report.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --format json

# 只生成图表
python scripts/07_lineage/lineage_report.py \
    --lineage data/lineage/lineage_v20260824_100000.json \
    --charts-only \
    --output-dir charts/
```

#### 报告内容

1. **摘要**：算子数量、总输入/输出/失败、失败率、总耗时
2. **算子状态分布**：success/partial/failed/skipped 数量
3. **算子详情**：每个算子的输入/输出/失败/耗时
4. **失败原因统计**：按原因分类的失败数量和占比
5. **数据集划分**：各划分的数量和来源
6. **瓶颈分析**：最耗时算子 Top 3

#### 图表（需安装 plotly）

- 算子耗时柱状图
- 算子失败率柱状图
- 失败原因饼图
- 数据集划分饼图
- 算子状态分布柱状图

## 五、与现有流水线集成

### 在 clean_pipeline.py 中使用

```python
from scripts.07_lineage.lineage_logger import LineageLogger

# 初始化血缘记录器
lineage = LineageLogger(
    dataset_version=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    output_path="data/lineage/lineage.json",
    auto_save=True  # 每次记录算子后自动保存
)

# Stage 1: 元数据清洗
with lineage.operator("metadata_cleaning", version="1.0") as op:
    op.set_input("data/00_raw_collect/audio_manifest.csv", count=500)
    # ... 执行清洗 ...
    op.set_output("data/00.5_cleaned/cleaned_manifest.csv", count=498, failed_count=2)
    op.set_config({"strategies": ["skip", "fill_default", "flag"]})

# Stage 2: 格式标准化
with lineage.operator("format_standardization", version="1.0", model_version="ffmpeg-4.3") as op:
    op.set_input("data/00.5_cleaned/cleaned_manifest.csv", count=498)
    # ... 执行转码 ...
    op.set_output("data/01_preprocess/processed_master/", count=498, failed_count=0)
    op.set_config({"sample_rate": 48000, "bit_depth": 24, "format": "flac"})

# ... 更多阶段 ...

# 记录数据集划分
lineage.log_splits({
    "train": {"count": 398, "source_manifest": "main_pool.csv"},
    "val": {"count": 50, "source_manifest": "main_pool.csv"},
    "test": {"count": 40, "source_manifest": "test_pool.csv"},
    "holdout": {"count": 10, "source_manifest": "holdout_pool.csv"}
})

# 最终保存
lineage.save()
lineage.print_summary()
```

## 六、设计原则

1. **算子级记录**：不记录每条样本，只记录批次（适合500-10000首规模）
2. **增量追加**：多次运行合并到同一个lineage文件（支持断点续跑）
3. **自动记录**：上下文管理器自动记录耗时和状态
4. **失败可追溯**：记录失败样本ID和失败原因
5. **配置可复现**：记录算子配置参数，支持完全复现
6. **防泄露校验**：支持检查train/test/holdout的audio_id重叠

## 七、与轻量血缘的对比

| 特性 | 轻量血缘（v1.0） | 算子级血缘（v2.0） |
|------|-----------------|-------------------|
| 记录粒度 | 阶段级样本数变化 | 算子级执行日志 |
| 算子信息 | ❌ 无 | ✅ 名称/版本/模型/配置 |
| 失败样本 | ❌ 无 | ✅ 样本ID+原因 |
| 耗时记录 | ❌ 无 | ✅ 自动记录 |
| 溯源能力 | ⚠️ 阶段级 | ✅ 算子级 |
| 复现能力 | ⚠️ 部分 | ✅ 完全（配置+顺序） |
| 防泄露 | ✅ 有 | ✅ 有（增强） |
| 复杂度 | 低 | 中 |

## 八、未来扩展（v3.0）

- **样本级血缘**：记录每条样本的完整处理路径（适合>10000首规模）
- **算子嵌套**：支持算子A调用算子B的嵌套关系
- **实时监控**：Web UI实时查看血缘和算子执行状态
- **DAG可视化**：算子依赖关系的有向无环图可视化
