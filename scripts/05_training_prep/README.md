# Stage 5: 训练准备（切片 + 特征提取）

> 本文档说明 Stage 5 的职责、时机和参数。依据：docs/PIPELINE_OVERVIEW_V2.md

## 时机（重要）

- **必须在 Stage 4 数据划分之后执行**
- 只对 `train/val` 的 `audio_id` 切片，不切 `test/holdout/ood`
- 切片时已有结构标签（来自 L3 黄金集传播），可按乐段边界切

## 为什么切片放在 Stage 5 而不是 Stage 1？

| 维度 | Stage 1 切片（错误） | Stage 5 切片（正确） |
|------|---------------------|---------------------|
| 数据范围 | 全部样本（含 fail/test） | 只切 train/val |
| 结构信息 | 无（预标注未做） | 有（L3 结构标签已传播） |
| 切片数量 | 盲目 3038 片 | 与训练集精确匹配 |
| 资源浪费 | 切了不会进训练的样本 | 只切需要的 |

## 脚本

| 脚本 | 用途 |
|------|------|
| `01_audio_chunker.py` | 按结构边界切片（15s默认，短曲不切） |
| `02_extract_features.py` | 提取 mel/chroma/MFCC 128维特征 |

## 关键参数

### 01_audio_chunker.py

```bash
# 正确用法：只切 train/val
python scripts/05_training_prep/01_audio_chunker.py \
  --manifest data/00_raw_collect/audio_manifest.csv \
  --output-dir data/05_training_prep/segments \
  --only-train-val \
  --splits data/04_final_dataset/splits

# 测试用法：切全部（不推荐用于正式训练）
python scripts/05_training_prep/01_audio_chunker.py \
  --manifest data/00_raw_collect/audio_manifest.csv \
  --output-dir data/05_training_prep/segments
```

| 参数 | 说明 |
|------|------|
| `--manifest` | 输入 audio_manifest.csv 路径（必填） |
| `--output-dir` | 切片输出目录 |
| `--chunk-sec` | 切片长度（秒），默认15秒 |
| `--overlap` | 滑动窗口重叠比例（0-1），默认0.5 |
| `--only-train-val` | **只切 train/val 的样本**（推荐） |
| `--splits` | 数据划分目录，默认 data/04_final_dataset/splits |
| `--limit` | 只处理前N个音频（用于测试） |
| `--dry-run` | 预览模式，不实际生成切片文件 |

### 02_extract_features.py

```bash
python scripts/05_training_prep/02_extract_features.py \
  --segments data/05_training_prep/segments \
  --output data/05_training_prep/features/audio_features.csv
```

## 输出目录

```
data/05_training_prep/
├── segments/              # 训练切片（FLAC 16-bit/44.1kHz）
│   ├── chunks_manifest_*.csv
│   └── *.flac
└── features/              # 特征文件
    └── audio_features.csv
```

## 注意事项

1. **切片格式**：训练切片用 FLAC 16-bit/44.1kHz 即可，不需要 24-bit（节省 33% 空间）
2. **测试集不切片**：test/holdout/ood 保持整首音频，用于完整曲目评估
3. **历史切片处理**：之前在 Stage 1 切的 3038 片标记为 `pre_split_chunks`（历史产物），不用于正式训练，确认 Stage 5 逻辑正确后可删除
