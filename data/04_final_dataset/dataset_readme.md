# 最终数据集说明

## 数据集概览

- **数据集名称**：音乐语料数据集
- **版本**：v1.0
- **更新日期**：2026-08-07
- **总样本数**：待填充

## 目录结构

```
04_final_dataset/
├── final_audio/              # 音频文件（软链接，不复制大文件）
└── final_metadata/
    ├── corpus_full_meta.csv  # 全部样本完整元数据
    ├── train_split.csv       # 训练集
    ├── val_split.csv         # 验证集
    ├── test_split.csv        # 测试集
    └── holdout_gold.csv      # 黄金留出集
```

## 数据集划分

| 子集 | 比例 | 用途 | 说明 |
|------|------|------|------|
| train | 70% | 模型训练 | 用于模型参数学习 |
| val | 10% | 验证调参 | 用于超参数调优、早停 |
| test | 10% | 测试评估 | 用于最终模型性能评估 |
| holdout_gold | 10% | 黄金留出 | 全程不参与训练调参，仅最终测评 |

> **重要**：holdout_gold 是黄金标准集，从数据最前面切分，全程不参与训练、调参，只做最终测评使用。

## 元数据字段说明

| 字段 | 说明 |
|------|------|
| sample_id | 样本唯一ID |
| audio_path | 音频文件路径 |
| annotations_raw | 完整标注结果（JSON字符串，训练时json.loads解析） |

## 生成方式

```bash
# 从 Label Studio 导出的 jsonl 文件生成
python3 scripts/03_labelstudio/convert_ls_jsonl.py
```

## 随机种子

数据集切分使用固定随机种子 `42`，保证可复现。
