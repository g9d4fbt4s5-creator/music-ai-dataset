# 数据集版本 v20260820_112723

## 基本信息

| 项目 | 内容 |
|------|------|
| 版本号 | v20260820_112723 |
| 创建时间 | 2026-08-20T11:27:23.998581+08:00 |
| 样本总数 | 0 |
| 备注 | mtg-jamendo 20首特征提取(beats+f0+汇总特征) |
| 源快照 | /Users/m.jian/music_corpus_project/snapshots/gpu_backup_20260820_173500 |

## 目录结构

```
v20260820_112723/
├── final_metadata/
│   ├── corpus_full_meta.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── holdout_gold.csv
└── readme.md
```

## 数据集划分

| 子集 | 比例 | 用途 |
|------|------|------|
| train | 70% | 模型训练 |
| val | 10% | 验证调参 |
| test | 10% | 测试评估 |
| holdout_gold | 10% | 黄金留出集（全程不参与训练调参） |

## 注意事项

- 本版本冻结后只读，不再修改
- latest 软链接指向当前生产版本
- 如需回滚到旧版本，修改 latest 软链接即可

---

*由 freeze_version.py 自动生成*
