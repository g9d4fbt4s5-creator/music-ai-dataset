# Snapshots 目录说明

## 用途

本目录存放从 AutoDL GPU 通过 rsync 拉回的快照，包含 GPU 端的推理输出、预处理结果等。

## 目录结构

```
snapshots/
├── gpu_backup_YYYYMMDD_HHMMSS/    # 单个快照（按时间命名）
│   ├── csv/                          # CSV 格式数据
│   ├── features/                     # 特征提取结果（按 track 分子目录）
│   │   ├── all_features.csv          # 汇总特征
│   │   ├── track_XXXXXXX/
│   │   │   ├── beats.csv             # 节拍时间点
│   │   │   └── f0.csv                # 基频序列
│   │   └── ...
│   └── logs/                         # GPU 端运行日志
├── snapshot_retention.toml          # 快照轮转配置
└── README.md                        # 本文件
```

## 快照轮转规则

配置文件：`snapshot_retention.toml`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_snapshots` | 5 | 本地保留最新 5 个快照 |
| `warning_threshold` | 80% | 磁盘使用率超过 80% 警告 |
| `critical_threshold` | 90% | 磁盘使用率超过 90% 严重告警 |
| `require_oss_verified` | true | 清理前必须确认 `.oss_verified` 标记 |

## 管理工具

### 查看磁盘状态和快照列表

```bash
python scripts/utils/disk_guard.py --check-only
```

### 执行快照轮转（清理旧快照）

```bash
# 预览模式（不实际删除）
python scripts/utils/disk_guard.py --dry-run

# 强制执行轮转
python scripts/utils/disk_guard.py --enforce-retention
```

### 从 GPU 拉回新快照

```bash
bash scripts/utils/backup_gpu_mirror.sh
```

## 重要说明

1. **快照不提交 git**：`snapshots/gpu_backup_*/` 已被 `.gitignore` 忽略
2. **清理前必须确认 OSS 备份**：`disk_guard.py` 清理前会检查 `.oss_verified` 标记
3. **快照是只读的**：拉回后不再修改，用于追溯和灾难恢复
4. **磁盘空间管理**：定期运行 `disk_guard.py --check-only` 监控磁盘使用率

## 灾难恢复

如果本地数据丢失，可以从 OSS 恢复：

```bash
python scripts/utils/disaster_recovery.py --full-restore
```

使用 `OSS_RECOVERY` 只读密钥从 OSS 下载备份，恢复后与 `raw_audio_checksums.csv` 比对校验。
