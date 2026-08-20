# 项目结构说明

## 目录树

```
music-corpus-project/
├── .git/                          # Git 仓库（自动生成，不要手动修改）
├── .gitignore                     # Git 忽略规则（音频/模型/日志/密钥不提交）
├── .env.example                   # 配置模板（真实密钥在 ~/.config/music-corpus/.env）
├── README.md                      # 项目主文档（架构/流程/快速开始）
├── environment.yml                # 本地 Mac conda 环境（music-corpus-local）
├── environment_gpu.yml            # GPU 服务器 conda 环境（music-corpus-gpu）
│
├── configs/                       # 配置文件
│   ├── label_studio/
│   │   └── labeling_interface.xml # Label Studio 标注界面配置
│   └── schema/                    # 数据规范
│       ├── audio_id_schema.json   # audio_id 规范（ULID + md5 散列）
│       └── corpus_format_spec.yaml # 入库音频格式标准
│
├── scripts/                       # 脚本代码
│   ├── 00_collect/                # 数据采集阶段
│   │   ├── import_audio.py        # 音频入库（格式校验+ULID+散列迁移）
│   │   └── gen_raw_checksum.py    # 生成完整性基线+全局索引
│   │
│   ├── 01_preprocess/             # 预处理阶段（待开发）
│   │   ├── resample.py            # 重采样（待开发）
│   │   ├── loudness_normalize.py  # 响度归一化（待开发）
│   │   ├── segment.py             # 音频切片（待开发）
│   │   └── demucs_separate.py    # Demucs 源分离（待开发）
│   │
│   ├── 02_preannotation/          # 预标注阶段
│   │   ├── run_preann_infer.py    # 推理缓存→预标注转换（无 OSS 逻辑）
│   │   ├── tag_mapping_musiccaps.py # MusicCaps 标签映射
│   │   ├── custom_audio_preprocess.py # 自有音频预标注
│   │   └── stat_unmapped.py       # 未映射标签统计
│   │
│   ├── 03_labelstudio/            # Label Studio 阶段
│   │   ├── convert_mapped_to_labelstudio.py # 映射结果→LS 导入格式
│   │   ├── convert_ls_jsonl.py    # LS 导出→数据集切分
│   │   └── ls_backup_export.py    # 标注数据备份
│   │
│   ├── 04_dataset/                # 数据集版本管理
│   │   └── freeze_version.py      # 版本冻结（vYYYYMMDD_HHMMSS + latest 软链接）
│   │
│   └── utils/                     # 通用工具
│       ├── config_loader.py       # 统一配置加载器（三优先级）
│       ├── get_audio_physical_path.py # 统一散列路径计算
│       ├── pipeline_lock.py       # 并发锁（fcntl.flock）
│       ├── clear_stale_lock.py    # 僵尸锁清理
│       ├── disk_guard.py          # 磁盘管控（水位告警+快照轮转）
│       ├── verify_oss_upload.py   # OSS 上传完整性校验
│       ├── disaster_recovery.py   # 灾难恢复（OSS_RECOVERY 只读密钥）
│       ├── upload_cache_to_oss.py # 推理缓存上传 OSS
│       ├── oss_gpu_backup_client.py # GPU 端 OSS 备份客户端
│       ├── oss_local_client.py    # Mac 端 OSS 备份客户端
│       └── backup_gpu_mirror.sh   # rsync 拉回 GPU 快照
│
├── data/                          # 数据目录（音频不提交 git）
│   ├── 00_raw_collect/            # 原始采集数据
│   │   ├── raw_audio/             # 原始音频（md5 两层散列，不提交 git）
│   │   │   ├── f0/b8/             # 散列目录示例
│   │   │   └── ...
│   │   ├── raw_audio_checksums.csv # 完整性基线（只读，永不修改）
│   │   ├── audio_manifest.csv     # 全局业务索引（可更新 status）
│   │   ├── rejected/              # 拒绝入库的音频（格式不合格）
│   │   └── raw_metadata/          # 原始元数据
│   │
│   ├── 01_preprocess/             # 预处理输出（不提交 git）
│   │   ├── processed_audio/       # 标准化切片 wav
│   │   └── demucs_stems/          # Demucs 分轨
│   │
│   ├── 02_preannotation/          # 预标注数据
│   │   ├── model_output_cache/    # 模型推理输出缓存（不提交 git）
│   │   ├── label_mapping/         # 标签映射
│   │   │   ├── label_mapping_dict.json # 标签映射字典
│   │   │   └── unmapped_tag_report.json # 未映射标签报告
│   │   └── preann_csv/            # 预标注 CSV
│   │
│   ├── 03_human_review/           # 人工标注
│   │   ├── labelstudio_tasks.json # LS 导入任务
│   │   └── backups/               # 标注备份（不提交 git）
│   │
│   └── 04_final_dataset/          # 最终数据集
│       ├── dataset_readme.md      # 数据集说明
│       ├── final_metadata/        # 当前生产版本元数据
│       ├── vYYYYMMDD_HHMMSS/     # 历史版本（只读）
│       └── latest -> vYYYYMMDD_HHMMSS/ # 软链接，指向当前生产版本
│
├── snapshots/                     # GPU 快照（不提交 git）
│   ├── gpu_backup_YYYYMMDD_HHMMSS/ # rsync 拉回的快照
│   └── snapshot_retention.toml    # 快照轮转配置
│
├── notebooks/                     # Jupyter 分析脚本
│
├── docs/                          # 项目文档
│   └── oss_ram_setup.md           # OSS RAM 权限分离配置指引
│
├── logs/                          # 运行日志（不提交 git）
│
└── envs/                          # 环境相关
    └── wheel_cache/               # dev 版本 wheel 包（必须提交，保证复现）
```

---

## 目录职责说明

### 阶段编号规则

脚本和数据目录都使用两位数编号，表示流水线阶段：

| 编号 | 阶段 | 说明 |
|------|------|------|
| 00 | collect | 数据采集、入库、完整性基线 |
| 01 | preprocess | 预处理（重采样、响度归一、切片、源分离） |
| 02 | preannotation | 预标注（模型推理、标签映射） |
| 03 | labelstudio | 人工标注（LS 导入/导出/备份） |
| 04 | dataset | 数据集版本管理（冻结、切分） |

### 关键目录详解

#### `data/00_raw_collect/raw_audio/`

**原始音频存储目录，使用 md5(audio_id)[0:4] 两层散列。**

- 散列规则：`raw_audio/{hash[0:2]}/{hash[2:4]}/{hash_full}_{audio_id}.{ext}`
- 目的：避免单目录文件过多导致性能问题
- **只读**：入库后永不修改、永不删除
- **不提交 git**：音频文件大，.gitignore 已覆盖

#### `data/00_raw_collect/raw_audio_checksums.csv`

**完整性基线文件，只读，永不修改。**

字段：`audio_id, file_relative_path, original_filename, sha256, file_bytes, sample_rate, bit_depth, channels, duration_seconds, imported_at`

用途：
- 灾难恢复后校验文件完整性
- 检测文件损坏
- 新协作者验证数据一致性

#### `data/00_raw_collect/audio_manifest.csv`

**全局业务索引，可更新 status，不删除历史行。**

字段：`audio_id, file_relative_path, original_filename, status, quality_flags, sha256, file_bytes, sample_rate, bit_depth, channels, duration_seconds, imported_at, updated_at`

status 取值：
- `active`：正常可用
- `rejected`：拒绝入库
- `archived`：归档（不参与训练）

**重要**：所有脚本必须从 manifest 读取音频列表，**禁止 ls/find 扫描目录**。

#### `data/04_final_dataset/`

**最终数据集版本化管理。**

- `vYYYYMMDD_HHMMSS/`：历史版本，只读
- `latest`：软链接，指向当前生产版本
- `final_metadata/`：当前生产版本元数据（latest 的实际内容）

版本冻结由 `scripts/04_dataset/freeze_version.py` 执行。

#### `snapshots/`

**GPU 快照目录，rsync 拉回的 GPU 数据。**

- 命名：`gpu_backup_YYYYMMDD_HHMMSS/`
- 本地保留最新 5 个（由 `snapshot_retention.toml` 配置）
- 清理前必须检查 `.oss_verified` 标记
- **不提交 git**

#### `logs/`

**运行日志目录。**

- 命名：`{script_name}_{YYYYMMDD_HHMMSS}.log`
- 每次运行生成新日志，不覆盖旧日志
- **不提交 git**

---

## Git 提交规则

### 必须提交的文件

- ✅ 所有脚本代码（`scripts/`）
- ✅ 配置文件（`configs/`、`environment.yml`、`.gitignore`、`.env.example`）
- ✅ 数据元数据（CSV、JSON，不含音频）
- ✅ 文档（`README.md`、`docs/`、`PROJECT_STRUCTURE.md`）
- ✅ 标签映射字典（`label_mapping_dict.json`）
- ✅ dev 版本 wheel 包（`envs/wheel_cache/`）

### 绝对不能提交的文件

- ❌ 音频文件（wav/mp3/flac/ogg 等）
- ❌ 模型权重（pt/pth/bin/safetensors/ckpt 等）
- ❌ 密钥文件（.env、*.key、*.pem）
- ❌ 日志文件（logs/、*.log）
- ❌ Python 缓存（__pycache__/、*.pyc）
- ❌ Jupyter 缓存（.ipynb_checkpoints/）
- ❌ IDE 配置（.vscode/、.idea/）
- ❌ macOS 系统文件（.DS_Store）
- ❌ GPU 快照（snapshots/gpu_backup_*/）
- ❌ 模型推理缓存（model_output_cache/）
- ❌ 标注备份（backups/）

---

## 路径访问规则

### 音频路径

**禁止**在业务代码中手动拼接音频路径，必须调用：

```python
from scripts.utils.get_audio_physical_path import get_audio_physical_path

# 根据 audio_id 获取物理路径
audio_path = get_audio_physical_path(audio_id)
```

### 音频列表

**禁止**使用 `ls`、`find`、`glob` 扫描音频目录，必须读取：

```python
import pandas as pd

# 从 manifest 读取音频列表
manifest = pd.read_csv("data/00_raw_collect/audio_manifest.csv")
active_audios = manifest[manifest["status"] == "active"]
```

### 配置

**禁止**硬编码密钥，必须使用统一配置加载器：

```python
from scripts.utils.config_loader import get_oss_config

# 获取 OSS 备份账号配置
oss_config = get_oss_config("backup")
```

---

## 并发与安全规则

### pipeline_lock

- 长时间运行的流水线必须获取 `pipeline_lock`
- 锁文件位置：`data/.pipeline_lock`
- 超时阈值：2 小时（仅告警，不自动删除）
- 僵尸锁必须手动运行 `clear_stale_lock.py --force-stale-clean`

### disk_guard

- 磁盘使用率 > 80% 警告，> 90% 严重
- 快照轮转：保留最新 5 个
- 清理前必须确认 `.oss_verified` 标记
- 建议每天运行一次 `disk_guard.py --check-only`

### OSS 安全

- 备份账号（OSS_BACKUP）：只写权限，不能下载/删除
- 恢复账号（OSS_RECOVERY）：只读权限，不能上传/删除
- 密钥存储在 `~/.config/music-corpus/.env`，权限 600
- 项目目录只留 `.env.example` 模板
