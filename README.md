# 音乐 AI 数据集建设项目 (Music AI Dataset)

> 求职作品集 | 目标岗位: 腾讯音乐娱乐集团 AI创新内容运营 (深圳)
> 端到端音乐数据集构建: 采集→清洗→预标注→人工校验→训练评估

## 项目亮点

| 维度 | 说明 |
|------|------|
| **7阶段流水线** | 采集入库→元数据清洗→格式标准化→质量清洗→去重→辅助清洗→预处理输出，严格顺序执行 |
| **L1-L4分层预标注** | 物理特征→语义候选→多模态结构标注→KNN传播融合，成本分层(5%多模态+95%文本) |
| **来源隔离数据集** | main_pool/test_pool/holdout_pool/ood_pool 四池独立，test/holdout 不参与训练 |
| **三层计算架构** | Mac主节点 + AutoDL GPU重计算 + iOS科学上网采集，苹果生态互通 |
| **工程化文档** | ARCHITECTURE.md + DATASET_TAXONOMY.md + DECISIONS.md(10条ADR) + EXECUTION_PLAN.md |

## 核心架构

### 7阶段数据流水线

```
Stage 0 采集入库 → Stage 1 元数据清洗 → Stage 2 格式标准化(GPU)
→ Stage 3 质量清洗(三分支pass/marginal/fail) → Stage 4 近似去重
→ Stage 5 辅助清洗(5.1语言过滤/5.2歌词转写/5.3风格聚类)
→ Stage 6 预处理输出(切片+训练特征)
```

### L1-L4 分层预标注

```
L1 物理标签: BPM/调性/SNR/LUFS (librosa, 全量)
L2 语义候选: MERT 768d嵌入 + CLAP zero-shot (GPU, 全量)
L3 结构标注: Qwen-Omni多模态, 段落/乐器/情绪/Caption (5%黄金集)
L4 传播融合: DeepSeek全量文本 + KNN传播(量化阈值) + 规则融合 (全量)
→ ls_preannotations.jsonl → Label Studio 人工校验
```

### 模型选型与成本

| 层级 | 模型 | 单首成本 | 覆盖 |
|------|------|----------|------|
| L3 | Qwen3.5-Omni-Flash (多模态, 国内直连) | ~¥0.1 | 5% |
| L4 | DeepSeek V4 Flash (文本) | ~¥0.001 | 100% |
| L4 | KNN(cosine) | 免费 | 100% |

## 文档索引

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 完整7阶段+L1-L4架构, 三分支决策, 融合矩阵, QC清单 |
| [DATASET_TAXONOMY.md](docs/DATASET_TAXONOMY.md) | 数据集分类体系, 核心四划分, 功能标签, 样本均衡, 统计评测 |
| [DECISIONS.md](docs/DECISIONS.md) | 10条关键决策记录(ADR), 背景/选项/决策/理由/后果 |
| [EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) | 执行计划活文档, P0-P3进度跟踪, 已知问题, 环境配置 |
| [IOS_MAC_SYNC.md](docs/IOS_MAC_SYNC.md) | iOS→Mac数据采集同步工作流, 科学上网方案 |

## 快速开始

### 环境准备

```bash
# Mac 本地环境
conda env create -f environment.yml
conda activate labelstudio-env

# GPU 环境 (AutoDL)
conda env create -f environment_gpu.yml
```

### 端到端运行 (27首测试数据)

```bash
# Stage 0-1: 入库 + 元数据清洗
python3 scripts/00_collect/import_audio.py --source-dir data/incoming/
python3 scripts/00.5_cleaning/clean_pipeline.py --stages 1

# Stage 2-6: GPU 重计算 (SSH到GPU执行)
# 格式标准化 → 质量清洗 → 去重 → 辅助清洗 → 预处理

# L1-L4 预标注 (Mac)
python3 scripts/02_preannotation/l1_physical/l1_physical_features.py
python3 scripts/02_preannotation/l4_propagated/l4_knn_propagation.py \
  --embeddings-dir data/.../l2_mert_embedding \
  --l4-deepseek-dir data/02_preannotation/l4_deepseek \
  --l3-golden-dir data/02_preannotation/l3_structural \
  --output-dir data/02_preannotation/l4_propagated \
  --ls-output data/02_preannotation/ls_preannotations.jsonl

# Label Studio 人工校验
label-studio start --allow-local-files
```

### 数据集统计与可视化

```bash
# 数据集统计评测
python3 scripts/06_evaluation/dataset_stats.py \
  --manifest data/00_raw_collect/audio_manifest.csv \
  --l4-dir data/02_preannotation/l4_propagated \
  --mert-dir data/.../l2_mert_embedding \
  --output-dir data/.../stats/

# MERT 聚类可视化 (plotly交互HTML)
python3 scripts/05_visualization/visualize_mert_clustering.py \
  --embeddings-dir data/.../l2_mert_embedding \
  --labels-dir data/02_preannotation/l4_propagated \
  --output data/.../stats/mert_clustering.html
```

---

## 存储架构与核心约束

> 以下为项目底层存储架构，详细配置见各章节。

| 约束 | 违规后果 |
|------|----------|
| **OSS 只上传，业务绝不读取音频** | 若流水线从 OSS 拉音频，直接破坏本地磁盘为主数据源的设计 |
| **raw_audio 只读，永不修改移动** | 任何原地编辑原始音频 = 破坏完整性基线，无法回溯 |
| **禁止 ls/find 扫描音频目录** | 10万+文件时文件系统卡死；永远读元数据表 |
| **latest 软链接只允许脚本修改** | 人工改链接会导致训练/标注读取错误版本 |
| **磁盘清理前必须确认 OSS 已备份** | 未校验 `.oss_verified` 就删本地 = 数据永久丢失 |

---

## 存储架构

### 架构定位

| 组件 | 定位 | 职责 |
|------|------|------|
| 本地 Mac 磁盘 | **唯一业务数据源** | 所有业务流水线只读本地磁盘 |
| AutoDL GPU | 重计算节点 | 预处理、demucs、模型推理，结果通过 rsync 拉回 |
| 阿里云 OSS | **纯备份归档** | 仅上传备份，业务绝不读取，灾难恢复时才用 |

### 数据传输方式

```
GPU 服务器 ──rsync/SCP──→ 本地 Mac（业务数据源）
     │                            │
     │ 上传备份（外网）            │ 上传备份（外网）
     ▼                            ▼
  阿里云 OSS（纯备份，业务不读）
```

- **GPU ↔ Mac**：rsync 直接传输，不走 OSS
- **GPU → OSS**：外网上传备份（只写密钥）
- **Mac → OSS**：外网上传备份（只写密钥）
- **OSS → Mac**：仅灾难恢复时使用（只读密钥）

### 完整流水线流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     AutoDL GPU 服务器                        │
│  1. 读取本地 raw_audio（GPU 本地副本）                       │
│  2. 预处理：重采样、-14LUFS、切片、demucs源分离              │
│  3. MERT / CLAP / MOSS-Music 推理                           │
│  4. 输出到 GPU 本地快照目录                                   │
│  5. （可选）上传备份到 OSS（外网，只写密钥）                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ rsync/SCP 直接拉回（不走 OSS）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       本地 Mac（业务数据源）                   │
│  1. rsync 拉回 GPU 快照到 ./snapshots/                       │
│  2. 读取本地 model_output_cache 推理缓存                      │
│  3. 运行 run_preann_infer.py，输出 preann_csv                │
│  4. 标签映射、黑名单过滤                                      │
│  5. 特征提取：读取本地音频，librosa/essentia/madmom计算       │
│  6. Label-Studio 导入 tasks json，audio 字段填本地路径        │
│  7. 人工标注完成导出 jsonl，生成数据集元数据、划分数据集       │
│  8. （可选）上传备份到 OSS（外网，只写密钥）                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ 上传备份（外网，只写密钥）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    阿里云 OSS（纯备份归档）                    │
│  ├── raw_audio/            原始音频备份                       │
│  ├── processed_audio/      预处理音频备份                     │
│  ├── demucs_stems/         分轨备份                           │
│  ├── model_output_cache/   推理缓存备份                       │
│  └── snapshots/            GPU 快照备份                       │
│                                                              │
│  ⚠️ 业务绝不从 OSS 读取音频                                   │
│  ⚠️ 仅灾难恢复时使用 OSS_RECOVERY 只读密钥下载                │
└─────────────────────────────────────────────────────────────┘
```

---

## OSS 配置说明

### 密钥权限分离（最小权限原则）

| 账号 | 用途 | 权限 |
|------|------|------|
| `OSS_BACKUP_*` | 日常上传备份 | `PutObject` + `ListObjects`（前缀限制），**禁止 GetObject/DeleteObject** |
| `OSS_RECOVERY_*` | 灾难恢复下载 | `GetObject` + `ListObjects`（前缀限制） |

> 禁止给备份账号 `GetObject` / `DeleteObject`，防止密钥泄露后音频被盗或误删。

### 密钥位置

- 真实密钥：`~/.config/music-corpus/.env`（项目目录外）
- 项目根目录：仅存 `.env.example` 模板

### Endpoint

- **全部使用外网 Endpoint**：`https://oss-cn-hangzhou.aliyuncs.com`
- 新架构关闭内网 Endpoint 音频访问（GPU 同地域内网免流量，但业务不通过 OSS 读音频）

---

## 环境准备

### 本地 Mac 环境（完整功能）

```bash
# 1. 下载 dev 版本 wheel 包（首次运行）
pip download essentia==2.1b6.dev1389 madmom-onnx==0.17.dev0 -d ./envs/wheel_cache

# 2. 创建环境
conda env create -f environment.yml

# 3. 激活
conda activate music-corpus-local
```

### GPU 服务器环境（AutoDL）

```bash
# 创建环境
conda env create -f environment_gpu.yml

# 激活
conda activate music-corpus-gpu
```

GPU 环境不包含 librosa / essentia / madmom-onnx / label-studio，只负责重计算任务。

---

## 项目目录结构

```
music_corpus_project/
├── configs/
│   ├── model_configs/                  # 模型训练、推理yaml配置
│   ├── label_studio/
│   │   └── labeling_interface.xml      # Label-Studio标注界面模板
│   └── schema/                         # 数据规范
│       ├── audio_id_schema.json        # ULID 规范 + 冲突策略
│       └── corpus_format_spec.yaml     # 入库音频格式标准
├── scripts/
│   ├── 00_collect/                     # 数据采集、入库
│   │   ├── import_audio.py             # 采集入库，前置格式校验
│   │   └── gen_raw_checksum.py         # 生成 checksum + audio_manifest.csv
│   ├── 01_preprocess/                  # 音频预处理
│   ├── 02_preannotation/                # 机器预标注
│   │   ├── tag_mapping_musiccaps.py
│   │   ├── custom_audio_preprocess.py
│   │   ├── stat_unmapped.py
│   │   └── run_preann_infer.py         # 只做推理，不上传（新架构）
│   ├── 03_labelstudio/                 # LS导入/导出
│   │   ├── convert_mapped_to_labelstudio.py
│   │   ├── convert_ls_jsonl.py
│   │   └── ls_backup_export.py
│   ├── 04_dataset/                     # 数据集构建、版本冻结
│   └── utils/                          # 公共工具库
│       ├── get_audio_physical_path.py  # 统一散列路径计算
│       ├── pipeline_lock.py            # 并发锁
│       ├── clear_stale_lock.py         # 僵尸锁清理
│       ├── disk_guard.py               # 磁盘管控 + 快照轮转
│       ├── verify_oss_upload.py        # OSS上传完整性校验
│       ├── disaster_recovery.py        # 灾难恢复
│       ├── upload_cache_to_oss.py      # 独立上传缓存到OSS
│       ├── oss_local_client.py         # Mac端OSS备份客户端
│       ├── oss_gpu_backup_client.py    # GPU端OSS备份客户端
│       └── backup_gpu_mirror.sh        # rsync拉回GPU快照
├── envs/
│   └── wheel_cache/                    # dev版本wheel包缓存（提交git）
├── data/
│   ├── .pipeline_lock                   # 运行时锁文件（gitignore）
│   ├── 00_raw_collect/
│   │   ├── raw_audio/                    # 两层散列：md5(audio_id)[0:2]/[2:4]/
│   │   ├── raw_audio_checksums.csv       # 完整性基线（只读，永不修改）
│   │   ├── audio_manifest.csv            # 全局业务索引（可更新status）
│   │   └── rejected/                     # 格式不合规拒绝入库（定期清理）
│   ├── 01_preprocess/
│   │   ├── processed_audio/segments/
│   │   └── demucs_stems/
│   ├── 02_preannotation/
│   │   ├── preann_csv/
│   │   ├── model_output_cache/           # 推理缓存（disk_guard管控）
│   │   └── label_mapping/
│   ├── 03_human_review/
│   └── 04_final_dataset/
│       ├── v20260820_173500/             # 版本化目录（精确到秒）
│       └── latest -> ./v20260820_173500/ # 软链接（仅脚本可修改）
├── snapshots/                            # GPU快照（rsync拉回）
│   ├── gpu_backup_20260820_173500/
│   │   └── .oss_verified                 # OSS校验通过标记
│   ├── snapshot_retention.toml           # 快照轮转配置
│   └── README.md
├── models/
├── notebooks/
├── logs/
├── README.md
├── environment.yml                       # 本地Mac环境
├── environment_gpu.yml                   # GPU服务器环境
├── .env.example                          # 密钥模板（真实密钥在~/.config/）
└── .gitignore
```

---

## 快速开始

### 1. GPU 端：推理 + 快照

```bash
# SSH 到 GPU
ssh root@your-gpu-ip

# 运行推理（输出到 GPU 本地快照目录）
cd /workspace/music_corpus_project
conda activate music-corpus-gpu
python scripts/02_preannotation/run_preann_infer.py

# （可选）上传备份到 OSS
python scripts/utils/upload_cache_to_oss.py
```

### 2. 本地 Mac：rsync 拉回快照

```bash
cd /Users/m.jian/music_corpus_project

# rsync 直接拉回 GPU 快照（不走 OSS）
./scripts/utils/backup_gpu_mirror.sh root@your-gpu-ip /workspace/data/model_output_cache

# 快照会拉回到 ./snapshots/gpu_backup_YYYYMMDD_HHMMSS/
```

### 3. 本地 Mac：预标注转换

```bash
conda activate music-corpus-local

# 将拉回的推理缓存复制/链接到 data/02_preannotation/model_output_cache/
# 然后运行转换
python3 scripts/02_preannotation/run_preann_infer.py
```

### 4. Label Studio 标注

```bash
# 启动 Label Studio（允许本地文件访问）
label-studio start --allow-local-files

# 导入 tasks json（audio 字段是本地磁盘路径）
```

### 5. 数据集切分

```bash
# Label Studio 导出 JSON-MIN / raw jsonl
# 放到 data/03_human_review/labelstudio_export/

python3 scripts/03_labelstudio/convert_ls_jsonl.py
```

---

## 关键脚本速查

### 00 采集入库
| 脚本 | 职责 | 输出 |
|------|------|------|
| `import_audio.py` | 格式校验 + 拒绝不合规文件 → `rejected/` | — |
| `gen_raw_checksum.py` | 生成 `raw_audio_checksums.csv` + `audio_manifest.csv` | 校验基线 + 全局索引 |

### 02 预标注
| 脚本 | 职责 | 注意 |
|------|------|------|
| `run_preann_infer.py` | 只做推理，输出到 `model_output_cache/` | **绝不包含上传逻辑** |
| `upload_cache_to_oss.py` | 独立上传缓存到 OSS | 上传后调用校验，写入 `.oss_verified` |

### utils（全局工具）
| 脚本 | 职责 | 使用场景 |
|------|------|----------|
| `get_audio_physical_path(audio_id)` | 统一计算 md5 散列路径 | **所有代码必须调用，禁止手动拼接** |
| `pipeline_lock.py` | 创建/释放锁，含 PID + 时间戳 | 任何修改 data/ 的流水线前后调用 |
| `clear_stale_lock.py` | 显式清理僵尸锁 | 带 `--force-stale-clean` 参数 |
| `disk_guard.py` | 水位告警 + 过期缓存清理 | 定期 cron 执行 |
| `verify_oss_upload.py` | 上传后校验 OSS 完整性 | 快照轮转、缓存上传后必须调用 |
| `disaster_recovery.py` | 从 OSS 恢复本地数据 | 磁盘损坏 / 新协作者初始化 |
| `backup_gpu_mirror.sh` | rsync 拉回 GPU 快照 | GPU 推理完成后调用 |

---

## 关键文件 / 标记文件清单

| 文件路径 | 类型 | 说明 |
|----------|------|------|
| `data/.pipeline_lock` | 运行时锁 | 存在 = 有流水线在跑，禁止启动新任务 |
| `data/00_raw_collect/raw_audio_checksums.csv` | 只读基线 | 采集生成，永不修改，完整性校验唯一标准 |
| `data/00_raw_collect/audio_manifest.csv` | 业务索引 | 可更新 `status` 字段，灾难恢复蓝图 |
| `data/04_final_dataset/latest` | 软链接 | 指向当前生产版本，**禁止手动修改** |
| `snapshots/gpu_backup_*/.oss_verified` | 标记文件 | 上传 + 校验全部通过，本地才可删除该快照 |
| `snapshots/snapshot_retention.toml` | 配置 | 本地保留快照数量、OSS 前缀等 |
| `~/.config/music-corpus/.env` | 密钥 | 真实 OSS 密钥，**项目目录仅存 `.env.example`** |

---

## 目录散列规则（统一封装，禁止手动拼接）

```python
# 唯一正确方式：调用工具函数
from utils import get_audio_physical_path
rel_path = get_audio_physical_path("01ARZ3NDEKTSV4RRFFQ69G5FAV")
# 返回：raw_audio/a1/b2/a1b2c3d4..._01ARZ3NDEKTSV4RRFFQ69G5FAV.mp3
# 其中 a1b2 = md5(audio_id)[0:4]
```

**禁止行为**：业务代码手动拼接 `f"raw_audio/{audio_id[:2]}/..."`

---

## 版本号规则

```
vYYYYMMDD_HHMMSS          # 基础格式，如 v20260820_173500
vYYYYMMDD_HHMMSS-001      # 同一秒内冲突，自动追加序列号
latest -> ./v20260820_173500/   # 软链接指向当前生产版本
```

---

## 故障排查速查

| 现象 | 排查步骤 | 解决 |
|------|----------|------|
| 流水线启动失败，提示 lock 存在 | `cat data/.pipeline_lock` 看 PID + 时间戳 | 确认无进程在跑后，`python scripts/utils/clear_stale_lock.py --force-stale-clean` |
| 磁盘告警 90% | `python scripts/utils/disk_guard.py` | 自动清理过期缓存；检查 `snapshots/` 是否超 retention |
| 快照上传后本地不敢删 | 检查目录内是否有 `.oss_verified` | 无标记 → 运行 `verify_oss_upload.py` |
| 灾难恢复后文件损坏 | `python scripts/utils/disaster_recovery.py --verify` | 与 `raw_audio_checksums.csv` 比对 sha256 |
| 新协作者初始化项目 | 先拉代码 → `disaster_recovery.py --full-restore` → 校验 checksum | 使用 OSS_RECOVERY 只读密钥 |
| audio_id 冲突 | 检查 `configs/schema/audio_id_schema.json` | 强制 ULID，冲突拒绝入库 |

---

## 禁止行为清单（红线）

- ❌ 从 OSS 读取/播放音频用于业务流水线
- ❌ 修改、移动、重命名 `data/00_raw_collect/raw_audio/` 内任何文件
- ❌ 业务代码手动拼接音频物理路径（必须用 `get_audio_physical_path`）
- ❌ 用 `ls` / `find` 扫描音频目录做遍历（永远读 `audio_manifest.csv`）
- ❌ 人工修改 `data/04_final_dataset/latest` 软链接
- ❌ 删除本地快照前未确认 `.oss_verified` 标记
- ❌ 把真实 `.env` 密钥提交到 git
- ❌ 无条件自动删除 `.pipeline_lock`（超时仅告警，显式参数才清理）
- ❌ `raw_audio_checksums.csv` 入库后再次修改
- ❌ `rejected/` 目录放任不管不清理（会膨胀占磁盘）

---

## 实施注意边界（编码阶段参考）

1. 物理分片路径统一封装为工具函数 `get_audio_physical_path(audio_id)`，禁止业务代码手动拼接路径；所有音频访问通过 audio_id 查表，禁止 ls/find 扫描音频目录。
2. `.oss_verified` 仅代表上传当时校验通过；OSS侧对象被外部删除不会同步更新本地标记；高危快照清理可加 `--recheck-oss` 实时重校验。
3. `pipeline_lock`：fcntl.flock负责进程崩溃自动释放；锁超时仅告警，不无条件自动删除僵尸锁，提供 `--force-stale-clean` 显式清理入口，防止误中断长时任务。
4. `final_dataset` 版本冲突：同一秒多次冻结自动追加 `-001/-002`；latest软链接仅允许脚本修改。
5. `raw_audio_checksums.csv`为只读基线；`audio_manifest.csv`为业务索引，只更新status，不删除历史行。
6. `rejected`目录存放格式不合格原始素材，需要人工定期清理，不会自动回收磁盘。
7. OSS RAM全部开启前缀限制，最小权限原则。

---

## 运行约定

> **所有 python 脚本必须在项目根目录 `music_corpus_project/` 下执行。**
> **Mac + Miniconda 环境统一用 `python3`，不要用 `python`。**

```bash
cd /Users/m.jian/music_corpus_project
```

## 日志说明

所有脚本运行时自动生成日志文件到 `logs/` 目录，文件名带时间戳：
```
logs/run_preann_20260820_142030.log
logs/upload_oss_20260820_153045.log
```
