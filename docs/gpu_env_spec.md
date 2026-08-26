# GPU 环境规范文件（GPU Environment Specification）

> **用途**：记录 AutoDL GPU 实例的完整环境信息，作为所有 GPU 任务启动前的必读检查清单。
> **维护规则**：每次 GPU 环境变化（安装依赖、下载模型、修改配置）后，必须更新此文件。
> **最后更新**：2026-08-26

---

## 1. 实例基本信息

| 项目 | 值 |
|------|-----|
| 平台 | AutoDL |
| GPU 型号 | NVIDIA RTX 4090（24GB 显存） |
| SSH 端口 | 49530（当前会话，可能变化） |
| SSH 主机 | connect.westb.seetacloud.com |
| SSH 用户 | root |
| 免密登录 | ~/.ssh/id_rsa（已配置） |
| 项目路径 | /workspace/music_corpus_project/ |
| 自定义镜像 | music-ai-dataset-env-v1（已保存） |

---

## 2. 网络状态

| 项目 | 状态 |
|------|------|
| 外网访问 | ❌ **不可达**（`Network is unreachable`） |
| HuggingFace | ❌ 无法下载模型 |
| PyPI | ❌ 无法 pip install |
| 内网/AutoDL 镜像源 | ✅ 可用（conda/pip 需配置国内源） |

**重要**：所有模型必须提前下载到本地路径，通过 `--model-path` 参数指定，**禁止依赖运行时下载**。

---

## 3. Conda 环境

| 项目 | 值 |
|------|-----|
| 环境名 | labelstudio-env |
| Python | 3.11.13 |
| 激活命令 | `source /root/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env` |

### 3.1 核心依赖包（已安装）

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | 2.2.2+cu121 | 深度学习框架（CUDA 可用） |
| transformers | 4.40.2 | HuggingFace 模型加载 |
| librosa | 0.11.0 | 音频处理 |
| numpy | 1.26.4 | 数值计算 |
| pandas | 3.0.5 | 数据处理 |
| soundfile | 0.14.0 | 音频读写 |
| scikit-learn | （已安装） | KNN/机器学习 |
| laion_clap | （已安装） | CLAP 零样本分类 |
| ulid-py | （已安装） | ULID 生成 |

**验证命令**：
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers, librosa, numpy, pandas, soundfile; print('all core deps OK')"
python -c "import laion_clap; print('laion_clap OK')"
```

---

## 4. 预下载模型（本地路径）

> **重要**：GPU 无外网，所有模型必须在此列表中。启动脚本时必须用 `--model-path` 指定。

| 模型 | 路径 | 大小 | 用途 | 下载日期 |
|------|------|------|------|---------|
| CLAP (630k-audioset-fusion) | `/root/autodl-tmp/models/clap_fusion/630k-audioset-fusion-best.pt` | 1.8 GB | L2 零样本标注（genre/mood） | 2026-08-20 |
| MERT-v1-95M | HuggingFace 缓存（`m-a-p/MERT-v1-95M`） | ~400 MB | L2 嵌入提取 | 运行时缓存（首次需下载） |

### 4.1 MERT 模型说明

MERT 模型通过 `transformers` 从 HuggingFace 缓存加载。如果缓存中不存在，**首次运行需要外网下载**。

**检查缓存**：
```bash
ls ~/.cache/huggingface/hub/models--m-a-p--MERT-v1-95M/ 2>/dev/null && echo "MERT cached" || echo "MERT NOT cached"
```

**如果未缓存**：需要在有外网的环境下载后上传到 GPU，或使用 AutoDL 的模型市场。

---

## 5. 数据目录结构

```
/workspace/music_corpus_project/
├── data/
│   ├── 00_raw_collect/           # 原始采集（manifest.csv）
│   ├── 00.5_cleaned/reports/     # QC Gate 报告
│   ├── 01_preprocess/
│   │   └── processed_master/      # 母版音频（MP3 格式，GPU 同步时转码）
│   ├── 02_preannotation/
│   │   ├── l1_physical/           # L1 物理特征（JSON + CSV）
│   │   ├── l2_embedding/          # L2 MERT 嵌入（.npy，768维）
│   │   ├── l2_embedding_clap/     # L2 CLAP 嵌入（.npy，512维）
│   │   ├── l2_semantic/           # L2 CLAP 零样本（.json，genre/mood top-5）
│   │   ├── l3_structural/         # L3 结构标注（.json）
│   │   └── l4_propagated/         # L4 KNN 传播结果
│   ├── 03_human_annotation/
│   │   └── golden_set/             # 黄金集（manifest + annotations）
│   └── 04_final_dataset/splits/   # 数据划分（train/val/test/holdout）
├── scripts/                        # 项目脚本
└── logs/                           # 运行日志
```

---

## 6. GPU 任务启动前检查清单（Mandatory Pre-flight Checklist）

> **每次启动 GPU 任务前，必须逐项检查并确认。**

### 6.1 环境检查

- [ ] SSH 连接正常：`bash scripts/utils/gpu_ssh.sh <port> "echo OK"`
- [ ] Conda 环境可用：`conda activate labelstudio-env`
- [ ] CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 项目路径存在：`ls /workspace/music_corpus_project/`

### 6.2 数据检查

- [ ] 母版音频已同步：`ls /workspace/music_corpus_project/data/01_preprocess/processed_master/*.mp3 | wc -l`（应为 85）
- [ ] **无 AppleDouble 文件**：`ls /workspace/music_corpus_project/data/01_preprocess/processed_master/._* 2>/dev/null | wc -l`（应为 0）
- [ ] manifest 已同步：`ls /workspace/music_corpus_project/data/00_raw_collect/audio_manifest.csv`

### 6.3 模型检查（关键！）

- [ ] **CLAP 模型存在**：`ls -lh /root/autodl-tmp/models/clap_fusion/630k-audioset-fusion-best.pt`（1.8G）
- [ ] **MERT 模型已缓存**：`ls ~/.cache/huggingface/hub/models--m-a-p--MERT-v1-95M/ 2>/dev/null`
- [ ] 启动脚本时使用 `--model-path` 指定本地模型，**禁止默认下载**

### 6.4 输出目录检查

- [ ] 输出目录存在：`mkdir -p data/02_preannotation/l2_embedding/ data/02_preannotation/l2_semantic/ logs/`
- [ ] 清理旧输出（如需重新运行）：`rm -f data/02_preannotation/l2_embedding/*.npy data/02_preannotation/l2_semantic/*.json`

---

## 7. 常见问题与解决方案

### 7.1 AppleDouble 文件（`._*`）

**问题**：Mac 上 `tar` 打包的文件在 Linux 解压时产生 `._*` 资源分叉文件，音频脚本匹配到这些无效文件导致 `Failed to open input`。

**解决方案**：
```bash
# 解压后立即清理
find /workspace/music_corpus_project/data/01_preprocess/processed_master/ -name '._*' -delete

# 或在 Mac 打包时使用 COPYFILE_DISABLE
COPYFILE_DISABLE=1 tar czf master_mp3.tar.gz -C /tmp/master_mp3 .
```

### 7.2 CLAP 模型下载失败

**问题**：GPU 无外网，CLAP 脚本默认从 HuggingFace 下载 `630k-audioset-fusion-best.pt`。

**解决方案**：使用本地模型路径：
```bash
python scripts/02_preannotation/l2_semantic/l2_clap_zero_shot.py \
  --model-path /root/autodl-tmp/models/clap_fusion/630k-audioset-fusion-best.pt \
  --device cuda
```

### 7.3 后台进程日志目录不存在

**问题**：`nohup python script.py > logs/script.log 2>&1 &` 时，如果 `logs/` 目录不存在，日志重定向失败，进程立即退出。

**解决方案**：启动前 `mkdir -p logs/`。

### 7.4 GPU 同步 FLAC 太慢

**问题**：85 首 FLAC（3.4G）通过 rsync 传输速度约 1-2 MB/s，需 30+ 分钟。

**解决方案**：本地转 MP3（320kbps，约 500MB）后打包传输：
```bash
# 本地：并行转 MP3
python /tmp/convert_mp3.py  # ThreadPoolExecutor 4线程

# 打包
COPYFILE_DISABLE=1 tar czf /tmp/master_mp3.tar.gz -C /tmp/master_mp3 .

# 传输
scp -P <port> /tmp/master_mp3.tar.gz root@connect.westb.seetacloud.com:/workspace/

# GPU 解压 + 清理 AppleDouble
ssh -p <port> root@connect.westb.seetacloud.com "cd /workspace && tar xzf master_mp3.tar.gz -C /workspace/music_corpus_project/data/01_preprocess/processed_master/ && find /workspace/music_corpus_project/data/01_preprocess/processed_master/ -name '._*' -delete"
```

---

## 8. 脚本参数规范

### 8.1 MERT 嵌入提取

```bash
python scripts/02_preannotation/l2_embedding/extract_mert_embedding.py \
  --input-dir data/01_preprocess/processed_master/ \
  --output data/02_preannotation/l2_embedding/ \
  --device cuda \
  --chunk-sec 30
```

- 输出：`{hash_audioid}_mert_embedding.npy`（768维）
- 注意：audio_id 是 hash_audioid 格式，需通过 manifest 的 master_path 映射到 ULID

### 8.2 CLAP 零样本标注

```bash
python scripts/02_preannotation/l2_semantic/l2_clap_zero_shot.py \
  --input-dir data/01_preprocess/processed_master/ \
  --output data/02_preannotation/l2_semantic/ \
  --embedding-output data/02_preannotation/l2_embedding_clap/ \
  --model-path /root/autodl-tmp/models/clap_fusion/630k-audioset-fusion-best.pt \
  --device cuda \
  --top-k 5
```

- 输出：`{hash_audioid}_semantic.json`（genre/mood top-5）+ `{hash_audioid}_clap_embedding.npy`（512维）
- **必须指定 `--model-path`**，禁止默认下载

---

## 9. 维护日志

| 日期 | 变更内容 | 操作人 |
|------|---------|--------|
| 2026-08-26 | 初始创建，记录 RTX 4090 实例环境、CLAP 模型路径、常见问题 | 豆包agent |

---

## 10. 相关文档

- [V3 流水线架构](PIPELINE_OVERVIEW_V3.md)
- [ADR-003 数据划分与源隔离](../adr/ADR-003-DATA-SPLIT-AND-SOURCE-ISOLATION.md)
- [GPU 同步脚本](../../scripts/utils/gpu_sync.sh)
- [GPU SSH 脚本](../../scripts/utils/gpu_ssh.sh)
