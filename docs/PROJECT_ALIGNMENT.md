# Music AI Dataset 项目全景对齐文档

> 生成时间：2026-08-21
> 用途：与其他 AI / 协作者快速对齐项目现状、架构、成果与待办

---

## 一、项目概述

**项目名称**：music-ai-dataset（音乐语料库数据集构建）

**核心目标**：构建标准化、可复现、高质量的音乐语料数据集，覆盖数据采集→清洗→预处理→预标注→人工标注→数据集版本化全流程。

**技术栈**：
- 本地：MacBook Pro 2020 Intel i5 / 16GB RAM / x86_64
- GPU：AutoDL RTX 4090 24GB / 503GB内存 / 46GB磁盘
- 存储：阿里云 OSS（纯备份归档，业务不读取音频）
- 版本管理：GitHub 私有仓库（SSH 认证）

**GitHub 仓库**：`git@github.com:g9d4fbt4s5-creator/music-ai-dataset.git`

---

## 二、目录结构

```
music_corpus_project/
├── .git/                          # Git 仓库
├── .gitignore                     # Git 黑名单（音频/模型/日志/密钥）
├── .env.example                   # 环境变量模板（真实密钥在 ~/.config/music-corpus/.env）
├── environment.yml                # Mac 本地 conda 环境（music-corpus-local）
├── environment_gpu.yml            # GPU conda 环境（music-corpus-gpu）
├── README.md                      # 项目说明
│
├── configs/
│   ├── cleaning_config.yaml       # 数据清洗6阶段配置
│   ├── schema/                    # 数据规范（audio_id_schema.json, corpus_format_spec.yaml）
│   ├── model_configs/             # 模型配置
│   └── label_studio/
│       └── labeling_interface.xml # Label Studio 标注界面配置
│
├── scripts/
│   ├── 00_collect/                # 数据采集
│   │   ├── import_audio.py        # 音频导入+校验
│   │   └── gen_raw_checksum.py    # 原始音频 checksum 生成
│   │
│   ├── 00.5_cleaning/             # 数据清洗（6阶段）
│   │   ├── clean_pipeline.py      # 6阶段主流程（集成YAMNet）
│   │   ├── field_standardizer.py  # Stage1: 字段标准化（GM128/VAD/三级流派）
│   │   ├── format_normalize.py    # Stage2: 格式标准化
│   │   ├── quality_check.py       # Stage3: 质量检查+自动修复+降噪候选
│   │   ├── content_filter.py      # Stage3: 内容过滤（非音乐/人声/安全）
│   │   ├── approximate_dedup.py   # Stage4: 近似去重（chroma+余弦，已悬置）
│   │   ├── pii_remover.py         # Stage5: PII移除（9类正则）
│   │   ├── language_filter.py     # Stage5: 语言过滤（Whisper base）
│   │   ├── audio_chunker.py       # Stage6: 切片分块
│   │   ├── extract_features.py    # Stage6: 特征提取（128维）
│   │   └── yamnet_infer.py        # YAMNet推理（yamnet_env专用，支持并行）
│   │
│   ├── 01_preprocess/             # 预处理
│   │   ├── generate_master.py     # 母版生成（mp3→FLAC 48k/24bit/stereo）
│   │   └── batch_denoise.py       # 降噪候选批量处理（noisereduce）
│   │
│   ├── 02_preannotation/          # 预标注
│   │   ├── tag_mapping_musiccaps.py    # 标签映射（MusicCaps格式）
│   │   ├── custom_audio_preprocess.py  # 自有音频预标注
│   │   ├── stat_unmapped.py            # 未映射标签统计
│   │   └── run_preann_infer.py         # 预标注推理（CLAP/MERT）
│   │
│   ├── 03_labelstudio/            # Label Studio
│   │   ├── convert_mapped_to_labelstudio.py  # 映射→LS导入格式
│   │   ├── convert_ls_jsonl.py              # LS导出→切分csv
│   │   └── ls_backup_export.py              # LS标注备份导出
│   │
│   ├── 04_dataset/                # 数据集
│   │   └── join_meta.py           # 元数据join
│   │
│   └── utils/                     # 工具
│       ├── get_audio_physical_path.py    # 统一音频路径解析
│       ├── oss_local_client.py            # OSS本地客户端（上传备份）
│       ├── oss_gpu_backup_client.py       # OSS GPU备份客户端
│       ├── upload_cache_to_oss.py         # 推理缓存上传OSS
│       ├── verify_oss_upload.py           # OSS上传完整性校验
│       ├── disaster_recovery.py           # 灾难恢复
│       ├── disk_guard.py                  # 磁盘管控
│       ├── pipeline_lock.py               # 并发控制（fcntl.flock）
│       └── clear_stale_lock.py            # 清理过期锁
│
├── data/
│   ├── 00_raw_collect/            # 原始采集（只读，永不修改）
│   │   ├── raw_audio/             # 原始音频（散列目录 md5(audio_id)[0:4]两层）
│   │   ├── raw_audio_checksums.csv  # 原始音频checksum基线
│   │   └── audio_manifest.csv     # 全局音频索引（含master_path）
│   │
│   ├── 00.5_cleaned/              # 清洗产物
│   │   └── reports/               # 清洗报告
│   │       ├── quality_check_report.csv
│   │       ├── quality_check_report_noise_candidates.csv
│   │       ├── yamnet_output.csv
│   │       └── yamnet_input_list.csv
│   │
│   ├── 01_preprocess/             # 预处理产物
│   │   ├── processed_master/      # 统一母版 FLAC 48k/24bit/stereo
│   │   ├── segments/              # 切片片段（缓存）
│   │   ├── demucs_stems/          # Demucs分轨（缓存，按track_id分目录）
│   │   └── denoised_audio/        # 降噪后音频
│   │
│   ├── 02_preannotation/          # 预标注产物
│   │   ├── label_mapping/
│   │   │   └── label_mapping_dict.json  # 标签映射字典（v2.0, 80标签）
│   │   ├── model_output_cache/    # 模型推理缓存
│   │   └── features/
│   │       └── audio_features.csv # 128维音频特征
│   │
│   └── 04_final_dataset/          # 最终数据集（版本化）
│       └── v20260820_112723/     # 第一个数据集版本
│
├── snapshots/                      # GPU快照备份
│   ├── snapshot_retention.toml    # 快照轮转规则
│   └── README.md
│
├── models/                         # 模型权重（不进git）
├── notebooks/                      # Jupyter分析笔记本
├── logs/                           # 运行日志（带时间戳，不进git）
│
└── envs/
    └── wheel_cache/                # dev版本wheel包缓存（essentia/madmom-onnx）
```

---

## 三、数据清洗6阶段完成状态

### Stage 1: 元数据清洗 ✅ 已实现
- **脚本**：`field_standardizer.py`
- **功能**：字段标准化（乐器GM128 / 情绪VAD / 流派三级分类）、缺失补全、冲突消解、无效样本剔除
- **复用**：`label_mapping_dict.json`（v2.0, 80标签）

### Stage 2: 格式标准化 ✅ 已实现
- **脚本**：`format_normalize.py`
- **功能**：统一为 WAV/FLAC，44.1kHz/48kHz，16/24-bit
- **母版生成**：`generate_master.py`（mp3/wav/m4a → FLAC 48k/24bit/stereo，ffmpeg转码，checksum校验）

### Stage 3: 音频质量清洗 ✅ 已实现
- **脚本**：`quality_check.py` + `content_filter.py`
- **硬门槛（直接剔除）**：文件损坏 / 时长<5秒 / 削波>5% / 完全静音>99%
- **软标记（只警告不剔除）**：SNR<15dB / 静音占比>70% / 动态范围<10dB
- **自动修复**：AudioRepairer（格式转换/重采样/响度归一化）
- **降噪候选**：SNR<15dB 或 频谱平坦度>0.5 → 标记为降噪候选，人工审核后批量跑 noisereduce
- **内容过滤**：非音乐检测（librosa+规则兜底+YAMNet）/ 人声检测（librosa+Demucs）/ 文本安全检测
- **YAMNet集成**：`clean_pipeline.py` 通过 subprocess 调用 `yamnet_env`，输出CSV，主环境读取合并到df

### Stage 4: 多级去重 ⚠️ 部分实现（近似去重已悬置）
- **精确去重** ✅：MD5/SHA-256 文件级哈希比对
- **近似去重** ⚠️：chroma特征+余弦相似度已实现，但用户要求悬置，待后续重启
- **片段级去重** ❌：未实现
- **跨集泄露防控** ❌：未实现

### Stage 5: 辅助清洗 ⚠️ 部分实现
- **PII移除** ✅：`pii_remover.py`（9类正则：手机号/身份证/邮箱/银行卡/IP/URL/QQ/微信/地址）
- **语言过滤** ✅：`language_filter.py`（Whisper base detect_language()）
- **歌词/人声内容清洗** ❌：未实现（Demucs→FunASR歌唱版/faster-whisper，放GPU）
- **风格一致性聚类** ❌：未实现（MERT/CLAP嵌入 + DBSCAN，GPU嵌入+CPU聚类）

### Stage 6: 预处理输出 ✅ 已实现
- **脚本**：`audio_chunker.py` + `extract_features.py`
- **切片分块**：5-30秒片段，滑动窗口重叠50%（已测试通过）
- **特征提取**：128维特征（Mel/CQT/Chroma）
- **母版+派生架构**：原始采集→FLAC母版→segments/demucs_stems（缓存），重采样实时做不永久存

---

## 四、环境配置

### 4.1 Mac 本地环境（labelstudio-env）
- **Python**：3.11.13
- **核心包**：
  - torch 2.2.2 + torchaudio 2.2.2
  - transformers 4.40.2
  - librosa 0.11.0
  - essentia 2.1b6.dev1389（dev版，wheel缓存）
  - madmom-onnx 0.17.dev0（dev版，wheel缓存）
  - laion_clap 1.1.7
  - demucs 3.0.6
  - torchcrepe 0.0.24
  - label-studio 1.23.0
  - numpy 1.26.4（降级到1.x，CLAP不兼容2.x）
  - scipy 1.13.0 / scikit-learn 1.9.0
  - pyloudnorm（响度归一化）
  - oss2（阿里云OSS SDK）
  - python-ulid（audio_id生成）
  - noisereduce 3.0.3（降噪）
- **已下载模型**：MERT-v1-95M / LAION CLAP（HTSAT-base, 2.35GB）/ Demucs mdx_extra_q / bert-base-uncased / roberta-base
- **激活方式**：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env`

### 4.2 GPU 环境（AutoDL，labelstudio-env）
- **硬件**：RTX 4090 24GB / 503GB内存 / 46GB磁盘可用
- **连接**：`ssh -p 43107 root@connect.westb.seetacloud.com`
- **已装包**：funasr 1.4.2（中文歌唱歌词转写）/ faster-whisper 1.2.1（非中文歌词/口语）/ jiwer 4.0.0（WER/CER评估）/ noisereduce 3.0.3
- **数据**：jazz_500_audio-low（500首mp3）/ jazz_500_features（20首特征产物）/ models（模型权重）
- **注意**：无卡模式仍按小时计费，不能跑CLAP/torchcrepe推理

### 4.3 YAMNet 独立环境（yamnet_env）
- **目的**：与主环境完全隔离，避免 TensorFlow 与 PyTorch/numpy 版本冲突
- **Python**：3.11
- **核心包**：tensorflow 2.15.0 / tensorflow-hub 0.15.0 / numpy 1.24.4（锁死，TF不支持2.x）/ librosa 0.11.0 / soundfile
- **模型**：YAMNet（~15MB，521类音频事件）
- **激活方式**：`source /opt/miniconda3/etc/profile.d/conda.sh && conda activate yamnet_env`
- **调用方式**：主环境通过 subprocess 调用，输出CSV，主环境只读CSV不import tensorflow

---

## 五、关键架构决策

### 5.1 OSS 定位：纯备份归档（08-20架构转向）
- **旧方案**：OSS 数据中转 + 永久存储，GPU内网免流量读取音频
- **新方案**：OSS 纯备份归档，**业务绝不从OSS读取音频**
- **传输方式**：GPU快照通过 rsync 拉回本地，OSS仅异步备份
- **GPU读写OSS**：强制使用内网Endpoint（同地域免流量），只上传不下载音频
- **密钥位置**：`~/.config/music-corpus/.env`（权限600），项目根目录只留 `.env.example`

### 5.2 母版+派生架构
- **原始采集文件**：`00_raw_collect/raw_audio/`（mp3等），只读永不修改
- **统一母版**：`01_preprocess/processed_master/`（FLAC 48k/24bit/stereo），所有派生从这里出
- **派生缓存**：segments（切片）/ demucs_stems（分轨）必须缓存，重采样实时做不永久存
- **散列规则**：`md5(audio_id)[0:2]/md5(audio_id)[2:4]/` 两层目录

### 5.3 音频路径统一解析
- **核心红线**：禁止 ls/find 扫描音频目录，永远读 `audio_manifest.csv`
- **所有音频路径**：必须调用 `get_audio_physical_path(audio_id)`
- **audio_id**：ULID格式，全局唯一

### 5.4 质量检查硬门槛/软标记分离
- **硬门槛（直接剔除）**：文件损坏 / 时长<5秒 / 削波>5% / 完全静音>99%
- **软标记（只警告不剔除，人工审核）**：SNR<15dB / 静音占比>70% / 动态范围<10dB
- **背景**：原阈值（SNR≥20dB/静音≤50%/DR≥30dB）误杀正常音乐（7个中剔除5个，但只有1个真不是音乐），人工听后修正

### 5.5 YAMNet 策略：全都跑YAMNet（已废弃双路径）
- **旧方案**：数据来源可控用规则兜底，混杂用YAMNet（双路径）
- **新方案**：全都跑YAMNet，因为YAMNet比规则兜底快12-20倍（0.3-0.5秒/首 vs 5.9秒/首）
- **非音乐判定**：is_music=False 且 (has_speech=True 或 has_noise=True) → 剔除
- **并行处理**：20首以内串行，>100首用 `--parallel N` multiprocessing

### 5.6 ASR 双轨制
- **中文歌唱**：FunASR 歌唱版（paraformer-zh）
- **非中文/口语**：faster-whisper small
- **语言检测**：Whisper base detect_language()（轻量，<1GB显存）
- **歌词转写输入**：必须是 Demucs vocals stem（不是原曲），实测WER从60%降到25%

---

## 六、数据产物清单

### 6.1 已入库音频（7个，本地）
| audio_id | 格式 | 时长 | 说明 |
|----------|------|------|------|
| 01M0E9X162CTB4D15WZQ5D8FVX | wav | 30.0s | 正常音乐 |
| 01M0E9X17RZ522YBF6Z913H4VJ | wav | 30.0s | 正常音乐 |
| 01M0E9X19DMAXGT6QD167G1YJX | wav | 30.0s | 正常音乐 |
| 01M0E9X1AGP5JXX7YEA7NAV25V | wav | 30.0s | 正常音乐 |
| 01M0E9X1BHAGVNQYTXE4CMNJV0 | wav | 30.0s | 正常音乐 |
| 01M0E9X1CNR601CZY38WVQB82E | wav | 2.0s | 短音频（硬门槛剔除） |
| 01M0E9X1D5FB4ZND9B13XSD71K | mp3 | 98.5s | 正常音乐 |

### 6.2 核心数据文件
- `data/00_raw_collect/audio_manifest.csv`：全局音频索引（7条，含master_path/master_md5）
- `data/00_raw_collect/raw_audio_checksums.csv`：原始音频checksum基线
- `data/01_preprocess/processed_master/`：7个FLAC母版（已生成）
- `data/00.5_cleaned/reports/quality_check_report.csv`：质量检查报告（硬门槛/软标记分离后）
- `data/00.5_cleaned/reports/yamnet_output.csv`：YAMNet检测结果（7条）
- `data/02_preannotation/label_mapping/label_mapping_dict.json`：标签映射字典（v2.0, 80标签）
- `data/02_preannotation/features/audio_features.csv`：7音频128维特征
- `data/04_final_dataset/v20260820_112723/`：第一个数据集版本

### 6.3 GPU 数据
- `/root/autodl-tmp/jazz_500_audio-low/`：500首mp3（MTG-Jamendo jazz子集）
- `/root/autodl-tmp/jazz_500_features/`：20首特征产物
- `/root/autodl-tmp/models/`：模型权重

---

## 七、速度基准测试

### 7.1 质量检查+降噪候选标记（按时长归一化）
| 平台 | 测试音频总时长 | 实际耗时 | 处理速度 | 1000首30秒估算 |
|------|-------------|---------|---------|--------------|
| Mac 本地 | 250.5秒 | 9.12秒 | 27.46 音频秒/秒 | 18.2分钟 |
| GPU (AutoDL) | 2324.3秒 | 38.50秒 | 60.37 音频秒/秒 | 8.3分钟 |

**结论**：GPU比Mac快约2.2倍（按时长归一化）。质量检查是CPU密集型，但GPU实例CPU性能更强，长音频批量处理效率更高。

### 7.2 YAMNet vs 规则兜底
| 方法 | 单首耗时 | 1000首估算 | 准确率 |
|------|---------|-----------|--------|
| YAMNet（CPU） | 0.3-0.5秒 | 5-8分钟 | 高（521类预训练） |
| 规则兜底（librosa） | ~5.9秒 | 99分钟 | 低（仅静音/时长/频谱平坦度） |

**结论**：YAMNet快12-20倍，准确率更高，全都跑YAMNet。

---

## 八、未完成项与下一步

### P0（必须先做）
1. **Stage 5.2 歌词转写流水线**：Demucs分离→FunASR歌唱版（中文）/faster-whisper（非中文），放GPU
2. **demucs_stems目录结构调整**：从按stem类型分目录改为按track_id分目录

### P1（重要）
3. **Stage 5.3 风格一致性聚类**：MERT/CLAP嵌入 + DBSCAN，GPU嵌入+CPU聚类
4. **Stage 4 近似去重重启**：用户要求悬置，待后续决定方案（Chromaprint/AcoustID vs chroma余弦）
5. **audio_manifest.csv完善**：添加更多元数据字段（来源/版权/语言等）

### P2（可选）
6. **OSS RAM子账号创建**：备份账号（只写）+ 恢复账号（只读），权限最小化
7. **envs/wheel_cache下载**：essentia和madmom-onnx的dev版wheel包
8. **CI/CD**：.github/workflows/ 脚本语法检查
9. **snapshots/README.md**：快照用途和retention规则说明

### 已废弃/冻结
- ~~OSS数据中转~~ → 改为纯备份归档
- ~~双路径YAMNet策略~~ → 全都跑YAMNet
- ~~processed_audio设public-read~~ → 业务不通过OSS播放音频
- ~~近似去重（chroma余弦）~~ → 悬置，待重启

---

## 九、Git 状态

- **当前分支**：main
- **最新commit**：`060b75a`（feat: 母版生成脚本+降噪批量处理+YAMNet并行+速度对比）
- **远程仓库**：`git@github.com:g9d4fbt4s5-creator/music-ai-dataset.git`（私有，SSH认证）
- **.gitignore覆盖**：音频/模型权重/日志/密钥/数据集/缓存，保留配置文件和脚本

---

## 十、统一配置中心

**位置**：`~/.config/music-corpus/.env`（权限600）

| 字段 | 状态 | 说明 |
|------|------|------|
| OSS_BACKUP_ACCESS_KEY_ID/SECRET | ✅ 已填 | 读写账号，用于日常上传备份 |
| OSS_RECOVERY_ACCESS_KEY_ID/SECRET | ✅ 已填 | 只读账号，用于灾难恢复（list/head） |
| OSS_BUCKET | ✅ music-ai-dataset-2026 | OSS Bucket名称 |
| OSS_REGION | ✅ cn-hangzhou | OSS地域 |
| AUTODL_HOST/PORT | ✅ connect.westb.seetacloud.com:43107 | AutoDL GPU连接 |
| AUTODL_API_TOKEN | ✅ 已填 | 弹性部署API |

**三优先级加载**：`~/.config/music-corpus/.env` → `cwd/.env` → 环境变量（GPU上.bashrc的环境变量继续生效，零破坏）

---

## 十一、关键联系人/参考

- **飞书文档（架构评估）**：https://feishu.doubao.com/docx/SMq1dUZx9o3VJQxljKDcpHKsnIc
- **项目根路径**：`/Users/m.jian/music_corpus_project/`
- **运行约定**：所有脚本在项目根目录执行，统一用 `python3`，conda环境用 `labelstudio-env`

---

*本文档由项目当前状态自动生成，可直接复制给其他AI/协作者对齐。*
