# 执行计划与进度跟踪 (Execution Plan)

> 版本: v1.0.0 | 更新: 2026-08-25 | 状态: 持续更新
> 这是项目的活文档(living document)，每完成一项打勾，Git commit 时同步更新。

## 一、项目目标

构建音乐AI数据集(music_corpus_project)，作为求职作品集。
目标岗位: 腾讯音乐娱乐集团 AI创新内容运营(深圳)，内推人凯子鱼(余总)。
GitHub: https://github.com/g9d4fbt4s5-creator/music-ai-dataset

## 二、P0 工程化骨架 (2小时内完成)

- [x] **ARCHITECTURE.md** — 完整7阶段+L1-L4预标注架构
  - 三分支决策(pass/marginal/fail)
  - L4融合矩阵(cosine_dist阈值)
  - 10项QC清单
  - 模型成本表
  - Mac/GPU/iOS三层架构
- [x] **DATASET_TAXONOMY.md** — 数据集分类体系
  - 核心四划分(train/val/test/holdout)
  - 功能标签(黄金集/OOD/badcase/marginal)
  - Train内部分层(pretrain/sft/dpo)
  - 样本均衡策略
  - 统计评测规范
- [x] **qc_gate.py** — 统一QC三分支决策
  - 整合YAMNet+librosa+时长+源质量
  - fail不进下游，marginal标记flag_for_review
- [x] **ls_import_converter.py** — L3/L4→Label Studio转换
  - TimeSeries段落标注
  - 分类标签+Caption
  - 来源追溯元数据
- [x] **dataset_stats.py** — 数据集统计评测
  - 分布统计(流派/BPM/调性/响度/时长/人声/来源)
  - 质量统计(坏样本/边际样本/告警)
  - 标注统计(置信度/来源/IAA)
  - OOD统计(分布距离)
  - 嵌入可视化(t-SNE/UMAP)
- [x] **recipe.json** — 训练配方v1.0
  - 数据混合(含OOD 5%+黄金集定位)
  - 特征参数(n_mels/hop_length等具体值)
  - 增强策略(初期禁用，渐进式启用)
  - 标签Schema
  - 训练任务(pretrain/sft/dpo)
  - 评估指标
- [x] **labeling_interface.xml v2** — Label Studio标注模板
  - 9字段+结构段落+审核决策
  - 黄金集标记+边际样本警告
  - 元数据展示区

## 三、P1 数据流水线 (已完成27首端到端验证)

### Stage 0-6 完整流水线
- [x] Stage 0 采集入库 (27首，ULID主键，散列存储)
- [x] Stage 1 元数据清洗 (27首active)
- [x] Stage 2 格式标准化 (GPU ffmpeg FLAC 48kHz)
- [x] Stage 3 质量清洗 (YAMNet+librosa，21音乐/5人声/1bad)
- [x] Stage 4 多级去重 (MD5 0重复，4近似候选悬置)
- [x] Stage 5.1 语言过滤 (Whisper，5首有人声)
- [x] Stage 5.2 歌词转写 (Demucs+Whisper，5首)
- [x] Stage 5.3 风格聚类 (MERT 768d嵌入+DBSCAN)
- [x] Stage 6 预处理输出 (862切片+862特征)

### 预标注 L1-L4
- [x] L1 物理标签 (BPM/调性/SNR/LUFS，27首)
- [x] L2 语义候选 (MERT嵌入+规则推断，27首)
- [x] L3 结构标注 (Qwen-Omni多模态，2首黄金集+2首DeepSeek文本)
- [x] L4 传播融合 (DeepSeek全量+KNN传播，27首)
- [x] ls_preannotations.jsonl (27条)

## 四、P2 待完成事项

### 4.1 500首全量扩展
- [ ] 采集500首Jazz音频(MTG-Jamendo为主)
- [ ] 按修正后架构完整跑Stage 0-6
- [ ] QC Gate阈值微调(当前27首0 pass，需放宽)
- [ ] L3黄金集扩展到25首(5%)
- [ ] L4全量DeepSeek+KNN传播

### 4.2 人工标注
- [ ] 启动Label Studio本地部署
- [ ] 导入ls_preannotations.jsonl
- [ ] 配置reviewer角色+审核工作流
- [ ] 人工校验27首(标注一致性IAA统计)
- [ ] 疑难样本用Sonic Visualiser精细分析

### 4.3 数据集划分
- [ ] split_dataset.py来源隔离版(检查现有脚本)
- [ ] main_pool/test_pool/holdout_pool/ood_pool目录创建
- [ ] 分层抽样(genre/BPM/响度/时长/人声)
- [ ] 类别重加权计算
- [ ] 生成splits/目录(train.csv/val.csv/test.csv/holdout.csv/ood.csv)

### 4.4 模型训练
- [ ] Pretrain: MERT masked modeling + CLAP对比学习
- [ ] SFT: 流派分类+情绪分类+乐器检测+Caption生成
- [ ] DPO: Caption偏好对齐(待人工偏好标注)
- [ ] OOD评估(MAESTRO/Ballroom)
- [ ] 基线模型对比

### 4.5 环境固化
- [ ] AutoDL自定义镜像保存(清理→关机→控制台保存)
- [ ] Dockerfile.gpu头部添加注释(AutoDL不支持DinD)
- [ ] .env.example更新(新增Qwen-Omni配置)
- [ ] requirements.txt/conda环境锁定

## 五、P3 求职作品集

- [ ] README.md重写(项目概述+架构图+成果展示)
- [ ] WorkBuddy使用体验文章(凯子鱼建议，应聘附件)
- [ ] 项目演示视频/截图
- [ ] 技术博客(音乐AI数据集构建实践)
- [ ] 简历项目描述更新

## 六、关键决策记录 (ADR)

| # | 决策 | 理由 | 日期 |
|---|------|------|------|
| 1 | ULID作主键+track_slug作可读标识 | 工业标准，排序友好 | 2026-08-20 |
| 2 | YAMNet has_vocals是Demucs唯一触发器 | Jazz场景librosa粗筛误判萨克斯/钢琴 | 2026-08-23 |
| 3 | 入库阈值1200秒(20分钟) | 爵士长即兴/古典乐章常见 | 2026-08-24 |
| 4 | L3用Qwen-Omni多模态直接输入音频 | DeepSeek文本API只能推断，无法听音频 | 2026-08-24 |
| 5 | L4 DeepSeek全量+KNN传播融合 | 成本分层：5%多模态高成本+95%文本低成本 | 2026-08-25 |
| 6 | MD5精确去重移到Stage0 | 避免重复文件浪费Stage2/3 GPU算力 | 2026-08-25 |
| 7 | Stage2先记原始指标再loudnorm | SNR/DR不受增益影响，质检用原始值 | 2026-08-25 |
| 8 | L1物理标签在Stage6切片前提取 | BPM/调性需要整首音频全局信息 | 2026-08-25 |
| 9 | 增强策略初期禁用 | 保持数据原始性，待baseline后逐步添加 | 2026-08-25 |
| 10 | OOD池5%独立采集 | 测泛化能力，不参与训练验证 | 2026-08-25 |

## 七、已知问题与风险

| # | 问题 | 状态 | 影响 |
|---|------|------|------|
| 1 | CLAP嵌入提取全部失败(format错误) | 待修复 | L2 zero-shot分类不可用 |
| 2 | QC Gate 阈值偏严(27首0 pass/21 marginal/6 fail) | 500首全量前需根据分布微调 | 避免marginal队列膨胀导致人工审核压力爆炸 |
| 3 | Stage4 4个近似重复候选未验证 | 悬置 | 可能有重复数据 |
| 4 | 第6首2秒超短bad样本仍在下游 | 待排除 | 影响聚类和训练切片 |
| 5 | Mac无法科学上网 | iOS方案替代 | Gemini/GPT需iOS端调用 |
| 6 | AutoDL DinD不可行 | 自定义镜像替代 | 环境固化用保存镜像 |
| 7 | openai库Mac依赖冲突 | requests替代 | Qwen-Omni用requests直接调用 |
| 8 | L3段落标签中英文混合 | 待统一 | Qwen-Omni输出中文，设计用英文 |

## 八、API Key 与环境

> 注意: .env已在.gitignore中，以下仅记录key名称不记录值

| 服务 | 环境变量 | 用途 | 状态 |
|------|----------|------|------|
| DeepSeek | DEEPSEEK_API_KEY | L4全量文本标签 | ✅ 可用 |
| Qwen-Omni | DASHSCOPE_API_KEY | L3多模态结构标注 | ✅ 可用 |
| Qwen Workspace | QWEN_WORKSPACE_ID | ws-799rqc6h5qsqmorr | ✅ 已配置 |
| 旧DashScope | — | 已失效(InvalidApiKey) | ❌ 废弃 |

## 九、GPU 环境

- AutoDL RTX 4090 24GB
- SSH: ssh -p 49530 root@connect.westb.seetacloud.com
- conda: /root/miniconda3/envs/labelstudio-env (Python 3.11)
- 模型权重: /root/autodl-tmp/models/
- 项目: /root/autodl-tmp/music-ai-dataset/

## 十、Mac 环境

- 项目: /Users/m.jian/music_corpus_project/
- conda: /opt/miniconda3/envs/labelstudio-env (Python 3.11)
- yamnet_env: /opt/miniconda3/envs/yamnet_env (TF 2.15)
- 已装: plotly, scikit-learn, umap-learn, librosa
