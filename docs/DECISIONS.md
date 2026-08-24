# 关键决策记录 (Architecture Decision Records)

> 版本: v1.0.0 | 更新: 2026-08-25
> 记录项目中10条关键技术决策的背景、选项、理由和后果，面试时可直接引用。

---

## ADR-001: ULID 作主键 + track_slug 作可读标识

**日期**: 2026-08-20
**状态**: 已采纳

### 背景
音频数据集需要唯一标识符，同时需要人类可读的名称用于文件命名和调试。

### 选项
1. 自增整数 ID — 简单但分布式环境冲突
2. UUID v4 — 唯一但无序，数据库索引性能差
3. **ULID (Universally Unique Lexicographically Sortable Identifier)** — 唯一+时间有序+26字符可读
4. 文件名哈希 — 依赖文件名，重命名后ID变化

### 决策
采用 ULID 作主键(audio_id)，同时生成 track_slug(artist_title_slug) 作可读标识。

### 理由
- ULID 按时间排序，散列存储时目录顺序合理
- 26字符 Crockford Base32，比 UUID 短且无歧义字符
- 不依赖文件名，重命名/转码后ID不变
- track_slug 用于日志和调试，不用于主键

### 后果
- 所有表/文件以 audio_id 关联
- 散列存储用 audio_id 前2级作目录(如 `01/M0/01M0RFPVN7FD6R26.flac`)
- 面试加分点：工业级标识符设计

---

## ADR-002: YAMNet has_vocals 是 Demucs 唯一触发器

**日期**: 2026-08-23
**状态**: 已采纳

### 背景
Jazz 数据集中需要判断哪些音频有人声，以决定是否运行 Demucs 人声分离和 Whisper 转写。

### 选项
1. librosa 粗筛人声(RMS+频谱质心) — 快但误判率高
2. **YAMNet 分类(vocal_score > 0.1)** — 预训练模型，准确率高
3. 全部跑 Demucs — 准确但 GPU 算力浪费(纯器乐也跑)
4. 人工标注 — 准确但成本高

### 决策
仅对 YAMNet 判定 has_vocals=True 的样本运行 Demucs+Whisper，纯器乐跳过 Stage 5.1/5.2 直接进 5.3。

### 理由
- Jazz 场景中 librosa 粗筛会把萨克斯/钢琴高频误判为人声
- YAMNet 在 AudioSet 上预训练，人声/乐器区分准确率 >90%
- 27首中仅5首有人声，跳过22首节省 80% Demucs 算力
- 纯器乐直接进 5.3 风格聚类，不阻塞流水线

### 后果
- Stage 5.1/5.2 变为条件分支，非全量执行
- has_vocals=False 的样本无歌词字段
- 面试加分点：条件分支设计+算力优化

---

## ADR-003: 入库阈值 1200 秒(20分钟)

**日期**: 2026-08-24
**状态**: 已采纳

### 背景
音频入库时需要设定时长上限，避免超长 DJ-mix/现场录音占用过多存储和算力。

### 选项
1. 600秒(10分钟) — 保守，但会拒绝爵士长即兴/古典乐章
2. **1200秒(20分钟)** — 平衡，覆盖绝大多数音乐作品
3. 1800秒(30分钟) — 宽松，但 DJ-mix 也会进入
4. 无上限 — 简单但存储/算力不可控

### 决策
入库阈值设为 1200 秒(20分钟)，超过则拒绝并记录到 rejected/。

### 理由
- 爵士长即兴独奏常见 15-20 分钟，600秒会误拒
- 古典乐章(如贝多芬交响曲)单乐章可达 15-20 分钟
- DJ-mix 通常 >30 分钟，1200秒能有效过滤
- 被拒音频可通过 import_audio.py --force 重新入库

### 后果
- 27首中1首688秒被拒后重入(01M0RHT5QBG7MD9C)
- QC Gate 中 >900秒标记 long_form，>1800秒标记 dj_mix
- 面试加分点：领域知识驱动的阈值设计

---

## ADR-004: L3 用 Qwen-Omni 多模态直接输入音频

**日期**: 2026-08-24
**状态**: 已采纳

### 背景
L3 结构标注需要识别段落结构(Intro/Theme/Improv/Outro)、乐器、情绪和 Caption。

### 选项
1. **DeepSeek 文本 API(输入L1特征JSON)** — 便宜但无法听音频，结构靠推断
2. **Qwen3.5-Omni-Flash 多模态(直接输入音频)** — 能听音频，结构准确
3. Gemini 2.0 Flash — 质量高但需科学上网
4. GPT-4o — 质量最高但贵且需科学上网

### 决策
L3 默认用 Qwen3.5-Omni-Flash 多模态直接输入音频，仅对5%黄金集调用。高精度需求时在 iOS 端调用 Gemini/GPT-4o，结果回传 Mac。

### 理由
- DeepSeek 文本API只看 BPM/调性数字，无法判断段落边界和乐器
- Qwen-Omni 国内直连，20分钟音频上限覆盖绝大多数作品
- 5%黄金集(500首→25首)成本可控(~¥2.5)
- Mac 无法科学上网，iOS 端调用 Gemini/GPT-4o 是务实替代

### 后果
- L3 输出含精确时间戳的段落结构(如第一首4段落: 0-42.4s/42.4-133.8s/...)
- 乐器识别准确(萨克斯/钢琴/贝斯/架子鼓/电吉他)
- 音频需预处理: FLAC>8MB→转MP3 320kbps→仍>10MB→取前3分钟
- 面试加分点：多模态模型选型+成本分层+跨设备协作

---

## ADR-005: L4 DeepSeek 全量 + KNN 传播融合

**日期**: 2026-08-25
**状态**: 已采纳

### 背景
L4 需要为全量样本生成最终预标注标签，同时利用 L3 黄金集的高质量标注。

### 选项
1. 全量用 Qwen-Omni — 质量高但成本爆炸(500首×¥0.1=¥50)
2. 全量用 DeepSeek 文本 — 便宜但无结构信息
3. **DeepSeek 全量文本标签 + KNN 传播 L3 黄金集标签 + 规则融合** — 成本低+质量可控
4. 仅 KNN 传播 — 无黄金集的样本无标签

### 决策
L4 采用三层融合：
- DeepSeek V4 Flash 全量生成 genre/mood/instruments/caption(输入L1+L2特征JSON)
- KNN(cosine) 将 L3 黄金集的 mood/instruments 传播到相似样本
- 按字段差异化阈值融合：genre <0.4 传播，mood/instruments <0.25 传播，caption 不传播

### 理由
- DeepSeek 文本API ¥0.001/首，500首仅¥0.5
- KNN 传播利用 MERT 嵌入空间相似性，黄金集标签可覆盖全量
- Caption 不传播(每首独立生成，避免黄金集描述被错误传播)
- 差异化阈值：genre 稳定可放宽，mood/instruments 不稳定需严格

### 后果
- 27首中4首黄金集 + 23首KNN传播 + 0首DeepSeek-only
- 标签含 source 追溯(deepseek/knn/golden)，支持审计
- 面试加分点：成本分层+融合策略+可追溯设计

---

## ADR-006: MD5 精确去重移到 Stage 0

**日期**: 2026-08-25
**状态**: 已采纳

### 背景
原架构中 MD5 精确去重在 Stage 4，意味着重复文件已经经历了 Stage 2/3 的 GPU 处理，浪费算力。

### 选项
1. MD5 在 Stage 4 — 简单但浪费算力
2. **MD5 在 Stage 0(入库时)** — 重复文件不进下游，节省算力
3. 不做精确去重 — 简单但数据冗余

### 决策
MD5 精确去重移到 Stage 0 入库时执行，Stage 4 只保留近似去重(Chromaprint/chroma)。

### 理由
- MD5 不需要音频内容解码，只读文件哈希，秒级完成
- 重复文件在 Stage 2 ffmpeg 转码时浪费 GPU 时间
- 近似去重需要音频特征(chroma/指纹)，必须在 Stage 2 后执行
- 工业界标准做法：入库时精确去重，清洗时近似去重

### 后果
- Stage 0 增加 MD5 计算步骤
- Stage 4 简化为仅近似去重
- 面试加分点：算力优化+流水线位置设计

---

## ADR-007: Stage 2 先记原始指标再 loudnorm

**日期**: 2026-08-25
**状态**: 已采纳

### 背景
响度归一化(loudnorm)会整体改变音频增益，低电平噪声也会被同步放大，导致归一化后 SNR 数值失真。

### 选项
1. loudnorm 在 Stage 2，SNR 在 Stage 3(归一化后) — SNR 失真
2. **Stage 2 先计算原始指标(SNR/DR/LUFS)，再执行 loudnorm** — 质检用原始值
3. loudnorm 在 Stage 3 质检后 — 功能正确但位置分散
4. 不做 loudnorm — 简单但训练时响度不一致

### 决策
Stage 2 分两步：先计算并记录原始指标(orig_snr/orig_dr/orig_lufs)到 audio_manifest.csv，再执行 ffmpeg loudnorm 转码。Stage 3 质检使用原始指标。

### 理由
- SNR/DR 反映原始录音质量，不应受归一化影响
- 母版 FLAC 是归一化后的(用于训练和特征提取)，但元数据保留原始值
- 两步在同一 Stage 内完成，不增加流水线阶段数
- 面试加分点：信号处理细节+指标溯源设计

### 后果
- audio_manifest.csv 增加 orig_* 字段
- Stage 3 qc_gate.py 读取 orig_snr/orig_dr 而非归一化后的值
- 面试加分点：工程严谨性

---

## ADR-008: L1 物理标签在 Stage 6 切片前提取

**日期**: 2026-08-25
**状态**: 已采纳

### 背景
原架构中 L1 物理标签在 Stage 6 之后提取，但 BPM/调性需要整首音频的全局信息，切片后无法准确计算。

### 选项
1. L1 在 Stage 6 后(切片上提取) — BPM/调性不准确
2. **L1 在 Stage 5.3 与 Stage 6 之间(整首音频提取)** — 准确且不阻塞
3. L1 在 Stage 3 质检时顺便提取 — 位置早但与质检耦合

### 决策
L1 物理标签(BPM/调性/SNR/DR/LUFS/频谱质心)在 Stage 5.3b 聚类之后、Stage 6 切片之前提取，基于整首母版 FLAC。

### 理由
- BPM 计算需要整首音频的节拍周期性，15秒切片可能落在无节拍段落
- 调性识别需要整首的和声进行，切片可能转调
- L1 可在 GPU 跑 Stage 5.3a 嵌入的同时，Mac 端并行提取，不增加总时间
- L4 DeepSeek 生成标签时 L1 特征已就绪，不阻塞

### 后果
- L1 提取脚本在 Mac 端运行(librosa/madmom)
- Stage 6 切片仅用于训练特征(mel-spec/MFCC)，不用于 L1
- 面试加分点：全局vs局部特征区分+并行优化

---

## ADR-009: 增强策略初期禁用，渐进式启用

**日期**: 2026-08-25
**状态**: 已采纳

### 背景
数据增强(time_stretch/pitch_shift/add_noise/spec_augment)可以提升模型泛化，但部分增强会改变标签(BPM/调性)，且初期需要建立基线模型。

### 选项
1. 初期全部启用增强 — 泛化好但基线不可比
2. **初期禁用，待 baseline 建立后逐步启用** — 可控可对比
3. 仅启用 label_invariant 增强 — 安全但泛化提升有限

### 决策
recipe.json 中 augmentation.enabled=false，初期不做任何增强，保持数据原始性。待 baseline 模型建立后，根据效果逐步启用：先 add_noise/spec_augment(label_invariant)，再 time_stretch/pitch_shift(pretrain_only，需同步修改标签)。

### 理由
- 基线模型需要在原始数据上评估，增强会混淆效果归因
- time_stretch 改变 BPM，pitch_shift 改变调性，训练节奏/调性敏感任务时标签会错
- label_invariant 标记明确哪些增强安全，哪些需要同步改标签
- 渐进式启用符合工程实践，每步可对比

### 后果
- recipe.json 中所有增强策略 enabled=false
- 训练代码读取 recipe.json，enabled=false 时跳过增强
- 面试加分点：实验设计严谨性+标签感知增强

---

## ADR-010: OOD 池 5% 独立采集，不参与训练

**日期**: 2026-08-25
**状态**: 已采纳

### 背景
需要测试模型在分布外数据上的泛化能力，OOD(Out-of-Distribution)集应与训练数据风格差异较大。

### 选项
1. 从 main_pool 随机切分作 OOD — 简单但分布相同，无泛化测试意义
2. **单独建立 ood_pool，从风格差异大的来源采集，5%比例，不参与训练** — 真正测泛化
3. 不设 OOD 集 — 简单但无法评估泛化能力

### 决策
单独建立 ood_pool(5%比例)，从与 main_pool 风格差异较大的来源采集(不同年代/录音方式/子流派)，不参与训练和验证，仅用于最终泛化评估。候选数据集：MAESTRO(古典钢琴)、Ballroom(标准舞曲)、GTZAN(流派多样性)。

### 理由
- 从 main_pool 切分的"OOD"实际是同分布，无法测泛化
- 5%比例足够统计显著性，又不会占用过多采集资源
- 不参与训练/验证，确保评估结果无泄漏
- 候选数据集公开可用，无需自行采集

### 后果
- 数据池设计：main_pool / test_pool / holdout_pool / ood_pool 四池隔离
- recipe.json 中 ood.ood_pool="5%"，含候选数据集说明
- 评估时计算 in-domain vs OOD 的 accuracy_drop 和 calibration_error
- 面试加分点：泛化能力评估+数据隔离设计

---

## 决策索引

| ADR | 主题 | 日期 |
|-----|------|------|
| 001 | ULID作主键+track_slug | 2026-08-20 |
| 002 | YAMNet has_vocals是Demucs唯一触发器 | 2026-08-23 |
| 003 | 入库阈值1200秒(20分钟) | 2026-08-24 |
| 004 | L3用Qwen-Omni多模态直接输入音频 | 2026-08-24 |
| 005 | L4 DeepSeek全量+KNN传播融合 | 2026-08-25 |
| 006 | MD5精确去重移到Stage 0 | 2026-08-25 |
| 007 | Stage 2先记原始指标再loudnorm | 2026-08-25 |
| 008 | L1物理标签在Stage 6切片前提取 | 2026-08-25 |
| 009 | 增强策略初期禁用，渐进式启用 | 2026-08-25 |
| 010 | OOD池5%独立采集，不参与训练 | 2026-08-25 |
