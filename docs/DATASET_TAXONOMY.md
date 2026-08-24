# 数据集分类体系 (Dataset Taxonomy)

> 版本: v1.0.0 | 更新: 2026-08-25

## 一、核心四划分（空间维度）

| 子集 | 用途 | 能否反复看 | 来源 | 冻结策略 |
|------|------|-----------|------|----------|
| **Train** | 模型参数学习 | ✅ 反复训练 | main_pool | 不冻结，持续追加 |
| **Val** | 调参、早停、选模型 | ✅ 反复验证 | main_pool | 不冻结，随train版本更新 |
| **Test** | 最终性能评估 | ❌ 只跑一次 | **test_pool 独立采集** | 版本冻结，不参与调参 |
| **Hold-out** | 跨版本对比、论文发表 | ❌ 全程冻结 | **holdout_pool 独立采集** | 永久冻结，不调参不偷看 |

> Test 和 Hold-out 在工业界常是同一个东西的不同叫法：Hold-out 强调"冻结"，Test 强调"评估"。

### 来源隔离原则

```
data/00_raw_collect/
├── main_pool/          # 训练+验证，持续追加
├── test_pool/          # 测试集，独立采集，冻结
└── holdout_pool/       # 最终评估集，长期封存
```

- **test_pool**: 与 main_pool 不同采集批次/不同来源，确保分布独立
- **holdout_pool**: 从项目初期就封存，不参与任何调参决策
- **禁止**: 从 main_pool 随机切分作为 test/holdout（会导致分布泄漏）

## 二、功能标签（从空间中抽取）

> 核心四划分是"空间"，黄金集/OOD/badcase 是"功能标签"（从空间里抽出来做特定用途）。不要混为一谈。

### 黄金集 Golden Set

- **不是与 train/val/test 并列的"第四划分"，也不是独立数据池**
- **双重身份**：物理上存在于 main_pool 中，从 main_pool 抽样约5%（50–500条）人工精标
- **划分时不做特殊处理**：它可能进 train，也可能进 val，取决于随机划分，不强制锁在 train 里
- 用途: few-shot 示例、prompt 校准、Reward Model 训练、IAA 基准、KNN传播种子
- **本项目**: L3 Qwen-Omni 的 5% 采样 = 黄金集
- 存储: `data/03_human_annotation/reviewed_good/`
- **面试表述**: "黄金集物理上在 main_pool 里，但划分时正常分布，我们不特殊处理它的归属"

### OOD 集 Out-of-Distribution

- **不是独立划分，是 Test 的特殊形态**
- 定义: 与训练数据风格/来源完全不同的测试集
- 用途: 考验泛化能力（Jazz 训练 → Classical 测试）
- **本项目**: 可用 MAESTRO（钢琴古典）或 Ballroom（标准舞曲）做 Jazz 模型的 OOD 测试
- 存储: `data/00_raw_collect/ood_pool/`

### Badcase 集

- 从 Train/Val 中抽取的质检失败/RM 低分样本
- 用途: DPO rejected、错误分析、主动学习
- 存储: `data/03_human_annotation/badcase/`

### Marginal 集

- confidence 低/标注分歧大的样本
- 用途: 主动学习候选、人工重点审核
- 存储: `data/03_human_annotation/marginal/`

## 三、Train 内部的子集（很多人漏掉）

```
Train (main_pool)
├── Pretrain 子集: 无监督/自监督
│   ├── CLAP 对比学习（audio-text配对）
│   ├── MERT masked modeling
│   └── 数据: 全量无标注音频
├── SFT 子集: 监督微调
│   ├── 单样本: prompt + audio + label
│   └── 数据: 有标注的子集（L4预标注+人工校验）
├── DPO/RLHF 子集: 成对偏好
│   ├── prompt + chosen + rejected
│   └── 数据: 人工偏好标注 + badcase自动生成
├── Golden 子集: 从上面抽样的精标小集合（5%）
│   └── 用途: few-shot/prompt校准/KNN种子
├── Badcase 子集: 质检失败/RM低分
│   └── 用途: DPO rejected/错误分析
└── Marginal 子集: confidence低/标注分歧大
    └── 用途: 主动学习候选
```

## 四、样本均衡策略

### 4.1 分层抽样（Stratified Sampling）

划分 train/val/test 时，按以下维度分层：

| 维度 | 分层方式 | 目的 |
|------|----------|------|
| genre | 按流派分层 | 避免某流派全在test |
| BPM | 分档: <80/80-120/120-160/>160 | 节奏分布均衡 |
| 响度 | 分档: 低/中/高 | 避免test全是安静曲目 |
| 时长 | 分档: 短/中/长 | 结构分布均衡 |
| vocal_presence | instrumental/vocal/mixed | 人声分布均衡 |
| 来源 | 按采集批次/网站 | 来源隔离 |

### 4.2 类别重加权（Class Re-weighting）

训练时对低频类别上采样或加权：

```python
# 计算类别权重（逆频率）
class_weights = {
    genre: total_count / (num_classes * genre_count)
    for genre, genre_count in genre_distribution.items()
}

# 损失函数加权
loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)
```

### 4.3 数据增强（Label-Aware Augmentation）

| 增强方式 | label_invariant | 适用阶段 | 注意事项 |
|----------|----------------|----------|----------|
| add_noise | ✅ true | 全阶段 | 不改变任何标签 |
| time_stretch | ❌ false | pretrain_only | 改变BPM，需同步改标签 |
| pitch_shift | ❌ false | pretrain_only | 改变调性，需同步改key |
| spec_augment | ✅ true | 全阶段 | 频谱掩码，不改变标签 |
| mixup | ⚠️ partial | pretrain_only | 标签需混合加权 |

### 4.4 主动学习（Active Learning）

对 Marginal 集（低置信度样本）优先人工标注：

1. 模型预测 confidence < 0.6 的样本 → 加入 Marginal 池
2. 标注者间一致性（IAA）< 0.7 的样本 → 加入 Marginal 池
3. 每次迭代从 Marginal 池抽样标注 → 模型重训 → 重新评估

## 五、数据集统计评测

### 5.1 分布统计

每次数据集版本发布时，自动生成统计报告：

| 统计维度 | 指标 | 可视化 |
|----------|------|--------|
| 流派分布 | 各类别数量/占比 | 柱状图/饼图 |
| BPM分布 | 均值/中位数/分位数/直方图 | 直方图+箱线图 |
| 调性分布 | 12调性+大小调占比 | 环形图 |
| 响度分布 | LUFS均值/分位数 | 直方图 |
| 时长分布 | 均值/分位数/超长占比 | 直方图 |
| 嵌入分布 | MERT t-SNE/UMAP降维 | 散点图(按流派着色) |
| 人声分布 | instrumental/vocal/mixed占比 | 饼图 |
| 来源分布 | 各采集网站/批次占比 | 堆叠柱状图 |

### 5.2 质量统计

| 统计项 | 指标 | 阈值告警 |
|--------|------|----------|
| 坏样本率 | fail占比 | >5% 告警 |
| 边际样本率 | marginal占比 | >15% 告警 |
| 重复率 | 精确+近似重复占比 | >2% 告警 |
| 损坏率 | 解码失败占比 | >1% 告警 |
| 静音率 | silence_ratio>0.8占比 | >5% 告警 |

### 5.3 标注统计

| 统计项 | 指标 | 说明 |
|--------|------|------|
| 预标注覆盖率 | L4标签覆盖比例 | 目标100% |
| 人工校验率 | 已人工标注/总量 | 目标≥20% |
| 标注一致性(IAA) | Cohen's Kappa / Krippendorff's Alpha | 目标≥0.7 |
| 置信度分布 | low/medium/high占比 | high应>60% |
| 标签来源分布 | deepseek/knn/golden/l1占比 | 审计用 |
| KNN传播率 | 有KNN邻居的样本占比 | 目标>80% |

### 5.4 OOD 统计

| 统计项 | 指标 | 说明 |
|--------|------|------|
| OOD占比 | OOD样本/总测试集 | 建议10-20% |
| 分布距离 | train vs OOD 的 MMD/KS检验 | 量化分布差异 |
| 性能下降 | in-domain vs OOD 的指标差 | 泛化能力评估 |

## 六、面试标准回答

> 被问到"数据集有哪些"时，标准回答顺序：

1. **先答核心四划分 + 来源隔离原则**
   - "Train/Val/Test/Hold-out，Test和Hold-out独立采集，不与Train同批次"

2. **再答 Train 内部的 pretrain/sft/dpo 分层**
   - "Train内部又分Pretrain（无监督）、SFT（监督微调）、DPO（偏好对齐）"

3. **最后答黄金集/badcase/ood 的功能定位**
   - "黄金集是从Train抽5%人工精标，做few-shot和KNN种子；OOD是Test的特殊形态，测泛化；badcase做DPO rejected"

4. **一句话区分**
   - "核心四划分是空间，黄金集/OOD/badcase是功能标签，不要混为一谈"
