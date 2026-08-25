# ADR-004: L1-L4 预标注分层架构

**状态**: Accepted  
**日期**: 2026-08-26  
**决策者**: 数据工程团队  
**影响范围**: data/02_preannotation/l1_physical/, l2_semantic/, l3_structural/, l4_propagated/

---

## 背景

音乐数据集的预标注不是"跑一个模型出标签"那么简单，而是一个分层工程体系。不同层级的标签需要不同的模型、不同的输入、不同的置信度评估方式。

本项目设计了 L1-L4 四层预标注架构，从底层物理特征到高层语义标签逐层递进，每层有明确的输入输出契约和置信度评估。

---

## 决策

### 1. 四层架构总览

| 层级 | 名称 | 核心任务 | 输入 | 输出 | 模型/方法 |
|------|------|---------|------|------|----------|
| L1 | Physical 物理特征 | 提取可复现的底层声学特征 | 原始音频 | bpm, key, loudness, snr, spectral_centroid, zero_crossing_rate | librosa / essentia 规则计算 |
| L2 | Semantic 语义标签 | 基于嵌入的风格/情绪分类 | L1 特征 + 音频嵌入 | genre_candidates, mood_candidates, vocal_presence | MERT 嵌入 + 规则/分类器 |
| L3 | Structural 结构分段 | 时间轴上的乐段/乐器/情绪标注 | 音频 + L1/L2 特征 | segments: start, end, label, instruments, emotion, confidence, reasoning | 结构分析模型 + LLM 推理 |
| L4 | Propagated 传播标签 | KNN 传播 + 人工校验后的完整标签 | L1-L3 + 黄金集 + KNN 邻居 | genre, subgenre, mood, instrumentation, vocal_presence, quality_assessment, caption, segments | KNN 传播 + 映射字典 + LLM Caption |

---

### 2. 各层输入输出契约

#### L1 Physical（物理特征）

**输入**：原始音频文件（mp3/wav/flac）

**输出**（`{audio_id}_physical.json`）：
```json
{
  "audio_id": "xxx",
  "duration_sec": 210.39,
  "bpm": 103.4,
  "key": "D",
  "loudness_db": -17.78,
  "snr_db": 32.1,
  "spectral_centroid_hz": 937.8,
  "zero_crossing_rate": 0.0349
}
```

**特点**：
- 完全可复现，相同音频每次计算结果一致
- 不依赖模型，纯规则/信号处理
- 是 L2-L4 的基础输入
- 置信度：100%（计算确定性）

#### L2 Semantic（语义标签）

**输入**：L1 物理特征 + MERT 音频嵌入

**输出**（`{audio_id}_semantic.json`）：
```json
{
  "audio_id": "xxx",
  "genre_candidates": [
    {"label": "Jazz", "confidence": 0.85},
    {"label": "Blues", "confidence": 0.15}
  ],
  "mood_candidates": [
    {"label": "melancholic", "confidence": 0.7}
  ],
  "vocal_presence": "instrumental",
  "source": "mert_embedding + l1_rules"
}
```

**特点**：
- 基于嵌入的分类，输出候选列表而非单一标签
- 每个候选带置信度
- vocal_presence 是三分类：instrumental / vocal / unknown
- 置信度：模型输出，需人工校验

#### L3 Structural（结构分段）

**输入**：音频 + L1 物理特征 + L2 语义标签

**输出**（`{audio_id}_structure.json`）：
```json
{
  "audio_id": "xxx",
  "segments": [
    {
      "start": 0.0,
      "end": 42.35,
      "label": "即兴独奏",
      "instruments": ["电吉他"],
      "emotion": "平静",
      "confidence": 0.95,
      "reasoning": "单把电吉他以蓝调风格进行缓慢、抒情的独奏..."
    }
  ]
}
```

**特点**：
- 时间轴上的分段标注，每段有独立的标签/乐器/情绪
- 带 reasoning 字段，说明为什么这么分段
- 置信度：分段级别的模型置信度
- LLM 参与推理，需人工校验

#### L4 Propagated（传播标签）

**输入**：L1-L3 预标注 + 黄金集（人工标注） + KNN 邻居标签

**输出**（`{audio_id}_full_tags.json`）：
```json
{
  "audio_id": "xxx",
  "genre": "jazz",
  "subgenre": "smooth jazz",
  "mood": ["舒缓", "怀旧"],
  "instrumentation": ["电吉他"],
  "vocal_presence": "instrumental",
  "quality_assessment": "good",
  "caption": "这是一段纯粹的电吉他爵士乐独奏...",
  "segments": [...]
}
```

**特点**：
- 最终完整标签，用于训练和评测
- KNN 传播：基于音频嵌入相似度，从黄金集传播标签
- 映射字典：将原始标签映射到标准标签体系
- LLM Caption：生成自然语言描述
- 置信度：传播来源 + 邻居相似度 + 人工校验状态

---

### 3. 层级间数据流

```
原始音频
    ↓
L1 Physical（librosa/essentia 规则计算）
    ↓ 输出：bpm, key, loudness, snr, ...
L2 Semantic（MERT 嵌入 + 分类器）
    ↓ 输出：genre_candidates, mood_candidates, vocal_presence
L3 Structural（结构分析 + LLM 推理）
    ↓ 输出：segments（时间轴分段）
L4 Propagated（KNN 传播 + 映射字典 + LLM Caption）
    ↓ 输出：完整标签（genre, subgenre, mood, instrumentation, caption, ...）
    ↓
人工校验（Label Studio）
    ↓
黄金集 / 训练集
```

---

### 4. 置信度与人工校验策略

| 层级 | 置信度来源 | 校验策略 | 校验比例 |
|------|-----------|---------|---------|
| L1 | 计算确定性（100%） | 不校验 | 0% |
| L2 | 模型分类置信度 | 低置信度（<0.6）样本人工校验 | 10-20% |
| L3 | 分段模型置信度 + LLM reasoning | 全量人工校验（分段错误影响大） | 100%（抽检） |
| L4 | KNN 传播来源 + 邻居相似度 | 低相似度（<0.7）传播样本人工校验 | 20-30% |

**关键原则**：
- L1 不校验（确定性计算）
- L2/L4 按置信度阈值抽检
- L3 全量抽检（分段错误会传播到 caption 和训练标签）

---

### 5. KNN 传播机制（L4 核心）

**传播条件**：
- 音频嵌入与黄金集样本的余弦相似度 > 阈值（默认 0.7）
- 邻居数量 ≥ 3（避免单邻居误传播）
- 邻居标签一致性 > 60%（避免标签矛盾）

**传播来源标记**：
```json
{
  "propagation_source": "knn",
  "nearest_neighbors": [
    {"audio_id": "golden_xxx", "similarity": 0.85, "genre": "jazz"},
    {"audio_id": "golden_yyy", "similarity": 0.82, "genre": "jazz"},
    {"audio_id": "golden_zzz", "similarity": 0.78, "genre": "jazz"}
  ],
  "propagation_confidence": 0.82
}
```

**不传播的情况**：
- 相似度 < 0.7 → 标记为 unmapped，进入人工审核队列
- 邻居标签矛盾 → 标记为 disputed，进入人工审核队列
- 邻居数量 < 3 → 标记为 low_confidence，进入人工审核队列

---

### 6. 标签映射字典

L4 传播后的原始标签需要通过映射字典转换为标准标签体系：

```json
{
  "genre_mapping": {
    "Jazz": "jazz",
    "jazz": "jazz",
    "smooth jazz": "jazz",
    "bebop": "jazz",
    "Blues": "blues",
    "blues": "blues"
  },
  "mood_mapping": {
    "melancholic": "忧郁",
    "舒缓": "舒缓",
    "calm": "平静",
    "平静": "平静"
  },
  "blacklist_tags": ["noise", "speech", "silence", "low quality", "distorted"]
}
```

**映射类型枚举**（merge_mapping.py 只认 mapping_type，不用字符串内容猜测）：
- `genre`：流派映射
- `mood`：情绪映射
- `instrument`：乐器映射
- `blacklist`：黑名单标签

---

### 7. 岗位边界

- **数据岗**：维护 L1-L4 预标注流水线，KNN 传播参数调优，映射字典维护，置信度阈值设置。
- **标注岗**：人工校验 L2/L3/L4 的低置信度样本，维护黄金集。
- **评测岗**：使用预标注标签做模型评测，评估预标注准确率。

---

## 参考

- [MERT: 音乐音频表示学习](https://arxiv.org/abs/2306.00107)
- [KNN 分类器](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [音乐结构分析](https://arxiv.org/abs/2107.00107)
- [Label Studio 标注平台](https://labelstud.io/)
