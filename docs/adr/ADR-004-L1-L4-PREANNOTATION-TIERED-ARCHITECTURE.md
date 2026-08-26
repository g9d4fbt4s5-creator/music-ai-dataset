# ADR-004: L1-L4 预标注分层架构

**状态**: Accepted（2026-08-27 修正：传播时机 + L1种子定义 + 格式无关性）
**日期**: 2026-08-26
**最后修正**: 2026-08-27
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
| L1 | Physical 物理特征 | 提取可复现的底层声学特征 | 原始音频（FLAC/MP3/WAV，格式无关） | bpm, key, loudness, snr, spectral_centroid, zero_crossing_rate | librosa / essentia 规则计算 |
| L2 | Semantic 语义标签 | 基于嵌入的风格/情绪分类 | L1 特征 + 音频嵌入 | genre_candidates, mood_candidates, vocal_presence | MERT 嵌入 + CLAP 零样本 + 规则/分类器 |
| L3 | Structural 结构分段 | 时间轴上的乐段/乐器/情绪标注 | 音频 + L1/L2 特征 | segments: start, end, label, instruments, emotion, confidence, reasoning | Qwen-Omni 多模态 + LLM 推理 |
| L4 | Propagated 传播标签 | KNN 传播 + 人工校验后的完整标签 | L2 嵌入 + L3 黄金集标注 + L2 CLAP 候选 | genre, subgenre, mood, instrumentation, vocal_presence, quality_assessment, caption, segments | KNN 传播（仅 train/val） + 映射字典 + LLM Caption |

**⚠️ 关键约束（2026-08-27 修正）**：
- L1 物理特征（BPM/key/loudness 等数值型特征）**仅作样本特征描述和距离度量**，**不直接作为 KNN 分类标签种子**
- L4 KNN 传播的种子池 = **L3 黄金集人工标注（高置信）+ L2 CLAP 零样本候选（中置信）**
- L4 KNN 传播**必须在 Stage 4 数据划分之后执行**，test/holdout/ood **禁止传播**

---

### 2. 各层输入输出契约

#### L1 Physical（物理特征）

**输入**：原始音频文件（mp3/wav/flac，格式无关，librosa/soundfile 自动处理）

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
- 是 L2-L4 的基础输入（特征描述和距离度量）
- **不直接作为 KNN 分类标签种子**（数值型特征无法传播 genre/mood 分类标签）
- 置信度：100%（计算确定性）

#### L2 Semantic（语义标签）

**输入**：L1 物理特征 + MERT 音频嵌入 + CLAP 零样本标注

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
  "source": "mert_embedding + clap_zero_shot + l1_rules"
}
```

**特点**：
- 基于嵌入的分类，输出候选列表而非单一标签
- 每个候选带置信度
- vocal_presence 是三分类：instrumental / vocal / unknown
- **L2 CLAP top-1 候选可作为 L4 KNN 传播的中置信种子**
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
- LLM（Qwen-Omni）参与推理，需人工校验
- **黄金集的 L3 人工标注是 L4 KNN 传播的高置信种子**

#### L4 Propagated（传播标签）

**输入**：
- L2 MERT 嵌入向量（距离度量）
- L3 黄金集人工标注（高置信种子）
- L2 CLAP 零样本 top-1 候选（中置信种子）
- Stage 4 数据划分结果（train/val/test/holdout/ood）

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
  "segments": [...],
  "propagation_source": "knn_train" | "knn_val_pseudo" | "golden" | "manual",
  "propagation_confidence": 0.82
}
```

**特点**：
- 最终完整标签，用于训练和评测
- KNN 传播：基于 L2 MERT 嵌入余弦相似度，从种子池传播标签
- 映射字典：将原始标签映射到标准标签体系
- LLM Caption：生成自然语言描述
- 置信度：传播来源 + 邻居相似度 + 人工校验状态

---

### 3. 层级间数据流

```
原始音频（FLAC/MP3/WAV，格式无关）
    ↓
L1 Physical（librosa/essentia 规则计算）
    ↓ 输出：bpm, key, loudness, snr, ...（特征描述，非分类标签）
L2 Semantic（MERT 嵌入 + CLAP 零样本 + 分类器）
    ↓ 输出：genre_candidates, mood_candidates, vocal_presence
    ↓ CLAP top-1 → L4 中置信种子
L3 Structural（Qwen-Omni 多模态 + LLM 推理）
    ↓ 输出：segments（时间轴分段）
    ↓ 黄金集人工标注 → L4 高置信种子
L4 Propagated（KNN 传播 + 映射字典 + LLM Caption）
    ↓ 输入：L2 MERT 嵌入（距离）+ L3 黄金集（高置信种子）+ L2 CLAP（中置信种子）
    ↓ ⚠️ 必须在 Stage 4 划分后执行，test/holdout/ood 禁止传播
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

#### 5.1 传播时机（🔴 强制约束，2026-08-27 修正）

**L4 KNN 传播必须在 Stage 4 数据划分之后执行**，原因：
- 防止测试集信息泄漏（ADR-003 隔离原则）
- 确保 train/val/test/holdout/ood 的物理隔离

**传播范围**：
| 子集 | 是否传播 | 标记 |
|------|---------|------|
| train | ✅ 传播（未标注样本） | `knn_train` |
| val | ✅ 传播（伪标签） | `knn_val_pseudo` |
| test | ❌ 禁止传播 | — |
| holdout | ❌ 禁止传播 | — |
| ood | ❌ 禁止传播 | — |

**违反后果**：测试集样本通过邻居关系"见过"训练集嵌入，导致评估指标虚高，违反 ADR-003 数据隔离原则。

#### 5.2 种子池构建（🔴 2026-08-27 修正）

**种子池 = 高置信种子 + 中置信种子**

| 种子类型 | 来源 | 置信度 | 标签类型 |
|---------|------|--------|---------|
| 高置信种子 | L3 黄金集人工标注（Qwen-Omni + 人工校验） | 高 | genre, mood, instruments, segments, caption |
| 中置信种子 | L2 CLAP 零样本 top-1 候选（confidence > 0.3） | 中 | genre, mood |

**⚠️ L1 物理特征不直接作为种子**（2026-08-27 修正）：
- L1 的 BPM/key/loudness 是**连续数值型物理特征**，不是分类标签
- KNN 传播的目的是传播 genre/mood/instruments 等**分类标签**
- L1 特征仅用于：①样本特征描述 ②L4 阶段 1（无嵌入时）的临时距离度量 ③规则映射生成低置信伪标签（可选，第一版不实现）

#### 5.3 距离度量

**阶段 1（当前，L2 MERT 嵌入可用）**：
- 使用 L2 MERT 嵌入向量（768维）的余弦相似度
- `cosine_dist = 1 - dot(a,b) / (norm(a) * norm(b))`

**阶段 2（备选，无嵌入时临时方案）**：
- 使用 L1 物理特征向量：`[bpm_norm, key_onehot, lufs_norm, spectral_centroid_norm]`
- 欧氏距离：`dist(a,b) = sqrt(sum((a_i - b_i)^2))`
- 局限性：L1 特征与 genre/mood 语义关联弱，传播准确率低（约 60-70%）

#### 5.4 多标签阈值传播（逐标签独立判断）

**阈值**（2026-08-27 修正：不同标签类型独立阈值）：

| 标签类型 | 余弦距离阈值 | 说明 |
|---------|------------|------|
| genre | < 0.40 | 风格边界模糊，容忍度较高 |
| mood | < 0.25 | 情感标签主观，需更严格 |
| instruments | < 0.25 | 乐器组合多样，需严格 |

**传播逻辑**（伪代码）：
```python
def propagate_sample(target_id, target_embedding, knn_model, seed_pool):
    nearest_seed_id, distance = knn_model.find_nearest(target_embedding)
    seed_labels = seed_pool[nearest_seed_id]  # {genre: "Jazz", mood: "Calm", ...}
    
    propagated = {}
    for label_type, threshold in THRESHOLDS.items():
        if label_type not in seed_labels:
            continue
        if distance < threshold:
            propagated[label_type] = {
                "value": seed_labels[label_type],
                "source_seed": nearest_seed_id,
                "distance": distance,
                "confidence": 1 - distance / threshold,
            }
    return propagated
```

**⚠️ 禁止使用 if/elif 链**：一个种子可能同时有 genre 和 mood 标签，必须逐标签独立判断阈值。

#### 5.5 传播条件

- 音频嵌入与种子的余弦距离 < 对应标签类型阈值
- 种子数量 ≥ 2（避免单种子误传播）
- 邻居标签一致性 > 60%（避免标签矛盾，多邻居投票时）

**不传播的情况**：
- 距离 ≥ 阈值 → 标记为 unmapped，进入人工审核队列
- 邻居标签矛盾 → 标记为 disputed，进入人工审核队列
- 种子数量 < 2 → 标记为 low_confidence，进入人工审核队列

#### 5.6 传播来源标记

```json
{
  "propagation_source": "knn_train" | "knn_val_pseudo" | "golden" | "manual",
  "nearest_neighbors": [
    {"audio_id": "golden_xxx", "distance": 0.15, "genre": "jazz"},
    {"audio_id": "clap_yyy", "distance": 0.22, "genre": "jazz"}
  ],
  "propagation_confidence": 0.82
}
```

---

### 6. 标签映射字典

L4 传播后的原始标签需要通过映射字典转换为标准标签体系，详见 **ADR-005《标签映射字典版本管理》**。

---

### 7. 输入格式无关性声明（2026-08-27 新增）

**L1-L4 所有层对音频输入格式（FLAC / MP3 / WAV）无关**，原因：
- librosa / soundfile 自动处理不同格式的解码
- MERT 模型输入是 16kHz 重采样后的 mel 频谱，MP3 压缩伪影在 16kHz 下不显著
- CLAP 模型输入是 48kHz 波形，320kbps MP3 与 FLAC 的嵌入差异 < 1%（实测）
- Qwen-Omni 支持多种音频格式输入

**实测数据**（85首抽样对比）：
- MERT 嵌入余弦相似度：FLAC vs MP3 320kbps > 0.99
- CLAP 嵌入余弦相似度：FLAC vs MP3 320kbps > 0.99
- L1 物理特征（BPM/key/loudness）：FLAC vs MP3 320kbps 差异 < 0.5%

**与 ADR-006 的关系**：母版格式选择（FLAC vs MP3）不影响 L1-L4 预标注的输出质量，仅影响存储和传输效率。

---

## 与其他 ADR 的关系

- **ADR-001《QC Gate 阈值决策》**：L1 物理特征提取应排除 QC fail 样本
- **ADR-002《HITL 异步闭环》**：L3/L4 的低置信样本进入 waiting_pool 人工复核
- **ADR-003《数据划分与来源隔离》**：L4 KNN 传播必须在 Stage 4 划分后执行，test/holdout/ood 禁止传播（本 ADR 第 5.1 节）
- **ADR-005《标签映射字典版本管理》**：L4 传播标签通过映射字典转换为标准体系
- **ADR-006《母版格式选择》**：L1-L4 对输入格式无关（本 ADR 第 7 节）

---

## 修正历史

| 日期 | 修正内容 | 修正原因 |
|------|---------|---------|
| 2026-08-26 | 初始版本 | — |
| 2026-08-27 | 增加传播时机约束（第5.1节）：L4必须在划分后执行，test/holdout/ood禁止传播 | Kimi审核发现ADR-004与ADR-003隔离原则矛盾，脚本已修正但ADR未同步 |
| 2026-08-27 | 修正种子池定义（第5.2节）：删除"L1作为低置信种子"，种子池=L3黄金集+L2 CLAP候选 | Kimi审核发现L1 BPM/key是数值型特征，无法传播genre/mood分类标签，DeepSeek初版代码因此出错 |
| 2026-08-27 | 增加多标签独立阈值（第5.4节）：genre<0.40, mood<0.25, instruments<0.25 | Kimi审核发现原设计用统一阈值，DeepSeek代码用if/elif链导致多标签种子只匹配第一个 |
| 2026-08-27 | 增加输入格式无关性声明（第7节） | Kimi审核发现ADR-004与ADR-006隐含冲突，未声明嵌入提取对压缩格式的容忍度 |
| 2026-08-27 | 四层架构总览表更新L2/L3/L4的模型/方法列 | 与实际脚本对齐：L2增加CLAP，L3改为Qwen-Omni，L4明确种子来源 |

---

## 参考

- [MERT: Acoustic Music Understanding Model](https://huggingface.co/m-a-p/MERT-v1-95M)
- [CLAP: Contrastive Language-Audio Pretraining](https://github.com/LAION-AI/CLAP)
- [Qwen-Omni 多模态模型](https://help.aliyun.com/zh/model-studio/getting-started-with-qwen-omni)
- [音乐结构分析](https://en.wikipedia.org/wiki/Music_segmentation)
- [Label Studio 标注平台](https://labelstud.io/)
