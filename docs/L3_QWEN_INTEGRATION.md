# L3 Qwen-Omni 与 V4 框架对接规范

> 版本: v1.0 | 更新: 2026-08-25
> 模型: Qwen3.5-Omni-Flash (多模态, 国内直连)
> 角色: L3 结构标注层, 5% 黄金集

---

## 一、核心结论

| 维度 | 结论 |
|------|------|
| 数据结构适配 | ✅ 完全兼容（时间区间 + 多选标签 + Caption） |
| 标签词汇适配 | ⚠️ 需要后处理映射层（中文→英文/GM128/VAD） |
| 黄金集人工审核 | ✅ 必须，Qwen 只是预填候选 |
| Label Studio 模板 | ✅ 同一套 V4 模板，通过预标注 JSON 控制预填内容 |
| 按数据集切换模板 | ❌ 不需要，所有音乐标注字段体系统一 |

---

## 二、Qwen-Omni 输出格式

### 2.1 结构标注表格

```
时间        段落        乐器                          情绪
0-13.8s     前奏        钢琴 / 贝斯 / 架子鼓          平静
13.8-37.6s  主题呈示    萨克斯 / 钢琴 / 贝斯 / 架子鼓  欢快
37.6-56.6s  即兴独奏    萨克斯 / 钢琴 / 贝斯 / 架子鼓  欢快
...
243.8-259.6s 尾奏        萨克斯 / 钢琴 / 贝斯 / 架子鼓  欢快
```

### 2.2 Caption

> "这是一首典型的爵士摇摆乐（Swing Jazz）。乐曲由钢琴、贝斯和架子鼓组成的 rhythm section 开场，随后萨克斯管奏出明快活泼的主旋律..."

### 2.3 JSON 输出格式

```json
{
  "audio_id": "01M0RFPVNJ21SF5Y7VHPP8QV6T",
  "segments": [
    {
      "start": 0.0,
      "end": 13.8,
      "paragraph": "前奏",
      "instruments": ["钢琴", "贝斯", "架子鼓"],
      "emotion": "平静"
    }
  ],
  "caption": "这是一首典型的爵士摇摆乐...",
  "confidence": 0.85
}
```

---

## 三、后处理映射层

### 3.1 段落标签映射

| Qwen 输出 (中文) | V4 模板 (英文) | 说明 |
|-------------------|-----------------|------|
| 前奏 | Intro | |
| 主题呈示 | Verse | 或 Chorus，取决于音乐理论定义 |
| 即兴独奏 | Solo | |
| 尾奏 | Outro | |
| 间奏 | Bridge | |
| 未知段落 | Unknown | V4 模板新增兜底标签 |

### 3.2 乐器名称映射（中文 → GM128）

| 中文 | 英文 | GM128 |
|------|------|-------|
| 钢琴 | piano | GM001 |
| 木吉他 | acoustic guitar | GM025 |
| 电吉他 | electric guitar | GM027 |
| 贝斯 | bass | GM033 |
| 架子鼓 | drums | GM118 |
| 萨克斯 | saxophone | GM065 |
| 小号 | trumpet | GM056 |
| 长号 | trombone | GM057 |
| 弦乐 | strings | GM048 |
| 小提琴 | violin | GM040 |
| 大提琴 | cello | GM042 |
| 长笛 | flute | GM073 |
| 合成器 | synth | GM098 |
| 人声演唱 | vocals | GM091 |

> 映射字典 `label_mapping_dict.json` 的 `instrument_gm128_map` 同时支持中英文 key。

### 3.3 情绪标签映射（中文 → V4 mood + VAD）

| Qwen 输出 | V4 mood 选项 | VAD (V,A,D) |
|-----------|-------------|-------------|
| 平静 | 温柔舒缓 Calm | (0.5, 0.2, 0.5) |
| 舒缓 | 温柔舒缓 Calm | (0.6, 0.2, 0.3) |
| 欢快 | 欢快活泼 Joyful | (0.8, 0.7, 0.6) |
| 激昂 | 激昂热血 Intense | (0.7, 0.9, 0.8) |
| 紧张 | 紧张悬疑 Tense | (0.3, 0.8, 0.7) |
| 神秘 | 神秘 Mysterious | (0.5, 0.5, 0.4) |
| 忧郁 | 忧郁伤感 Melancholic | (0.2, 0.3, 0.3) |
| 浪漫 | 浪漫甜蜜 Romantic | (0.75, 0.4, 0.45) |

---

## 四、V4 模板最小修改

在 `structure` Labels 中新增 `Unknown` 作为兜底：

```xml
<Labels name="structure" toName="audio_source" choice="multiple" showInline="true">
    <Label value="Intro" background="#7dd3fc"/>
    <Label value="Verse" background="#fde047"/>
    ...
    <Label value="Silence" background="#e5e7eb"/>
    <Label value="Unknown" background="#d1d5db"/>  <!-- 新增：Qwen 未知段落兜底 -->
</Labels>
```

---

## 五、黄金集人工审核工作流

### 5.1 黄金集必须人工精标

Qwen-Omni 只是预填候选，不是最终真值。直接拿 Qwen 输出当黄金集会导致 KNN 传播把错误放大到全量。

### 5.2 审核重点（与普通样本不同）

| 审核项 | 容差 | 说明 |
|--------|------|------|
| 段落边界精度 | ±2s | Qwen 的 13.8s 切换点是否准确 |
| 乐器区间遗漏 | — | Qwen 是否漏了某段隐藏的钢琴 |
| 情绪一致性 | — | 同一乐段内情绪是否突变（可能是幻觉） |
| Caption 准确性 | — | 是否有幻觉或技术参数罗列 |

### 5.3 审核后黄金集标准

```
review_decision = "approve" 或 "approve_with_edits"
golden_set = "yes 加入黄金集"
review_flag = "golden_standard"
```

修正后的 structure 区间成为 KNN 传播的真值种子。

### 5.4 同一套 V4 模板

| 样本类型 | 预标注 JSON 表现 | 模板表现 |
|----------|-----------------|----------|
| 普通样本 | structure 为空 | 结构段落区空白，人工从零标注 |
| 黄金集 | structure 预填 Qwen 结果 | 结构段落区预填，人工修正边界 |
| 边际样本 | meta.marginal_display="block" | 顶部黄色警告横幅 |
| KNN 传播 | meta.propagation_source + knn_sim | 元数据区显示传播来源+相似度 |

核心原则：一个模板覆盖所有样本类型，通过预标注 JSON 控制"预填多少"。

---

## 六、L4 KNN 传播参数

### 6.1 量化阈值

| 字段 | 传播条件 | cosine_dist | cosine_sim | gold_confidence |
|------|----------|-------------|------------|-----------------|
| genre | 稳定字段，放宽 | < 0.40 | > 0.60 | ≥ medium |
| mood | 主观，严格 | < 0.25 | > 0.75 | = high |
| instruments | 存在性，严格 | < 0.25 | > 0.75 | = high |
| caption | ❌ 不传播 | — | — | — |

> cosine_dist = 1 - cosine_sim。对外统一用 cosine distance（越小越相似）。

### 6.2 阈值常量（文件顶部，不硬编码）

```python
# scripts/02_preannotation/l4_propagated/l4_knn_propagation.py
DIST_THRESHOLD_GENRE = 0.40
DIST_THRESHOLD_MOOD = 0.25
DIST_THRESHOLD_INSTRUMENTS = 0.25
GOLD_CONFIDENCE_GENRE = {"high", "medium"}
GOLD_CONFIDENCE_MOOD = {"high"}
GOLD_CONFIDENCE_INSTRUMENTS = {"high"}
```

500 首全量时如果传播率太低，直接改常量重跑，不用翻逻辑。

---

## 七、完整数据流

```
Qwen-Omni API (多模态, 5%黄金集)
    ↓ 输出: 中文段落/乐器/情绪 + Caption
l3_qwen_postprocess.py
    ├── 段落: 前奏→Intro, 主题呈示→Verse, 即兴独奏→Solo, 未知→Unknown
    ├── 乐器: 钢琴→GM001, 萨克斯→GM065 (映射字典中文key)
    ├── 情绪: 平静→温柔舒缓 + VAD(0.5,0.2,0.5)
    └── Caption: 直接填入
    ↓
V4 预标注 JSON (含 structure/instruments/mood/caption 区间)
    ↓
Label Studio V4 模板 (同一套, 黄金集显示🌟横幅)
    ├── 人工修正段落边界 (±2s容差)
    ├── 人工确认/补充乐器
    ├── 人工修正情绪和Caption
    └── 审核决策: approve / approve_with_edits
    ↓
黄金集 (人工精标, review_flag=golden_standard)
    ↓
L4 KNN 传播 (genre<0.4, mood/instruments<0.25, caption不传播)
    ↓
全量预标注 → 人工校验 → 训练数据集
```

---

## 八、面试表述

> "L3 用 Qwen3.5-Omni-Flash 多模态模型对 5% 黄金集做结构标注，输出中文段落/乐器/情绪表格。后处理脚本将中文映射为 V4 标准（段落→英文标签，乐器→GM128，情绪→V4选项+VAD三元组），未知段落用 Unknown 兜底。黄金集必须人工精标，Qwen 只是预填——结构边界允许 ±2s 容差，乐器和情绪需要确认。所有样本用同一套 V4 模板，通过预标注 JSON 控制预填内容，不按数据集切换模板。L4 KNN 传播用量化阈值：genre cosine_dist < 0.4，mood/instruments < 0.25，caption 不传播。阈值抽成文件顶部常量，500 首全量时可根据传播率微调。"
