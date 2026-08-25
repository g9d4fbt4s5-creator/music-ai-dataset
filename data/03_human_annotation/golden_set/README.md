# Golden Set（黄金集）

人工精标的高质量样本子集，用于 KNN 传播种子、few-shot 示例、回归测试、IAA 基准。

**版本**: v1.0
**创建日期**: 2026-08-26
**样本数**: 3（待人工精标）
**相关 ADR**: [ADR-003](../adr/ADR-003-DATA-SPLIT-AND-SOURCE-ISOLATION.md) 第6节 | [ADR-004](../adr/ADR-004-L1-L4-PREANNOTATION-TIERED-ARCHITECTURE.md)

---

## 用途

| 用途 | 说明 |
|------|------|
| KNN 传播种子 | L4 预标注的 KNN 传播起点，从黄金集向未标注样本传播标签 |
| Few-shot 示例 | L3 结构化预标注（大模型 caption）的 few-shot 参考样本 |
| 回归测试 | 模型/流水线变更后，在黄金集上验证性能不退化 |
| IAA 基准 | 标注者间一致性（Inter-Annotator Agreement）评估基准 |

---

## 当前样本

| audio_id | 时长 | 特点 | 标注状态 |
|----------|------|------|---------|
| 01M0RFPVQSC0HP4QJKKVCQYKCX | 526s | 长曲，可能现场/长即兴 | pending_annotation |
| 01M0RFPVTTSMAV6XT49PFATG84 | 160s | 标准单曲长度 | pending_annotation |
| 01M0RHT5QBG7MD9CBDYF68TKPY | 688s | 超长曲，大概率现场录音 | pending_annotation |

---

## 标注字段

每个样本的精标 JSON 包含以下字段：

| 字段 | 说明 |
|------|------|
| genre_level1/2/3 | 三级流派标签 |
| instruments | 乐器列表（文本） |
| instruments_gm128_ids | 乐器 GM128 标准编号 |
| vocal_presence | 有人声/无人声/纯器乐 |
| mood_tags | 情绪标签列表 |
| mood_vad | 情绪 VAD 三维度（效价/唤醒度/支配度） |
| tempo_bpm | 速度（BPM） |
| time_signature | 拍号 |
| key | 调性 |
| is_cover | 是否翻唱 |
| is_live | 是否现场录音 |
| quality_notes | 质量备注 |
| special_features | 特殊特征标签 |

---

## 标注流程

1. 从 `annotations/` 目录读取待标注样本的 JSON 模板
2. 听音频后填写 `annotation` 部分的所有字段
3. 填写 `annotator` 和 `annotation_timestamp`
4. 将 `review_status` 改为 `annotated`
5. 提交到 Git（黄金集纳入版本控制）

---

## 划分策略（ADR-003 第6节）

- 黄金集物理上在 main_pool 中
- 划分时**不特殊处理**，正常分布在 train/val 中
- 不强制锁在 train（避免训练集偏差）
- `split_dataset.py` 的 `--protect-golden` 参数确保黄金集不进入 test/holdout/ood

---

## 扩展计划

- 500 首全量时，抽样 5%（约 25 首）作为黄金集
- 覆盖不同流派、时长、录音条件、人声/器乐
- 每个样本至少 2 人标注，计算 IAA
