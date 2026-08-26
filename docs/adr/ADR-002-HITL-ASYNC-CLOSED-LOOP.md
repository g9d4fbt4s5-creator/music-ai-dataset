# ADR-002: HITL 异步闭环与自适应听检架构

**状态**: Accepted  
**日期**: 2026-08-26  
**决策者**: 数据工程团队  
**影响范围**: scripts/utils/hitl_scheduler.py, scripts/utils/badcase_collector.py, scripts/utils/adaptive_listening_check.py, data/waiting_pool/

---

## 背景

音乐数据集构建过程中，质量检查（QC Gate）会产出 marginal 样本（边缘样本），这些样本不直接过滤，也不直接进训练集，需要人工听检确认。

传统做法是人工定期导出 marginal 列表，手动创建听检任务，听检后手动更新阈值。这种方式效率低、容易遗漏、无法形成闭环。

本项目设计了 HITL（Human-In-The-Loop）异步闭环架构，实现：
- 自动监测异常信号
- 自动生成针对性听检任务
- 自动收集 badcase
- 半自动规则更新（建议自动生成，人工确认后生效）

---

## 决策

### 1. 核心原则

#### 1.1 绝对异步
主流水线不阻塞，听检并行进行。
```
主流水线 ──→ 产出 waiting_pool ──→ 继续跑下一批次
                ↓
            HITL 听检（人工，慢）
                ↓
            结果回流（下一批次生效）
```
**禁止**：主流水线停下来等人工听检结果。

#### 1.2 数据驱动触发
不是定时触发，而是阈值/统计异常触发：
- marginal 率突增 → 自动创建 SNR 听检任务
- unmapped 标签高频出现 → 自动创建映射审核任务
- 新簇出现 → 自动创建聚类验证任务

#### 1.3 规则更新半自动
```
听检结果 → 脚本生成更新建议 → 人工一键确认 → 写入配置
```
**禁止**：听检结果直接自动修改 qc_gate.py 阈值。

---

### 2. 9 种听检任务类型

| # | 任务类型 | 触发阶段 | 触发条件 | 优先级 |
|---|---------|---------|---------|--------|
| 1 | qc_snr_calibration | Stage 3 QC | marginal 率 > 25% | P0 |
| 2 | qc_content_boundary | Stage 3 QC | content fail 但 top5 含人声标签 | P0 |
| 3 | dedup_similarity | Stage 4 | 指纹相似度 0.92-0.99 | P1 |
| 4 | knn_propagation_verify | L4 传播 | 新批次传播 > 50 首 | P1 |
| 5 | unmapped_tag_review | L4 映射 | unmapped 标签频次 > 5 | P1 |
| 6 | cluster_validation | Stage 5.3 | 新簇或簇内方差异常 | P1 |
| 7 | segment_boundary | Stage 6 | 切片后抽检 5% | P1 |
| 8 | data_source_quality | Stage 0 | 新增采集源 | P1 |
| 9 | post_threshold_audit | Stage 3 后 | QC 阈值被修改 | P0 |

**设计原则**：9 种框架先搭好，按需启用，避免过度工程。当前 27 首试点阶段只需要 qc_snr_calibration 和 qc_content_boundary（已完成），其他 7 种为后续 Stage 和 500 首全量预留。

---

### 3. 混合方案：预定义 XML 骨架 + 动态填充数据

**决策**：不调用 DeepSeek API 生成 XML 结构，而是把 9 种听检模板的骨架预定义在代码的 TEMPLATE_REGISTRY 中，运行时只动态填充数据字段。

| 维度 | 方案 A（对话生成） | 方案 B（DeepSeek API） | 方案 C（混合，采用） |
|------|-------------------|----------------------|---------------------|
| XML 格式可靠性 | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| 自动化程度 | ❌ | ✅ | ✅ |
| 运行成本 | 0 | API 费用 | 0 |
| 可维护性 | 每次找我 | 需写验证层 | 模板骨架清晰 |

**核心逻辑**：听检任务的判断维度是固定的（SNR 是否可接受、KNN 标签是否正确、去重样本是否真重复），真正变化的是预填数据（具体的 SNR 值、DR 值、KNN 距离、标签名等）。所以 XML 骨架不需要每次重新生成，只需要在固定骨架上动态填充数据。

**模板版本管理**：TEMPLATE_REGISTRY 中每个模板包含 version、decision_field、last_verified 字段，当模板骨架迭代时可追踪版本，避免旧任务和新任务混用同一模板导致决策口径不一致。

---

### 4. waiting_pool 目录结构

```
data/waiting_pool/
├── qc_snr/           # 等 SNR 听检
├── qc_content/       # 等 content 听检
├── dedup_pairs/      # 等去重确认
├── knn_verify/       # 等 KNN 验证
├── unmapped_tags/    # 等标签映射审核
├── cluster_validate/ # 等聚类验证
├── segment_quality/  # 等切片检查
├── source_quality/   # 等数据源评估
└── post_threshold/   # 等阈值后抽检
```

**超时机制**：waiting_pool 中样本超过 7 天未听检，自动降级处理（按保守策略判 marginal）。

---

### 5. Badcase 收集机制

#### 5.1 什么是 badcase
Badcase 是"预期应该正确，但实际错了"的样本，用于归因分析和系统改进。

**收集**：
- ✅ YAMNet 误杀正常音乐（top5 含人声但判非音乐）
- ✅ has_vocals 与 top5_events 不一致
- ✅ KNN 传播标签错误
- ✅ 预标注流派错误
- ✅ 标签矛盾（两个标注员判定不一致）
- ✅ 合成样本混入（Ace Studio 生成 + Demucs 分轨）

**不收集**：
- ❌ 普通质量 fail（SNR<10dB、静音>80%、削波>5%）— 这是正常清洗

#### 5.2 Badcase 生命周期
```
各环节自动捕获（collector）
    ↓
data/03_human_annotation/badcase/auto_collected.jsonl（过程态）
    ↓
人工/Agent 定期审查
    ├── 双确认 → 迁移到 data/04_final_dataset/badcase_pool/（终态）
    ├── 误报 → 标记 ignored
    └── 需深入分析 → 保留过程态，写分析笔记
```

#### 5.3 终态 badcase_pool 的用途
- DPO 负样本（chosen vs rejected）
- QC 规则迭代输入（如"这类样本以后直接 fail"）
- 面试作品集素材（归因分析案例）

---

### 6. 完整 HITL 闭环架构

```
┌─────────────────────────────────────────────────────────┐
│                    主流水线（自动）                        │
│  Stage 0 → 1 → 2 → 3 → 4 → 5 → 6 → L1 → L2 → L3 → L4  │
└──────────────────────┬──────────────────────────────────┘
                       │ 产出 marginal / 不确定 / 候选样本
                       ▼
┌─────────────────────────────────────────────────────────┐
│              waiting_pool/（自动填充）                     │
│  qc_snr/ qc_content/ dedup_pairs/ knn_verify/ ...       │
└──────────────────────┬──────────────────────────────────┘
                       │ 触发条件满足
                       ▼
┌─────────────────────────────────────────────────────────┐
│           HITL Scheduler（自动调度）                       │
│  检查触发条件 → 生成听检任务 → 创建 Label Studio 项目      │
└──────────────────────┬──────────────────────────────────┘
                       │ 人工听检/审核
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Label Studio 人工标注/审核                      │
│  听检判断 → 导出结果                                       │
└──────────────────────┬──────────────────────────────────┘
                       │ 解析结果
                       ▼
┌─────────────────────────────────────────────────────────┐
│         Badcase Collector（自动收集）                      │
│  解析导出结果 → 识别异常 → 写入 badcase pool               │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    更新 QC 阈值   更新映射字典   更新 KNN 参数
    （人工确认）   （人工确认）   （人工确认）
          │            │            │
          └────────────┼────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│          下一批次流水线（应用新规则）                      │
└─────────────────────────────────────────────────────────┘
```

---

## 黄金集 pending 状态下的 L4 降级策略（2026-08-27 新增）

### 背景

L4 KNN 传播的高置信种子来自 L3 黄金集人工标注（详见 ADR-004 第 5.2 节）。但黄金集标注是人工密集型任务，可能长期处于 `pending_annotation` 状态。如果 L4 传播必须等待黄金集标注完成，会阻塞整个流水线。

**当前状态（2026-08-27）**：3 首黄金集 `review_status=pending_annotation`，L4 种子池因此缺少高置信种子。

### 降级策略

当黄金集处于 `pending_annotation` 状态时，L4 KNN 传播按以下优先级降级：

| 优先级 | 种子来源 | 置信度 | 触发条件 | 标签覆盖 |
|--------|---------|--------|---------|---------|
| **P0（首选）** | L3 黄金集人工标注（`review_status=completed`） | 高 | 黄金集标注完成 | genre, mood, instruments, segments, caption |
| **P1（降级）** | L3 Qwen-Omni 自动标注（`review_status=qwen_omni_annotated`） | 中高 | 黄金集已通过 Qwen-Omni 自动标注，但未经人工校验 | genre, mood, instruments, segments, caption（需标记 `auto_annotated=true`） |
| **P2（再降级）** | L2 CLAP 零样本 top-1 候选（confidence > 0.3） | 中 | 黄金集完全未标注，或 Qwen-Omni 标注不可用 | genre, mood（无 instruments/segments/caption） |
| **P3（最低）** | L1 物理特征规则映射（可选，第一版不实现） | 低 | 以上全部不可用 | genre 粗分类（如 BPM>120 → 可能 pop/electronic） |

### 降级标记

L4 传播结果必须明确标记种子来源，便于后续人工校验和质量评估：

```json
{
  "audio_id": "xxx",
  "genre": "jazz",
  "propagation_source": "knn_train",
  "seed_source": "l2_clap_candidate",  // 标记种子来源层级
  "seed_confidence_level": "medium",     // high / medium_high / medium / low
  "golden_pending": true,                // 标记传播时黄金集是否处于 pending 状态
  "pending_degradation": "P2"            // 实际使用的降级层级
}
```

### 降级状态下的质量控制

1. **传播阈值收紧**：降级状态下（P2/P3），KNN 传播的距离阈值收紧 20%（如 genre 从 0.40 收紧到 0.32），降低错误传播风险
2. **人工校验优先级提升**：降级状态下传播的样本，进入 waiting_pool 的优先级提升为 P1（正常为 P2）
3. **黄金集标注完成后重跑**：黄金集 `review_status` 变为 `completed` 后，必须重跑 L4 传播，用高置信种子替换降级种子
4. **不可用于 test/holdout/ood**：无论降级层级如何，test/holdout/ood 始终禁止传播（ADR-004 第 5.1 节）

### 与 ADR-004 的关系

本章节是 ADR-004《L1-L4 预标注分层架构》第 5.2 节（种子池构建）的补充，明确了黄金集 pending 状态下的降级流程。ADR-004 定义了种子池的组成，本 ADR 定义了种子池不可用时的降级策略。

---

## 阈值后抽检（post_threshold_audit）

阈值调整不是"一放了之"。让一批原本被拦在门外的 marginal/fail 样本进入训练池，如果它们实际上质量不达标，会污染下游的 KNN 传播和模型训练。

**抽检对象**：阈值漂移样本（因本次阈值调整而改变分支的样本）。

**分层抽检优先级**：
| 优先级 | 样本特征 | 抽检比例 |
|--------|---------|---------|
| P0 | 阈值边界附近（如 SNR 12.0-13.0dB） | 50% |
| P1 | 阈值漂移样本中同时有其他 marginal 标记 | 30% |
| P2 | 纯阈值漂移样本（仅 SNR 一项触发，其他指标优秀） | 10% |

**决策规则**：
- 劣质率 ≤ 5% → 阈值调整有效，正式生效
- 劣质率 5-20% → 阈值微调（如 12dB → 13dB），重新抽检
- 劣质率 > 20% → 回滚阈值，漂移样本恢复旧分支

**当前 27 首不需要 post_threshold_audit**：因为 SNR 12-15dB 的漂移样本（6首）已全部在 9 首 marginal 听检中覆盖。该任务类型为 500 首全量预留。

---

## 岗位边界

- **数据岗**：批量计算全套指标，执行 pass/marginal/fail 逻辑，输出 marginal 待审核清单，统计分布；维护 HITL 调度器和 badcase 收集器。
- **评测岗**：使用 marginal 样本做鲁棒性测试，测试模型在噪声、失真样本的表现；使用 badcase_pool 做 DPO 负样本。

---

## 参考

- [Label Studio 文档](https://labelstud.io/guide/)
- [Active Learning 综述](https://arxiv.org/abs/2009.00236)
- [Human-In-The-Loop 机器学习](https://arxiv.org/abs/2107.01349)
