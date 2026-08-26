# Music AI Dataset — 流水线架构总览 V3

> 版本: v3.0.0 | 更新: 2026-08-26 | 状态: 生产级可落地
>
> 本文档整合四方审核意见（用户 / 豆包agent / DeepSeek / Kimi），是当前流水线的**权威架构文档**。
> 取代 V2（docs/PIPELINE_OVERVIEW_V2.md），V2 文档保留作为历史参考。

---

## 一、核心修正（相对 V2）

| # | 修正内容 | 提出方 | 原因 |
|---|---------|--------|------|
| 1 | **明确区分两种切片**：时间戳切片（Stage 0）vs 训练切片（Stage 5） | Kimi | V2 混淆了两种切片，导致架构逻辑混乱 |
| 2 | **L4 KNN 传播移到 Stage 5**，且 test/holdout/ood 不传播 | Kimi + DeepSeek | 防止测试集信息泄漏，避免伪标签污染评估 |
| 3 | **母版规格降为 48kHz/16bit** | 豆包agent（Kimi确认） | CLAP 输入 48kHz 无需重采样，16bit 比 24bit 节省 33% 空间 |
| 4 | **Bilibili 时间戳用硬指标校验**（覆盖+连续） | Kimi | 替代模糊置信度，可量化、可测试 |
| 5 | **切片级跨集去重当前跳过** | Kimi | Chromaprint 对 15 秒切片不可靠，85 首 audio_id 去重已足够 |
| 6 | **3038 片旧切片归档**（非删除） | 豆包agent | 保留可追溯性，确认 Stage 5 新切片正确后可删除归档 |

---

## 二、两种切片的明确区分

> 这是 V3 最重要的架构澄清。V2 中"切片"一词被混用，导致用户和四方审核都产生困惑。

| 维度 | 时间戳切片 | 训练切片 |
|------|-----------|---------|
| **发生时机** | Stage 0 入库时 | Stage 5 数据划分后 |
| **输入** | Bilibili 长音频合集（30-60 分钟） | 独立单曲（3-5 分钟母版） |
| **输出** | 独立单曲（3-5 分钟） | 15 秒训练片段 |
| **用途** | 把合集拆成单曲，用于后续所有处理 | 模型训练输入 |
| **脚本** | `scripts/utils/split_by_timestamps.py` | `scripts/05_training_prep/01_audio_chunker.py` |
| **参数** | 时间戳配置（configs/split_timestamps.json） | 15 秒 / 0.5 重叠 / --only-train-val |
| **长音频处理** | 长音频保留在 raw_audio/ 作为归档 | 不涉及 |

### 修正后的流程

```
Stage 0 入库时：
  Bilibili 长音频 → 时间戳切片（入库时完成）→ 独立单曲进入 manifest
  长音频保留在 raw_audio/，标记 is_master_recording=true

Stage 1-4：在独立单曲（3-5 分钟）上做母版、预标注、划分

Stage 5：只对 train/val 的独立单曲做 15 秒训练切片
```

---

## 三、完整流水线（V3，编号统一）

```
Stage 0:  采集入库（含时间戳切片）
  ├── Bilibili: API获取元数据 → 时间戳硬指标校验 → 按时间戳切片为独立单曲
  ├── 长音频保留 raw_audio/ 作为归档（标记 is_master_recording=true）
  ├── 独立单曲入 manifest
  ├── 本地音频: 直接入库（保持原始格式，永不转码）
  └── 输出: raw_audio/ + manifest（CSV）

Stage 0.5: QC Gate（软门槛）
  ├── YAMNet 内容分类 + librosa 音质检查
  ├── 输出 pass/marginal/fail，fail 移入 rejected/
  └── 输出 QC 报告（qc_gate_report.csv，含 final_branch 列）

Stage 1:  母版生成（只处理 pass）
  ├── 统一 FLAC 48kHz/16-bit（V3 修正，非 24bit）
  ├── 排除 fail 样本（--qc-report 参数，读取 final_branch 列）
  └── 输出 processed_master/

Stage 2:  预标注 L1+L2+L3（整首母版，不含 L4）
  ├── L1 物理标签（本地 CPU）: BPM/调性/SNR/DR/LUFS/频谱质心
  ├── L2 MERT/CLAP 嵌入（GPU，整首母版，同步时可转 MP3）
  ├── L3 结构标注（黄金集，API，短曲目优先，<3 分钟）
  └── 输出 L1/L2/L3 结果

Stage 3:  人工审核 + 黄金集精标（Label Studio）
  ├── 黄金集覆盖所有关键维度（genre/mood/instruments/structure）
  ├── marginal 样本进入 waiting_pool 人工复核
  └── 输出精标标签

Stage 4:  数据划分
  ├── artist 隔离（同一 artist_id 不跨集）
  ├── audio_id 级别跨集去重（Chromaprint，阈值 >0.5）
  ├── source_type 标记（AI生成/分轨人声标记不排除，mark_only）
  ├── 黄金集抽样（--sample-golden，优先选短曲目）
  └── 输出 train/val/test/holdout/ood CSV

Stage 5:  L4 KNN 传播 + 训练准备（只对 train/val）
  ├── L4 KNN 传播（V3 修正，防止泄漏）:
  │   ├── train: 拟合 + 传播 → 用于训练
  │   ├── val: 预测（明确标记为伪标签）→ 超参调优
  │   └── test/holdout/ood: ❌ 不传播，仅用 L1 物理标签 + L3 人工标注
  ├── 15 秒训练切片（固定窗口，--only-train-val，beat-sync 留后续）
  ├── 切片级跨集去重: 当前跳过（85 首 audio_id 去重已足够）
  ├── 特征提取（mel/chroma/MFCC → npy，LMDB 留后续）
  └── 输出训练数据包

Stage 6:  模型训练
  └── 模型训练 + 评估
```

---

## 四、L4 KNN 传播防泄漏方案（V3 核心修正）

### 问题

V2 中 L4 KNN 传播在 Stage 2 全量数据（含 test/holdout）上做，导致测试集样本的标签通过邻居关系"见过"训练集信息，评估虚高。

### 修正方案

| 子集 | L4 KNN 传播 | 用途 | 标签质量 |
|------|------------|------|---------|
| **train** | ✅ 拟合 + 传播 | 训练 | 传播标签（可接受） |
| **val** | ✅ 预测（标记伪标签） | 超参调优 | 伪标签（需知晓质量） |
| **test** | ❌ 不传播 | 定量评估 | 仅 L1 物理标签 + L3 人工标注 |
| **holdout** | ❌ 不传播 | 最终评估 | 仅 L1 物理标签 + L3 人工标注 |
| **ood** | ❌ 不传播 | 泛化分析 | 仅 L1 物理标签 |

### 实现

- `l4_knn_propagation.py` 增加 `--train-split` / `--val-split` / `--exclude-splits` 参数
- 从 Stage 4 划分结果读取样本 ID，避免全量泄漏
- val 的 L4 标签在输出中标记 `source="knn_pseudo"`，与 train 的 `source="knn"` 区分

### 注意事项

- 若 test 集没有 L3 标注，则 test 集**不用于定量评估**，只做定性分析，直到黄金集覆盖 test
- 黄金集抽样时应确保部分黄金集样本落在 test/holdout 中，用于评估

---

## 五、母版规格（V3 修正）

### 最终规格

| 参数 | 值 | 理由 |
|------|-----|------|
| 格式 | FLAC | 无损压缩，开源支持好 |
| 采样率 | **48 kHz** | CLAP 模型输入需要 48kHz，母版 48kHz 加载时无需重采样 |
| 位深 | **16-bit**（V3 修正，非 24bit） | 16-bit 比 24-bit 节省 33% 空间，MERT/CLAP 对 16-bit 无感知 |
| 声道 | 立体声（2ch） | 保留空间信息 |

### 规格对比

| 方案 | CLAP 重采样 | MERT 重采样 | 空间节省 | 85首大小估算 |
|------|------------|------------|---------|------------|
| 48kHz/24bit（V2） | 无需 | 重采样到16kHz | 0% | 3.4 GB |
| 44.1kHz/16bit（DeepSeek原建议） | 需重采样到48kHz | 重采样到16kHz | 68% | 1.1 GB |
| **48kHz/16bit（V3，采纳）** | **无需** | 重采样到16kHz | **33%** | **2.3 GB** |

### 实现

- `scripts/01_preprocess/01_generate_master.py`
  - `MASTER_BIT_DEPTH = 16`（已修改，commit f0bb32b）
  - ffmpeg `-sample_fmt s16`（已修改）
- 采样率保留配置参数 `MASTER_SAMPLE_RATE`，允许未来按需调整

---

## 六、Bilibili 时间戳切片硬指标校验（V3 修正）

### 问题

V2 中 Bilibili 元数据提取方案过于乐观，未考虑时间戳格式不统一、缺漏、不连续等现实问题。

### 硬指标方案（采纳 Kimi 建议）

自动切片必须**同时满足**以下条件：

```
1. 第一首 start = 0（容差 2 秒）
2. 最后一首 end ≈ 视频时长（容差 10 秒）
3. 第 i 首 end ≈ 第 i+1 首 start（容差 5 秒）
4. 艺术家信息可从标题/简介提取（非空）
```

**任一条件不满足 → 标记 `manual_review`，不自动切片，转人工确认。**

### 多源提取优先级

| 优先级 | 来源 | 置信度 | 说明 |
|--------|------|--------|------|
| P1 | 视频章节 API | 0.95 | 如果 UP 主设置了章节，最可靠 |
| P2 | 简介正则 | 0.70 | 时间戳格式多样，需正则匹配 |
| P3 | 评论正则 | 0.50 | 评论区时间戳常被淹没 |
| P4 | LLM 兜底 | 0.60 | DeepSeek API 提取，成本约 ¥0.01-0.05/视频 |

### 缓存机制

- API 结果缓存到 `.cache/bilibili/{bvid}.json`
- 避免重复请求，降低被封风险

### 实现

- `scripts/utils/split_by_timestamps.py` 升级，增加硬指标校验逻辑
- 时间戳配置仍用 `configs/split_timestamps.json`，但增加 `confidence` 和 `review_status` 字段

---

## 七、存储策略（V3 修正）

### 音频数据存储决策表

| 数据类型 | 格式 | 调用频率 | 删留决策 | 位置 |
|---------|------|---------|---------|------|
| 原始采集（Bilibili长音频） | 保持原始格式（永不转码） | 低（归档） | **保留** | raw_audio/ |
| 原始采集（独立单曲） | 保持原始格式 | 低 | **保留** | raw_audio/ |
| 母版（Stage 1） | FLAC 48kHz/16bit | 中（预标注输入） | **保留** | processed_master/ |
| 训练切片（Stage 5） | FLAC 16-bit | 高（训练输入） | **保留**（可重生成） | 05_training_prep/segments/ |
| 特征文件（Stage 5） | npy / fp16 | 高（训练输入） | **保留**（可重生成） | 05_training_prep/features/ |
| L2 嵌入（MERT/CLAP） | npy / npz | 中（L4传播+检索） | **保留** | 02_preannotation/embeddings/ |
| QC fail 样本 | 原始格式 | 极低 | **移入 rejected/** | 00.5_cleaned/rejected/ |
| 旧切片（Stage 1时期） | WAV/FLAC | 无 | **已归档**（确认后可删） | _archive/pre_split_chunks_20260826/ |
| 已从manifest移除的原始合集 | 原始格式 | 无 | **归档或删除** | _archive/ 或删除 |

### GPU 同步策略

| 数据 | 同步方式 | 大小估算 | 时间 |
|------|---------|---------|------|
| 母版 FLAC（48kHz/16bit） | rsync 直接传 | ~2.3 GB（85首） | 3-5 分钟 |
| 母版转 320kbps MP3（临时） | rsync 传 MP3，GPU 端验证后删除 | ~500 MB | 1-2 分钟 |
| raw 音频 + GPU 端生成母版 | rsync raw，GPU 端 ffmpeg 转码 | ~1 GB | 2 分钟 + 1 分钟转码 |

**当前推荐**：母版直接 rsync（2.3 GB），大规模数据时改用 MP3 临时转码或 raw+端侧转码。

---

## 八、已知债务与暂缓事项（P2，500 首时处理）

| # | 事项 | 暂缓原因 | 触发条件 |
|---|------|---------|---------|
| 1 | Stage 0 入库前硬门槛 QC（min_duration=30s, min_sample_rate=16kHz） | 当前手动采集，QC Gate 拦截足够 | 自动化采集时 |
| 2 | 切片级嵌入去重（MERT/CLAP 余弦相似度） | 85 首 audio_id 去重已足够，Chromaprint 对 15s 不可靠 | 500 首时 |
| 3 | manifest 改 SQLite/Parquet | CSV 已够用，引入数据库增加复杂度 | 500 首时评估 |
| 4 | DVC 数据版本控制 | 个人项目 Git + manifest 备份足够 | 论文阶段引入 |
| 5 | 特征提取改用 LMDB/TFRecord | 训练阶段未到，npy 对 500 首影响不大 | 训练阶段 |
| 6 | 切片策略优化（beat-synchronous） | 固定窗口可作为 baseline，beat-sync 需额外开发 | 训练质量优化时 |
| 7 | 并发锁、--resume、幂等性 | 单机单用户非阻塞 | 多用户/自动化时 |

---

## 九、四方审核记录

| 审核方 | 审核日期 | 关键贡献 |
|--------|---------|---------|
| 用户 | 2026-08-26 | 项目负责人，最终决策，提供数据和元数据 |
| 豆包agent | 2026-08-26 | 执行本地操作，维护项目文件，V2 文档初稿，母版 48kHz/16bit 建议 |
| DeepSeek | 2026-08-26 | 架构方案，L4 KNN 防泄漏代码，Bilibili 时间戳硬指标代码 |
| Kimi | 2026-08-26 | 独立架构审核，发现时间戳切片混淆、L4 测试集泄漏、切片级去重不可靠等关键漏洞 |

---

## 十、相关文档

- `docs/ARCHITECTURE.md` — 系统架构文档（顶部指向本文档）
- `docs/adr/ADR-003-DATA-SPLIT-AND-SOURCE-ISOLATION.md` — 数据划分与来源隔离（含切片时机决策）
- `docs/DECISIONS.md` — 决策记录
- `scripts/05_training_prep/README.md` — Stage 5 脚本说明
- `docs/PIPELINE_OVERVIEW_V2.md` — V2 历史文档（保留参考）
