# ADR-003: 数据划分与来源隔离

## 状态
Accepted（2026-08-27 修正：黄金集抽取流程 + 入选标准 + 退役机制 + 鲁棒性测试池）

## 日期
2026-08-26（创建）, 2026-08-27（修正）

## 决策者
数据工程团队

## 影响范围
- `data/00_raw_collect/`（四池物理隔离）
- `data/04_final_dataset/splits/`（train/val/test 划分）
- `scripts/utils/split_dataset.py`（划分脚本）
- `data/00_raw_collect/audio_manifest.csv`（元数据字段）

---

## 背景

音乐数据集构建过程中，数据划分和来源隔离直接决定模型评估结果的可信度。
MIR（Music Information Retrieval）领域最常见的评估陷阱是数据泄漏：
同一艺术家、同一首歌的不同版本出现在训练集和测试集中，导致模型学到的是
音色/录音环境特征而非真正的音乐特征，评估指标虚高。

本 ADR 固化数据划分策略、来源隔离原则、防泄漏字段设计和跨集去重规则。

---

## 决策

### 1. 四池物理隔离（空间维度）

```
data/00_raw_collect/
├── raw_audio/          # main_pool：训练+验证，持续追加
├── test_pool/          # 测试集：独立采集，版本冻结
├── holdout_pool/       # 最终评估集：独立采集，永久冻结
└── ood_pool/           # 域外测试集：独立采集，只做泛化分析
```

| 池 | 用途 | 来源 | 冻结策略 | 能否调参 |
|----|------|------|---------|---------|
| **main_pool** | 训练+验证 | 持续采集 | 不冻结 | ✅ |
| **test_pool** | 最终性能评估 | 独立采集（不同批次/来源） | 版本冻结 | ❌ |
| **holdout_pool** | 跨版本对比/论文发表 | 独立采集（项目初期封存） | 永久冻结 | ❌ |
| **ood_pool** | 域外泛化测试 | 独立采集（不同风格/来源） | 版本冻结 | ❌ |

**硬性约束**：
- test_pool / holdout_pool / ood_pool **必须独立采集**，禁止从 main_pool 随机切分
- 禁止在调参过程中查看 holdout_pool 指标（防止偷看）
- ood_pool 只做泛化分析，**不参与官方排名**

### 2. 划分比例

#### 500 首规模（当前目标）

| 子集 | 数量 | 比例 | 来源 |
|------|------|------|------|
| **train** | 400 | 80% | main_pool 划分 |
| **val** | 50 | 10% | main_pool 划分 |
| **test** | 50 | 10% | main_pool 划分（或独立采集） |
| **holdout** | 50-100 | — | holdout_pool 独立采集 |
| **ood** | 20-50 | — | ood_pool 独立采集 |

#### 规模自适应原则

| 数据规模 | train/val/test | 说明 |
|---------|---------------|------|
| <1k（当前27首） | 不划分，全量用于试点 | 量太小，划分无统计意义 |
| 1k-100k（500首目标） | 80/10/10 | 中等规模标准比例 |
| >100k | 95/2.5/2.5 或 98/1/1 | 大规模，val/test 各几千条足够 |

### 3. 防泄漏字段设计

在 `audio_manifest.csv` 中增加以下字段：

| 字段 | 类型 | 说明 | 划分用途 |
|------|------|------|---------|
| **audio_id** | string | 文件级唯一标识（基于内容hash） | 主键 |
| **artist_id** | string | 演奏者/乐队唯一标识 | **艺术家级隔离**：同一 artist_id 的所有录音必须在同一子集 |
| **song_group_id** | string | 歌曲级标识（同一首歌的不同版本/切片/翻唱共享） | **歌曲级隔离**：同一 song_group_id 的所有样本必须在同一子集 |
| **source_type** | string | 数据来源类型（normal / ace_studio_generated / demucs_vocals / ace_studio_generated_demucs_vocals） | 来源过滤：合成/分轨样本排除出训练集 |
| **source_batch** | string | 采集批次标识 | 批次级隔离：同一批次尽量不跨集 |

#### artist_id 处理规则

| 场景 | artist_id 取值 | 说明 |
|------|---------------|------|
| 有明确艺术家元数据 | `artist:<name>` | 如 `artist:Miles Davis` |
| 乐队/组合 | `band:<name>` | 如 `band:Yellowjackets` |
| AI 生成（Ace Studio等） | `ai:<generator>` | 如 `ai:ace_studio`，此类样本排除出训练集 |
| 无元数据/未知 | `unknown:<hash前8位>` | 按音频内容hash分组，防止同一未知艺术家跨集 |

#### song_group_id 处理规则

| 场景 | song_group_id 取值 | 说明 |
|------|-------------------|------|
| 有明确歌曲名+艺术家 | `song:<artist>:<title>` | 如 `song:Miles Davis:So What` |
| 不同版本/翻唱/remix | 同一 song_group_id | 划分时必须在同一子集 |
| 同一首歌的不同切片 | 同一 song_group_id | 防止切片跨集泄漏 |
| 无元数据/未知 | `unknown_song:<hash前8位>` | 按音频内容hash分组 |
| 纯器乐即兴（无固定曲名） | `improvisation:<artist>:<date>` | 按艺术家+日期分组 |

### 4. 分层抽样维度

划分 train/val/test 时，按以下维度分层，确保各子集分布一致：

| 维度 | 分层方式 | 优先级 |
|------|---------|--------|
| **artist_id** | 艺术家级隔离（同一艺术家不跨集） | P0 必备 |
| **song_group_id** | 歌曲级隔离（同一首歌不跨集） | P0 必备 |
| **genre** | 按流派分层 | P1 |
| **source_batch** | 按采集批次分层 | P1 |
| **vocal_presence** | instrumental/vocal/mixed | P2 |
| **BPM** | 分档：<80 / 80-120 / 120-160 / >160 | P2 |
| **duration** | 分档：短(<3min) / 中(3-8min) / 长(>8min) | P2 |

**划分算法**：
1. 按 artist_id 分组，确保同一艺术家的所有录音在同一子集
2. 按 song_group_id 分组，确保同一首歌的所有版本在同一子集
3. 在组级别进行分层抽样，按 genre/source_batch 分层
4. 调整各组分配，使 train/val/test 比例接近目标比例
5. 输出划分结果，记录每组的分配决策

### 5. 跨集指纹去重

#### 去重层级

| 层级 | 方法 | 阈值 | 动作 |
|------|------|------|------|
| **精确重复** | Chromaprint 完全匹配 | 100% | 删除重复，只保留一份 |
| **近似重复** | Chromaprint 相似度 | ≥0.92 | 标记为 marginal，人工确认是否同一首歌 |
| **跨集去重** | train vs test/holdout 指纹相似度 | ≥0.5 | 将 test/holdout 中的样本移到 marginal 或丢弃 |

#### 跨集去重流程

```
1. 对 main_pool 全部样本提取 Chromaprint 指纹
2. 划分 train/val/test 后
3. 计算 train 与 test 的指纹相似度矩阵
4. 如果相似度 > 0.5，将 test 中的样本标记为 cross_set_duplicate
5. cross_set_duplicate 样本处理：
   - 移到 marginal 候选池（人工确认是否真的是同一首歌）
   - 或直接从 test 中移除（如果确认是同一首歌的不同版本）
6. 对 holdout_pool 和 ood_pool 同样执行跨集去重
```

**为什么阈值是 0.5 而不是 0.92**：
- 0.92 用于检测"几乎完全相同"的重复（同一录音的不同编码）
- 0.5 用于检测"同一首歌的不同版本"（cover/remix/不同演奏），这类样本虽然不是精确重复，但模型仍然"见过"类似的旋律/和声进行，会导致评估虚高

### 6. 黄金集定位

| 维度 | 决策 |
|------|------|
| **定义** | 从 main_pool 抽样约 5%（50-500条），人工精标 |
| **物理位置** | `data/03_human_annotation/golden_set/` |
| **划分处理** | **不特殊处理**，正常分布在 train/val 中，不强制锁在 train |
| **用途** | few-shot 示例、prompt 校准、KNN 传播种子、IAA 基准、回归测试 |
| **与 KNN 传播关系** | 黄金集是 KNN 传播的种子源，通过相似度向未标注样本传播标签 |

**为什么不强制锁在 train**：
- 黄金集的核心价值是"高质量标注"，不是"训练数据"
- 如果强制锁在 train，会导致 val/test 中缺少高质量标注样本，无法验证 KNN 传播在未见过样本上的效果
- 正常分布在 train/val 中，可以同时用于训练和验证

#### 6.1 黄金集抽取流程（🔴 2026-08-27 修正：用户主动选择，非脚本预设）

**历史教训**：项目早期曾由脚本单方面预设3首Jazz作为黄金集，未经过聚类和用户确认，导致风格覆盖不全（全是Jazz）、2首被截断标注、用户完全不知情。此做法违反本ADR的"人工精标"原则，已纠正。

**正确流程（强制执行）**：

```
Step 1: 聚类
  └─ 用 L2 MERT 嵌入（768维）做 K-Means / HDBSCAN 聚类
  └─ 将 cluster_id 写回 manifest
  └─ 输出每个 cluster 的大小、风格分布、代表性样本

Step 2: 分布展示
  └─ 向用户展示每个 cluster 的：
     - 样本数量、主要风格（L2 CLAP genre top-1）
     - 距 cluster 中心最近的 Top-5 候选（含时长、artist、genre）
     - 标注哪些候选满足硬性门槛（时长120-360s、QC pass、artist已知）

Step 3: 用户主动选择
  └─ 用户从每个 cluster 中挑选 1-2 首代表性样本
  └─ 用户可基于：音乐熟悉度、风格代表性、结构清晰度、未来可展示性
  └─ 禁止脚本单方面预设黄金集，必须用户确认

Step 4: 标记与标注
  └─ manifest 中标记 is_golden=True, golden_version=vX.X
  └─ 运行 Qwen-Omni 自动标注（长音频分段调用，避免截断）
  └─ 用户人工复核高置信标注，低置信样本重新标注
```

**黄金集数量建议**：
| 数据规模 | 黄金集数量 | 依据 |
|---------|-----------|------|
| 85首（当前试点） | 5-8首 | 每cluster 1首 + 1-2首边缘案例 |
| 500首（目标） | 15-25首 | 3-5%比例，按cluster分层 |
| 学术界常规 | 3-5% | FMA: 106,574首中5,000首精标（4.7%） |

#### 6.2 黄金集入选标准（2026-08-27 补充：学术界+工业界混合标准）

**硬性门槛（必须全部满足）**：

| # | 标准 | 筛选逻辑 | 依据 |
|---|------|---------|------|
| 1 | 时长 120s ~ 360s | 排除过长（>6分钟，标注成本高）和过短（<2分钟，结构不完整） | SALAMI数据集平均4.2分钟 |
| 2 | QC = pass | 排除 fail/marginal 样本 | ADR-001 final_branch=pass |
| 3 | artist_id 已知 | 排除 unknown_*，确保可溯源 | MusicNet/FMA标注规范 |
| 4 | 非 AI 生成 | 排除 source_type=ace_studio_generated* | 合成样本不适合作为真值基准 |
| 5 | 聚类分层覆盖 | 每个 cluster 至少选1首，确保风格覆盖 | ADR-003分层抽样原则 |

**质量优选（软约束，满足越多越好）**：

| # | 标准 | 学术界依据 | 工业界依据 |
|---|------|-----------|-----------|
| 6 | cluster 中心点优先 | 该cluster的"最典型"样本，KNN传播效果最佳 | 高ROI：标注1首能代表整簇 |
| 7 | 音频质量高（SNR>20dB，无削波） | 标注员一致性（IAA）>0.8 | 降低标注员疲劳，提升一致性 |
| 8 | 流派标签明确（L2 genre置信度>0.7） | 减少人工判断歧义 | 标注成本低，一致性高 |
| 9 | 乐器多样性（≥2种乐器） | 结构标注信息量丰富 | 覆盖多乐器场景 |
| 10 | 用户熟悉/能听出结构 | 便于人工复核 | 用户参与度高，标注质量好 |

**满足 8/10 项即可入选。**

#### 6.3 黄金集退役机制（2026-08-27 补充）

当黄金集需要重新选择时（如聚类更新、风格覆盖不足、标注质量问题），旧黄金集按以下流程退役：

| 操作 | 说明 |
|------|------|
| **保留文件** | annotation JSON 文件不删除，保留历史标注结果 |
| **移除标记** | manifest 中 is_golden=False, golden_version=None |
| **状态变更** | annotation 中 review_status=auto_annotated, golden_status=retired_vX |
| **退役原因** | annotation 中记录 retired_reason（如"聚类后重新选择，风格覆盖不全"） |
| **不参与 L4** | l4_knn_propagation.py 读取时过滤 golden_status != active |
| **可重新启用** | 未来人工复核后，可重新标记为 active 并纳入新版本黄金集 |

**退役版本号**：retired_v1, retired_v2, ... 与 golden_version 对应。

#### 6.4 鲁棒性测试池（Stress Test Pool，2026-08-27 补充）

**与黄金集的区别**：黄金集 = good case（KNN种子、真值基准）；鲁棒性测试池 = bad/edge case（压力测试，不参与KNN传播）。

| 维度 | 黄金集 (Golden Set) | 鲁棒性测试池 (Stress Test) | OOD 集 |
|------|---------------------|---------------------------|--------|
| **定义** | 高质量人工精标，KNN种子源 | 训练分布内的困难/边缘样本 | 训练分布外的未知风格/来源 |
| **样本类型** | good case（结构清晰、音质好） | bad case（噪声大、混音糊）+ edge case（极短、纯打击乐、无调性） | 域外风格（电子、古典、民乐） |
| **是否参与KNN** | ✅ 是（种子源） | ❌ 否 | ❌ 否 |
| **是否参与训练** | ✅ 正常分布在train/val | ❌ 不参与训练/调参 | ❌ 不参与训练/调参 |
| **用途** | few-shot、prompt校准、KNN种子、IAA基准 | Stage 6后模型压力测试，看"翻车率" | 泛化测试，看模型是否过拟合 |
| **标注方式** | Qwen-Omni + 人工精标（高投入） | Qwen-Omni自动标注（定性观察）或人工真值（定量评测） | 自动标注或人工标注 |
| **物理位置** | `data/03_human_annotation/golden_set/` | `data/04_final_dataset/stress_test/` | `data/04_final_dataset/ood_pool/` |

**鲁棒性测试池来源**：
- QC marginal 样本（SNR 12-15dB、动态范围<5dB）
- HDBSCAN/K-Means 的 outlier 离群簇样本
- 用户投诉/模型置信度低的样本
- 人工挑选的边缘案例（纯打击乐、无调性、环境噪音混入）

**85首试点建议**：3首鲁棒性测试池（从marginal/outlier中选），成本低但价值高。500首时扩展到10-15首。

**评测指标设计**：
| 指标 | 计算集合 | 作用 |
|------|---------|------|
| 主F1 | val/test（正常分布） | 模型基本能力 |
| 黄金集IAA | golden set（good case） | 标注质量基准 |
| 鲁棒性翻车率 | stress test（bad/edge） | 模型失效边界 |
| OOD泛化率 | ood_pool | 域外能力 |

### 7. source_type 过滤规则

| source_type | 训练集 | 验证集 | 测试集 | holdout | ood | 说明 |
|-------------|--------|--------|--------|---------|-----|------|
| **normal** | ✅ | ✅ | ✅ | ✅ | ✅ | 正常采集的原始音乐 |
| **ace_studio_generated** | ❌ | ❌ | ❌ | ❌ | ⚠️ | AI 生成音乐，排除出训练/评估，可作为 OOD 分析 |
| **demucs_vocals** | ❌ | ❌ | ❌ | ❌ | ⚠️ | Demucs 分轨后的人声单轨，排除出训练/评估 |
| **ace_studio_generated_demucs_vocals** | ❌ | ❌ | ❌ | ❌ | ⚠️ | AI 生成+分轨人声，YAMNet 域外样本，排除 |

**过滤时机**：Stage 1（采集入库）时根据 source_type 标记自动排除，不进入下游 Stage 2-6。

---

## 数据泄漏防范清单

| 防范措施 | 实现方式 | 状态 |
|---------|---------|------|
| 艺术家级隔离 | artist_id 字段 + 划分时按 artist_id 分组 | ✅ 本 ADR 定义 |
| 歌曲级隔离 | song_group_id 字段 + 划分时按 song_group_id 分组 | ✅ 本 ADR 定义 |
| 来源隔离 | test/holdout/ood 独立采集 | ✅ 目录已预留 |
| 跨集去重 | Chromaprint 相似度 >0.5 移除 | ✅ 本 ADR 定义 |
| 批次隔离 | source_batch 字段 + 分层抽样 | ✅ 本 ADR 定义 |
| Holdout 冻结 | 永久冻结，不参与调参 | ✅ 本 ADR 定义 |
| OOD 不参与排名 | 只做泛化分析 | ✅ 本 ADR 定义 |

---

## 切片时机决策（2026-08-26 修正；2026-08-27 补充：两种切片的明确区分）

### 两种切片的定义（2026-08-27 补充）

本项目存在两种完全不同的"切片"操作，必须明确区分，禁止混淆：

| 维度 | 时间戳切片（Timestamp Split） | 训练切片（Training Chunk） |
|------|-------------------------------|---------------------------|
| **执行阶段** | Stage 0（采集入库阶段） | Stage 5（训练准备阶段） |
| **输入** | Bilibili/YouTube 合集长音频（30分钟-2小时） | 单曲母版（2-15分钟） |
| **输出** | 单曲 FLAC（每首一个文件） | 15秒 wav 训练片段（每首多个） |
| **切片依据** | 视频简介中的时间戳列表（如 `00:00 Song A, 05:30 Song B`） | 固定时长（15秒）+ 重叠（50%），或按乐段边界智能切片 |
| **脚本** | `scripts/00_raw_collect/split_by_timestamps.py` | `scripts/05_training_prep/01_audio_chunker.py` |
| **数据划分关系** | 在划分之前执行，切片后才入库和划分 | **必须在 Stage 4 数据划分之后执行**，只对 train/val 切片 |
| **防泄漏要求** | 同一合集的不同单曲可能是同一艺术家，需确保 artist_id 正确填充 | test/holdout/ood **禁止切片**，保持整首音频用于完整曲目评估 |
| **硬指标校验** | 覆盖完整性（第一首start=0，最后一首end≈时长）、连续性（相邻曲目gap<5s）、艺术家非空 | 切片时长一致性、重叠率、音频格式统一 |

**常见混淆错误（历史教训）**：
- ❌ 将训练切片放在 Stage 1 执行，导致对全部样本（含 fail/test/holdout）切片，3038片旧切片包含 test/holdout 杂质
- ❌ 将时间戳切片的输出直接作为训练样本，跳过 Stage 1-4 的 QC、母版生成、预标注、数据划分
- ✅ 正确流程：时间戳切片（Stage 0）→ QC → 母版 → 预标注 → 数据划分（Stage 4）→ 训练切片（Stage 5，仅 train/val）

### 训练切片时机（原 2026-08-26 修正）

**切片必须在 Stage 4 数据划分之后执行（Stage 5），只对 train/val 的 audio_id 切片。**

### 背景

早期设计将切片放在 Stage 1（预处理阶段），导致以下问题：

1. **数据范围错误**：对全部样本（含 fail/test/holdout/ood）切片，浪费算力
2. **结构信息缺失**：预标注（L1-L4）未完成，无法按乐段边界智能切片
3. **切片数量盲目**：85首切出3038片，但实际进入训练集的可能只有60-70首
4. **与数据划分矛盾**：切片后再划分，可能导致同一首歌的不同切片跨集泄漏

### 修正后的流程

```
Stage 0-3: 采集 → QC → 母版 → 预标注 → 人工审核
    ↓
Stage 4: 数据划分（train/val/test/holdout/ood + 跨集去重 + artist隔离）
    ↓
Stage 5: 切片 + 特征提取（只对 train/val，按结构边界切）
    ↓
Stage 6: 模型训练
```

### 实现

- 脚本位置：`scripts/05_training_prep/01_audio_chunker.py`
- 关键参数：`--only-train-val`（只切 train/val）、`--splits`（数据划分目录）
- 测试集不切片：test/holdout/ood 保持整首音频，用于完整曲目评估

### 历史切片处理

之前在 Stage 1 切的 3038 片标记为 `pre_split_chunks`（历史产物），不用于正式训练。确认 Stage 5 逻辑正确后可删除。

---

## 与其他 ADR 的关系

- **ADR-001《QC Gate 阈值决策》**：域外样本（Ace Studio 生成）不调整 YAMNet 阈值，而是通过 source_type 标记排除，参见本 ADR 第7节
- **ADR-002《HITL 异步闭环》**：marginal 样本（包括跨集去重疑似重复）进入 waiting_pool 人工复核
- **ADR-004《L1-L4 预标注分层》**：黄金集是 L4 KNN 传播的种子源
- **ADR-005《标签映射字典版本管理》**：source_type 过滤与映射字典的 hard_blacklist/soft_blacklist 协同工作，合成/分轨样本在 source_type 层排除，低质量标签在映射字典层过滤

---

## 参考

- MIR 领域标准做法：艺术家级隔离（artist-level split）
- FMA 数据集：分层抽样 + 保持艺术家完整性
- MTG-Jamendo：约 90/5/5 划分，按艺术家分层
- MulTTiPop：dev/test 7:3，无艺术家跨集
- 数据泄漏防范：以独立样本ID（艺术家/歌曲）为边界分层抽样
