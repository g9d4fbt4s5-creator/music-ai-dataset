# ADR-003: 数据划分与来源隔离

## 状态
Accepted

## 日期
2026-08-26

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

## 与其他 ADR 的关系

- **ADR-001《QC Gate 阈值决策》**：域外样本（Ace Studio 生成）不调整 YAMNet 阈值，而是通过 source_type 标记排除，参见本 ADR 第7节
- **ADR-002《HITL 异步闭环》**：marginal 样本（包括跨集去重疑似重复）进入 waiting_pool 人工复核
- **ADR-004《L1-L4 预标注分层》**：黄金集是 L4 KNN 传播的种子源

---

## 参考

- MIR 领域标准做法：艺术家级隔离（artist-level split）
- FMA 数据集：分层抽样 + 保持艺术家完整性
- MTG-Jamendo：约 90/5/5 划分，按艺术家分层
- MulTTiPop：dev/test 7:3，无艺术家跨集
- 数据泄漏防范：以独立样本ID（艺术家/歌曲）为边界分层抽样
