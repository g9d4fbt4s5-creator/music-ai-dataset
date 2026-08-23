# 预标注模块（Branch D 增强版）

分层混合式预标注架构，结合 GPU 本地模型 + API 大模型 + KNN 传播，实现低成本、高质量的音乐标签预标注。

## 架构概览

```
02_preannotation/
├── l1_physical/                     # L1 物理层（已有）
│   └── BPM/调性/SNR/语言/时长等基础特征
├── l2_semantic/                     # L2 语义层（GPU：CLAP zero-shot）
│   └── 流派/情绪/乐器候选标签 + 置信度
├── l3_structural/                   # L3 结构层（API）
│   ├── l3_deepseek_label_extraction.py  # L3a: DeepSeek V4 Flash 文本标签（全量）
│   ├── l3_qwen_audio_structure.py       # L3c: Qwen-Audio 真实音频结构（5%黄金集）
│   └── text_labels/                      # L3a 输出
│   └── corrected_labels/                 # L3b 输出（DeepSeek V4 Pro 纠错，10%）
│   └── audio_structure/                  # L3c 输出
├── l4_propagated/                    # L4 传播层（Mac：KNN + 融合）
│   └── l4_knn_propagation.py            # KNN 传播 + 多源加权融合
├── run_preannotation_pipeline.py    # 主流水线（L2→L3→L4）
├── run_preann_infer.py              # 推理结果格式转换（已有）
├── tag_mapping_musiccaps.py         # MusicCaps 标签映射（已有）
└── ls_preannotations.jsonl          # 最终输出（Label Studio 可导入）
```

## 各层说明

### L1 物理层（已有）
- **内容**：BPM、调性、SNR、语言检测、时长、采样率、响度
- **运行位置**：Mac 本地（librosa/essentia）
- **状态**：✅ 已实现

### L2 语义层（GPU：CLAP zero-shot）
- **内容**：流派/情绪/乐器候选标签 + 置信度
- **模型**：CLAP（laion/clap-htsat-unfused）
- **运行位置**：GPU（AutoDL）
- **输入**：母版 FLAC（48k/24bit）
- **输出**：候选标签 JSON + CLAP 嵌入向量（768维）
- **成本**：GPU 顺手跑，几乎无额外成本
- **状态**：⚠️ 脚本待实现（可复用 style_consistency_clustering.py 中的 CLAP 嵌入提取）

### L3 结构层（API）

#### L3a: DeepSeek V4 Flash 文本标签提取（全量）
- **内容**：结构化标签（流派/情绪/乐器/人声/速度/年代/子流派）+ 伪 Caption
- **模型**：DeepSeek V4 Flash（deepseek-chat）
- **运行位置**：Mac（API 调用，国内直连）
- **输入**：L1 + L2 特征 JSON
- **输出**：结构化标签 JSON
- **特性**：
  - 并发处理（默认 10 workers）
  - 速率限制（默认 60 req/min）
  - 指数退避重试（默认 3 次）
  - 结果缓存（避免重复调用）
- **成本**：500首约 ¥1~2
- **状态**：✅ 已实现（l3_deepseek_label_extraction.py）

#### L3b: DeepSeek V4 Pro 疑难样本纠错（10%抽样）
- **内容**：对低置信度样本进行标签纠错
- **模型**：DeepSeek V4 Pro
- **抽样策略**：按置信度抽样，最低 10% 样本优先
- **成本**：50首约 ¥0.5
- **状态**：✅ 已实现（复用 l3_deepseek_label_extraction.py，--mode correction）

#### L3c: Qwen-Audio 真实音频结构（5%黄金集）
- **内容**：段落边界、乐器区间、真实 Caption、人声片段、速度变化
- **模型**：Qwen-Audio（阿里云百炼/DashScope）
- **抽样策略**：随机抽样 5%（最少 10 首）
- **输入**：音频片段（FLAC，降采样到 16kHz，最长 30 秒）
- **成本**：25首约 ¥25（可通过降低采样或短片段控制）
- **状态**：⚠️ 脚本待实现（l3_qwen_audio_structure.py）

### L4 传播层（Mac：KNN + 融合）
- **内容**：KNN 标签传播 + 多源加权融合
- **KNN 配置**：
  - K=5 近邻
  - 余弦相似度
  - 基于 CLAP 嵌入
  - 距离加权
- **融合策略**：加权投票
  - L2 CLAP: 0.3
  - L3 文本标签: 0.4
  - L3 纠错标签: 0.2
  - L3 音频结构: 0.1
  - KNN 传播: 0.2（额外）
- **输出**：最终预标注 JSONL（Label Studio 可导入）
- **状态**：✅ 已实现（l4_knn_propagation.py）

## 成本估算（500首 Jazz）

| 阶段 | 模型 | 样本数 | 成本 |
|------|------|--------|------|
| L2 | CLAP zero-shot | 500 | ¥0（GPU顺手跑） |
| L3a | DeepSeek V4 Flash | 500 | ¥1~2 |
| L3b | DeepSeek V4 Pro | 50（10%） | ¥0.5 |
| L3c | Qwen-Audio | 25（5%） | ¥25 |
| L4 | KNN + 融合 | 500 | ¥0（本地） |
| **总计** | | | **约 ¥27** |

**预计耗时**：约 20 分钟（API 并发）

## 使用方法

### 1. 配置 API Key

```bash
# DeepSeek API Key
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# 阿里云百炼 API Key（Qwen-Audio）
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

### 2. 完整流水线（L2→L3→L4）

```bash
# 在 GPU 上运行 L2（CLAP zero-shot）
python scripts/02_preannotation/run_preannotation_pipeline.py \
    --config configs/preannotation/preannotation_config.yaml \
    --stages l2

# 在 Mac 上运行 L3 + L4（API 调用 + KNN 融合）
python scripts/02_preannotation/run_preannotation_pipeline.py \
    --config configs/preannotation/preannotation_config.yaml \
    --stages l3_text,l3_correction,l3_audio,l4
```

### 3. 单独运行各阶段

```bash
# L3a: DeepSeek 文本标签提取（全量）
python scripts/02_preannotation/l3_structural/l3_deepseek_label_extraction.py \
    --input-dir data/02_preannotation/l1_physical \
    --l2-dir data/02_preannotation/l2_semantic \
    --output data/02_preannotation/l3_structural/text_labels \
    --config configs/preannotation/preannotation_config.yaml

# L3b: DeepSeek Pro 疑难纠错（10%）
python scripts/02_preannotation/l3_structural/l3_deepseek_label_extraction.py \
    --input-dir data/02_preannotation/l3_structural/text_labels \
    --output data/02_preannotation/l3_structural/corrected_labels \
    --mode correction \
    --correction-ratio 0.10

# L4: KNN 传播 + 多源融合
python scripts/02_preannotation/l4_propagated/l4_knn_propagation.py \
    --embeddings-dir data/02_preannotation/model_output_cache/clap_embeddings \
    --l3-text-dir data/02_preannotation/l3_structural/text_labels \
    --l3-corrected-dir data/02_preannotation/l3_structural/corrected_labels \
    --output data/02_preannotation/l4_propagated \
    --ls-output data/02_preannotation/ls_preannotations.jsonl
```

### 4. 试运行（不调用 API）

```bash
python scripts/02_preannotation/run_preannotation_pipeline.py \
    --config configs/preannotation/preannotation_config.yaml \
    --dry-run
```

## 配置文件

配置文件位置：`configs/preannotation/preannotation_config.yaml`

主要配置项：
- `global`: 全局路径配置
- `l1_physical`: L1 物理层配置
- `l2_semantic`: L2 CLAP 配置（模型、标签、阈值）
- `l3_structural`: L3 API 配置（DeepSeek/Qwen、并发、缓存、成本控制）
- `l4_propagated`: L4 KNN + 融合配置（K值、权重、融合策略）
- `cost_estimation`: 成本估算

## 关键特性

### 1. 并发 + 速率限制
- DeepSeek API 支持多线程并发（默认 10 workers）
- 内置速率限制（默认 60 req/min），避免触发 API 限流
- 429 错误自动退避重试

### 2. 结果缓存
- API 调用结果自动缓存到本地（基于特征内容哈希）
- 重复运行时直接读取缓存，不重复调用 API
- 缓存目录：`data/02_preannotation/api_cache/`

### 3. 指数退避重试
- API 调用失败自动重试（默认 3 次）
- 指数退避：1s → 2s → 4s → ...（最大 10s）
- 网络异常、超时、5xx 错误均会重试

### 4. KNN 标签传播
- 从黄金集（有 L3 标签的样本）传播到全量
- 基于 CLAP 嵌入的余弦相似度
- 距离加权投票
- 可配置 K 值和距离度量

### 5. 多源加权融合
- 整合 L2/L3a/L3b/L3c/KNN 多层标签
- 可配置各源权重
- 列表类型标签取 top-3，单值类型取最高权重
- 输出置信度分数

## 输出格式

### 单个样本标签（JSON）
```json
{
  "sample_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "labels": {
    "genre": ["jazz", "bebop", "hard bop"],
    "mood": ["melancholic", "introspective"],
    "instrumentation": ["saxophone", "piano", "double bass", "drums"],
    "vocal_presence": false,
    "tempo_category": "medium",
    "era": "1950s",
    "subgenre": "hard bop",
    "caption": "A melancholic bebop jazz piece featuring saxophone and piano...",
    "confidence": 0.75,
    "_fusion_info": {
      "method": "weighted_voting",
      "n_sources": 4,
      "sources": [...]
    }
  }
}
```

### Label Studio 预标注（JSONL）
```json
{
  "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "data": {
    "audio": "/data/audio/01ARZ3NDEKTSV4RRFFQ69G5FAV.wav",
    "sample_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"
  },
  "predictions": [{"result": [], "model_version": "l4_knn_fusion_v1"}],
  "preannotations": {
    "genre": ["jazz", "bebop"],
    "mood": ["melancholic"],
    ...
  }
}
```

## 待实现项

- [ ] L2: `l2_clap_zero_shot.py`（可复用 style_consistency_clustering.py）
- [ ] L3c: `l3_qwen_audio_structure.py`（Qwen-Audio 真实音频结构）
- [ ] 端到端测试（7首测试音频完整跑通 L2→L3→L4）
- [ ] 与 Label Studio 的集成测试

## 参考

- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [阿里云百炼 Qwen-Audio](https://help.aliyun.com/zh/model-studio/)
- [CLAP: Contrastive Language-Audio Pretraining](https://github.com/LAION-AI/CLAP)
- [Label Studio 文档](https://labelstud.io/)
