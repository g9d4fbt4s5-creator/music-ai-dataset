# Music AI Dataset — 系统架构文档

> 版本: v1.0.0 | 更新: 2026-08-25 | 状态: 生产级可落地

## 一、设计原则

1. **来源隔离**: test/holdout 独立采集，不与 train 同批次
2. **严格顺序**: Stage 编号小的跑完才能进入大的，禁止跳步
3. **三分支决策**: 每个质检环节输出 pass/marginal/fail，fail 不进下游
4. **不重复计算**: MERT/CLAP 嵌入在 Stage 5.3a 提取一次，L2/L4 复用
5. **成本分层**: L3 多模态仅 5% 黄金集，L4 文本 API 全量覆盖
6. **可追溯**: 每个标签带 source 字段（deepseek/knn/golden/l1），支持审计

## 二、完整流水线

```
Stage 0  采集入库（Mac）
├── 生成 ULID / audio_id
├── 散列存储原始音频 (hash前缀两级目录)
└── 精确去重（MD5，秒级）
     └── 重复 → 丢弃/合并，不进下游
    ↓
Stage 1  元数据清洗（Mac）
├── 缺失值处理（3种策略: fill_default/mark_unknown/reject）
├── 字段标准化（artist/genre/duration/license）
└── PII 移除（如果有）
    ↓
Stage 2  格式标准化（GPU ffmpeg）
├── Step 1: 先计算原始指标（只读，不写入文件）
│   └── ffmpeg ebur128 → orig_lufs / orig_dr / orig_peak
│   └── librosa → orig_snr / orig_clip_ratio / orig_silence_ratio
│   └── 写入 audio_manifest.csv（orig_* 字段）
├── Step 2: 转 FLAC 48kHz/24bit 母版
└── Step 3: 响度归一化（loudnorm=I=-16:TP=-1.5:LRA=11）
    ↓
Stage 3  质量清洗（Mac YAMNet + librosa）
├── YAMNet 内容分类（music/speech/noise/vocal，双阈值）
│   ├── music_score > 0.7 → 直接通过
│   ├── music_score < 0.3 → 直接拒绝
│   └── 0.3-0.7 → 兜底检查（SNR/时长）
├── librosa 音质检查（使用 orig_* 原始指标，不受 loudnorm 影响）
│   ├── SNR < 10dB → fail
│   ├── clip_ratio > 5% → fail
│   ├── silence_ratio > 80% → fail/marginal
│   ├── DR < 3 或 DR > 20 → marginal
│   ├── duration < 5s → fail
│   ├── duration > 900s(15min) → tag=long_form
│   └── duration > 1800s(30min) → tag=dj_mix（需切片）
└── 三分支决策
     ├── pass → Stage 4
     ├── marginal → Stage 4（标记 flag_for_review=true）
     └── fail → 丢弃（不进下游，不浪费 GPU 算力）
    ↓
Stage 4  近似去重（需要音频特征）
├── Chromaprint 音频指纹（感知级，防 MP3→FLAC 转码重复）
└── chroma + 余弦相似度（阈值 > 0.95）
    ↓
Stage 5  辅助清洗（条件分支）
├── has_vocals=True → 5.1 语言过滤(Whisper) → 5.2 歌词转写(Demucs+Whisper)
└── has_vocals=False ──────────────────────────────→ 5.3a
└── 5.3a MERT/CLAP 嵌入提取（GPU，768d/512d）
     ↓
     5.3b DBSCAN 聚类（Mac CPU，秒级，eps=5.0 min_samples=2）
    ↓
【L1 物理标签】BPM/调性/SNR/DR/LUFS/频谱质心（整首音频，Mac，可与5.3a并行）
    ↓
【L3 结构标注】Qwen-Omni（5%黄金集，整首音频，MP3压缩后上传）
├── 输入: 整首音频（<10MB Base64，超限转MP3 320kbps或取片段）
├── 输出: 段落结构(Intro/Theme/Improv/Outro)/乐器/情绪/Caption
└── 仅5%黄金集，成本可控
    ↓
Stage 6  预处理输出（GPU）
├── 切片 15s / 50% overlap
└── 训练特征提取（mel-spec / chroma / MFCC，128维，用于模型训练输入）
    ↓
【L2 语义候选】MERT嵌入/CLAP zero-shot（不重复计算，来自5.3a）
    ↓
【L4 传播融合】DeepSeek全量 + KNN传播（量化规则）+ 规则融合
├── DeepSeek V4 Flash: 全量文本标签（genre/mood/instruments/caption）
├── KNN传播: L3黄金集标签传播到相似样本（cosine_dist阈值）
└── 融合规则: 按字段稳定性差异化阈值（见下方L4融合矩阵）
    ↓
ls_preannotations.jsonl → Label Studio 人工校验
    ↓
split_dataset.py（来源隔离: main_pool/test_pool/holdout_pool）
    ↓
04_final_dataset/
```

## 三、L4 融合决策矩阵

```python
def fuse_label(field, deepseek_result, knn_result):
    # Caption 不传播（每首独立生成）
    if field == "caption":
        return {"value": deepseek_result["value"], "source": "deepseek"}

    # Genre：稳定字段，传播阈值放宽
    if field == "genre":
        if knn_result["cosine_dist"] < 0.4 and knn_result["gold_confidence"] in ["high", "medium"]:
            return {"value": knn_result["value"], "source": "knn", "ref": knn_result["neighbor_id"]}
        return {"value": deepseek_result["value"], "source": "deepseek"}

    # Mood/Instruments：不稳定字段，传播阈值严格
    if field in ["mood", "instruments"]:
        if knn_result["cosine_dist"] < 0.25 and knn_result["gold_confidence"] == "high":
            return {"value": knn_result["value"], "source": "knn", "ref": knn_result["neighbor_id"]}
        return {"value": deepseek_result["value"], "source": "deepseek"}

    return {"value": deepseek_result["value"], "source": "deepseek"}
```

| cosine_dist | 含义 | genre传播 | mood/instruments传播 |
|-------------|------|-----------|----------------------|
| < 0.25 | 非常相似 | ✅ | ✅ (需 gold_confidence=high) |
| 0.25–0.40 | 较相似 | ✅ | ❌ |
| > 0.40 | 不太相似 | ❌ | ❌ |

## 四、QC 过滤完整清单

| 检查项 | 阈值 | 分支 | 工具 |
|--------|------|------|------|
| 损坏音频 | 解码失败/文件=0 | fail | ffmpeg |
| 静音 | silence_ratio > 0.8 | fail/marginal | librosa |
| 爆音/削波 | clip_ratio > 0.05 | fail | librosa |
| 超短 | duration < 5s | fail | ffprobe |
| 超长(DJ-mix) | duration > 1800s | tag=dj_mix | ffprobe |
| 长曲 | duration > 900s | tag=long_form | ffprobe |
| 非音乐 | YAMNet music_score < 0.3 | fail | YAMNet |
| 严重低SNR | snr_db < 10 | fail | librosa |
| 动态范围异常 | DR < 3 或 DR > 20 | marginal | librosa |
| 低质量源 | sr < 22050 / 单声道 / br < 128k | marginal | ffprobe |

## 五、模型分工与成本

| 层级 | 模型 | 输入 | 输出 | 单首成本 | 覆盖比例 |
|------|------|------|------|----------|----------|
| L1 | librosa/madmom | 音频 | BPM/调性/SNR | 免费 | 100% |
| L2 | MERT-v1-95M | 音频 | 768d嵌入 | GPU免费 | 100% |
| L3 | Qwen3.5-Omni-Flash | 音频直接输入 | 段落/乐器/情绪/Caption | ~¥0.1 | 5% |
| L4 | DeepSeek V4 Flash | L1+L2特征JSON | genre/mood/instruments/Caption | ~¥0.001 | 100% |
| L4 | KNN(cosine) | MERT嵌入 | 黄金集标签传播 | 免费 | 100% |

### 替代方案（利用 iOS 科学上网）

| 模型 | 音频上限 | 质量 | 科学上网 | 用途 |
|------|----------|------|----------|------|
| Qwen3.5-Omni-Flash | 20min | ⭐⭐⭐ | ❌ 直连 | 默认主力 |
| Qwen3.5-Omni-Plus | 3h | ⭐⭐⭐⭐ | ❌ 直连 | 长音频/精细结构 |
| Gemini 2.0 Flash | 9.5h | ⭐⭐⭐⭐ | ✅ 需要 | iOS端调用，高质量需求 |
| GPT-4o | 10min | ⭐⭐⭐⭐⭐ | ✅ 需要 | iOS端调用，疑难样本 |

## 六、Mac/GPU/iOS 三层架构

```
Mac（主节点，无科学上网）
├── Stage 0/1/3/4/5.3b/L1/L4: 本地计算
├── Label Studio: 本地部署
├── 代码管理: git
└── 数据存储: 主副本 + OSS备份

GPU（流式计算节点，AutoDL RTX 4090）
├── Stage 2/5.1/5.2/5.3a/6: 重计算
├── 模型权重: /root/autodl-tmp/models/
└── 产物回传: scp/rsync → Mac

iOS（科学上网节点，iPhone/iPad）
├── 数据采集: YouTube/SoundCloud 下载
├── API调用: Gemini/GPT-4o（需要科学上网的模型）
├── 同步: AirDrop / iCloud Drive → Mac
└── 不跑重计算（MERT/Demucs等在Mac/GPU）
```

## 七、数据生命周期

```
原始音频 → Stage2母版FLAC → Stage6切片/特征 → 训练完成
   │           │                    │
   │           ├── 保留(母版)       ├── 保留(训练输入)
   │           └── 生成后可删原始   └── 训练完成后可删切片
   └── OSS冷库备份(永久)
```

- **GPU清理时机**: 原始音频→母版生成后删；母版→切片/嵌入生成后删
- **Mac保留**: 母版FLAC + 嵌入 + 标注 + 元数据
- **OSS备份**: 原始音频 + 母版 + 标注（只读账号）
