# MIR 评测模块（mir_eval 标准评测库）

> 用 mir_eval 标准评测库，批量计算模型/工具预测错误率，写入 manifest 统计。

## 一、模块结构

```
scripts/06_evaluation/
├── eval_bpm.py          # BPM/节拍检测评测（essentia/madmom/librosa vs 真值）
├── eval_f0.py           # 基频检测评测（torchcrepe vs 真值）【待实现】
├── eval_onset.py        # onset检测评测（madmom vs 真值）【待实现】
├── eval_pipeline.py     # 批量评测主脚本【待实现】
└── README.md            # 本文档
```

## 二、真值来源（不需要人工现标）

| 来源 | 适用指标 | 获取方式 | 推荐度 |
|------|---------|---------|--------|
| **公开数据集自带标注** | BPM / key / onset / f0 | GTZAN、Ballroom、SMC、JCS、MAESTRO | ⭐⭐⭐⭐⭐ |
| **MIDI 文件导出的时间戳** | onset / beat / note | Lakh MIDI、MAESTRO、自己制作的 MIDI | ⭐⭐⭐⭐ |
| **节拍跟踪软件的"高置信"输出** | BPM / beat | madmom 的 beat 结果，人工抽检 10% 后作为伪真值 | ⭐⭐⭐ |
| **工具分歧仲裁** | 全部 | 全部工具输出差异大的样本，人工听 50-100 首仲裁 | ⭐⭐ |

### 具体公开数据集（可直接下载真值）

| 数据集 | 规模 | 有真值的指标 | 适用场景 | 下载 |
|--------|------|------------|---------|------|
| **JCS** (Jazz-Choro-Samba) | 200+ 首 | BPM / beat 真值 | **爵士场景，最适合你的项目** | 搜索 "JCS dataset beat tracking" |
| **Ballroom** | 698 首 | BPM 真值 | 舞曲 BPM，essentia vs madmom 对比 | 搜索 "Ballroom dataset BPM" |
| **SMC MIREX** | 217 首 | beat / onset 真值 | 节拍跟踪评测 | 搜索 "SMC MIREX dataset" |
| **MAESTRO** | 200+ 小时 | note / onset / beat（MIDI） | 钢琴音乐，MIDI 作为真值 | Magenta 官网 |
| **GTZAN** | 1000 首 | genre（无 BPM 真值） | 流派分类，不适合 BPM 评测 | 公开下载 |

> **对于你的 Jazz 项目，最推荐 JCS 数据集**——它专门包含爵士、choro、samba 的 BPM 和 beat 真值，与你的场景完全匹配。

## 三、人工在 mir_eval 中的真实角色

人工不是"从零标注 BPM"，而是做**仲裁和校验**：

```
工具 A (essentia) 预测 BPM = 120
工具 B (madmom)   预测 BPM = 87
差异 > 5% → 触发人工仲裁
    ↓
人工听 10 秒，判断哪个更接近真实速度
    ↓
人工裁决 = 120 → 写入真值库
```

**人工只处理"有争议的样本"，而不是全部重标。**

## 四、没有真值时的替代方案：工具间一致性

如果你的 Jazz 500 首确实没有任何真值，mir_eval 仍然可以做**工具间一致性分析**：

```python
# 不计算"误差"，计算"工具间分歧"
essentia_bpm = [120, 95, 87, ...]
madmom_bpm   = [121, 96, 87, ...]

# 计算两者差异
disagreement = np.abs(np.array(essentia_bpm) - np.array(madmom_bpm))

# 分歧大的样本 → 人工仲裁候选
high_disagreement = [i for i, d in enumerate(disagreement) if d > 5]
```

**价值**：找出"工具们打起来"的样本，优先人工复核，逐步积累真值。

## 五、eval_bpm.py 使用方法

### 5.1 用公开数据集评测（需要下载数据集）

```bash
# Ballroom 数据集评测
python scripts/06_evaluation/eval_bpm.py \
    --audio-dir data/datasets/ballroom/audio \
    --truth-dir data/datasets/ballroom/truth \
    --dataset-type ballroom \
    --tools essentia,madmom \
    --report-json reports/bpm_eval_ballroom.json \
    --report-html reports/bpm_eval_ballroom.html

# JCS 数据集评测（最适合爵士）
python scripts/06_evaluation/eval_bpm.py \
    --audio-dir data/datasets/jcs/audio \
    --truth-dir data/datasets/jcs/truth \
    --dataset-type jcs \
    --tools essentia,madmom,librosa \
    --report-csv reports/bpm_eval_jcs.csv
```

### 5.2 工具间一致性分析（无真值）

```bash
# 对自己的数据集做一致性分析
python scripts/06_evaluation/eval_bpm.py \
    --audio-dir data/01_preprocess/processed_master \
    --tools essentia,madmom,librosa \
    --consistency-only \
    --high-disagreement-threshold 5 \
    --export-disagreement reports/high_disagreement_samples.csv \
    --report-csv reports/bpm_consistency.csv
```

### 5.3 只评测单个工具

```bash
python scripts/06_evaluation/eval_bpm.py \
    --audio-dir data/datasets/jcs/audio \
    --truth-dir data/datasets/jcs/truth \
    --tools madmom \
    --report-json reports/bpm_eval_madmom.json
```

### 5.4 限制处理数量（测试用）

```bash
python scripts/06_evaluation/eval_bpm.py \
    --audio-dir data/01_preprocess/processed_master \
    --tools librosa,essentia \
    --consistency-only \
    --limit 10
```

## 六、评测指标（mir_eval.beat）

| 指标 | 说明 | 范围 |
|------|------|------|
| **F1-score** | 节拍检测的F1分数 | 0-1 |
| **Cemgil** | Cemgil et al. 2007 的评测指标 | 0-1 |
| **Goto** | Goto and Muraoka 1997 的评测指标 | 0-1 |
| **P-score** | McKinney et al. 2007 的评测指标 | 0-1 |
| **BPM 绝对误差** | 预测BPM与真值BPM的绝对差 | BPM |
| **BPM 相对误差** | 预测BPM与真值BPM的相对差 | % |
| **BPM 准确率(±5%)** | 预测在真值±5%范围内的比例 | % |
| **BPM 准确率(含倍频)** | 考虑2倍/0.5倍频的准确率 | % |

## 七、支持的 BPM 提取工具

| 工具 | 算法 | 准确率 | 速度 | 依赖 |
|------|------|--------|------|------|
| **madmom** | RNN + TempoEstimation | ⭐⭐⭐⭐⭐ | 慢 | madmom, onnxruntime |
| **essentia** | RhythmExtractor2013 + PercivalBpmEstimator | ⭐⭐⭐⭐ | 中 | essentia |
| **librosa** | beat_track | ⭐⭐⭐ | 快 | librosa |

## 八、输出文件格式

### 8.1 JSON 报告

```json
{
  "n_audio_files": 200,
  "tools": ["essentia", "madmom"],
  "n_truth": 200,
  "n_evaluated": 400,
  "essentia_mean_abs_error": 3.2,
  "essentia_mean_rel_error": 0.025,
  "essentia_accuracy_1x": 0.85,
  "essentia_accuracy_2x": 0.92,
  "essentia_f1_score": 0.78,
  "madmom_mean_abs_error": 1.8,
  "madmom_mean_rel_error": 0.014,
  "madmom_accuracy_1x": 0.92,
  "madmom_accuracy_2x": 0.96,
  "madmom_f1_score": 0.85
}
```

### 8.2 CSV 详细结果

```csv
audio_id,tool,predicted_bpm,truth_bpm,bpm_absolute_error,bpm_relative_error,bpm_correct_1x,bpm_correct_2x,f1_score
track_001,essentia,120.5,120,0.5,0.004,True,True,0.82
track_001,madmom,119.8,120,0.2,0.002,True,True,0.88
...
```

### 8.3 一致性分析 CSV

```csv
audio_id,n_tools,tools,mean_bpm,std_bpm,min_bpm,max_bpm,range_bpm,diff_essentia_madmom
track_001,2,"essentia,madmom",120.15,0.49,119.8,120.5,0.7,0.7
track_002,2,"essentia,madmom",128.5,45.96,95.7,161.5,65.8,65.8
...
```

## 九、求职作品集展示建议

对于求职作品集（腾讯音乐AI创新内容运营岗），建议展示：

1. **用 JCS 数据集跑 essentia vs madmom 的 BPM 对比**
   - 展示评测指标表格（F1/Cemgil/Goto/P-score）
   - 展示误差分布直方图
   - 结论：madmom 比 essentia 准确率高 X%，但速度慢 Y 倍

2. **对自己的 Jazz 500 首做工具间一致性分析**
   - 展示高分歧样本比例
   - 展示高分歧样本的人工仲裁结果
   - 结论：X% 的样本工具间一致，Y% 的样本需要人工仲裁

3. **写一篇评测报告**
   - 标题：《Jazz 音乐 BPM 检测工具评测报告：essentia vs madmom vs librosa》
   - 包含：实验设置、评测指标、结果分析、结论与建议
   - 作为应聘附件，展示数据分析和评测能力

## 十、依赖安装

```bash
# 核心依赖
pip install mir_eval numpy scipy pandas

# BPM 提取工具（按需安装）
pip install librosa          # 基础BPM检测
pip install essentia         # 高级BPM检测（RhythmExtractor2013）
pip install madmom-onnx     # 深度学习BPM检测（最准确）

# 全部安装
pip install mir_eval librosa essentia madmom-onnx
```

## 十一、待实现

- [ ] eval_f0.py — 基频检测评测（torchcrepe vs 真值）
- [ ] eval_onset.py — onset检测评测（madmom vs 真值）
- [ ] eval_pipeline.py — 批量评测主脚本（统一入口）
- [ ] 可视化报告生成（plotly 交互式图表）
- [ ] JCS 数据集下载脚本
- [ ] 评测结果写入 audio_manifest.csv

---

**一句话总结**：人工不会逐首标 BPM，mir_eval 的真值来自公开数据集（JCS/Ballroom 最适合爵士）、MIDI 导出、工具分歧仲裁。对于求职作品集，用 JCS 数据集跑 essentia vs madmom 的 BPM 对比就够了，不需要自己造真值。
