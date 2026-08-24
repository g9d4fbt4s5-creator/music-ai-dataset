# 标注员操作手册 (Labeling Guide)

> 版本: V4 | 更新: 2026-08-25
> 模板: `configs/label_studio/labeling_interface_v4.xml`
> 适用: 音乐数据集人工校验 (HITL 闭环)

---

## 一、标注流程总览

```
打开任务 → 听音频 → 检查预标注 → 修正错误 → 填写审核决策 → 提交
   ↑                                                          ↓
   └──────── 边际样本/黄金集会有顶部提示 ────────────────────┘
```

**核心原则：预标注已填充 80% 内容，你只需要校验和修正，不需要从零标注。**

---

## 二、字段说明

### 2.1 元数据只读区（不可编辑）

顶部灰色区域显示预标注结果，仅供参考：

| 字段 | 含义 | 来源 |
|------|------|------|
| BPM | 每分钟节拍数 | L1 librosa/madmom |
| 调性 | 主音+调式 | L1 essentia 三重投票 |
| 响度 | LUFS 积分响度 | L1 ffmpeg ebur128 |
| SNR | 信噪比 (dB) | L1 原始音频计算 |
| 来源批次 | 数据采集来源 | Stage 0 入库记录 |
| 传播来源 | KNN 传播自哪个黄金集 | L4 融合 |
| KNN相似度 | 与最近黄金集的余弦相似度 | L4 融合 |
| 融合策略 | 该字段用 DeepSeek 还是 KNN | L4 融合 |
| L2置信度 | CLAP zero-shot 置信度 | L2 语义候选 |
| 标注来源 | golden_set / knn / deepseek | L4 融合 |

### 2.2 音乐结构段落（可拖动边界）

在波形上拖动区间边界，标注 10 类结构：

| 标签 | 含义 | 典型位置 |
|------|------|----------|
| Intro | 引子 | 开头，建立氛围 |
| Verse | 主歌 | 叙事段落 |
| Pre-Chorus | 预副歌 | Verse→Chorus 过渡 |
| Chorus | 副歌 | 高潮/记忆点 |
| Bridge | 桥段 | 对比段落，常转调 |
| Instrumental | 器乐段 | 无人声的演奏段 |
| Solo | 独奏段 | 某乐器华彩 |
| Breakdown | 分解段 | 极简配器， buildup 前 |
| Outro | 尾奏 | 结尾淡出 |
| Silence | 静音段 | 纯静音/环境音 |

**操作技巧**：
- 点击标签 → 在波形上拖拽创建区间
- 拖动区间边缘调整边界
- 区间可以重叠（如 Solo 在 Instrumental 内）
- 预标注已填充，只需修正边界位置

### 2.3 乐器配器区间（GM128 标准）

与结构区间在同一波形上，用**不同颜色**区分：

| 标签 | GM编号 | 颜色 |
|------|--------|------|
| 钢琴 Piano | GM001 | 蓝 |
| 木吉他 Acoustic Guitar | GM025 | 绿 |
| 电吉他 Electric Guitar | GM027 | 橙 |
| 弦乐 Strings | GM048 | 紫 |
| 鼓组 Drum Kit | GM118 | 红 |
| 贝斯 Bass | GM033 | 深蓝 |
| 人声演唱 Lead Vocal | GM091 | 黄绿 |
| 和声 Backup Vocal | GM092 | 黄 |
| 合成器 Synth | GM098 | 青 |
| 管乐 Brass | GM061 | 粉红 |

**注意**：乐器区间和结构区间都画在同一波形上。结构区间在上半区创建，乐器区间在下半区创建，视觉上用颜色和位置区分。

### 2.4 流派（主标签 + 次标签）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| genre_primary | 单选 | ✅ | 最主要的流派 |
| genre_secondary | 多选 | ❌ | 融合的次要流派 |

**主标签选择规则**：
- 听感上最突出的流派选为主标签
- 融合风格选最核心的那个（如 Jazz Fusion → 主标签 Jazz，次标签 Fusion）
- 不确定时选最接近的大类

**次标签选择规则**：
- 明显融合了第二种风格时才选
- 纯风格不选次标签
- 最多选 2 个次标签

### 2.5 情绪（多选 + VAD 备注）

| 字段 | 类型 | 必填 |
|------|------|------|
| mood | 多选 | ❌（建议至少选1个） |
| mood_vad | 文本 | ❌ |

10 类情绪：欢快活泼 / 温柔舒缓 / 激昂热血 / 忧郁伤感 / 浪漫甜蜜 / 空灵治愈 / 紧张悬疑 / 大气史诗 / 神秘 / 怀旧

**VAD 备注**（可选）：
- Valence (效价): 0=负面, 1=正面
- Arousal (唤醒度): 0=平静, 1=激动
- Dominance (支配度): 0=被动, 1=主动
- 格式：`Valence=0.8, Arousal=0.3, Dominance=0.5`

### 2.6 调性与和声（可选，L1 已预填）

| 字段 | 类型 | 说明 |
|------|------|------|
| key_tonic | 单选 | 主音 (C/B/Atonal等) |
| key_mode | 单选 | 调式 (Major/Minor/Modal/Atonal) |
| key_modulation | 文本 | 转调记录（可选） |

**标注规则**：
- L1 已自动提取，听一下确认是否正确
- 不确定时选 "Atonal 无调性" 或留空
- 转调记录格式：`时间点 从 X 转至 Y`，如 `0:45 从 C Major 转至 F Minor`
- 无转调留空

### 2.7 人声

| 选项 | 含义 |
|------|------|
| 纯器乐 Instrumental | 全程无人声 |
| 有人声 Vocal | 以人声为主 |
| 人声+器乐混合 Mixed | 人声和器乐段落交替/并存 |

### 2.8 质量评估（A-E 级）

| 等级 | 分数 | 含义 | 自动路由 |
|------|------|------|----------|
| A级 | 5分 | 母带级，音质完美 | 正常通过 |
| B级 | 4分 | 良好，轻微瑕疵不影响 | 正常通过 |
| C级 | 3分 | 可接受，有明显瑕疵 | 正常通过 |
| D级 | 2分 | 较差，底噪/削波/失真 | ⚠️ 自动标记需二次复核 |
| E级 | 1分 | 极差/废弃，无法使用 | ⚠️ 自动触发 reject |

**注意**：D/E 级会在导出时自动设置 `review_decision = reject` 和 `review_flag = needs_second_review`。

### 2.9 Caption（自然语言描述）

用音乐评论家的口吻写一句听后感，**禁止罗列技术参数**。

✅ 正确示例：
> "一首慵懒的午夜爵士三重奏，萨克斯在钢琴的铺陈下娓娓道来，贝斯线条沉稳而富有弹性。"

❌ 错误示例：
> "BPM 120，C大调，SNR 28dB，有萨克斯和钢琴。"

**修正规则**：预标注的 Caption 由 DeepSeek 生成，可能有幻觉或不准确，听后修正。

### 2.10 审核决策

| 字段 | 选项 | 说明 |
|------|------|------|
| review_decision | approve / approve_with_edits / needs_revision / reject | 必填 |
| golden_set | yes / no | 是否加入黄金集 |
| review_flag | no_review_needed / needs_second_review / disagreement_detected / golden_standard | 复核标记 |

**决策规则**：
- `approve`：预标注完全正确，无需修改
- `approve_with_edits`：修正了部分标签，但整体可用
- `needs_revision`：错误较多，需要重新标注
- `reject`：音频本身有问题（损坏/非音乐/极低质量）

**黄金集选择**：
- 标注质量高、结构清晰、有代表性的样本选 `yes`
- 黄金集用于 L3 多模态标注和 KNN 传播源
- 建议黄金集占总量 5-10%

### 2.11 修正说明

填写你改了什么、为什么改。格式示例：
> "预标注流派为 Bebop，实际听感为 Cool Jazz，已修正主流派。结构段落边界微调了 Chorus 起始点。"

---

## 三、预标注修正规范

### 3.1 必须修正的情况

| 情况 | 处理 |
|------|------|
| 流派明显错误 | 修正 genre_primary |
| 人声判断错误（有人声标纯器乐） | 修正 vocal_presence |
| 结构段落边界偏差 >5秒 | 拖动修正 |
| Caption 与听感不符 | 重写 |
| 质量等级与实际不符 | 修正 quality_grade |

### 3.2 可以保留的情况

| 情况 | 处理 |
|------|------|
| 次标签多一个或少一个 | 不强制修改 |
| 情绪多选 2-3 个都合理 | 保留 |
| 调性 L1 结果与听感接近 | 保留 |
| 结构边界偏差 <3秒 | 保留 |

### 3.3 绝对不要做的事

- ❌ 不要修改元数据只读区（那是 L1/L4 自动生成的）
- ❌ 不要因为"不确定"就全部选第一个选项
- ❌ 不要跳过音频直接提交（至少听 30 秒）
- ❌ 不要在 Caption 里写技术参数

---

## 四、效率技巧

### 4.1 Label Studio 快捷键

| 快捷键 | 功能 |
|--------|------|
| `空格` | 播放/暂停 |
| `←` / `→` | 快退/快进 5秒 |
| `↑` / `↓` | 音量 +/- |
| `Ctrl+Enter` | 提交标注 |
| `Ctrl+Z` | 撤销 |
| `鼠标滚轮` | 缩放波形 |

### 4.2 批量处理建议

1. **先听后标**：完整听一遍，心里有数再开始修正
2. **从结构开始**：先确认/修正结构段落，再填标签
3. **利用预标注**：80% 的内容已经填好，只需要检查和微调
4. **边际样本优先**：顶部有 ⚠️ 警告的样本需要更仔细检查
5. **黄金集标记**：遇到特别好的样本记得选 `golden_set = yes`

### 4.3 单首标注时间目标

| 样本类型 | 目标时间 |
|----------|----------|
| 预标注准确，只需确认 | 1-2 分钟 |
| 需要修正部分标签 | 3-5 分钟 |
| 边际样本/结构复杂 | 5-8 分钟 |
| 黄金集精标 | 8-12 分钟 |

---

## 五、疑难案例

### 5.1 Atonal（无调性）怎么标？

- 听不出明确主音 → `key_tonic = Atonal 无调性`
- 现代爵士/自由即兴常见
- 不要强行选一个主音

### 5.2 Fusion 风格怎么选主次？

| 实际风格 | 主标签 | 次标签 |
|----------|--------|--------|
| Jazz Fusion (爵士融合) | 爵士 Jazz | Fusion跨界 |
| Jazz-Rock | 爵士 Jazz 或 摇滚 Rock | Fusion跨界 |
| Electronic Jazz | 电子 Electronic | 爵士 Jazz |
| Classical Crossover | 古典 Classical | Fusion跨界 |

原则：听感上哪个更突出选哪个为主。

### 5.3 纯器乐有人声采样怎么办？

- 人声采样/念白作为音色使用 → `纯器乐 Instrumental`
- 有明确旋律的人声演唱 → `有人声 Vocal`
- 人声段落和器乐段落交替 → `人声+器乐混合 Mixed`

### 5.4 多段落情绪变化大怎么标？

- 选最突出的 1-2 个情绪
- 如果全程变化大，可以多选 3 个
- 在 mood_vad 里备注变化：`前段平静(0.3)，后段激昂(0.8)`

### 5.5 质量 D/E 级但音乐本身很好？

- D/E 级只评估**音频质量**（底噪/削波/失真/压缩），不评估音乐价值
- 低质量录音但音乐好 → 仍标 D/E，但 review_decision 可以选 `approve_with_edits`
- 在 annotation_note 里说明："音乐质量好，但录音底噪大"

---

## 六、V4 字段与 L1-L4 预标注映射

| 模板字段 | 数据来源 | 预标注阶段 | 人工是否需修正 |
|----------|----------|------------|----------------|
| qc_flags / marginal_display | QC Gate 输出 | Stage 3 | 只读 |
| bpm / key / lufs / snr | librosa/madmom/essentia | L1 物理标签 | key 需确认 |
| structure 区间 | Qwen-Omni / Gemini API | L3 结构标注 (黄金集) | ✅ 修正边界 |
| instruments 区间 | YAMNet + Demucs 辅助 | L2 语义候选 | ✅ 修正 |
| genre_primary / genre_secondary | CLAP zero-shot + KNN 传播 | L2/L4 融合 | ✅ 修正 |
| mood + mood_vad | DeepSeek 文本生成 + KNN | L4 融合 | ✅ 修正 |
| key_tonic / key_mode | L1 essentia 预填 | L1 物理标签 | 确认即可 |
| vocal_presence | YAMNet vocal_score | L1/L2 | ✅ 修正 |
| quality_grade | librosa 质检 + 人工 | Stage 3 + 人工 | ✅ 修正 |
| caption | DeepSeek 伪描述 | L4 融合 | ✅ 重写 |
| propagation_source / knn_sim | KNN 传播元数据 | L4 融合 | 只读 |
| golden_set / review_flag | 人工审核后写入 | HITL 闭环输出 | 人工填写 |

---

## 七、多标签权重的工程实现

Label Studio 原生不支持标签权重。权重在**导出脚本**中按业务规则赋予：

```python
# scripts/03_labelstudio/export_annotations.py
def parse_genre_weights(annotation):
    """主标签权重=1.0，次标签权重=0.3"""
    primary = annotation["genre_primary"]      # 单选
    secondaries = annotation.get("genre_secondary", [])  # 多选
    weights = {primary: 1.0}
    for sec in secondaries:
        weights[sec] = 0.3
    return weights
```

**面试表述**："模板层用主/次字段分离实现多标签，权重在导出脚本中按业务规则赋予，不依赖标注工具原生能力。"

---

## 八、D/E 级自动路由规则

导出时自动处理低质量样本：

```python
def auto_route_by_quality(annotation):
    grade = annotation.get("quality_grade", "")
    if "D级" in grade or "E级" in grade:
        annotation["review_decision"] = "reject 拒绝"
        annotation["review_flag"] = "needs_second_review 需二次复核"
    return annotation
```

---

## 九、版本历史

| 版本 | 定位 | 状态 |
|------|------|------|
| V1 | 基础7类区间，无工程对接 | ❌ 废弃 |
| V2 | 业务需求文档，无交互设计 | 📄 仅参考 |
| V3 | 界面草图，工程对接好但业务字段不足 | 🦴 骨架保留 |
| V4 | 整合版，当前采用 | ✅ 使用中 |
