# 轻量级决策索引 (DECISIONS.md)

非架构级决策记录在此，与 5 份核心 ADR 互补。
ADR 记录不可逆的架构决策，本文档记录可调整的参数、阈值和工程约定。

**最后更新**: 2026-08-26
**相关 ADR**: [ADR-001](docs/adr/ADR-001-QC-GATE-THRESHOLDS.md) | [ADR-002](docs/adr/ADR-002-HITL-ASYNC-CLOSED-LOOP.md) | [ADR-003](docs/adr/ADR-003-DATA-SPLIT-AND-SOURCE-ISOLATION.md) | [ADR-004](docs/adr/ADR-004-L1-L4-PREANNOTATION-TIERED-ARCHITECTURE.md) | [ADR-005](docs/adr/ADR-005-LABEL-MAPPING-VERSIONING.md)

---

## 1. 预标注与传播阈值

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| L4 KNN 传播阈值 | genre_dist < 0.40, mood/instruments_dist < 0.25 | 2026-08-25 | ADR-004 | cosine 距离阈值，低于此值才传播标签 |
| L4 融合权重 | L1 物理特征 0.3, L2 语义 0.4, L3 结构 0.3 | 2026-08-25 | ADR-004 | 加权融合，L2 语义权重最高 |
| KNN 邻居数 | k=5 | 2026-08-25 | ADR-004 | 5 个最近邻投票 |
| 黄金集抽样比例 | 5%（50-500条） | 2026-08-26 | ADR-003/004 | 从 main_pool 抽样人工精标，作为 KNN 种子 |

---

## 2. 数据划分与采样

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| 划分比例（500首） | 80/10/10（400 train / 50 val / 50 test） | 2026-08-26 | ADR-003 | holdout 独立 50-100 首，ood 独立 20-50 首 |
| 跨集去重阈值 | song_group_id 精确匹配 / 指纹相似度 ≥ 0.5 | 2026-08-26 | ADR-003 | 0.5 检测同一首歌的不同版本（cover/remix） |
| 近似去重阈值 | Chromaprint 相似度 ≥ 0.92 | 2026-08-21 | ADR-003 | 标记 marginal，人工确认是否重复 |
| 分布对齐策略 | P0/P1/P2 三层优先级，加权不删样本 | 2026-08-25 | ADR-003 | 加权采样，不删除少数类样本 |
| 艺术家级隔离 | 同一 artist_id 必须在同一子集 | 2026-08-26 | ADR-003 | MIR 防泄漏底线 |
| 歌曲级隔离 | 同一 song_group_id 必须在同一子集 | 2026-08-26 | ADR-003 | 不同版本/翻唱/remix 不跨集 |

---

## 3. QC 与质量阈值

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| SNR pass 阈值 | ≥ 12 dB | 2026-08-24 | ADR-001 | 从 15dB 放宽，9首听检 100% 可接受 |
| SNR marginal | 10 ~ 12 dB | 2026-08-24 | ADR-001 | < 10dB fail |
| 静音比例 marginal | 60% ~ 80% | 2026-08-25 | ADR-001 | 从 50% 放宽，爵士长前奏/间奏正常 |
| 静音比例 fail | > 80% | 2026-08-25 | ADR-001 | 无有效音乐内容 |
| LUFS pass | -28 ~ -8 LUFS | 2026-08-25 | ADR-001 | 音乐标准响度区间 |
| LUFS fail | < -36 或 > -4 | 2026-08-25 | ADR-001 | fail 边界从 -6 放宽到 -4（金属/电子正常） |
| 削波比例 marginal | 2% ~ 5% | 2026-08-25 | ADR-001 | 经验阈值，待 500 首验证 |
| 削波比例 fail | > 5% | 2026-08-25 | ADR-001 | 严重失真 |
| YAMNet music_score pass | > 0.7 | 2026-08-24 | ADR-001 | 高置信度直接放行 |
| YAMNet music_score fail | < 0.3 | 2026-08-24 | ADR-001 | 判为非音乐，不因域外样本调整 |
| 时长 fail | < 5s | 2026-08-25 | ADR-001 | 碎片无音乐信息 |
| 时长 long_form | > 15min | 2026-08-25 | ADR-001 | 标记 marginal，不直接 fail |

---

## 4. HITL 与听检

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| 听检任务类型 | 9 种（qc_snr/qc_content/dedup/knn/unmapped/cluster/segment/source/post_threshold） | 2026-08-25 | ADR-002 | 按阶段按需启用，不全做 |
| waiting_pool 超时 | 7 天未听检自动降级 | 2026-08-25 | ADR-002 | 按保守策略处理 |
| marginal 率触发听检 | > 25% 自动创建 SNR 听检任务 | 2026-08-25 | ADR-002 | 数据驱动触发 |
| unmapped 标签触发 | 频次 > 5 自动创建映射审核 | 2026-08-25 | ADR-002 | 数据驱动触发 |
| 阈值后抽检回滚 | 劣质率 > 20% 自动回滚 | 2026-08-25 | ADR-002 | post_threshold_audit |
| badcase 双确认 | 过程态 → 双确认 → 终态 badcase_pool | 2026-08-25 | ADR-002 | 避免误报污染 |
| badcase 收集边界 | 只收集"流程出错"样本，不收集普通质量 fail | 2026-08-26 | ADR-002 | fail 是正常清洗，badcase 是流程异常 |

---

## 5. 切片与增强

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| 切片策略 | 先标注后切片，短曲不切，乐段边界优先 | 2026-08-24 | ADR-004 | 避免在 solo 中间切断 |
| 切片长度 | 30s / 60s 两档 | 2026-08-24 | ADR-004 | 根据任务选择 |
| 数据增强 label_invariant | 增强后标签不变（音量/噪声/速度微调） | 2026-08-25 | ADR-004 | recipe.json 注释，待实现 |

---

## 6. 标签映射与标准化

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| 映射字典唯一真相源 | configs/label_mapping_dict.json | 2026-08-25 | ADR-005 | 禁止硬编码映射 |
| 版本号规则 | 语义化版本 v<major>.<minor> | 2026-08-25 | ADR-005 | merge_mapping.py 自动升级 |
| 合并原子性 | 预校验 + 原子写入，任何错误整体取消 | 2026-08-25 | ADR-005 | 不提供 --force |
| hard_blacklist | speech, silence, podcast, white noise 等 | 2026-08-25 | ADR-005 | L4 传播时直接丢弃 |
| soft_blacklist | noise, low quality, distorted, clipping 等 | 2026-08-25 | ADR-005 | 标记 marginal，人工确认 |
| 乐器映射标准 | GM128 (General MIDI Level 1) | 2026-08-20 | ADR-005 | instrument_gm128_map，33 个乐器 |

---

## 7. 来源与域外样本

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| source_type 排除集合 | ace_studio_generated, demucs_vocals, ace_studio_generated_demucs_vocals 等 | 2026-08-26 | ADR-003 | Stage 1/划分时自动排除 |
| YAMNet 误杀根因 | Ace Studio 生成 + Demucs 分轨人声单轨，属域外样本 | 2026-08-25 | ADR-001/003 | 不调 YAMNet 阈值，从源头排除 |
| 噪音来源扩展 | 历史录音 + 曲风/乐器/效果器/演奏法正常特征 | 2026-08-24 | ADR-001 | 听检模板选项已更新 |
| OOD 集用途 | 只做泛化分析，不参与官方排名 | 2026-08-26 | ADR-003 | 独立采集，风格/来源与 main_pool 差异大 |

---

## 8. 基础设施与工程约定

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| Label Studio 本地文件服务 | LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true, recursive_scan=True | 2026-08-25 | — | 音频 404 问题已解决 |
| Label Studio 音频路径格式 | /data/local-files/?d=相对路径 | 2026-08-25 | — | 不能用本地绝对路径 |
| Label Studio API 认证 | Legacy API Token（40位十六进制） | 2026-08-25 | — | JWT refresh token 不可用 |
| .gitignore 数据产物 | data/00.5_cleaned/reports/ 下 CSV 不跟踪 | 2026-08-25 | — | 本地保留，Git 移除跟踪 |
| 物理文件不删除 | fail 样本物理归档，仅逻辑排除 | 2026-08-25 | ADR-001 | 防止误判 |
| conda 环境 | labelstudio-env (Python 3.11.13) | 2026-08-20 | — | 所有脚本在此环境运行 |

---

## 9. 分布统计

| 决策 | 内容 | 日期 | 相关 ADR | 备注 |
|------|------|------|----------|------|
| P0 必统计子集 | train, val, golden, holdout | 2026-08-25 | ADR-003 | 用于域偏移告警 |
| P1 按需统计 | ood, marginal | 2026-08-25 | ADR-003 | 仅输出统计，不参与偏移告警 |
| fail 统计 | 仅统计数量，不参与子集对比 | 2026-08-25 | ADR-003 | 已逻辑剔除 |
| 域偏移告警阈值 | P0 子集间某特征均值差异 > 1 个标准差 | 2026-08-25 | ADR-003 | 写入 distribution_warnings.csv |

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-08-26 | 初始创建，汇总 9 大类决策 |
