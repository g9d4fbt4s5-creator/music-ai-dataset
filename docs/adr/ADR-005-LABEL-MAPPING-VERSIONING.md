# ADR-005: 标签映射字典版本管理

## 状态
Accepted

## 日期
2026-08-26

## 决策者
数据工程团队

## 影响范围
- `configs/label_mapping_dict.json`（唯一真相源）
- `scripts/02_preannotation/merge_mapping.py`（原子合并脚本）
- `scripts/02_preannotation/tag_mapping_musiccaps.py`（MusicCaps映射）
- `data/02_preannotation/l4_propagated/`（L4传播标签输出）

---

## 背景

音乐数据集构建过程中，标签映射字典是连接"模型原始输出"和"标准化标签体系"的唯一真相源。
模型（CLAP/MERT/DeepSeek）输出的标签是自由文本，需要通过映射字典转换为标准化的
genre/instrument/mood 等标签，才能用于训练和评估。

映射字典会随着数据量增长和人工审核持续迭代。如果没有严格的版本管理，会导致：
- 不同批次的数据使用不同版本的映射，标签不一致
- 映射变更后无法追溯"为什么这个标签被映射到那个标准标签"
- 下游评测脚本因映射字典格式变更而崩溃
- 无法回滚到之前的映射版本

本 ADR 固化标签映射字典的版本管理规则、合并流程和回滚机制。

---

## 决策

### 1. 唯一真相源（Single Source of Truth）

| 维度 | 决策 |
|------|------|
| **文件位置** | `configs/label_mapping_dict.json` |
| **格式** | JSON，UTF-8 编码 |
| **访问方式** | 所有脚本通过读取此文件获取映射规则，禁止硬编码映射 |
| **副本** | 不允许在其他位置维护映射副本（如脚本内的硬编码字典） |

**当前结构**（v2.0）：

```json
{
  "version": "v2.0",
  "updated_at": "2026-08-25",
  "changelog": [...],
  "compatible_preannotation_versions": [...],
  "version_rule": {"major": "...", "minor": "..."},
  "hard_blacklist": [...],
  "soft_blacklist": [...],
  "instrument_gm128_map": {...},
  "genre_map": {...},
  "mood_map": {...},
  "genre_level_bridge": {...}
}
```

### 2. 版本号规则

采用语义化版本（Semantic Versioning）：`v<major>.<minor>`

| 版本类型 | 触发条件 | 示例 |
|---------|---------|------|
| **major（大版本）** | 映射规则变更 / 大规模黑名单调整 / 字段结构变化 / 不兼容的格式变更 | v1.0 → v2.0 |
| **minor（小版本）** | 新增/修改少量标签映射 / 扩展黑白名单 / 兼容性格式追加 | v2.0 → v2.1 |

**版本升级自动化**：
- `merge_mapping.py --apply` 默认升级 minor 版本（v2.0 → v2.1）
- `merge_mapping.py --apply --major` 升级 major 版本（v2.0 → v3.0）
- 版本号自动计算，不允许手动修改

### 3. Changelog 记录

每次版本升级必须在 `changelog` 数组中追加一条记录：

```json
{
  "date": "2026-08-25",
  "version": "v2.0",
  "action": "add",  // init / add / modify / remove / refactor
  "note": "新增 genre_level_bridge 歧义覆盖、compatible_preannotation_versions、blacklist_tags 扩展"
}
```

| 字段 | 说明 | 必填 |
|------|------|------|
| date | 变更日期（YYYY-MM-DD） | ✅ |
| version | 变更后的版本号 | ✅ |
| action | 变更类型（init/add/modify/remove/refactor） | ✅ |
| note | 变更说明（人类可读） | ✅ |

### 4. 兼容的预标注版本

`compatible_preannotation_versions` 字段声明当前映射字典兼容的预标注版本：

```json
"compatible_preannotation_versions": [
  "l1_physical_v1",
  "l2_clap_v1",
  "l3_qwen_omni_v1",
  "l4_deepseek_knn_v1"
]
```

**规则**：
- L1-L4 预标注脚本输出时，必须记录使用的预标注版本号
- 映射字典加载时，检查预标注版本是否在兼容列表中
- 不兼容的预标注版本 → 警告或报错，防止标签错位
- 预标注版本升级时，必须同步更新映射字典的兼容列表（minor 版本升级）

### 5. 黑白名单机制

| 名单类型 | 用途 | 处理方式 | 示例 |
|---------|------|---------|------|
| **hard_blacklist** | 绝对非音乐标签 | 直接排除，不进入映射 | speech, silence, podcast, white noise |
| **soft_blacklist** | 低质量/模糊标签 | 标记为 marginal，人工确认 | noise, low quality, distorted, clipping |

**规则**：
- hard_blacklist 中的标签 → L4 传播时直接丢弃，不写入最终标签
- soft_blacklist 中的标签 → 标记 `flag_for_review`，进入 HITL 人工复核
- 黑白名单调整属于 minor 版本变更（少量调整）或 major 版本变更（大规模调整）

### 6. 合并流程（原子性保证）

使用 `merge_mapping.py` 进行映射字典的合并更新，流程如下：

```
1. 人工审核产出 mapping_updates_pending.json
   （每条更新包含 mapping_type 字段：add_instrument / add_genre / modify_mapping / blacklist_add 等）
         │
         ▼
2. 预校验（--apply 前自动执行）
   - 检查所有条目的 mapping_type 是否合法
   - 检查目标标签是否存在冲突
   - 检查格式是否正确
   - 任何错误 → 整体取消，不写入（原子性）
         │
         ▼
3. 执行合并（--apply）
   - 全部通过预校验后，原子写入
   - 自动升级版本号（minor 或 major）
   - 追加 changelog 记录
   - 更新 updated_at
         │
         ▼
4. 验证
   - 重新加载映射字典，验证格式正确
   - 输出变更摘要（新增X条，修改Y条，删除Z条）
```

**安全原则**：
- 完全依赖 `mapping_type` 字段，不做字符串内容猜测
- 未知 mapping_type → 直接抛 ValueError，阻断而非猜测
- 预校验 + 原子写入：任何错误都不产生半更新状态
- 合并前自动备份当前版本到 `configs/mappings/backup/label_mapping_dict_v<version>.json`

### 7. 回滚机制

| 场景 | 回滚方式 |
|------|---------|
| 合并后发现错误 | 从 `configs/mappings/backup/` 恢复上一版本 |
| Git 版本控制 | `git revert` 到上一个 commit |
| 预校验失败 | 自动取消，不产生任何变更 |

**备份策略**：
- 每次合并前，自动备份当前映射字典到 `configs/mappings/backup/`
- 备份文件名：`label_mapping_dict_v<version>_<timestamp>.json`
- 保留最近 10 个备份，更早的自动清理

### 8. 映射类型枚举

`merge_mapping.py` 支持的 `mapping_type` 枚举：

| mapping_type | 说明 | 版本影响 |
|-------------|------|---------|
| `add_instrument` | 新增乐器映射 | minor |
| `add_genre` | 新增流派映射 | minor |
| `add_mood` | 新增情绪映射 | minor |
| `modify_mapping` | 修改现有映射 | minor |
| `remove_mapping` | 删除映射 | minor（少量）/ major（大量） |
| `blacklist_add` | 新增黑白名单 | minor |
| `blacklist_remove` | 移除黑白名单 | minor |
| `refactor_structure` | 重构字段结构 | major |
| `compatibility_update` | 更新兼容预标注版本 | minor |

---

## 与其他 ADR 的关系

| ADR | 关系 |
|-----|------|
| **ADR-001《QC Gate 阈值决策》** | soft_blacklist 标签触发 marginal，进入 HITL 复核，参见 ADR-002 |
| **ADR-002《HITL 异步闭环》** | unmapped_tag_review 听检任务产出映射更新建议，经人工确认后通过 merge_mapping.py 合并 |
| **ADR-003《数据划分与来源隔离》** | 映射字典版本是数据集元数据的一部分，划分时记录使用的映射版本 |
| **ADR-004《L1-L4 预标注分层》** | L4 传播标签依赖映射字典，compatible_preannotation_versions 确保版本兼容 |

---

## 工程约束

- **禁止硬编码映射**：所有脚本必须从 `configs/label_mapping_dict.json` 读取映射，禁止在代码中硬编码映射字典
- **禁止手动修改版本号**：版本号只能通过 `merge_mapping.py` 自动升级
- **禁止跳过预校验**：`--apply` 前必须通过预校验，不提供 `--force` 跳过选项
- **合并必须有 changelog**：每次版本升级必须追加 changelog 记录，不允许空 changelog
- **映射字典纳入 Git**：`configs/label_mapping_dict.json` 必须提交到 Git，每次合并对应一个 commit
- **下游脚本版本检查**：使用映射字典的脚本必须检查版本兼容性，不兼容时警告或报错

---

## 参考

- 语义化版本（Semantic Versioning 2.0.0）
- 配置即代码（Configuration as Code）
- 原子性提交（Atomic Commit）原则

## 与其他 ADR 的关系

- **ADR-002《HITL 异步闭环》**：映射字典更新是 HITL 听检任务的一种产出（unmapped_tag_review），人工审核后通过 merge_mapping.py 原子合并
- **ADR-003《数据划分与来源隔离》**：source_type 过滤与映射字典的 hard_blacklist/soft_blacklist 协同工作，合成/分轨样本在 source_type 层排除，低质量标签在映射字典层过滤
- **ADR-004《L1-L4 预标注分层》**：映射字典是 L2 语义候选标签的标准化层，L4 KNN 传播后的标签也需经过映射字典标准化
