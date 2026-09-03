# 五方协作规范（COLLABORATION_PROTOCOL.md）

> 版本：v1.2 | 生效日期：2026-09-03 | 适用范围：music_corpus_project 全生命周期
>
> 本文件为项目最高协作准则，任何 ADR、AGENTS.md、README 与本文件冲突时，以本文件为准。
> 
> **修订说明（v1.1）**：基于 DeepSeek 独立审核意见修正——验证包分层、WorkBuddy 产出外部审核、Kimi/DeepSeek P0 强制阻塞、门禁失效熔断。
>
> **修订说明（v1.2）**：基于用户裁定新增「封装义务」——P0/P1 验证包新增必填第 9 字段 encapsulation，一次性修补（one_time_patch）默认打回；新增第 3.4 节封装义务细则。

---

## 一、五方角色与绝对边界

| 角色 | 核心职责 | 绝对禁止 | 能力范围 |
|------|---------|---------|---------|
| **用户（User）** | 人工抉择节点（黄金集/Challenge 确认）、最终审批、控制台操作、拍板架构变更、**门禁失效熔断触发后仲裁** | 不写代码、不直接修改脚本、不替 agent 做验证 | 全项目读取 + 关键决策 |
| **豆包 Agent** | 编写脚本、修复 bug、执行流水线、产出数据 | **禁止**声称"完成"未经 workbuddy 验收；**禁止**写新 ADR/规范文档；**禁止**绕过门禁进入下一阶段 | Mac + GPU 执行环境 |
| **WorkBuddy** | **工序门禁系统**：前置检查、强制验证、指标确认、打回权、文档统一收口 | **禁止**编写模型训练代码；**禁止**直接修改数据；**禁止**与豆包做重叠执行工作；**禁止**对自己产出的候选池/文档自行验收 | Mac + GPU + OSS 全读取 + 本地写入（审计产出） |
| **Kimi** | 架构设计、技术方案评估、长程规划、复杂问题拆解、**独立审核 WorkBuddy 产出** | 不直接执行脚本、不替 agent 验证 | 基于对话上下文 |
| **DeepSeek** | 独立技术审核、风险预警、方案纠偏、边界仲裁、**独立审核 WorkBuddy 产出** | 不直接执行脚本、不替 agent 验证 | 基于对话上下文 |

**关键原则**：
- **WorkBuddy 是唯一门禁持有者**（对豆包产出）。任何工序产出未经 workbuddy 验收标记，视为"未完成"。
- **WorkBuddy 的产出（候选池、文档）必须由 Kimi 或 DeepSeek 独立审核**，workbuddy 无权自行验收自己的产出。
- **豆包的"完成"声明无效**，除非附带 workbuddy 的 `✅ 验收通过` 标记。
- **Kimi/DeepSeek 的 P0 标记 = 强制阻塞**：豆包必须先解决并提交验证包，workbuddy 验收通过后才能解锁下一阶段。
- **用户只在三种情况下介入**：（1）人工抉择节点（HITL）；（2）五方争议无法仲裁时；（3）**门禁失效熔断触发时**。

---

## 二、工序-门禁工作流（核心机制）

### 2.1 流水线阶段与门禁点

```
[阶段 N] 执行
    │
    ▼ 豆包产出
[验证包] 提交给 WorkBuddy
    │
    ▼ WorkBuddy 门禁检查
    ├── ❌ 不通过 → 打回豆包（附原因 + 复现命令）
    │
    └── ✅ 通过 → 写验收标记 → 解锁 [阶段 N+1]
                │
                ▼ 用户抽查（P0 强制抽查，P1 随机抽查）
```

### 2.2 工序风险分级与验证强度

| 风险等级 | 典型工序 | 验证包要求 | WorkBuddy 门禁 | 用户抽查 |
|---------|---------|-----------|---------------|---------|
| **P0** | L4 KNN 传播、数据划分、版本冻结、OSS 上传、黄金集采样 | **完整验证包**（8 字段全填） | 逐项检查清单 | **强制抽查** |
| **P1** | 脚本参数修正、路径修复、格式转换、特征提取 | **简化验证包**（git diff + 运行日志 + 关键数字） | 抽样检查 | 随机抽查 |
| **P2** | 文档修正、注释、日志格式调整 | **无需验证包**，git commit 即可 | 不检查 | 不抽查 |

**原则**：验证严格程度与工序风险成正比，不是所有事都走同一套重流程。

### 2.3 P0 强制门禁点（必须验收才能继续）

| 工序 | 豆包产出 | WorkBuddy 验收内容 | 用户抽查 |
|------|---------|-------------------|---------|
| **代码/脚本修改** | `.py` / `.sh` 文件 | 是否在 Git？是否符合 ADR？文档是否同步？ | — |
| **Stage 0 入库** | `audio_manifest.csv` | 字段完整性、track_id 唯一性、来源标记 | — |
| **Stage 2 格式标准化** | 母版 FLAC | 采样率/位深度/声道校验、checksum | — |
| **Stage 3 YAMNet 质检** | `yamnet_output.csv` | 统计合理性（如 has_noise 不应 >50%）、抽样复核 | — |
| **Stage 5 MERT/CLAP 嵌入** | `.npy` 嵌入文件 | 文件数=音频数、shape 一致性、抽样加载测试 | ✅ 强制 |
| **L4 KNN 传播** | 传播后标签 | **test 隔离验证**（test 样本传播数必须为 0）、传播覆盖率 | ✅ 强制 |
| **数据划分** | `train/val/test/holdout` CSV | artist 隔离 0 冲突、比例合规、黄金集在 train | ✅ 强制 |
| **黄金集采样** | `candidate_pool.json` | 分层采样覆盖度、簇分布均衡性 | ✅ **HITL 人工确认** |
| **Challenge 采样** | `challenge_candidates.json` | 非黄金集、分布代表性 | ✅ **HITL 人工确认** |
| **版本冻结** | `vYYYYMMDD_HHMMSS/` | 目录结构合规、元数据完整、血缘可追溯 | ✅ 强制 |
| **OSS 上传** | 云端对象 | 对象数=本地数、ETag/大小校验、`.oss_verified` 标记 | — |

### 2.4 打回标准（任一触发即打回）

1. **无验证证据**：豆包声称完成但未提供运行日志、产出样本、可复现命令
2. **环境不可复现**：脚本依赖未记录（如 `.bashrc` 隐式变量、未进 Git 的模块）
3. **文档矛盾**：修改违反现有 ADR，且 ADR 未同步更新
4. **指标缺失**：关键数字未量化（如"KNN 修好了"但未给传播样本数）
5. **下游阻塞**：产出格式不符合下一阶段输入要求（如 L4 输出 L5 读不了）
6. **安全红线**：OSS 上传了 mp3、密钥硬编码、看门狗含 shutdown

---

## 三、验证包规范（分层提交）

### 3.1 P0 完整验证包（8 字段）

豆包提交 P0 工序时，必须提供以下结构化验证包。**缺少任一字段，workbuddy 直接打回。**

```json
{
  "task_id": "L4_knn_propagation_fix_v2",
  "stage": "L4",
  "risk_level": "P0",
  "claim": "修复 KNN=0，test 隔离 0 冲突",

  "code_diff": {
    "files_modified": ["scripts/04_dataset/l4_knn_propagation.py"],
    "git_commit": "abc1234",
    "diff_summary": "修改了邻居搜索逻辑，排除 test 样本"
  },

  "execution_log": {
    "command": "python scripts/04_dataset/l4_knn_propagation.py --input ...",
    "full_output": "[粘贴完整终端输出，不少于最后 20 行]",
    "exit_code": 0,
    "execution_time_sec": 45
  },

  "output_sample": {
    "file_path": "data/04_final_dataset/v20260828/knn_propagated.csv",
    "first_5_rows": "[粘贴前5行内容]",
    "row_count": 84,
    "column_names": ["track_id", "propagated_genre", "confidence", "source"]
  },

  "verification_command": {
    "description": "验证 test 隔离和 KNN 传播数",
    "command": "python -c "..."",
    "expected_output": "test传播: 0\n传播覆盖率: 0.85"
  },

  "environment_snapshot": {
    "conda_env": "labelstudio-env",
    "key_packages": {"torch": "2.2.2", "scikit-learn": "1.9.0"},
    "implicit_deps": ["无 / .bashrc 已显式 export HF_ENDPOINT"]
  },

  "adr_impact": {
    "adr_affected": ["ADR-003"],
    "sync_status": "已更新 / 无需更新 / 待用户审批"
  },

  "encapsulation": {
    "root_cause": "本次问题的根因（一句话，禁止只描述表象）",
    "fix_type": "reusable_script | config_change | one_time_patch",
    "reusable_artifact": "封装产物：公共函数/脚本/配置文件的具体路径；无则填 none 并说明为何无法封装",
    "trigger": "下次同类问题如何被此封装产物自动拦截（如：回归测试命令 / 预检函数调用点）",
    "regression_test": "防再犯验证：已运行的测试命令与结果；无则填 none 并说明理由"
  }
}
```

**v1.2 铁律：`encapsulation` 为必填字段，缺少或留空直接打回。`fix_type = one_time_patch` 一律打回，唯一例外：该步骤本质是纯人工 HITL 决策（如用户听选定种子），须在 `reusable_artifact` 中说明。**

### 3.2 P1 简化验证包（3 字段）

```json
{
  "task_id": "fix_chunker_overlap",
  "stage": "Stage 6",
  "risk_level": "P1",
  "claim": "修复切片重叠率从 30% 改为 50%",

  "code_diff": {
    "files_modified": ["scripts/01_preprocess/audio_chunker.py"],
    "git_commit": "def5678"
  },

  "execution_log": {
    "command": "python scripts/01_preprocess/audio_chunker.py --input test.wav",
    "full_output": "[最后 10 行]",
    "exit_code": 0
  },

  "key_numbers": {
    "before_overlap": "0.30",
    "after_overlap": "0.50",
    "segments_generated": 12
  }
}
```

### 3.3 P2 免验证包

P2 工序只需 git commit，无需验证包。但 workbuddy 有权在代码审查时发现 P2 改动实际影响了 P0 行为，**升级风险等级**。

### 3.4 封装义务细则（v1.2 新增）

背景（2026-09-03 用户裁定）：豆包此前多轮修复均为「用户提问才做、跑出问题才手动修」，从不固化为可复用资产，导致同类问题反复出现（L2 ID 格式解析、缺依赖 import、.env 加载失败等）。自 v1.2 起：

1. **所有 P0/P1 修复必须通过第 9 字段 encapsulation 声明封装方式**；无法封装时必须说明理由，由 workbuddy 判定理由是否成立。
2. **优先封装顺序**：公共函数（消除重复逻辑）> 独立脚本（可复跑）> 预检/自检代码（在工序入口自动拦截）> 回归测试（证明不再犯）。
3. **workbuddy 验收动作**：核实 reusable_artifact 真实存在且被主流程调用（非死代码）；核实 regression_test 可复跑。只写文档不改代码的"封装"视为未封装。
4. **重复犯错升级**：同一根因第二次出现且上次 encapsulation 声明过，门禁自动升级为 P0 强制阻塞并记录熔断计数。

---

## 四、门禁检查清单（WorkBuddy 验收 SOP）

### 4.1 通用检查（所有 P0/P1）

- [ ] **Git 状态**：修改的文件是否已 `git add` 并附 commit hash？
- [ ] **文档同步**：若修改了行为，ADR/AGENTS.md 是否同步？（豆包无权改，需 workbuddy 统一收口）
- [ ] **环境可复现**：验证命令是否能在干净环境复现？有无隐式 `.bashrc` 依赖？
- [ ] **产出存在**：声称产出的文件是否真实存在？大小是否为 0？

### 4.2 数据工序专项检查（P0）

- [ ] **数量一致**：产出文件数 = 输入样本数
- [ ] **格式合规**：抽样加载测试通过
- [ ] **统计合理**：关键指标在合理区间
- [ ] **下游兼容**：产出能否被下一阶段脚本读取？
- [ ] **test 隔离**：P0 传播/划分工序必须验证 test 样本未被污染

### 4.3 安全红线检查

- [ ] **OSS 无 mp3**：上传列表中无 `.mp3` 后缀
- [ ] **密钥无硬编码**：代码中无 `AK=`、`SECRET=` 明文
- [ ] **看门狗无害**：`idle_watchdog.sh` 无危险命令
- [ ] **实例无裸奔**：关键脚本是否已 scp 回 Mac 或进 Git？

### 4.4 验收标记规范

**通过**：
```
✅ 验收通过 | task: L4_knn_propagation_fix_v2 | level: P0
- Git: abc1234
- 产出: 84 行, test 传播=0, 覆盖率=85%
- 验证命令已复现成功
- 解锁下一阶段: L5 结构标注
```

**打回**：
```
❌ 打回 | task: L4_knn_propagation_fix_v2 | level: P0
- 原因: 验证命令中 test 传播数未提供
- 要求: 补充 test 样本传播数的输出截图
- 参考: ADR-003 第 4.2 条
```

---

## 五、WorkBuddy 产出的独立审核机制

### 5.1 审核范围

WorkBuddy 以下产出**不得自行验收**，必须提交 Kimi 或 DeepSeek 独立审核：

| WorkBuddy 产出 | 审核方 | 审核内容 |
|---------------|--------|---------|
| `candidate_pool.json`（黄金集候选池） | Kimi / DeepSeek | 分层采样算法是否正确？簇覆盖是否均衡？ |
| `challenge_candidates.json`（Challenge 候选池） | Kimi / DeepSeek | 是否非黄金集？分布代表性？ |
| 文档修改（ADR/AGENTS/README） | Kimi / DeepSeek | 是否与现有架构矛盾？是否遗漏关键约束？ |
| 审计报告（如 script_execution_inventory.md） | Kimi / DeepSeek | 盘点是否完整？结论是否准确？ |

### 5.2 审核流程

```
WorkBuddy 产出候选池/文档
    │
    ▼
提交 Kimi 或 DeepSeek（轮流或按领域分配）
    │
    ├── ❌ 审核不通过 → 返回 WorkBuddy 修正
    │
    └── ✅ 审核通过 → WorkBuddy 写入验收标记 → 提交用户 HITL
```

### 5.3 审核意见约束力

- **Kimi/DeepSeek 的 P0 标记 = 强制阻塞**：该问题自动进入阻塞队列，豆包必须先解决
- **Kimi/DeepSeek 的 P1 标记 = 建议修复**：WorkBuddy 评估是否阻塞，可记录为技术债
- **双方分歧**：用户仲裁

---

## 六、门禁失效熔断机制

### 6.1 触发条件（任一满足即熔断）

| 熔断条件 | 说明 |
|---------|------|
| **豆包连续 2 次未经验收提交"完成"声明** | 绕过门禁 |
| **WorkBuddy 连续 2 次错误放行** | 门禁失效 |
| **Kimi/DeepSeek 连续 2 次标记 P0 但豆包未解决** | 阻塞失控 |
| **用户发现 P0 工序产出存在严重错误** | 抽查失效 |

### 6.2 熔断动作

1. **立即暂停流水线**：任何新工序不得启动
2. **WorkBuddy 启动紧急审计**：回溯最近 3 个工序的验收记录
3. **用户介入仲裁**：决定是修复门禁流程、降级风险等级、还是更换 agent
4. **熔断解除**：用户书面确认后恢复

### 6.3 熔断记录

每次熔断必须记录：
```json
{
  "fuse_triggered_at": "2026-08-28T10:00:00+08:00",
  "condition": "豆包连续 2 次未经验收提交完成声明",
  "evidence": ["task_id_1", "task_id_2"],
  "resolution": "用户确认修复 / 降级为 P1 / 更换 agent",
  "resolved_by": "user",
  "resolved_at": "2026-08-28T11:30:00+08:00"
}
```

---

## 七、文档权威机制

### 7.1 文档层级

| 优先级 | 文档 | 维护者 | 变更权限 |
|--------|------|--------|---------|
| 1 | **COLLABORATION_PROTOCOL.md**（本文件） | WorkBuddy 统一收口，Kimi/DeepSeek 审核 | 用户最终审批 |
| 2 | **ADR-xxx** | WorkBuddy 统一收口，Kimi/DeepSeek 审核 | 用户最终审批 |
| 3 | **AGENTS.md** | WorkBuddy 统一收口 | 用户最终审批 |
| 4 | **README.md** | WorkBuddy 统一收口 | 用户最终审批 |
| 5 | **docs/archive/** | 只读 | — |

### 7.2 文档变更流程

```
豆包/用户提出变更需求
    │
    ▼
WorkBuddy 评估影响范围
    │
    ├── 若影响协作规则 → 更新本文件 → Kimi/DeepSeek 审核 → 用户审批
    ├── 若影响架构 → 更新 ADR → Kimi/DeepSeek 审核 → 用户审批
    └── 若影响操作细节 → 更新 AGENTS.md → 用户审批
    │
    ▼
WorkBuddy 统一合并到唯一权威文档
    │
    ▼
豆包 agent 执行时只读最新版权威文档
```

**禁止**：任何 agent 直接修改 `.md` 文件。所有文档变更必须通过 workbuddy 收口。

---

## 八、争议升级路径

### 8.1 一级争议（技术实现分歧）

- **场景**：豆包认为方案 A 更好，workbuddy 认为方案 B 更好
- **仲裁**：Kimi / DeepSeek 独立评估
- **决策权**：用户拍板

### 8.2 二级争议（验收标准分歧）

- **场景**：豆包认为"产出可用"，workbuddy 认为"不符合规范"
- **仲裁**：引用本文件对应条款
- **决策权**：workbuddy 有**一票否决权**（打回权），用户可 override 但需承担风险

### 8.3 三级争议（架构方向分歧）

- **场景**：Kimi 建议方向 A，DeepSeek 建议方向 B
- **仲裁**：用户根据项目目标拍板
- **决策权**：用户唯一

### 8.4 四级争议（门禁失效）

- **场景**：熔断触发
- **仲裁**：用户直接介入，暂停流水线，审计门禁流程
- **决策权**：用户唯一

---

## 九、附则

### 9.1 生效与废止

- 本文件自用户确认后**立即生效**，此前所有 AGENTS.md、PROJECT_ALIGNMENT 中关于"分工"的条款以本文件为准。
- 废止文档：`PROJECT_ALIGNMENT.md`（分工部分）、`AGENTS.md`（分工部分）——保留其他操作细节。

### 9.2 修订记录

| 日期 | 版本 | 修订内容 | 审批人 |
|------|------|---------|--------|
| 2026-08-28 | v1.0 | 初始版本，确立五方门禁机制 | 用户 |
| 2026-08-28 | v1.1 | ① 验证包分层（P0/P1/P2）；② WorkBuddy 产出需 Kimi/DeepSeek 审核；③ Kimi/DeepSeek P0 强制阻塞；④ 门禁失效熔断机制 | 用户 |
| 2026-09-03 | v1.2 | ① P0/P1 验证包新增必填第 9 字段 encapsulation（root_cause/fix_type/reusable_artifact/trigger/regression_test）；② one_time_patch 默认打回（例外：纯 HITL 决策）；③ 新增 3.4 封装义务细则（封装优先级、验收动作、重复犯错升级） | 用户 |

### 9.3 五方确认签字区

- [ ] 用户：_________________（确认即生效）
- [ ] WorkBuddy：_________________（确认门禁职责 + 接受产出被审核）
- [ ] 豆包 Agent：_________________（确认提交验证包义务 + 接受分层验证）
- [ ] Kimi：_________________（确认独立审核角色 + P0 强制阻塞权）
- [ ] DeepSeek：_________________（确认独立审核角色 + P0 强制阻塞权）
