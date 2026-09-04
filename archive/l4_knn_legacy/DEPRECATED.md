# L4 KNN 传播 — DEPRECATED（已从生产流程移除）

## 移除日期
2026-09-04

## 移除原因

### 核心数据
| 指标 | 数值 |
|------|------|
| 单邻居 KNN 一致率（vs Qwen） | 41.2%（7/17） |
| k=3 投票 KNN 一致率 | 0%（0 传播） |
| k=5 投票 KNN 一致率 | 0%（3 首传播全错） |
| CLAP top-1 vs Qwen 一致率 | 28.6% |
| MERT 嵌入 genre 簇分布 | 不按 genre 聚类，声学相似 ≠ 风格相似 |

### 根因分析
1. **genre 是社会文化标签，不是纯声学属性**：听音频的模型（Qwen/CLAP/KNN）只能听到"有什么乐器、什么唱法"，听不到"这该叫 Classical Crossover 还是 Traditional Pop"
2. **MERT 嵌入空间里 genre 不按簇分布**：Bossa Nova 锚点赢家通吃，15/18 首 KNN 传播全是 Bossa Nova
3. **单邻居赢家通吃 bug**：种子池扩到 21 首也没用，只要 latin jazz/cool jazz 种子的嵌入吸力最强，单邻居机制就继续赢家通吃
4. **k≥3 投票导致 0 传播**：当前种子池 genre 分散（Traditional Pop/Cantopop/Bossa Nova/Trance/Pop），2/3 多数票永远达不到

### 在 84 首规模上的价值
0。传得越多，错得越多。

## 新的 L4 架构（2026-09-04 起生效）

```
L4 = 文本 LLM+搜索（P0，58 首有曲目名）
   + Qwen 听音频（P1，21 首 unknown）
   + 人工裁决（P2，冲突样本）
KNN → 不参与
```

### 来源优先级
| 优先级 | 条件 | 来源 | 权重 |
|--------|------|------|------|
| P0 | 有曲目名+艺术家 | 文本 LLM+搜索（Wikipedia/Discogs/AllMusic） | 1.0 直接采信 |
| P1 | 无曲目名（unknown） | Qwen-Omni 听音频 | 0.8 |
| P2 | 两个来源冲突 | 人工听（HITL） | 最终裁决 |
| P3 | 前两者都没有 | CLAP 零样本候选 | 0.3 只当候选 |
| EXCLUDED | ACE Studio 生成 | 排除 | - |

## 保留归档的原因

1. **500 首以上规模时重新评估**：文本 LLM 和 Qwen 的成本会上升，KNN 可能重新有经济价值
2. **作品集工程决策展示**："我们试过 KNN，一致率 0%，所以放弃了"——真实的工程决策比硬吹 KNN 更有说服力
3. **未来可能换嵌入**：如果训练一个 genre-specific embedding（如 contrastive learning），KNN 可能复活

## 归档内容

```
archive/l4_knn_legacy/
├── DEPRECATED.md                    # 本文件
├── scripts/
│   ├── l4_knn_propagation.py       # 静态 KNN 传播（防泄漏参数+泛类不传播+22首voided+ACE exclude）
│   ├── l4_knn_propagation_leakage_proof.py  # 防泄漏版本
│   ├── l4_iterative_knn.py         # v1 动态迭代 KNN（单邻居版本）
│   └── l4_iterative_knn_v2.py      # v2 动态迭代 KNN（k≥3投票+一致率）
└── data/
    ├── l4_iterative/                # v1 迭代产物（覆盖率 27.4%→53.6%，一致率 41.2%）
    ├── l4_iterative_v2/             # v2 迭代产物（k=3，一致率 0%）
    └── l4_iterative_v2_k5/          # v2 k=5 迭代产物（一致率 0%）
```

## 相关 commit
- `f6ac8d5`: feat(l4): 动态迭代KNN v2-k≥3投票+一致率唯一生存指标+交叉一致准入
- `3e6e1fc`: feat(l4): 动态迭代KNN架构-Active Learning种子池迭代扩大
- `b7bc18c`: fix(l4): 第五轮门禁六判据一次修完
- `a82db5b`: fix(l4): 第六轮门禁修复-ACE ID补全+generic_golden不误清种子+波切利Traditional Pop
