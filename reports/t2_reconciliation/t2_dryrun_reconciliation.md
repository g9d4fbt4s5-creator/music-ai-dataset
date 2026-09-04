# T2 Dry-Run 重放对账报告

- 生成时间：2026-09-04T15:50:41
- 方法：以现有L3产物为输入，临时目录dry-run重放L4，逐字段对账，不重调任何API
- Qwen API 调用：**0**；文本LLM API 调用：**0**
- run_pipeline 内网络符号：无

## 输入指纹（现有 L3 产物，未重新生成）

| 输入 | 角色 | 数量 | sha256(16) |
|------|------|------|-----------|
| genre_text_llm_annotations.json | P0文本LLM标注 | 58 | 07e94c224cece2f9 |
| user_rulings.json | 配置 | - | 56e35f8800cd5c72 |
| layered_conflict_resolutions.json | 配置 | - | 0e9b3622b081e222 |
| qwen_reannotation_manifest.json | 配置 | - | 8464c71657887391 |
| genre_annotation_plan.json | 配置 | - | 49df9c763362754a |
| l4_deepseek/ | P1 Qwen标注(目录名为历史遗留) | 77 | - |
| l3_structural/ | 黄金集Qwen精标(优先覆盖) | 5 | - |

## 对账结果

| 对账层 | 总数 | 一致 | 结果 |
|--------|------|------|------|
| genre_unified_final.json（样本） | 84 | 84 | ✅ |
| genre_unified_final.json（字段） | 905 | 905 | ✅ |
| l4_unified/（逐首文件） | 84 | 84 | ✅ |

**样本ID集合一致：✅**

### 总结论：✅ 100% 一致，封装合格
