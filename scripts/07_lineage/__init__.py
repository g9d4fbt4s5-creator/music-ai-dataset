"""
算子级血缘模块（Lineage v2.0）

模块结构：
- lineage_logger.py: 血缘记录器（核心，算子执行时自动记录）
- lineage_query.py: 血缘查询工具（按算子/样本/状态查询）
- lineage_verify.py: 血缘校验工具（完整性/一致性/失败率/防泄露）
- lineage_report.py: 血缘报告生成（Markdown/HTML/JSON + 图表）

设计原则：
- 算子级记录（不记录每条样本，只记录批次）
- 支持增量追加（多次运行合并到同一个lineage文件）
- 支持算子嵌套（算子A的输出是算子B的输入）
- 自动记录时间戳、算子版本、模型版本、耗时
- 失败样本记录和失败原因统计
- 标准 lineage.json（v2.0格式）
"""
from .lineage_logger import LineageLogger, OperatorRecord, OperatorContext

__all__ = ["LineageLogger", "OperatorRecord", "OperatorContext"]
__version__ = "2.0.0"
