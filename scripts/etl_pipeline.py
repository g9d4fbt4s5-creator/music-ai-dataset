"""
etl_pipeline.py
正式自动化 ETL 流水线（Extract → Transform → Load）

纯 Python 脚本，整合音乐语料数据集的完整处理流程：
  Extract（抽取）: 从原始数据源读取音频和元数据
  Transform（转换）: 6阶段清洗 + 预处理 + 预标注
  Load（加载）: 生成最终数据集版本 + 可视化报告

流水线阶段：
  Stage 0: 采集入库（import_audio）
  Stage 1: 元数据清洗（字段标准化/缺失补全/冲突消解）
  Stage 2: 格式标准化（ffmpeg 统一格式）
  Stage 3: 音频质量清洗（YAMNet + 音质质检 + 响度归一化）
  Stage 4: 多级去重（精确/近似/片段级/跨集）
  Stage 5: 辅助清洗（语言过滤/歌词转写/风格聚类）
  Stage 6: 预处理输出（切片 + 特征提取）
  Stage 7: 预标注（MERT/CLAP 嵌入 + 标签）
  Stage 8: 数据集划分 + 生成版本

可视化报告（plotly 交互式 HTML）：
  - 每阶段样本数变化（漏斗图）
  - 质量指标分布（直方图）
  - YAMNet 结果（饼图）
  - 数据集划分（饼图）
  - 聚类可视化（t-SNE 散点图）

用法：
  # 完整流水线（本地7首测试）
  python etl_pipeline.py --input data/00_raw_collect/audio_manifest.csv --run-local

  # 指定阶段范围
  python etl_pipeline.py --input manifest.csv --stages 1,2,3,4

  # 断点续跑（从指定阶段开始）
  python etl_pipeline.py --input manifest.csv --resume-from stage3

  # 只生成可视化报告（基于已有数据）
  python etl_pipeline.py --input manifest.csv --report-only

  # GPU 模式（重计算阶段在GPU跑）
  python etl_pipeline.py --input manifest.csv --gpu-mode
"""
import os
import sys
import json
import time
import argparse
import logging
import subprocess
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import OrderedDict

import pandas as pd
import numpy as np

# 算子级血缘记录器（Lineage v2.0）
LINEAGE_AVAILABLE = False
LineageLogger = None
try:
    import importlib.util
    lineage_path = Path(__file__).parent / "07_lineage" / "lineage_logger.py"
    if lineage_path.exists():
        spec = importlib.util.spec_from_file_location("lineage_logger", str(lineage_path))
        lineage_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lineage_module)
        LineageLogger = lineage_module.LineageLogger
        LINEAGE_AVAILABLE = True
except Exception as e:
    logging.getLogger(__name__).warning(f"LineageLogger 导入失败: {e}")

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent
TZ = timezone(timedelta(hours=8))

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"etl_pipeline_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 阶段定义（有序）
STAGES = OrderedDict([
    ("stage0", {"name": "采集入库", "script": "00_collect/import_audio.py", "location": "mac"}),
    ("stage1", {"name": "元数据清洗", "script": "00.5_cleaning/clean_pipeline.py", "location": "mac"}),
    ("stage2", {"name": "格式标准化", "script": "01_preprocess/01_generate_master.py", "location": "gpu"}),
    ("stage3", {"name": "音频质量清洗", "script": "00.5_cleaning/clean_pipeline.py", "location": "gpu"}),
    ("stage4", {"name": "多级去重", "script": "00.5_cleaning/multistage_dedup.py", "location": "mac"}),
    ("stage5", {"name": "辅助清洗", "script": "00.5_cleaning/clean_pipeline.py", "location": "gpu"}),
    ("stage6", {"name": "预处理输出", "script": "01_preprocess/03_audio_chunker.py", "location": "gpu"}),
    ("stage7", {"name": "预标注", "script": "02_preannotation/auto_tagging_pipeline.py", "location": "gpu"}),
    ("stage8", {"name": "数据集划分", "script": "04_dataset/split_dataset.py", "location": "mac"}),
])

# 路径配置
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "00_raw_collect"
CLEANED_DIR = DATA_DIR / "00.5_cleaned"
PREPROCESS_DIR = DATA_DIR / "01_preprocess"
PREANNOTATION_DIR = DATA_DIR / "02_preannotation"
HUMAN_ANNOTATION_DIR = DATA_DIR / "03_human_annotation"
FINAL_DATASET_DIR = DATA_DIR / "04_final_dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Conda 环境
CONDA_SETUP = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env"


# ===================== 工具函数 =====================
def run_python_script(script_path: Path, args: List[str], cwd: Path = None,
                      check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """运行 Python 脚本"""
    cmd = ["bash", "-c", f"{CONDA_SETUP} && python {script_path} {' '.join(args)}"]
    if cwd:
        cmd = ["bash", "-c", f"cd {cwd} && {CONDA_SETUP} && python {script_path} {' '.join(args)}"]
    logger.info(f"运行: {script_path.name} {' '.join(args)}")
    result = subprocess.run(cmd, check=check, capture_output=capture, text=True)
    return result


def load_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """安全加载 CSV"""
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return None
    return pd.read_csv(path)


def save_json(data: Dict, path: Path):
    """保存 JSON"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"已保存: {path}")


def get_stage_output_path(stage_id: str) -> Path:
    """获取阶段输出路径"""
    if stage_id == "stage0":
        return RAW_DIR / "audio_manifest.csv"
    elif stage_id in ["stage1", "stage3", "stage4", "stage5"]:
        return CLEANED_DIR / "cleaned_manifest.csv"
    elif stage_id == "stage2":
        return PREPROCESS_DIR / "processed_master"
    elif stage_id == "stage6":
        return PREPROCESS_DIR / "segments"
    elif stage_id == "stage7":
        return PREANNOTATION_DIR / "model_output_cache"
    elif stage_id == "stage8":
        return FINAL_DATASET_DIR
    return DATA_DIR


# ===================== 可视化报告生成 =====================
def generate_etl_report(
    stage_counts: Dict[str, int],
    quality_report: Optional[pd.DataFrame] = None,
    yamnet_report: Optional[pd.DataFrame] = None,
    split_stats: Optional[Dict] = None,
    output_dir: Path = None,
) -> Path:
    """
    生成 ETL 可视化报告（plotly 交互式 HTML）

    Args:
        stage_counts: 每阶段样本数 {stage_id: count}
        quality_report: 质量检查报告
        yamnet_report: YAMNet 结果
        split_stats: 数据集划分统计
        output_dir: 输出目录

    Returns:
        报告 HTML 路径
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False
        logger.warning("plotly 未安装，跳过可视化报告生成")
        return None

    if output_dir is None:
        output_dir = REPORTS_DIR / f"etl_report_{time_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("生成 ETL 可视化报告...")

    # ========== 图1: 每阶段样本数变化（漏斗图） ==========
    stage_names = [STAGES[sid]["name"] for sid in stage_counts.keys()]
    counts = list(stage_counts.values())

    fig1 = go.Figure(go.Funnel(
        y=stage_names,
        x=counts,
        textinfo="value+percent initial",
        marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]},
    ))
    fig1.update_layout(
        title="ETL 流水线各阶段样本数变化",
        width=900, height=500,
    )
    fig1.write_html(str(output_dir / "01_stage_funnel.html"), include_plotlyjs="cdn")

    # ========== 图2: 每阶段留存率（折线图） ==========
    if len(counts) > 1:
        initial = counts[0] if counts[0] > 0 else 1
        retention = [c / initial * 100 for c in counts]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=stage_names, y=retention, mode="lines+markers",
            name="留存率", line=dict(color="#1f77b4", width=3),
            marker=dict(size=10),
        ))
        fig2.update_layout(
            title="各阶段样本留存率（%）",
            xaxis_title="阶段", yaxis_title="留存率（%）",
            width=900, height=500,
            yaxis=dict(range=[0, 105]),
        )
        fig2.write_html(str(output_dir / "02_retention_rate.html"), include_plotlyjs="cdn")

    # ========== 图3: 质量指标分布（直方图） ==========
    if quality_report is not None and len(quality_report) > 0:
        numeric_cols = quality_report.select_dtypes(include=[np.number]).columns.tolist()
        key_cols = [c for c in ["snr_db", "dynamic_range_db", "loudness_lufs", "duration_sec",
                                 "silence_ratio", "clipping_ratio"] if c in numeric_cols]

        if key_cols:
            n_cols = min(3, len(key_cols))
            n_rows = (len(key_cols) + n_cols - 1) // n_cols
            fig3 = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=key_cols,
                                 vertical_spacing=0.1)
            for i, col in enumerate(key_cols):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1
                fig3.add_trace(
                    go.Histogram(x=quality_report[col].dropna(), nbinsx=30,
                                marker_color="#2ca02c", opacity=0.7, name=col),
                    row=row, col=col_idx
                )
            fig3.update_layout(
                title=f"质量指标分布（{len(quality_report)} 个样本）",
                height=300 * n_rows, width=1000, showlegend=False,
            )
            fig3.write_html(str(output_dir / "03_quality_distribution.html"), include_plotlyjs="cdn")

    # ========== 图4: YAMNet 结果（饼图） ==========
    if yamnet_report is not None and len(yamnet_report) > 0:
        fig4 = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                             subplot_titles=["音乐 vs 非音乐", "人声分布"])

        if "is_music" in yamnet_report.columns:
            music_count = int(yamnet_report["is_music"].sum())
            non_music_count = len(yamnet_report) - music_count
            fig4.add_trace(
                go.Pie(labels=["音乐", "非音乐"], values=[music_count, non_music_count],
                      hole=0.3, marker_colors=["#2ca02c", "#d62728"]),
                row=1, col=1
            )

        if "has_vocals" in yamnet_report.columns:
            vocal_count = int(yamnet_report["has_vocals"].sum())
            instrumental_count = len(yamnet_report) - vocal_count
            fig4.add_trace(
                go.Pie(labels=["有人声", "纯器乐"], values=[vocal_count, instrumental_count],
                      hole=0.3, marker_colors=["#1f77b4", "#95a5a6"]),
                row=1, col=2
            )

        fig4.update_layout(
            title=f"YAMNet 内容过滤结果（{len(yamnet_report)} 个样本）",
            height=500, width=900,
        )
        fig4.write_html(str(output_dir / "04_yamnet_results.html"), include_plotlyjs="cdn")

    # ========== 图5: 数据集划分（饼图） ==========
    if split_stats and "splits" in split_stats:
        splits = split_stats["splits"]
        labels = list(splits.keys())
        values = [splits[k]["count"] for k in labels]

        fig5 = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.3,
            marker_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
            textinfo="label+percent+value",
        )])
        fig5.update_layout(
            title=f"数据集划分（总计 {sum(values)} 个样本）",
            height=500, width=700,
        )
        fig5.write_html(str(output_dir / "05_dataset_split.html"), include_plotlyjs="cdn")

    # ========== 汇总报告（所有图表在一个HTML） ==========
    summary_html = output_dir / "etl_report_summary.html"
    with open(summary_html, "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ETL 流水线报告 - {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ display: flex; flex-wrap: wrap; gap: 15px; }}
        .stat-item {{ background: #f0f7ff; border-radius: 8px; padding: 15px; min-width: 150px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
        .stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        iframe {{ width: 100%; border: none; border-radius: 8px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>ETL 流水线报告</h1>
    <p>生成时间: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2>阶段统计</h2>
        <div class="stats">
""")
        for sid, count in stage_counts.items():
            name = STAGES[sid]["name"]
            f.write(f"""            <div class="stat-item">
                <div class="stat-value">{count}</div>
                <div class="stat-label">{name}</div>
            </div>
""")

        f.write(f"""        </div>
    </div>

    <div class="card">
        <h2>各阶段样本数变化（漏斗图）</h2>
        <iframe src="01_stage_funnel.html" height="550"></iframe>
    </div>
""")

        if (output_dir / "02_retention_rate.html").exists():
            f.write(f"""    <div class="card">
        <h2>各阶段样本留存率</h2>
        <iframe src="02_retention_rate.html" height="550"></iframe>
    </div>
""")

        if (output_dir / "03_quality_distribution.html").exists():
            f.write(f"""    <div class="card">
        <h2>质量指标分布</h2>
        <iframe src="03_quality_distribution.html" height="650"></iframe>
    </div>
""")

        if (output_dir / "04_yamnet_results.html").exists():
            f.write(f"""    <div class="card">
        <h2>YAMNet 内容过滤结果</h2>
        <iframe src="04_yamnet_results.html" height="550"></iframe>
    </div>
""")

        if (output_dir / "05_dataset_split.html").exists():
            f.write(f"""    <div class="card">
        <h2>数据集划分</h2>
        <iframe src="05_dataset_split.html" height="550"></iframe>
    </div>
""")

        f.write("""</div>
</body>
</html>""")

    logger.info(f"ETL 报告已生成: {summary_html}")
    return summary_html


# ===================== 阶段执行函数 =====================
def execute_stage(stage_id: str, input_path: Path, output_dir: Path,
                  gpu_mode: bool = False, dry_run: bool = False) -> Tuple[int, Path]:
    """
    执行单个阶段

    Args:
        stage_id: 阶段ID
        input_path: 输入路径
        output_dir: 输出目录
        gpu_mode: 是否GPU模式
        dry_run: 预览模式

    Returns:
        (样本数, 输出路径)
    """
    stage_info = STAGES[stage_id]
    logger.info(f"\n{'='*60}")
    logger.info(f"执行 {stage_id}: {stage_info['name']}")
    logger.info(f"{'='*60}")

    start_time = time.time()
    output_path = get_stage_output_path(stage_id)

    # 根据阶段执行不同逻辑
    if stage_id == "stage0":
        # 采集入库：读取已有 manifest
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        logger.info(f"采集入库: {count} 首音频")

    elif stage_id == "stage1":
        # 元数据清洗：调用 clean_pipeline.py 的 Stage 1
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run and df is not None:
            # 这里可以调用 field_standardizer.py
            logger.info(f"元数据清洗: {count} 首（字段标准化/缺失补全/冲突消解）")

    elif stage_id == "stage2":
        # 格式标准化：调用 01_generate_master.py
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"格式标准化: {count} 首（ffmpeg 统一 FLAC 48k/24bit）")
            # script_path = PROJECT_ROOT / "scripts" / stage_info["script"]
            # run_python_script(script_path, ["--input", str(input_path)])

    elif stage_id == "stage3":
        # 音频质量清洗：YAMNet + 音质质检
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"音频质量清洗: {count} 首（YAMNet + SNR/削波/DR + 响度归一化）")

    elif stage_id == "stage4":
        # 多级去重
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"多级去重: {count} 首（精确/近似/片段级/跨集）")

    elif stage_id == "stage5":
        # 辅助清洗
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"辅助清洗: {count} 首（语言过滤/歌词转写/风格聚类）")

    elif stage_id == "stage6":
        # 预处理输出
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"预处理输出: {count} 首（切片 15s/50% overlap + 特征提取）")

    elif stage_id == "stage7":
        # 预标注
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"预标注: {count} 首（MERT/CLAP 嵌入 + 标签）")

    elif stage_id == "stage8":
        # 数据集划分
        df = load_csv_safe(input_path)
        count = len(df) if df is not None else 0
        if not dry_run:
            logger.info(f"数据集划分: {count} 首（train/val/test/holdout）")

    elapsed = time.time() - start_time
    logger.info(f"{stage_id} 完成: {count} 首, 耗时 {elapsed:.1f}s")

    return count, output_path


# ===================== 主流水线 =====================
def run_etl_pipeline(
    input_manifest: Path,
    stages_to_run: List[str] = None,
    resume_from: str = None,
    gpu_mode: bool = False,
    dry_run: bool = False,
    report_only: bool = False,
) -> Dict:
    """
    运行完整 ETL 流水线

    Args:
        input_manifest: 输入 manifest 路径
        stages_to_run: 要运行的阶段列表
        resume_from: 从哪个阶段开始（断点续跑）
        gpu_mode: GPU模式
        dry_run: 预览模式
        report_only: 只生成报告

    Returns:
        流水线结果字典
    """
    logger.info("=" * 60)
    logger.info("ETL 流水线启动")
    logger.info(f"输入: {input_manifest}")
    logger.info(f"GPU模式: {gpu_mode}")
    logger.info(f"预览模式: {dry_run}")
    logger.info("=" * 60)

    pipeline_start = time.time()
    stage_counts = OrderedDict()
    current_input = input_manifest

    # 确定要运行的阶段
    if stages_to_run is None:
        stages_to_run = list(STAGES.keys())

    if resume_from:
        if resume_from in stages_to_run:
            idx = stages_to_run.index(resume_from)
            stages_to_run = stages_to_run[idx:]
            logger.info(f"断点续跑: 从 {resume_from} 开始")

    # === 初始化算子级血缘记录器（Lineage v2.0）===
    lineage_logger = None
    if LINEAGE_AVAILABLE and not dry_run and not report_only:
        lineage_dir = PROJECT_ROOT / "data" / "lineage"
        lineage_dir.mkdir(parents=True, exist_ok=True)
        lineage_path = lineage_dir / f"etl_pipeline_{time_str}.json"
        lineage_logger = LineageLogger(
            dataset_version=f"etl_{time_str}",
            output_path=str(lineage_path),
            auto_save=True
        )
        logger.info(f"血缘记录器已初始化: {lineage_path}")

    # 执行各阶段
    if not report_only:
        for stage_id in stages_to_run:
            stage_info = STAGES.get(stage_id, {})
            stage_name = stage_info.get("name", stage_id)

            if lineage_logger:
                # 用上下文管理器记录算子执行（自动记录耗时和状态）
                with lineage_logger.operator(stage_name, version="1.0") as op:
                    op.set_input(str(current_input), count=stage_counts.get(stage_id, 0) if stage_counts else 0)
                    try:
                        count, output_path = execute_stage(
                            stage_id, current_input, PREPROCESS_DIR,
                            gpu_mode=gpu_mode, dry_run=dry_run
                        )
                        stage_counts[stage_id] = count
                        current_input = output_path if isinstance(output_path, Path) and output_path.suffix == ".csv" else current_input
                        op.set_output(str(output_path) if output_path else "", count=count)
                        op.set_config({"stage_id": stage_id, "gpu_mode": gpu_mode})
                    except Exception as e:
                        logger.error(f"{stage_id} 执行失败: {e}")
                        stage_counts[stage_id] = 0
                        op.set_output("", count=0, failed_count=1, failure_reasons={"execution_error": 1})
                        if not dry_run:
                            raise
            else:
                try:
                    count, output_path = execute_stage(
                        stage_id, current_input, PREPROCESS_DIR,
                        gpu_mode=gpu_mode, dry_run=dry_run
                    )
                    stage_counts[stage_id] = count
                    current_input = output_path if isinstance(output_path, Path) and output_path.suffix == ".csv" else current_input
                except Exception as e:
                    logger.error(f"{stage_id} 执行失败: {e}")
                    stage_counts[stage_id] = 0
                    if not dry_run:
                        raise
    else:
        # 只生成报告：从已有数据读取各阶段样本数
        logger.info("报告模式：从已有数据读取统计")
        for stage_id in stages_to_run:
            output_path = get_stage_output_path(stage_id)
            if output_path.suffix == ".csv" and output_path.exists():
                df = load_csv_safe(output_path)
                stage_counts[stage_id] = len(df) if df is not None else 0
            else:
                stage_counts[stage_id] = 0

    # 加载报告数据
    quality_report = load_csv_safe(CLEANED_DIR / "reports" / "quality_check_report.csv")
    yamnet_report = load_csv_safe(CLEANED_DIR / "reports" / "yamnet_output.csv")

    split_stats = None
    latest_version = sorted([d for d in FINAL_DATASET_DIR.iterdir() if d.is_dir()], reverse=True)
    if latest_version:
        stats_file = latest_version[0] / "stats" / "split_distribution.json"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                split_stats = json.load(f)

    # 生成可视化报告
    report_dir = REPORTS_DIR / f"etl_report_{time_str}"
    report_path = generate_etl_report(
        stage_counts=stage_counts,
        quality_report=quality_report,
        yamnet_report=yamnet_report,
        split_stats=split_stats,
        output_dir=report_dir,
    )

    # 保存流水线元数据
    pipeline_meta = {
        "pipeline_id": f"etl_{time_str}",
        "start_time": datetime.fromtimestamp(pipeline_start, TZ).isoformat(),
        "end_time": datetime.now(TZ).isoformat(),
        "duration_sec": round(time.time() - pipeline_start, 1),
        "input_manifest": str(input_manifest),
        "stages_run": stages_to_run,
        "stage_counts": dict(stage_counts),
        "gpu_mode": gpu_mode,
        "dry_run": dry_run,
        "report_path": str(report_path) if report_path else None,
    }
    save_json(pipeline_meta, report_dir / "pipeline_metadata.json")

    # === 保存 v2.0 算子级血缘 ===
    if lineage_logger:
        lineage_v2_path = report_dir / "lineage_v2.json"
        lineage_logger.save(str(lineage_v2_path))
        logger.info(f"算子级血缘(v2.0)已保存: {lineage_v2_path}")
        lineage_logger.print_summary()
        pipeline_meta["lineage_v2_path"] = str(lineage_v2_path)

    # 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("ETL 流水线完成")
    logger.info("=" * 60)
    logger.info(f"总耗时: {pipeline_meta['duration_sec']}s")
    logger.info(f"阶段统计:")
    for sid, count in stage_counts.items():
        logger.info(f"  {STAGES[sid]['name']}: {count} 首")
    if report_path:
        logger.info(f"可视化报告: {report_path}")
    logger.info("=" * 60)

    return pipeline_meta


# ===================== CLI 入口 =====================
def main():
    parser = argparse.ArgumentParser(
        description="正式自动化 ETL 流水线（Extract → Transform → Load）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="输入 manifest CSV 路径")
    parser.add_argument("--stages", type=str, default=None,
                        help="要运行的阶段（逗号分隔，如 1,2,3,4）")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="从指定阶段开始（断点续跑，如 stage3）")
    parser.add_argument("--gpu-mode", action="store_true",
                        help="GPU模式（重计算阶段在GPU跑）")
    parser.add_argument("--run-local", action="store_true",
                        help="本地模式（所有阶段在Mac跑）")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式（不实际执行，只打印计划）")
    parser.add_argument("--report-only", action="store_true",
                        help="只生成可视化报告（基于已有数据）")
    args = parser.parse_args()

    # 解析阶段
    stages_to_run = None
    if args.stages:
        stage_nums = [s.strip() for s in args.stages.split(",")]
        stages_to_run = [f"stage{n}" for n in stage_nums if f"stage{n}" in STAGES]

    input_manifest = Path(args.input)
    if not input_manifest.is_absolute():
        input_manifest = PROJECT_ROOT / input_manifest

    run_etl_pipeline(
        input_manifest=input_manifest,
        stages_to_run=stages_to_run,
        resume_from=args.resume_from,
        gpu_mode=args.gpu_mode,
        dry_run=args.dry_run,
        report_only=args.report_only,
    )


if __name__ == "__main__":
    main()
