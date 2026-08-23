"""
download_datasets.py
MIR 公开数据集一键下载脚本

支持的数据集：
- JCS (Jazz-Choro-Samba): 爵士场景最佳，BPM/beat真值，200+首
- Ballroom: BPM检测标准基准，698首舞曲
- SMC MIREX: beat/onset真值，217首"难"样本
- MAESTRO: 钢琴MIDI即真值，200+小时
- GTZAN: 流派分类，1000首（无BPM真值）

用法：
    # 下载 JCS 数据集（爵士场景最佳）
    python download_datasets.py --dataset jcs --output-dir data/datasets

    # 下载 Ballroom 数据集（BPM检测标准基准）
    python download_datasets.py --dataset ballroom --output-dir data/datasets

    # 下载全部推荐数据集
    python download_datasets.py --dataset all --output-dir data/datasets

    # 只下载标注文件（不下载音频）
    python download_datasets.py --dataset jcs --annotations-only

    # 列出支持的数据集
    python download_datasets.py --list

注意：
- 部分数据集需要手动下载（如 McGill Billboard 音频不免费）
- 大文件下载可能需要较长时间，建议用 --limit 限制数量
- 下载失败的文件会记录到 failed_downloads.txt，可以重试
"""
import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 数据集配置
DATASETS = {
    "jcs": {
        "name": "JCS (Jazz-Choro-Samba)",
        "description": "爵士场景最佳，BPM/beat真值，200+首",
        "size": "~500MB",
        "audio_format": ".mp3/.wav",
        "annotations": "BPM + beat时间戳",
        "url": "https://github.com/CPJKU/JCS",
        "download_method": "git_clone",
        "repo_url": "https://github.com/CPJKU/JCS.git",
        "audio_subdir": "audio",
        "annotation_subdir": "annotations",
        "recommended": True,
        "scene": "jazz",
    },
    "ballroom": {
        "name": "Ballroom",
        "description": "BPM检测标准基准，698首舞曲",
        "size": "~2GB",
        "audio_format": ".wav",
        "annotations": "BPM + beat时间戳",
        "url": "http://mtg.upf.edu/ismir2004/contest/tempoContest/",
        "download_method": "manual",
        "recommended": True,
        "scene": "dance",
    },
    "smc": {
        "name": "SMC MIREX",
        "description": "beat/onset真值，217首'难'样本",
        "size": "~800MB",
        "audio_format": ".wav",
        "annotations": "beat + onset时间戳",
        "url": "https://zenodo.org/record/1442513",
        "download_method": "zenodo",
        "zenodo_record": "1442513",
        "recommended": False,
        "scene": "mixed",
    },
    "maestro": {
        "name": "MAESTRO",
        "description": "钢琴MIDI即真值，200+小时",
        "size": "~100GB",
        "audio_format": ".wav + .midi",
        "annotations": "note/onset/beat（MIDI即真值）",
        "url": "https://magenta.tensorflow.org/datasets/maestro",
        "download_method": "manual",
        "recommended": False,
        "scene": "piano",
    },
    "gtzan": {
        "name": "GTZAN",
        "description": "流派分类，1000首（无BPM真值）",
        "size": "~1.2GB",
        "audio_format": ".wav",
        "annotations": "genre标签（无BPM）",
        "url": "https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification",
        "download_method": "kaggle",
        "recommended": False,
        "scene": "genre",
    },
}


def list_datasets():
    """列出所有支持的数据集"""
    print("\n" + "=" * 80)
    print("  支持的 MIR 公开数据集")
    print("=" * 80)
    print(f"{'ID':<12} {'名称':<25} {'场景':<10} {'规模':<10} {'推荐':<6} {'下载方式'}")
    print("-" * 80)
    for dataset_id, info in DATASETS.items():
        recommended = "⭐⭐⭐" if info.get("recommended") else "⭐⭐"
        print(f"{dataset_id:<12} {info['name']:<25} {info.get('scene', ''):<10} "
              f"{info['size']:<10} {recommended:<6} {info['download_method']}")
    print("=" * 80)
    print("\n推荐组合：")
    print("  P0: jcs      — 爵士场景，BPM/beat真值，与你项目完全匹配")
    print("  P1: ballroom — BPM检测通用基准，适合 essentia vs madmom 对比")
    print("  P2: maestro  — 钢琴MIDI即真值，适合 onset 检测评测")
    print()


def download_git_clone(dataset_id: str, output_dir: Path,
                       annotations_only: bool = False) -> Tuple[bool, str]:
    """
    通过 git clone 下载数据集

    Args:
        dataset_id: 数据集ID
        output_dir: 输出目录
        annotations_only: 只下载标注文件

    Returns:
        (success, message)
    """
    info = DATASETS[dataset_id]
    repo_url = info.get("repo_url")
    if not repo_url:
        return False, f"{dataset_id} 没有 git 仓库地址"

    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"克隆 {info['name']} 仓库: {repo_url}")
    logger.info(f"目标目录: {dataset_dir}")

    try:
        # 浅克隆（只拉最新版本，节省时间和空间）
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dataset_dir)],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        if result.returncode == 0:
            logger.info(f"✅ {info['name']} 下载成功")
            # 列出下载的文件
            files = list(dataset_dir.rglob("*"))
            audio_files = [f for f in files if f.suffix.lower() in [".mp3", ".wav", ".flac", ".ogg"]]
            annotation_files = [f for f in files if f.suffix.lower() in [".csv", ".txt", ".beats", ".bpm", ".json"]]
            logger.info(f"  音频文件: {len(audio_files)}")
            logger.info(f"  标注文件: {len(annotation_files)}")
            return True, f"下载成功，{len(audio_files)} 个音频，{len(annotation_files)} 个标注"
        else:
            error_msg = result.stderr.strip()
            logger.error(f"❌ 克隆失败: {error_msg}")
            return False, f"克隆失败: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "克隆超时（超过10分钟）"
    except Exception as e:
        return False, f"克隆异常: {e}"


def download_zenodo(dataset_id: str, output_dir: Path,
                    annotations_only: bool = False) -> Tuple[bool, str]:
    """
    从 Zenodo 下载数据集

    Args:
        dataset_id: 数据集ID
        output_dir: 输出目录
        annotations_only: 只下载标注文件

    Returns:
        (success, message)
    """
    info = DATASETS[dataset_id]
    record_id = info.get("zenodo_record")
    if not record_id:
        return False, f"{dataset_id} 没有 Zenodo record ID"

    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"从 Zenodo 下载 {info['name']} (record: {record_id})")
    logger.info(f"目标目录: {dataset_dir}")

    # Zenodo API 获取文件列表
    api_url = f"https://zenodo.org/api/records/{record_id}"
    try:
        import urllib.request
        with urllib.request.urlopen(api_url, timeout=30) as response:
            data = json.loads(response.read().decode())

        files = data.get("files", [])
        logger.info(f"找到 {len(files)} 个文件")

        for file_info in files:
            filename = file_info.get("key", "unknown")
            file_url = file_info.get("links", {}).get("self")
            file_size = file_info.get("size", 0)

            # 如果只下载标注，跳过音频文件
            if annotations_only and filename.lower().endswith((".wav", ".mp3", ".flac", ".zip")):
                if "annotation" not in filename.lower() and "beat" not in filename.lower():
                    logger.info(f"  跳过（音频文件）: {filename}")
                    continue

            logger.info(f"  下载: {filename} ({file_size / 1024 / 1024:.1f} MB)")
            output_file = dataset_dir / filename

            try:
                urllib.request.urlretrieve(file_url, str(output_file))
                logger.info(f"    ✅ 完成")
            except Exception as e:
                logger.error(f"    ❌ 失败: {e}")
                # 记录失败
                with open(dataset_dir / "failed_downloads.txt", "a") as f:
                    f.write(f"{filename}\t{file_url}\t{e}\n")

        return True, f"Zenodo 下载完成"

    except Exception as e:
        return False, f"Zenodo 下载失败: {e}"


def download_manual(dataset_id: str, output_dir: Path,
                   annotations_only: bool = False) -> Tuple[bool, str]:
    """
    手动下载数据集（打印下载指南）

    Args:
        dataset_id: 数据集ID
        output_dir: 输出目录
        annotations_only: 只下载标注文件

    Returns:
        (success, message)
    """
    info = DATASETS[dataset_id]
    dataset_dir = output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 生成下载指南
    guide = f"""
{'='*60}
  {info['name']} 手动下载指南
{'='*60}

数据集信息:
  描述: {info['description']}
  规模: {info['size']}
  音频格式: {info['audio_format']}
  标注: {info['annotations']}
  官网: {info['url']}

下载步骤:
  1. 访问官网: {info['url']}
  2. 下载音频文件到: {dataset_dir / 'audio'}
  3. 下载标注文件到: {dataset_dir / 'annotations'}

目录结构:
  {dataset_dir}/
  ├── audio/          # 音频文件
  │   ├── track_001.wav
  │   ├── track_002.wav
  │   └── ...
  └── annotations/    # 标注文件
      ├── track_001.bpm
      ├── track_001.beats
      └── ...

下载完成后，运行评测:
  python scripts/06_evaluation/eval_bpm.py \\
      --audio-dir {dataset_dir / 'audio'} \\
      --truth-dir {dataset_dir / 'annotations'} \\
      --dataset-type {dataset_id} \\
      --tools essentia,madmom,librosa

{'='*60}
"""
    print(guide)

    # 保存指南到文件
    guide_file = dataset_dir / "DOWNLOAD_GUIDE.txt"
    with open(guide_file, "w", encoding="utf-8") as f:
        f.write(guide)
    logger.info(f"下载指南已保存: {guide_file}")

    return True, "手动下载指南已生成"


def download_dataset(dataset_id: str, output_dir: Path,
                    annotations_only: bool = False) -> Tuple[bool, str]:
    """
    下载数据集（统一入口）

    Args:
        dataset_id: 数据集ID
        output_dir: 输出目录
        annotations_only: 只下载标注文件

    Returns:
        (success, message)
    """
    if dataset_id not in DATASETS:
        return False, f"不支持的数据集: {dataset_id}"

    info = DATASETS[dataset_id]
    method = info["download_method"]

    logger.info(f"开始下载: {info['name']}")
    logger.info(f"下载方式: {method}")
    logger.info(f"输出目录: {output_dir / dataset_id}")

    if method == "git_clone":
        return download_git_clone(dataset_id, output_dir, annotations_only)
    elif method == "zenodo":
        return download_zenodo(dataset_id, output_dir, annotations_only)
    elif method == "manual":
        return download_manual(dataset_id, output_dir, annotations_only)
    elif method == "kaggle":
        return download_manual(dataset_id, output_dir, annotations_only)  # Kaggle也需要手动
    else:
        return False, f"未知的下载方式: {method}"


def main():
    parser = argparse.ArgumentParser(
        description="MIR 公开数据集一键下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集ID（jcs/ballroom/smc/maestro/gtzan/all）")
    parser.add_argument("--output-dir", type=str, default="data/datasets",
                        help="输出目录（默认 data/datasets）")
    parser.add_argument("--annotations-only", action="store_true",
                        help="只下载标注文件（不下载音频）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有支持的数据集")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制下载数量（部分数据集支持）")
    args = parser.parse_args()

    # 列出数据集
    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.print_help()
        print("\n示例：")
        print("  # 下载 JCS 数据集（爵士场景最佳）")
        print("  python download_datasets.py --dataset jcs --output-dir data/datasets")
        print()
        print("  # 列出所有支持的数据集")
        print("  python download_datasets.py --list")
        return

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 下载全部推荐数据集
    if args.dataset == "all":
        recommended = [did for did, info in DATASETS.items() if info.get("recommended")]
        logger.info(f"下载全部推荐数据集: {recommended}")

        results = {}
        for dataset_id in recommended:
            success, msg = download_dataset(dataset_id, output_dir, args.annotations_only)
            results[dataset_id] = (success, msg)
            logger.info("")  # 空行分隔

        # 汇总
        print("\n" + "=" * 60)
        print("  下载汇总")
        print("=" * 60)
        for dataset_id, (success, msg) in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {dataset_id}: {msg}")
        print("=" * 60)

    else:
        # 下载单个数据集
        success, msg = download_dataset(args.dataset, output_dir, args.annotations_only)
        if success:
            logger.info(f"✅ {msg}")
        else:
            logger.error(f"❌ {msg}")
            sys.exit(1)


if __name__ == "__main__":
    main()
