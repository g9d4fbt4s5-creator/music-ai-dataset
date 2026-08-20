"""
freeze_version.py
数据集版本冻结工具

⚠️ 核心约束：
- latest 软链接只允许脚本修改，禁止人工手动修改
- 版本号精确到秒：vYYYYMMDD_HHMMSS
- 同一秒内多次冻结自动追加 -001、-002
- 冻结后版本目录只读，不再修改

功能：
1. 创建版本目录：vYYYYMMDD_HHMMSS/
2. 复制数据集元数据到版本目录
3. 生成版本说明 readme.md
4. 更新 latest 软链接指向新版本
5. 记录版本历史

用法：
    # 冻结当前数据集版本
    python freeze_version.py --note "第一次正式版本"

    # 指定源快照
    python freeze_version.py --src-snapshot ./snapshots/gpu_backup_20260820_173500

    # 预览模式（不实际创建）
    python freeze_version.py --dry-run

    # 查看版本历史
    python freeze_version.py --list
"""
import os
import sys
import json
import shutil
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 时区
TZ = timezone(timedelta(hours=8))

# 数据集目录
FINAL_DATASET_DIR = PROJECT_ROOT / "data" / "04_final_dataset"
FINAL_METADATA_DIR = FINAL_DATASET_DIR / "final_metadata"
LATEST_LINK = FINAL_DATASET_DIR / "latest"
VERSION_HISTORY_FILE = FINAL_DATASET_DIR / "version_history.json"

# -------- logging 配置 --------
LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"freeze_version_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_version_name() -> str:
    """
    生成版本号：vYYYYMMDD_HHMMSS
    同一秒内冲突自动追加 -001、-002
    """
    now = datetime.now(TZ)
    base_name = now.strftime("v%Y%m%d_%H%M%S")

    # 检查是否已存在
    version_dir = FINAL_DATASET_DIR / base_name
    if not version_dir.exists():
        return base_name

    # 同一秒冲突，追加序列号
    counter = 1
    while True:
        version_name = f"{base_name}-{counter:03d}"
        version_dir = FINAL_DATASET_DIR / version_name
        if not version_dir.exists():
            return version_name
        counter += 1


def load_version_history() -> List[Dict]:
    """加载版本历史"""
    if not VERSION_HISTORY_FILE.exists():
        return []
    try:
        with open(VERSION_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_version_history(history: List[Dict]):
    """保存版本历史"""
    with open(VERSION_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def list_versions():
    """列出所有版本"""
    history = load_version_history()

    if not history:
        logger.info("暂无版本记录")
        return

    logger.info("=" * 80)
    logger.info(f"{'版本号':<30} {'创建时间':<25} {'样本数':<10} {'备注'}")
    logger.info("-" * 80)

    for v in reversed(history):
        version_name = v.get("version", "")
        created_at = v.get("created_at", "")
        sample_count = v.get("sample_count", 0)
        note = v.get("note", "")
        is_latest = " (latest)" if v.get("is_latest") else ""
        logger.info(f"{version_name:<30} {created_at:<25} {sample_count:<10} {note}{is_latest}")

    logger.info("=" * 80)

    # 显示 latest 指向
    if LATEST_LINK.exists() and LATEST_LINK.is_symlink():
        target = os.readlink(LATEST_LINK)
        logger.info(f"latest -> {target}")


def count_samples(metadata_dir: Path) -> int:
    """统计数据集样本数"""
    train_csv = metadata_dir / "train_split.csv"
    val_csv = metadata_dir / "val_split.csv"
    test_csv = metadata_dir / "test_split.csv"
    holdout_csv = metadata_dir / "holdout_gold.csv"

    total = 0
    for csv_file in [train_csv, val_csv, test_csv, holdout_csv]:
        if csv_file.exists():
            try:
                import csv
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    total += max(0, sum(1 for _ in reader) - 1)  # 减去表头
            except Exception:
                pass

    return total


def generate_version_readme(version_name: str, note: str, sample_count: int,
                            src_snapshot: Optional[str] = None) -> str:
    """生成版本说明 readme.md"""
    now = datetime.now(TZ).isoformat()

    readme = f"""# 数据集版本 {version_name}

## 基本信息

| 项目 | 内容 |
|------|------|
| 版本号 | {version_name} |
| 创建时间 | {now} |
| 样本总数 | {sample_count} |
| 备注 | {note} |
| 源快照 | {src_snapshot or '无（直接从 final_metadata 冻结）'} |

## 目录结构

```
{version_name}/
├── final_metadata/
│   ├── corpus_full_meta.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   └── holdout_gold.csv
└── readme.md
```

## 数据集划分

| 子集 | 比例 | 用途 |
|------|------|------|
| train | 70% | 模型训练 |
| val | 10% | 验证调参 |
| test | 10% | 测试评估 |
| holdout_gold | 10% | 黄金留出集（全程不参与训练调参） |

## 注意事项

- 本版本冻结后只读，不再修改
- latest 软链接指向当前生产版本
- 如需回滚到旧版本，修改 latest 软链接即可

---

*由 freeze_version.py 自动生成*
"""
    return readme


def freeze_version(note: str = "", src_snapshot: Optional[str] = None,
                   dry_run: bool = False) -> Dict:
    """
    冻结数据集版本

    参数：
        note: 版本备注
        src_snapshot: 源快照路径（可选）
        dry_run: 预览模式

    返回：
        版本信息字典
    """
    result = {
        "version": None,
        "created": False,
        "sample_count": 0,
        "errors": [],
    }

    # 检查源数据
    if not FINAL_METADATA_DIR.exists():
        logger.error(f"❌ final_metadata 目录不存在: {FINAL_METADATA_DIR}")
        logger.error("   请先运行 convert_ls_jsonl.py 生成数据集切分")
        result["errors"].append("final_metadata not found")
        return result

    # 生成版本号
    version_name = generate_version_name()
    version_dir = FINAL_DATASET_DIR / version_name
    result["version"] = version_name

    logger.info(f"版本号: {version_name}")
    logger.info(f"版本目录: {version_dir}")

    # 统计样本数
    sample_count = count_samples(FINAL_METADATA_DIR)
    result["sample_count"] = sample_count
    logger.info(f"样本总数: {sample_count}")

    if dry_run:
        logger.info("[DRY-RUN] 预览模式，不实际创建版本")
        result["created"] = True
        return result

    # 创建版本目录
    version_dir.mkdir(parents=True, exist_ok=True)

    # 复制元数据
    target_metadata_dir = version_dir / "final_metadata"
    if target_metadata_dir.exists():
        shutil.rmtree(target_metadata_dir)
    shutil.copytree(FINAL_METADATA_DIR, target_metadata_dir)
    logger.info(f"✅ 已复制元数据到: {target_metadata_dir}")

    # 生成 readme.md
    readme_content = generate_version_readme(version_name, note, sample_count, src_snapshot)
    readme_path = version_dir / "readme.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info(f"✅ 已生成版本说明: {readme_path}")

    # 更新 latest 软链接
    if LATEST_LINK.exists() or LATEST_LINK.is_symlink():
        LATEST_LINK.unlink()
    # 使用相对路径创建软链接
    os.symlink(version_name, LATEST_LINK)
    logger.info(f"✅ 已更新 latest 软链接: latest -> {version_name}")

    # 更新版本历史
    history = load_version_history()
    # 标记旧版本为非 latest
    for v in history:
        v["is_latest"] = False
    # 添加新版本
    history.append({
        "version": version_name,
        "created_at": datetime.now(TZ).isoformat(),
        "sample_count": sample_count,
        "note": note,
        "src_snapshot": src_snapshot,
        "is_latest": True,
    })
    save_version_history(history)
    logger.info(f"✅ 已更新版本历史: {VERSION_HISTORY_FILE}")

    result["created"] = True
    return result


def main():
    parser = argparse.ArgumentParser(
        description="数据集版本冻结工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 冻结当前数据集版本
  python freeze_version.py --note "第一次正式版本"

  # 指定源快照
  python freeze_version.py --src-snapshot ./snapshots/gpu_backup_20260820_173500 --note "GPU推理结果"

  # 预览模式
  python freeze_version.py --dry-run --note "测试"

  # 查看版本历史
  python freeze_version.py --list
        """
    )
    parser.add_argument("--note", type=str, default="", help="版本备注")
    parser.add_argument("--src-snapshot", type=str, default=None, help="源快照路径")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，不实际创建")
    parser.add_argument("--list", action="store_true", help="列出所有版本")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("数据集版本冻结")
    logger.info("=" * 60)

    if args.list:
        list_versions()
        sys.exit(0)

    logger.info(f"备注: {args.note or '(无)'}")
    logger.info(f"源快照: {args.src_snapshot or '(无)'}")
    logger.info(f"预览模式: {'是' if args.dry_run else '否'}")
    logger.info("")

    # 执行冻结
    result = freeze_version(
        note=args.note,
        src_snapshot=args.src_snapshot,
        dry_run=args.dry_run,
    )

    logger.info("")
    logger.info("=" * 60)
    if result["created"]:
        logger.info(f"✅ 版本冻结成功: {result['version']}")
        logger.info(f"   样本数: {result['sample_count']}")
        logger.info(f"   日志文件: {log_file}")
        logger.info("")
        logger.info("💡 提示：")
        logger.info("   - 版本目录只读，不再修改")
        logger.info("   - latest 软链接已指向新版本")
        logger.info("   - 如需回滚，修改 latest 软链接即可")
    else:
        logger.error("❌ 版本冻结失败")
        for err in result["errors"]:
            logger.error(f"   - {err}")
    logger.info("=" * 60)

    sys.exit(0 if result["created"] else 1)


if __name__ == "__main__":
    main()
