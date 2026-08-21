"""
verify_batch_transfer.py
批次回传完整性校验脚本

功能：
- 校验 GPU → Mac 回传的文件完整性（文件数量/大小/checksum）
- 只有校验通过才触发 GPU 清理（避免回传失败导致数据丢失）
- 生成校验报告

用法：
    # 校验批次回传完整性
    python verify_batch_transfer.py \
      --gpu-dir /root/autodl-tmp/batch_000_out/ \
      --mac-dir /Users/m.jian/music_corpus_project/data/01_preprocess/gpu_batches/batch_000/ \
      --checksum

    # 只校验文件数量和大小（快速模式）
    python verify_batch_transfer.py \
      --gpu-dir batch_000_out/ \
      --mac-dir batch_000/ \
      --quick

    # 严格模式（校验失败时报错退出）
    python verify_batch_transfer.py \
      --gpu-dir batch_000_out/ \
      --mac-dir batch_000/ \
      --strict
"""
import os
import sys
import json
import hashlib
import logging
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ===================== 配置 =====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
TZ = timezone(timedelta(hours=8))

LOG_DIR = PROJECT_ROOT / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"verify_batch_transfer_{time_str}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_file_list(dir_path: Path, extensions: List[str] = None) -> List[Path]:
    """获取目录下所有文件（递归）"""
    if not dir_path.exists():
        return []
    files = []
    for f in dir_path.rglob("*"):
        if f.is_file():
            if extensions is None or f.suffix.lower() in extensions:
                files.append(f)
    return files


def get_relative_path(file_path: Path, base_dir: Path) -> str:
    """获取相对于 base_dir 的路径"""
    return str(file_path.relative_to(base_dir))


def calculate_sha256(file_path: Path) -> str:
    """计算文件 SHA256"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def scan_directory(dir_path: Path, calculate_checksum: bool = False) -> Dict[str, Dict]:
    """
    扫描目录，生成文件清单

    Args:
        dir_path: 目录路径
        calculate_checksum: 是否计算 checksum

    Returns:
        {relative_path: {"size": int, "sha256": str}}
    """
    logger.info(f"扫描目录: {dir_path}")
    files = get_file_list(dir_path)
    result = {}

    for f in files:
        rel_path = get_relative_path(f, dir_path)
        file_info = {"size": f.stat().st_size}
        if calculate_checksum:
            file_info["sha256"] = calculate_sha256(f)
        result[rel_path] = file_info

    logger.info(f"  找到 {len(result)} 个文件，总计 {sum(v['size'] for v in result.values()) / 1024 / 1024:.1f} MB")
    return result


def verify_transfer(
    gpu_dir: Path,
    mac_dir: Path,
    calculate_checksum: bool = False,
) -> Dict:
    """
    校验 GPU → Mac 回传完整性

    Args:
        gpu_dir: GPU 端目录（通过 SSH 访问，这里假设已挂载或本地路径）
        mac_dir: Mac 端目录
        calculate_checksum: 是否计算 checksum

    Returns:
        校验结果字典
    """
    logger.info("=" * 60)
    logger.info("批次回传完整性校验")
    logger.info(f"  GPU 目录: {gpu_dir}")
    logger.info(f"  Mac 目录: {mac_dir}")
    logger.info(f"  Checksum 校验: {'开启' if calculate_checksum else '关闭（快速模式）'}")
    logger.info("=" * 60)

    result = {
        "passed": True,
        "gpu_dir": str(gpu_dir),
        "mac_dir": str(mac_dir),
        "checksum_enabled": calculate_checksum,
        "gpu_files": 0,
        "mac_files": 0,
        "missing_files": [],
        "size_mismatch": [],
        "checksum_mismatch": [],
        "extra_files": [],
    }

    # 检查目录是否存在
    if not gpu_dir.exists():
        logger.error(f"❌ GPU 目录不存在: {gpu_dir}")
        result["passed"] = False
        result["error"] = f"GPU directory not found: {gpu_dir}"
        return result

    if not mac_dir.exists():
        logger.error(f"❌ Mac 目录不存在: {mac_dir}")
        result["passed"] = False
        result["error"] = f"Mac directory not found: {mac_dir}"
        return result

    # 扫描两个目录
    gpu_files = scan_directory(gpu_dir, calculate_checksum=calculate_checksum)
    mac_files = scan_directory(mac_dir, calculate_checksum=calculate_checksum)

    result["gpu_files"] = len(gpu_files)
    result["mac_files"] = len(mac_files)

    # 检查缺失文件（GPU 有但 Mac 没有）
    for rel_path, gpu_info in gpu_files.items():
        if rel_path not in mac_files:
            result["missing_files"].append({
                "path": rel_path,
                "gpu_size": gpu_info["size"],
            })
            result["passed"] = False
            logger.error(f"  ❌ 缺失文件: {rel_path} (GPU: {gpu_info['size']} bytes)")
        else:
            mac_info = mac_files[rel_path]
            # 检查文件大小
            if gpu_info["size"] != mac_info["size"]:
                result["size_mismatch"].append({
                    "path": rel_path,
                    "gpu_size": gpu_info["size"],
                    "mac_size": mac_info["size"],
                })
                result["passed"] = False
                logger.error(f"  ❌ 大小不匹配: {rel_path} (GPU: {gpu_info['size']}, Mac: {mac_info['size']})")
            # 检查 checksum
            elif calculate_checksum and gpu_info.get("sha256") != mac_info.get("sha256"):
                result["checksum_mismatch"].append({
                    "path": rel_path,
                    "gpu_sha256": gpu_info.get("sha256"),
                    "mac_sha256": mac_info.get("sha256"),
                })
                result["passed"] = False
                logger.error(f"  ❌ Checksum 不匹配: {rel_path}")

    # 检查多余文件（Mac 有但 GPU 没有）
    for rel_path in mac_files:
        if rel_path not in gpu_files:
            result["extra_files"].append({"path": rel_path})
            logger.warning(f"  ⚠️  多余文件（Mac有但GPU没有）: {rel_path}")

    # 输出总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("校验结果总结")
    logger.info("=" * 60)
    logger.info(f"  GPU 文件数: {result['gpu_files']}")
    logger.info(f"  Mac 文件数: {result['mac_files']}")
    logger.info(f"  缺失文件: {len(result['missing_files'])}")
    logger.info(f"  大小不匹配: {len(result['size_mismatch'])}")
    logger.info(f"  Checksum 不匹配: {len(result['checksum_mismatch'])}")
    logger.info(f"  多余文件: {len(result['extra_files'])}")
    logger.info(f"  总体: {'✅ 通过' if result['passed'] else '❌ 失败'}")
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="批次回传完整性校验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu-dir", type=str, required=True,
                        help="GPU 端目录路径")
    parser.add_argument("--mac-dir", type=str, required=True,
                        help="Mac 端目录路径")
    parser.add_argument("--checksum", action="store_true",
                        help="计算并校验 SHA256 checksum（较慢但更可靠）")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（只校验文件数量和大小，不计算 checksum）")
    parser.add_argument("--strict", action="store_true",
                        help="严格模式（校验失败时报错退出）")
    parser.add_argument("--output", type=str, default=None,
                        help="校验报告输出路径（JSON）")
    args = parser.parse_args()

    gpu_dir = Path(args.gpu_dir)
    mac_dir = Path(args.mac_dir)

    if not gpu_dir.is_absolute():
        gpu_dir = PROJECT_ROOT / gpu_dir
    if not mac_dir.is_absolute():
        mac_dir = PROJECT_ROOT / mac_dir

    # 执行校验
    result = verify_transfer(
        gpu_dir=gpu_dir,
        mac_dir=mac_dir,
        calculate_checksum=args.checksum and not args.quick,
    )

    # 保存报告
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = mac_dir / "transfer_verification_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"校验报告已保存: {output_path}")

    # 严格模式
    if args.strict and not result["passed"]:
        logger.error("❌ 严格模式：校验失败，退出")
        sys.exit(1)

    return result["passed"]


if __name__ == "__main__":
    main()
