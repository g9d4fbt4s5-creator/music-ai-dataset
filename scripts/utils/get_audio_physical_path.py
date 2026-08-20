"""
get_audio_physical_path.py
统一音频物理路径计算工具

⚠️ 核心约束：
- 所有音频路径访问必须通过此函数，禁止业务代码手动拼接路径
- 禁止用 ls/find 扫描音频目录，永远读 audio_manifest.csv
- 文件名前缀带上 hash 片段只是方便人工排查，业务代码不要解析文件名拿 hash

散列规则：
- 散列键：md5(audio_id)[0:4]
- 两层散列：{base_dir}/{hash[0:2]}/{hash[2:4]}/
- 文件名：{hash_full}_{audio_id}.{ext}
- 示例：raw_audio/a1/b2/a1b2c3d4..._01ARZ3NDEKTSV4RRFFQ69G5FAV.mp3
"""
import hashlib
import re
import os
from pathlib import Path
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ULID 格式校验：26个字符，Crockford's Base32（排除 I,L,O,U）
ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")

# 支持的 base_dir
SUPPORTED_BASE_DIRS = [
    "raw_audio",
    "processed_audio/segments",
    "demucs_stems",
]


def validate_audio_id(audio_id: str) -> bool:
    """
    校验 audio_id 是否为合法 ULID 格式

    参数：
        audio_id: 待校验的 ID

    返回：
        True 合法，False 非法
    """
    if not isinstance(audio_id, str):
        return False
    # 大小写不敏感，统一转大写
    return bool(ULID_PATTERN.match(audio_id.upper()))


def compute_hash(audio_id: str) -> str:
    """
    计算 audio_id 的 md5 哈希

    参数：
        audio_id: 音频 ID

    返回：
        md5 哈希字符串（32位十六进制）
    """
    return hashlib.md5(audio_id.encode("utf-8")).hexdigest()


def get_audio_physical_path(audio_id: str,
                             extension: Optional[str] = None,
                             base_dir: str = "raw_audio") -> str:
    """
    计算音频的相对物理路径（统一散列规则）

    参数：
        audio_id: 音频 ULID
        extension: 文件扩展名（如 "mp3", "wav"），不传则返回目录路径
        base_dir: 基础目录，默认 "raw_audio"
                  可选："raw_audio", "processed_audio/segments", "demucs_stems"

    返回：
        相对路径字符串，如 "raw_audio/a1/b2/a1b2c3d4..._audio_id.mp3"

    示例：
        >>> get_audio_physical_path("01ARZ3NDEKTSV4RRFFQ69G5FAV", "mp3")
        'raw_audio/a1/b2/a1b2c3d4e5f6..._01ARZ3NDEKTSV4RRFFQ69G5FAV.mp3'
    """
    # 校验 base_dir
    if base_dir not in SUPPORTED_BASE_DIRS:
        raise ValueError(
            f"不支持的 base_dir: {base_dir}，"
            f"可选: {SUPPORTED_BASE_DIRS}"
        )

    # 统一转大写
    audio_id = audio_id.upper()

    # 计算哈希
    hash_full = compute_hash(audio_id)
    hash_prefix = hash_full[:4]  # 前4位用于两层散列

    # 两层散列目录
    dir_level1 = hash_prefix[0:2]
    dir_level2 = hash_prefix[2:4]

    # 构建路径
    if extension:
        # 去掉扩展名前的点
        ext = extension.lstrip(".")
        filename = f"{hash_full}_{audio_id}.{ext}"
        rel_path = f"{base_dir}/{dir_level1}/{dir_level2}/{filename}"
    else:
        # 不传扩展名，返回目录路径
        rel_path = f"{base_dir}/{dir_level1}/{dir_level2}/"

    return rel_path


def get_audio_absolute_path(audio_id: str,
                             extension: Optional[str] = None,
                             base_dir: str = "raw_audio") -> Path:
    """
    计算音频的绝对物理路径（项目根目录下）

    参数：
        audio_id: 音频 ULID
        extension: 文件扩展名
        base_dir: 基础目录

    返回：
        Path 对象，绝对路径
    """
    rel_path = get_audio_physical_path(audio_id, extension, base_dir)
    return PROJECT_ROOT / "data" / "00_raw_collect" / rel_path \
        if base_dir == "raw_audio" \
        else PROJECT_ROOT / "data" / "01_preprocess" / rel_path


def get_directory_for_audio(audio_id: str, base_dir: str = "raw_audio") -> str:
    """
    获取音频所在的目录路径（不含文件名）

    参数：
        audio_id: 音频 ULID
        base_dir: 基础目录

    返回：
        目录相对路径
    """
    return get_audio_physical_path(audio_id, extension=None, base_dir=base_dir)


def ensure_directory_for_audio(audio_id: str, base_dir: str = "raw_audio") -> Path:
    """
    确保音频所在目录存在，不存在则创建

    参数：
        audio_id: 音频 ULID
        base_dir: 基础目录

    返回：
        目录的绝对路径 Path 对象
    """
    if base_dir == "raw_audio":
        base = PROJECT_ROOT / "data" / "00_raw_collect"
    else:
        base = PROJECT_ROOT / "data" / "01_preprocess"

    rel_dir = get_directory_for_audio(audio_id, base_dir)
    abs_dir = base / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    return abs_dir


# ===================== 命令行入口 =====================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python get_audio_physical_path.py <audio_id> [extension] [base_dir]")
        print("")
        print("示例:")
        print("  python get_audio_physical_path.py 01ARZ3NDEKTSV4RRFFQ69G5FAV mp3")
        print("  python get_audio_physical_path.py 01ARZ3NDEKTSV4RRFFQ69G5FAV wav processed_audio/segments")
        print("")
        print("校验 ULID 格式:")
        print("  python get_audio_physical_path.py --validate 01ARZ3NDEKTSV4RRFFQ69G5FAV")
        sys.exit(1)

    # 校验模式
    if sys.argv[1] == "--validate":
        if len(sys.argv) < 3:
            print("请提供要校验的 audio_id")
            sys.exit(1)
        test_id = sys.argv[2]
        is_valid = validate_audio_id(test_id)
        print(f"audio_id: {test_id}")
        print(f"合法 ULID: {'是' if is_valid else '否'}")
        if is_valid:
            print(f"md5 哈希: {compute_hash(test_id)}")
        sys.exit(0 if is_valid else 1)

    # 路径计算模式
    audio_id = sys.argv[1]
    extension = sys.argv[2] if len(sys.argv) > 2 else None
    base_dir = sys.argv[3] if len(sys.argv) > 3 else "raw_audio"

    # 校验
    if not validate_audio_id(audio_id):
        print(f"⚠️  警告: {audio_id} 不是合法的 ULID 格式")
        print("   合法格式: 26个字符，Crockford's Base32（0-9, A-H, J-K, M-N, P, R-T, V-Z）")
        print("   示例: 01ARZ3NDEKTSV4RRFFQ69G5FAV")
        print()

    # 计算路径
    rel_path = get_audio_physical_path(audio_id, extension, base_dir)
    abs_path = get_audio_absolute_path(audio_id, extension, base_dir)
    hash_full = compute_hash(audio_id)

    print(f"audio_id:    {audio_id}")
    print(f"md5 哈希:    {hash_full}")
    print(f"散列前缀:    {hash_full[:4]} (两层: {hash_full[0:2]}/{hash_full[2:4]})")
    print(f"基础目录:    {base_dir}")
    print(f"扩展名:      {extension or '(未指定)'}")
    print()
    print(f"相对路径:    {rel_path}")
    print(f"绝对路径:    {abs_path}")
    print()
    print(f"目录是否存在: {'是' if abs_path.parent.exists() else '否'}")
