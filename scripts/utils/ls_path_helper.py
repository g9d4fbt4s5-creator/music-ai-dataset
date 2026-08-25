#!/usr/bin/env python3
"""
Label Studio 本地文件路径转换工具

解决的核心问题：
    Label Studio 默认将 audio 字段的值视为 HTTP URL，不能直接读取本地文件路径。
    直接使用 /Users/.../audio.mp3 会导致 404 错误。

解决方案：
    1. 启动时启用 LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    2. 设置 LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT 为项目根目录
    3. audio 字段使用 /data/local-files/?d=相对路径 格式

用法：
    from ls_path_helper import local_to_ls_url, prepare_import_data

    # 单条路径转换
    url = local_to_ls_url("/Users/m.jian/music_corpus_project/data/audio.mp3")
    # → /data/local-files/?d=data/audio.mp3

    # 批量转换导入 JSON
    prepare_import_data("input.json", "output.json")
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any

# 项目根目录（Label Studio 本地文件服务的 DOCUMENT_ROOT）
PROJECT_ROOT = Path("/Users/m.jian/music_corpus_project")

# Label Studio 本地文件服务 URL 前缀
LS_LOCAL_FILES_PREFIX = "/data/local-files/?d="


def local_to_ls_url(local_path: Union[str, Path]) -> str:
    """
    将本地绝对路径转换为 Label Studio 可访问的 URL。

    Args:
        local_path: 本地文件路径，可以是绝对路径或相对路径

    Returns:
        Label Studio 本地文件服务 URL，格式为 /data/local-files/?d=相对路径

    Examples:
        >>> local_to_ls_url("/Users/m.jian/music_corpus_project/data/audio.mp3")
        '/data/local-files/?d=data/audio.mp3'

        >>> local_to_ls_url("data/audio.mp3")
        '/data/local-files/?d=data/audio.mp3'
    """
    path = Path(local_path)

    # 如果是绝对路径，转换为相对于 PROJECT_ROOT 的路径
    if path.is_absolute():
        try:
            rel_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            # 路径不在 PROJECT_ROOT 下，警告但仍返回原路径
            print(f"⚠️ 警告: 路径 {path} 不在项目根目录 {PROJECT_ROOT} 下，"
                  f"Label Studio 可能无法访问")
            return str(path)
    else:
        rel_path = path

    # 转换为 POSIX 格式（斜杠分隔）
    rel_posix = rel_path.as_posix()

    return f"{LS_LOCAL_FILES_PREFIX}{rel_posix}"


def ls_url_to_local(ls_url: str) -> Path:
    """
    反向转换：Label Studio URL → 本地绝对路径。

    Args:
        ls_url: Label Studio 本地文件服务 URL

    Returns:
        本地绝对路径

    Examples:
        >>> ls_url_to_local("/data/local-files/?d=data/audio.mp3")
        PosixPath('/Users/m.jian/music_corpus_project/data/audio.mp3')
    """
    if ls_url.startswith(LS_LOCAL_FILES_PREFIX):
        rel_path = ls_url[len(LS_LOCAL_FILES_PREFIX):]
    elif ls_url.startswith("/"):
        rel_path = ls_url[1:]
    else:
        rel_path = ls_url

    return PROJECT_ROOT / rel_path


def convert_import_data(import_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换导入数据列表中所有 audio 字段的路径。

    Args:
        import_data: Label Studio 导入数据列表，每个元素包含 data.audio 字段

    Returns:
        转换后的导入数据列表（原地修改并返回）
    """
    converted = 0
    for item in import_data:
        data = item.get("data", {})
        audio = data.get("audio", "")

        if audio:
            # 已经是 Label Studio URL 格式，跳过
            if audio.startswith(LS_LOCAL_FILES_PREFIX):
                continue

            # 转换路径
            new_url = local_to_ls_url(audio)
            data["audio"] = new_url
            converted += 1

    print(f"✅ 已转换 {converted}/{len(import_data)} 条音频路径")
    return import_data


def prepare_import_file(input_path: Union[str, Path], output_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    读取听检任务导入 JSON，将所有 audio 字段从本地绝对路径
    转换为 Label Studio 可访问的 URL，保存到新文件。

    Args:
        input_path: 输入 JSON 文件路径
        output_path: 输出 JSON 文件路径

    Returns:
        转换后的导入数据列表
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 读取输入
    data = json.loads(input_path.read_text(encoding="utf-8"))

    # 转换路径
    converted_data = convert_import_data(data)

    # 保存输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(converted_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"✅ 已保存到: {output_path}")
    return converted_data


def verify_audio_accessible(ls_url: str, base_url: str = "http://localhost:8080",
                            token: str = "") -> bool:
    """
    验证 Label Studio URL 是否可以正常访问音频文件。

    Args:
        ls_url: Label Studio 本地文件服务 URL
        base_url: Label Studio 服务地址
        token: API Token（可选）

    Returns:
        True 如果可以访问，False 否则
    """
    import requests

    headers = {}
    if token:
        headers["Authorization"] = f"Token {token}"

    try:
        resp = requests.get(f"{base_url}{ls_url}", headers=headers, stream=True, timeout=10)
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            print(f"✅ 音频可访问: HTTP 200, Content-Type: {content_type}")
            return True
        else:
            print(f"❌ 音频访问失败: HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ 音频访问异常: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Label Studio 路径转换工具")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # convert 子命令
    convert_parser = subparsers.add_parser("convert", help="转换导入 JSON 文件中的音频路径")
    convert_parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    convert_parser.add_argument("--output", required=True, help="输出 JSON 文件路径")

    # verify 子命令
    verify_parser = subparsers.add_parser("verify", help="验证音频 URL 是否可访问")
    verify_parser.add_argument("--url", required=True, help="Label Studio 音频 URL")
    verify_parser.add_argument("--base-url", default="http://localhost:8080", help="Label Studio 地址")
    verify_parser.add_argument("--token", default="", help="API Token")

    args = parser.parse_args()

    if args.command == "convert":
        prepare_import_file(args.input, args.output)
    elif args.command == "verify":
        verify_audio_accessible(args.url, args.base_url, args.token)
    else:
        parser.print_help()
