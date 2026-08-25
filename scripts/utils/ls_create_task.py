#!/usr/bin/env python3
"""
一键创建 Label Studio 听检项目：自动处理本地存储和路径转换

解决的问题：
    每次创建新的听检项目时，需要手动：
    1. 创建项目
    2. 创建本地文件存储（recursive_scan=True）
    3. 同步存储
    4. 转换音频路径为 /data/local-files/?d= 格式
    5. 导入任务
    6. 删除同步产生的多余任务

这个脚本把以上步骤封装为一键操作。

用法：
    python scripts/utils/ls_create_task.py \\
        --title "SNR阈值校准听检" \\
        --template data/listening_tasks/template.xml \\
        --import-json data/listening_tasks/import.json \\
        --audio-dir data/00_raw_collect/raw_audio

    # 或者使用 adaptive_listening_check.py 生成的任务
    python scripts/utils/ls_create_task.py \\
        --task-dir data/listening_tasks/qc_snr_calibration_20260825_205428
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests

# 添加 scripts/utils 到路径，以便导入 ls_path_helper
sys.path.insert(0, str(Path(__file__).parent))
from ls_path_helper import local_to_ls_url, convert_import_data, verify_audio_accessible

# ============================================================
# 配置
# ============================================================
BASE_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
TOKEN = os.environ.get("LABEL_STUDIO_TOKEN", "0c42572cb998a04808267af748b96fb88cde6fc3")
PROJECT_ROOT = Path("/Users/m.jian/music_corpus_project")


def get_headers() -> Dict[str, str]:
    """获取请求头"""
    return {
        "Authorization": f"Token {TOKEN}",
        "Content-Type": "application/json"
    }


def check_label_studio_running() -> bool:
    """检查 Label Studio 是否正在运行"""
    try:
        resp = requests.get(f"{BASE_URL}/api/projects/", headers=get_headers(), timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def create_project(title: str, template_xml: str) -> int:
    """
    创建 Label Studio 项目

    Args:
        title: 项目标题
        template_xml: Labeling Interface XML

    Returns:
        项目 ID
    """
    resp = requests.post(
        f"{BASE_URL}/api/projects/",
        json={"title": title, "label_config": template_xml},
        headers=get_headers()
    )

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"创建项目失败: HTTP {resp.status_code}, {resp.text[:200]}")

    project_id = resp.json()["id"]
    print(f"✅ 项目创建成功: ID={project_id}, 标题='{title}'")
    return project_id


def create_local_storage(project_id: int, audio_dir: str) -> int:
    """
    创建本地文件存储（递归扫描）

    Args:
        project_id: 项目 ID
        audio_dir: 音频目录路径（相对于项目根目录或绝对路径）

    Returns:
        存储 ID
    """
    # 转换为绝对路径
    audio_path = Path(audio_dir)
    if not audio_path.is_absolute():
        audio_path = PROJECT_ROOT / audio_path

    storage_payload = {
        "project": project_id,
        "path": str(audio_path),
        "recursive_scan": True,
        "use_blob_urls": True,
        "presign": True,
        "title": "本地音频文件",
        "description": f"自动创建：听检任务音频源 ({audio_dir})"
    }

    resp = requests.post(
        f"{BASE_URL}/api/storages/localfiles",
        json=storage_payload,
        headers=get_headers()
    )

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"创建本地存储失败: HTTP {resp.status_code}, {resp.text[:200]}")

    storage_id = resp.json()["id"]
    print(f"✅ 本地文件存储创建: ID={storage_id}, 路径={audio_path}")
    return storage_id


def sync_storage(storage_id: int) -> int:
    """
    同步本地文件存储

    Args:
        storage_id: 存储 ID

    Returns:
        同步的文件数量
    """
    resp = requests.post(
        f"{BASE_URL}/api/storages/localfiles/{storage_id}/sync",
        headers=get_headers()
    )

    if resp.status_code not in (200, 201):
        print(f"⚠️ 存储同步返回: HTTP {resp.status_code}")
        return 0

    # 等待同步完成
    time.sleep(2)

    # 查询同步结果
    resp = requests.get(
        f"{BASE_URL}/api/storages/localfiles/{storage_id}",
        headers=get_headers()
    )
    if resp.status_code == 200:
        file_count = resp.json().get("file_count", 0)
        print(f"✅ 存储同步完成: {file_count} 个文件")
        return file_count

    print(f"✅ 存储同步完成")
    return 0


def import_tasks(project_id: int, import_data: List[Dict[str, Any]]) -> int:
    """
    导入任务到项目

    Args:
        project_id: 项目 ID
        import_data: 导入数据列表

    Returns:
        导入的任务数量
    """
    resp = requests.post(
        f"{BASE_URL}/api/projects/{project_id}/import",
        json=import_data,
        headers=get_headers()
    )

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"导入任务失败: HTTP {resp.status_code}, {resp.text[:200]}")

    task_count = resp.json().get("task_count", len(import_data))
    print(f"✅ 任务导入成功: {task_count} 首")
    return task_count


def get_project_tasks(project_id: int) -> List[Dict[str, Any]]:
    """获取项目所有任务"""
    resp = requests.get(
        f"{BASE_URL}/api/projects/{project_id}/tasks?page_size=1000",
        headers=get_headers()
    )
    if resp.status_code == 200:
        data = resp.json()
        return data if isinstance(data, list) else data.get("tasks", [])
    return []


def cleanup_extra_tasks(project_id: int, expected_count: int) -> int:
    """
    删除同步产生的多余任务（只保留手动导入的任务）

    本地文件存储同步会自动为每个音频文件创建任务，
    但我们只需要手动导入的那几首。

    Args:
        project_id: 项目 ID
        expected_count: 期望保留的任务数量

    Returns:
        删除的任务数量
    """
    tasks = get_project_tasks(project_id)
    total = len(tasks)

    if total <= expected_count:
        print(f"✅ 任务数量正常: {total}/{expected_count}，无需清理")
        return 0

    # 识别手动导入的任务（包含特定字段，如 music_score、snr 等）
    # 同步产生的任务只有 audio 字段，没有其他元数据
    imported_tasks = []
    extra_tasks = []

    for task in tasks:
        data = task.get("data", {})
        # 手动导入的任务通常包含除 audio 外的其他字段
        extra_fields = {k: v for k, v in data.items() if k != "audio"}
        if extra_fields:
            imported_tasks.append(task)
        else:
            extra_tasks.append(task)

    # 如果无法通过字段区分，保留前 expected_count 个
    if len(imported_tasks) < expected_count:
        imported_tasks = tasks[:expected_count]
        extra_tasks = tasks[expected_count:]

    # 删除多余任务
    deleted = 0
    for task in extra_tasks:
        resp = requests.delete(
            f"{BASE_URL}/api/tasks/{task['id']}/",
            headers=get_headers()
        )
        if resp.status_code in (200, 204):
            deleted += 1

    print(f"🧹 清理多余任务: 删除 {deleted} 首，保留 {total - deleted} 首")
    return deleted


def verify_first_audio(project_id: int) -> bool:
    """验证项目中第一个任务的音频是否可访问"""
    tasks = get_project_tasks(project_id)
    if not tasks:
        print("⚠️ 项目中没有任务，跳过音频验证")
        return False

    first_task = tasks[0]
    audio_url = first_task.get("data", {}).get("audio", "")

    if not audio_url:
        print("⚠️ 第一个任务没有 audio 字段")
        return False

    return verify_audio_accessible(audio_url, BASE_URL, TOKEN)


def create_task(
    title: str,
    template_xml_path: Path,
    import_json_path: Path,
    audio_dir: str = "data/00_raw_collect/raw_audio",
    cleanup: bool = True,
    verify: bool = True
) -> int:
    """
    一键创建 Label Studio 听检项目

    Args:
        title: 项目标题
        template_xml_path: 模板 XML 文件路径
        import_json_path: 导入 JSON 文件路径
        audio_dir: 音频目录路径
        cleanup: 是否清理同步产生的多余任务
        verify: 是否验证音频可访问

    Returns:
        项目 ID
    """
    print("=" * 60)
    print(f"🚀 一键创建 Label Studio 听检项目")
    print(f"   标题: {title}")
    print(f"   模板: {template_xml_path}")
    print(f"   导入: {import_json_path}")
    print("=" * 60)

    # 0. 检查 Label Studio 是否运行
    if not check_label_studio_running():
        print("❌ Label Studio 未运行，请先执行: bash scripts/utils/start_labelstudio.sh")
        sys.exit(1)
    print("✅ Label Studio 运行正常")

    # 1. 读取模板
    template_xml = template_xml_path.read_text(encoding="utf-8")
    print(f"✅ 模板读取成功: {len(template_xml)} 字符")

    # 2. 读取并转换导入数据
    import_data = json.loads(import_json_path.read_text(encoding="utf-8"))
    print(f"✅ 导入数据读取: {len(import_data)} 首")
    import_data = convert_import_data(import_data)

    # 3. 创建项目
    project_id = create_project(title, template_xml)

    # 4. 创建本地文件存储
    storage_id = create_local_storage(project_id, audio_dir)

    # 5. 同步存储
    sync_storage(storage_id)

    # 6. 导入任务
    import_tasks(project_id, import_data)

    # 7. 清理多余任务
    if cleanup:
        time.sleep(1)
        cleanup_extra_tasks(project_id, len(import_data))

    # 8. 验证音频
    if verify:
        time.sleep(1)
        verify_first_audio(project_id)

    print("=" * 60)
    print(f"🎉 项目创建完成！")
    print(f"🔗 标注地址: {BASE_URL}/projects/{project_id}/")
    print("=" * 60)

    return project_id


def main():
    parser = argparse.ArgumentParser(
        description="一键创建 Label Studio 听检项目（自动处理本地存储和路径转换）"
    )
    parser.add_argument("--title", required=True, help="项目标题")
    parser.add_argument("--template", type=Path, required=True, help="模板 XML 文件路径")
    parser.add_argument("--import-json", type=Path, required=True, help="导入 JSON 文件路径")
    parser.add_argument("--audio-dir", default="data/00_raw_collect/raw_audio",
                        help="音频目录路径（默认: data/00_raw_collect/raw_audio）")
    parser.add_argument("--no-cleanup", action="store_true", help="不清理同步产生的多余任务")
    parser.add_argument("--no-verify", action="store_true", help="不验证音频可访问性")

    args = parser.parse_args()

    # 验证文件存在
    if not args.template.exists():
        print(f"❌ 模板文件不存在: {args.template}")
        sys.exit(1)
    if not args.import_json.exists():
        print(f"❌ 导入文件不存在: {args.import_json}")
        sys.exit(1)

    create_task(
        title=args.title,
        template_xml_path=args.template,
        import_json_path=args.import_json,
        audio_dir=args.audio_dir,
        cleanup=not args.no_cleanup,
        verify=not args.no_verify
    )


if __name__ == "__main__":
    main()
