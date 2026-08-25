# Label Studio 本地音频文件配置指南

## 已知坑点

Label Studio 默认将 `audio` 字段的值视为 HTTP URL，**不能直接读取本地文件路径**。

直接使用 `/Users/.../audio.mp3` 会导致 404 错误：
```
There was an issue loading URL from $audio value
HTTP error status: 404
URL: http://localhost:8080/projects/15/data/00_raw_collect/raw_audio/...
```

## 解决方案（已封装为脚本）

### 1. 启动 Label Studio（自动带本地文件服务）

```bash
bash scripts/utils/start_labelstudio.sh
```

这个脚本自动设置以下环境变量：
- `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` — 启用本地文件服务
- `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/m.jian/music_corpus_project` — 文件根目录
- `LABEL_STUDIO_TOKEN` — API Token

### 2. 一键创建听检项目（自动处理存储和路径）

```bash
python scripts/utils/ls_create_task.py \
    --title "SNR阈值校准听检" \
    --template data/listening_tasks/template.xml \
    --import-json data/listening_tasks/import.json
```

这个脚本自动完成：
1. 创建项目
2. 创建本地文件存储（`recursive_scan=True`，递归扫描子目录）
3. 同步存储
4. 转换音频路径为 `/data/local-files/?d=相对路径` 格式
5. 导入任务
6. 删除同步产生的多余任务（只保留手动导入的）
7. 验证音频可访问

### 3. 路径转换工具（单独使用）

```bash
# 转换导入 JSON 文件中的音频路径
python scripts/utils/ls_path_helper.py convert \
    --input data/listening_tasks/import_raw.json \
    --output data/listening_tasks/import_converted.json

# 验证音频 URL 是否可访问
python scripts/utils/ls_path_helper.py verify \
    --url "/data/local-files/?d=data/00_raw_collect/raw_audio/xx/xx/file.mp3"
```

## 手动配置（不使用脚本时）

如果需要手动配置，按以下步骤：

### 步骤1：设置环境变量并启动

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/m.jian/music_corpus_project
label-studio start --port 8080
```

### 步骤2：创建本地文件存储

通过 API 创建（必须设置 `recursive_scan=True`，因为音频在子目录中）：

```python
import requests

resp = requests.post(
    "http://localhost:8080/api/storages/localfiles",
    json={
        "project": project_id,
        "path": "/Users/m.jian/music_corpus_project/data/00_raw_collect/raw_audio",
        "recursive_scan": True,  # 关键：递归扫描子目录
        "use_blob_urls": True,
    },
    headers={"Authorization": "Token <your_token>"}
)
```

### 步骤3：同步存储

```python
requests.post(
    f"http://localhost:8080/api/storages/localfiles/{storage_id}/sync",
    headers={"Authorization": "Token <your_token>"}
)
```

### 步骤4：导入 JSON 中 audio 字段使用正确格式

```json
{
  "data": {
    "audio": "/data/local-files/?d=data/00_raw_collect/raw_audio/xx/xx/file.mp3"
  }
}
```

**注意**：路径必须是相对于 `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` 的相对路径。

## 常见问题排查

### Q: 音频返回 404

**原因**：
1. 本地文件服务未启用（缺少环境变量）
2. 路径格式错误（使用了绝对路径而非 `/data/local-files/?d=` 格式）
3. 本地文件存储未创建或未同步
4. `recursive_scan=False`，音频在子目录中无法被扫描到

**解决**：
1. 确认使用 `scripts/utils/start_labelstudio.sh` 启动
2. 使用 `scripts/utils/ls_create_task.py` 创建项目
3. 检查音频路径格式是否为 `/data/local-files/?d=相对路径`

### Q: 项目中有很多多余的任务

**原因**：本地文件存储同步会自动为每个音频文件创建任务。

**解决**：`ls_create_task.py` 会自动清理多余任务，只保留手动导入的任务。手动清理时，删除只有 `audio` 字段、没有其他元数据的任务。

### Q: API 返回 401

**原因**：Token 无效或未启用 Legacy Token。

**解决**：
1. 确认使用 Legacy Token（40位十六进制），而非 JWT refresh token
2. 确认数据库中 `legacy_api_tokens_enabled=1`
3. Token 从 `label_studio.sqlite3` 的 `authtoken_token` 表获取

## 文件清单

```
scripts/utils/
├── start_labelstudio.sh          # 启动脚本（自动带环境变量）
├── ls_path_helper.py             # 路径转换工具
├── ls_create_task.py             # 一键创建任务
└── adaptive_listening_check.py   # 自适应听检任务生成

docs/
└── LABEL_STUDIO_SETUP.md         # 本文档
```

## 快速开始（3步）

```bash
# 1. 启动 Label Studio
bash scripts/utils/start_labelstudio.sh

# 2. 生成听检任务（adaptive_listening_check.py）
python scripts/utils/adaptive_listening_check.py generate --task-type qc_snr_calibration ...

# 3. 一键创建项目+存储+导入
python scripts/utils/ls_create_task.py \
    --title "SNR阈值校准听检" \
    --template data/listening_tasks/xxx_template.xml \
    --import-json data/listening_tasks/xxx_import.json
```

完成后访问 `http://localhost:8080/projects/<id>/` 开始听检。
