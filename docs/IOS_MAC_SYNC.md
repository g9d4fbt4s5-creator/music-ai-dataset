# iOS → Mac 数据采集同步工作流

> 版本: v1.0.0 | 更新: 2026-08-25
> 场景: Mac 无法科学上网，iPhone/iPad 可以。利用苹果生态互通性完成外网数据采集。

## 一、整体架构

```
iPhone/iPad (科学上网)                    Mac (无科学上网)
┌─────────────────────┐                  ┌─────────────────────┐
│ 1. 下载外网音频      │   AirDrop/       │ 4. 入库              │
│    (YouTube/SC等)    │───iCloud──→     │    import_audio.py   │
│ 2. 转码/重命名        │   Files          │ 5. 跑流水线          │
│ 3. 调用Gemini/GPT    │                  │    Stage 0-6 + L1-L4│
│    API(高精度需求)    │                  │                      │
└─────────────────────┘                  └─────────────────────┘
```

## 二、数据采集流程

### 2.1 iPhone/iPad 端下载

#### 方案 A: Documents by Readdle (推荐)
1. App Store 安装 **Documents by Readdle**
2. 内置浏览器可科学上网访问 YouTube/SoundCloud
3. 下载音频到 App 本地文件
4. 支持后台下载、批量下载

#### 方案 B: Shortcuts 自动化
1. 创建快捷指令："下载音频并保存到 Files"
2. 输入 URL → 调用 yt-dlp(通过 a-Shell) → 保存到 iCloud Drive
3. 可分享菜单直接触发

#### 方案 C: a-Shell (终端)
1. App Store 安装 **a-Shell** (类Unix终端)
2. 安装 yt-dlp: `pip install yt-dlp`
3. 命令行下载: `yt-dlp -x --audio-format mp3 <URL>`
4. 文件保存在 a-Shell 沙盒，可分享到 Files

### 2.2 传输到 Mac

#### 方式 1: AirDrop (最快，适合少量文件)
1. iPhone/iPad: 文件 App → 长按音频 → 分享 → AirDrop → 选择 Mac
2. Mac: 自动保存到 `~/Downloads/`
3. 适合: 单次 <20 个文件

#### 方式 2: iCloud Drive (适合批量)
1. iPhone/iPad: 保存到 `Files → iCloud Drive → 音乐采集/`
2. Mac: 自动同步到 `~/Library/Mobile Documents/com~apple~CloudDocs/音乐采集/`
3. 适合: 批量下载，后台同步

#### 方式 3: 有线传输 (最稳定，大文件)
1. USB 线连接 iPhone/iPad → Mac
2. Finder → 设备 → 文件 → 拖出音频
3. 适合: >100MB 大文件，避免网络不稳定

### 2.3 Mac 端入库

```bash
# 移动到采集目录
mv ~/Downloads/*.mp3 ~/music_corpus_project/data/incoming/

# 运行入库脚本
cd ~/music_corpus_project
python scripts/00_collect/import_audio.py \
  --source-dir data/incoming/ \
  --source "youtube_ios" \
  --max-duration 1200
```

## 三、API 调用流程 (Gemini/GPT-4o)

### 3.1 为什么在 iOS 端调用
- Mac 无法科学上网，Gemini/GPT API 不可达
- iPhone/iPad 可以科学上网，直接调用 API
- Qwen-Omni/DeepSeek 国内直连，仍在 Mac 端调用

### 3.2 Pythonista 3 调用 API

```python
# Pythonista 3 脚本: call_gemini.py
import requests
import json
import clipboard

# 配置
GEMINI_KEY = "your_api_key"
AUDIO_PATH = "path/to/audio.mp3"  # 从 iCloud Drive 读取

# 上传音频到 Gemini
# (具体API调用参考 Gemini 文档)

# 结果保存到剪贴板 + iCloud
result = json.dumps(response, ensure_ascii=False, indent=2)
clipboard.set(result)

# 保存到 iCloud Drive
with open("/private/var/mobile/Library/Mobile Documents/com~apple~CloudDocs/l3_results/xxx.json", "w") as f:
    f.write(result)
```

### 3.3 Shortcuts 调用 API

1. 创建快捷指令："调用 Gemini 分析音频"
2. 动作: 获取文件 → 编码为 Base64 → 获取 URL 内容(POST) → 解析 JSON → 保存到文件
3. 可在分享菜单直接选择音频触发

### 3.4 结果回传 Mac

1. iOS 端结果保存为 JSON 到 iCloud Drive
2. Mac 端自动同步，读取后放入 `data/02_preannotation/l3_structural/`
3. 或 AirDrop 单个 JSON 文件

## 四、iOS 端能做 vs 不能做

| 任务 | iOS 端 | Mac 端 | 说明 |
|------|--------|--------|------|
| 下载外网音频 | ✅ | ❌ | Documents/a-Shell |
| ffmpeg 转码 | ✅ | ✅ | a-Shell 有 ffmpeg |
| yt-dlp | ✅ | ❌ | a-Shell pip install |
| 调用 Gemini/GPT | ✅ | ❌ | Pythonista/Shortcuts |
| 调用 Qwen/DeepSeek | ✅ | ✅ | 国内直连 |
| MERT 嵌入提取 | ❌ | ✅(GPU) | 重计算，需 GPU |
| Demucs 人声分离 | ❌ | ✅(GPU) | 重计算 |
| DBSCAN 聚类 | ⚠️ | ✅ | Pythonista 可跑小数据 |
| Label Studio | ❌ | ✅ | 需本地部署 |

### 原则
- **轻量任务**(下载/转码/API调用) → iOS 端
- **重计算**(MERT/Demucs/训练) → Mac/GPU 端
- **结果文件**通过 iCloud/AirDrop 同步

## 五、推荐工作流 (500首采集)

```
第1步: iPhone 批量下载 (每天50首)
  ├── Documents by Readdle 下载 YouTube/SoundCloud
  ├── 保存到 iCloud Drive/音乐采集/
  └── Mac 自动同步

第2步: Mac 入库 + 流水线
  ├── import_audio.py 批量入库
  ├── Stage 0-6 完整流水线 (GPU)
  └── L1-L4 预标注 (Mac)

第3步: iOS 高精度 L3 (可选，仅疑难样本)
  ├── 从 Mac 同步疑难音频到 iOS
  ├── Pythonista 调用 Gemini 2.0 Flash
  ├── 结果 JSON 回传 Mac
  └── 覆盖 Qwen-Omni 结果

第4步: Mac 人工校验
  ├── Label Studio 导入预标注
  ├── 人工校验 + 修正
  └── 导出最终标注
```

## 六、注意事项

1. **iCloud 存储空间**: 大量音频会占用 iCloud，建议定期清理已入库的源文件
2. **AirDrop 限制**: 单次 AirDrop 大量文件可能失败，建议分批(<20个)
3. **Pythonista 内存**: 处理 >50MB FLAC 可能内存不足，先在 Mac 转 MP3 再同步
4. **API Key 安全**: 不要在 iOS 脚本中硬编码 API Key，使用配置文件或环境变量
5. **版权合规**: 下载的音频仅用于研究，数据集说明中标注来源和授权状态
