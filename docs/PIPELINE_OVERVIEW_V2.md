# 音乐语料库端到端流水线 — 整合方案 V2

> 本文档整合了从采集入库到训练准备的完整流程，修正了切片时机矛盾，明确了各阶段职责与数据流向。
> 版本：V2 | 日期：2026-08-26 | 状态：已固化

---

## 一、修正后的完整流水线架构

```
Stage 0: 采集入库（原始MP3/M4A，不转FLAC）
    ↓
Stage 0.5: QC清洗（排除fail）
    ↓
Stage 1: 母版生成（只做母版FLAC 48kHz/24bit，不切片）← 修正：排除fail
    ↓
Stage 2: 预标注L1-L4（在整首母版上做）
    ├── L1 物理标签（本地CPU）
    ├── L2 MERT/CLAP嵌入（GPU，整首母版）
    ├── L3 结构标注（Qwen-Omni，仅黄金集，API）
    └── L4 KNN传播（本地CPU）
    ↓
Stage 3: 人工审核 + 黄金集精标（Label Studio HITL）
    ↓
Stage 4: 数据划分（train/val/test/holdout + 跨集去重 + artist隔离）
    ↓
Stage 5: 切片 + 特征提取（只对train/val，按结构边界切）← 切片移到这里
    ↓
Stage 6: 模型训练
```

**核心修正**：切片从 Stage 1 移到 Stage 5（数据划分之后），只对最终进入训练集/验证集的样本切片。

---

## 二、各阶段职责与产出

| 阶段 | 输入 | 输出 | 设备 |
|------|------|------|------|
| Stage 0 采集入库 | 原始音频文件 | raw_audio/ + manifest | Mac |
| Stage 0.5 QC | manifest | pass/marginal/fail 报告 | Mac |
| Stage 1 母版 | 原始音频 | 统一 FLAC 48kHz/24bit 母版 | Mac |
| Stage 2 预标注 | 整首母版 | L1-L4 标签 | Mac + GPU + API |
| Stage 3 人工审核 | 预标注结果 | 精标标签 + badcase | Mac（Label Studio） |
| Stage 4 数据划分 | manifest | train/val/test/holdout CSV | Mac |
| Stage 5 切片+特征 | train/val 母版 | 训练片段 + 特征 | Mac |
| Stage 6 训练 | 切片+标签 | 模型 | GPU |

---

## 三、Bilibili 入库自动化流水线

### 流程

```
1. 获取 Bilibili 链接
    ↓
2. 自动提取元数据（Bilibili API + LLM）
    ├── 视频标题 → artist 候选
    ├── 简介/评论 → 曲目列表 + 时间戳
    └── 输出 tracklist JSON
    ↓
3. 下载音频（yt-dlp，保持原始MP3/M4A格式）
    ↓
4. 入库长音频（生成 ULID，保存到 raw_audio/）
    ├── audio_id = ULID
    ├── artist_id = 从标题/简介提取
    ├── source_type = bilibili
    └── metadata_json 保存曲目列表
    ↓
5. 按时间戳切片（派生数据，非 raw）
    ├── 每首切出独立片段
    ├── 每个片段生成新 ULID
    ├── song_group_id = 长音频的 audio_id
    ├── parent_audio_id = 长音频ID
    └── 保存到 processed_audio/
    ↓
6. 写 manifest
```

### 元数据提取自动化

| 信息 | 来源 | 方法 |
|------|------|------|
| 时间戳/章节 | Bilibili API 简介/评论 | 正则 + LLM 兜底 |
| 艺术家 | 视频标题/简介 | 正则 + LLM |
| 曲目标题 | 简介/评论 | 正则 + LLM |

### 时间戳正则模式

```python
TIMESTAMP_PATTERNS = [
    r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—:]\s*(.+)',  # 00:00 - 歌曲名
    r'(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)',              # 00:00 歌曲名
    r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)',          # [00:00] 歌曲名
    r'(\d{1,2}:\d{2}(?::\d{2})?)\s*[｜|]\s*(.+)',     # 00:00｜歌曲名
]
```

### Bilibili API 获取简介和评论

```python
def get_bilibili_desc(bvid: str) -> str:
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    return resp.json()["data"]["desc"]

def get_bilibili_comments(bvid: str, limit: int = 20) -> List[str]:
    url = f"https://api.bilibili.com/x/v2/reply?type=1&oid={bvid}&sort=2&ps={limit}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    replies = resp.json().get("data", {}).get("replies", [])
    return [r["content"]["message"] for r in replies]
```

---

## 四、数据格式与存储策略

### 各阶段格式

| 数据 | 格式 | 存储位置 |
|------|------|---------|
| 原始采集（Bilibili下载） | 原始格式（MP3/M4A/FLAC） | raw_audio/ |
| 切片前长音频 | 原始格式，标记 is_master_recording=true | raw_audio/ |
| 切片后单曲 | 派生数据，标记 parent_audio_id | processed_audio/ |
| 母版 | FLAC 48kHz/24bit | processed_master/ |
| 训练切片 | FLAC 16-bit/44.1kHz | segments/ |
| GPU 同步 | 320kbps MP3 或 16-bit FLAC | 临时 |

### 切片时机修正

- **切片前的长音频 = raw，永不删除**
- **切片后的单曲 = 派生数据，可由长音频重新生成**

---

## 五、当前已切好的 3038 片处理

| 方案 | 建议 |
|------|------|
| 保留 | 标记为 pre_split_chunks（历史产物），不用于正式训练 |
| 删除 | 重新切片很快（几分钟），删除后可节省 5-8 GB 空间 |

**建议**：保留一段时间，确认 Stage 5 切片逻辑正确后删除。

---

## 六、文件大小与同步策略

### 当前数据量（85首试点）

| 数据 | 大小 |
|------|------|
| 原始采集 | 约 500 MB - 1 GB |
| 母版 FLAC 24-bit | 约 3-4 GB |
| 切片 FLAC | 约 5-8 GB |
| 特征 npy | 约 3-5 MB |

### GPU 同步优化

只传母版 FLAC（或转成 16-bit/MP3），不传切片和特征。

```
✅ 母版 FLAC（85 首，约 3-4 GB）→ 100 Mbps 约 5-8 分钟
❌ 切片（3038 片，5-8 GB）
❌ 特征文件（npy）
```

如果带宽有限：将母版临时转成 320kbps MP3（约 500 MB）再同步，MP3 对 MERT/CLAP 嵌入提取影响很小。

---

## 七、需要修复的代码问题

### 问题 1：Stage 1 排除 fail 样本

在 `01_generate_master.py` 中增加过滤：

```python
def load_manifest_with_qc_filter(manifest_path, qc_report_path=None):
    """加载 manifest，如果有 QC 报告则排除 fail 样本"""
    import pandas as pd
    
    df = pd.read_csv(manifest_path)
    
    if qc_report_path and Path(qc_report_path).exists():
        qc = pd.read_csv(qc_report_path)
        if "final_branch" in qc.columns:
            fail_ids = set(qc[qc["final_branch"] == "fail"]["audio_id"])
            before = len(df)
            df = df[~df["audio_id"].isin(fail_ids)]
            after = len(df)
            print(f"排除 fail 样本: {before} → {after} (移除 {before-after} 首)")
    
    return df
```

### 问题 2：切片/特征提取移到 Stage 5

```bash
# 新建目录
mkdir -p scripts/05_training_prep/

# 移动脚本
mv scripts/01_preprocess/03_audio_chunker.py scripts/05_training_prep/01_audio_chunker.py
mv scripts/01_preprocess/04_extract_features.py scripts/05_training_prep/02_extract_features.py
```

### 问题 3：切片脚本增加"只切 train/val"参数

```python
def load_train_val_ids(split_dir: Path) -> set:
    """从数据划分结果加载 train/val 的 audio_id"""
    ids = set()
    for split in ["train.csv", "val.csv"]:
        f = split_dir / split
        if f.exists():
            import pandas as pd
            df = pd.read_csv(f)
            if "audio_id" in df.columns:
                ids.update(df["audio_id"])
    return ids
```

---

## 八、内存安全优化

- 特征提取时用生成器（generator）逐首处理，不一次性加载
- 嵌入提取时分批处理（batch_size=8/16）
- 处理完一批释放内存再处理下一批

---

## 九、更新后的文档清单

| 文档 | 更新内容 |
|------|---------|
| ARCHITECTURE.md | 切片从 Stage 1 移到 Stage 5 |
| ADR-003 | 明确切片时机在数据划分后 |
| 脚本目录 | 01_preprocess 只保留母版；新增 05_training_prep |
| 新增 docs/BILIBILI_PIPELINE.md | Bilibili 采集自动化流程 |
| 本文档 PIPELINE_OVERVIEW_V2.md | 完整流程整合 |

---

## 十、下一步行动清单

| 优先级 | 事项 | 预计时间 |
|--------|------|---------|
| P0 | 修复 Stage 1 排除 fail 样本 | 10 分钟 |
| P0 | 移动切片/特征脚本到 Stage 5 | 5 分钟 |
| P0 | 切片脚本增加"只切 train/val"参数 | 15 分钟 |
| P1 | Bilibili 长音频不转 FLAC，保持原始格式入库 | 5 分钟 |
| P1 | 写 bilibili_metadata_extractor.py | 30 分钟 |
| P1 | 更新 ARCHITECTURE.md | 15 分钟 |
| P2 | GPU 同步只传母版（确认同步命令） | 5 分钟 |
| P2 | 3038 片历史切片归档/删除 | 5 分钟 |

---

## 十一、历史问题复盘

### 已解决的问题

| 问题 | 根因 | 修复 |
|------|------|------|
| 母版生成路径不一致 | md5(audio_id) vs sha256 命名 | 优先使用 manifest 的 file_relative_path |
| artist_id 全是 unknown | 入库脚本只生成占位符 | fill_artist_from_filename.py 解析文件名 |
| 特征提取输出路径错误 | --output 传了目录而非CSV | 改为 --output data/stage6_features/audio_features.csv |
| GPU 实例环境为空 | 每次新实例都要重装依赖 | 保存 AutoDL 自定义镜像 |
| 切片时机矛盾 | 切片放在 Stage 1，应在 Stage 5 | 本文档固化修正 |

### 当前试点数据状态（85首）

| 指标 | 数值 |
|------|------|
| manifest 总曲目 | 85 首 |
| QC 结果 | 84 pass + 1 fail |
| artist_id 真实数 | 46 首（54%） |
| 母版生成 | 85 首全部成功 |
| 历史切片 | 3038 片（待归档/删除） |

---

*本文档为 V2 版本，替代之前所有零散的流程描述。后续所有代码修改以此文档为基准。*
