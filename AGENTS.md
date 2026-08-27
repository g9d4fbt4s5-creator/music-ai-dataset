# AGENTS.md - 项目AI协作指南

> 本文件规定在本项目中写任何代码/脚本前必须执行的检查清单，以及项目环境配置的权威来源。
> 目的：避免因忘记环境配置、依赖版本、路径约定而导致的低级错误。

## 一、写任何代码前，必须先读这些文件

| 优先级 | 文件 | 内容 | 为什么必须读 |
|--------|------|------|-------------|
| P0 | `.env` | API keys (DEEPSEEK_API_KEY, DASHSCOPE_API_KEY) | 确认key存在，不要硬编码key |
| P0 | `environment.yml` | conda环境名、python版本、依赖版本 | 确认应该用哪个环境、哪些包已安装 |
| P0 | `docs/gpu_access.md` | GPU连接方式、路径、远程启动模板 | 确认SSH命令、conda环境名、模型路径 |
| P1 | `README.md` | 项目整体架构、Stage流程 | 确认脚本在流水线中的位置 |
| P1 | 目标脚本的 `--help` | 实际参数名和默认值 | 不要凭记忆写参数，用--help核对 |

## 二、本地环境配置（权威来源）

### Python环境
- **conda环境名**: `audio`（不是`music-corpus-local`，environment.yml中的名字是历史遗留）
- **python版本**: 3.11.13
- **关键依赖**: librosa 0.11.0, pandas, numpy, scikit-learn, python-dotenv, requests
- **激活方式**: `conda activate audio` 或 在脚本中 `export PATH="/opt/miniconda3/envs/audio/bin:$PATH"`

### 其他可用环境
- `labelstudio-env`: Label Studio标注工具
- `base`: conda基础环境（不推荐用于本项目）

### GPU环境（AutoDL）
- **连接**: `ssh -p 49530 root@connect.westb.seetacloud.com`（免密，id_rsa + id_ed25519）
- **项目目录**: `/root/autodl-tmp/music-ai-dataset/`
- **conda环境**: `labelstudio-env`（GPU上python3不在默认PATH，必须conda activate或用绝对路径）
- **模型路径**: `/root/autodl-tmp/models/clap_fusion/`, `/root/autodl-tmp/models/MERT-v1-95M/`
- **远程启动模板**: `setsid nohup python3 ... > log 2>&1 < /dev/null &`

## 三、写master脚本（如run_end_to_end.sh）前，额外检查

1. **每个Stage脚本是否真实跑过一次**（不是dry-run）
2. **参数是否与实际脚本一致**（用 `python3 scripts/xxx.py --help` 核对，不要凭记忆）
3. **远程执行命令是否在GPU上手动验证过**（conda路径、模型路径）
4. **脚本开头是否有环境自检**（python版本、关键依赖、.env存在性）
5. **每步结束后是否有输出验证**（文件数、字段完整性）

## 四、数据血缘核心文件（绝不能加.gitignore）

| 文件 | 说明 |
|------|------|
| `data/00_raw_collect/audio_manifest.csv` | 数据血缘骨架，记录audio_id/checksum/artist_id/sample_type |
| `docs/gpu_access.md` | 运维文档，不含密码，必须被Git跟踪 |
| `.env.example` | 环境变量模板，必须被Git跟踪 |

**可以加.gitignore的**: 原始音频、母版、L1/L2嵌入、切片、.env（含真实key）

## 五、GPU代码同步规范

| 规则 | 说明 |
|------|------|
| Mac是代码中心仓 | 改码、commit、push都在Mac完成 |
| GPU只pull代码 | GPU上执行 `git pull origin main`，不要在GPU上直接改代码 |
| GPU数据不进Git | GPU上`data/`目录是运行时产物，通过rsync同步，不做Git操作 |
| **禁止`git reset --hard`** | 会丢失GPU本地所有代码修改 |
| **禁止`git clean -fd`** | 会删除GPU上所有未被Git跟踪的文件，包括rsync过来的数据和L2派生产物 |

## 六、常见错误与预防

| 错误 | 根因 | 预防 |
|------|------|------|
| `ModuleNotFoundError: No module named 'librosa'` | 用了系统python或沙箱python，不是conda audio环境 | 脚本开头`export PATH="/opt/miniconda3/envs/audio/bin:$PATH"` + 环境自检 |
| L4黄金集加载为0 | L3输出格式`*_l3_qwen.json`+`annotation`字段，L4找`*_structure.json`+顶层字段 | 写代码前先`ls`实际输出目录，确认文件名和JSON结构 |
| L4黄金集索引为0 | MERT文件名`{hash}_{audioid}_mert_embedding.npy`，提取时没去掉前缀hash | 写文件名解析逻辑前先看实际文件名格式 |
| GPU上`python3: command not found` | GPU上python3不在默认PATH | 远程命令前加`source /root/miniconda3/etc/profile.d/conda.sh && conda activate labelstudio-env` |
| API key 401 | 脚本没读.env，或key已过期 | 脚本开头`load_dotenv()` + API key预检 |
| artist数据泄漏 | 划分脚本只报warning不自动处理 | Step 6后加artist隔离后处理 |

## 七、本文件的维护

- 环境配置变更时（如换GPU实例、换conda环境、换API endpoint），立即更新本文件和`docs/gpu_access.md`
- 发现新的常见错误时，补充到第六节
- 本文件本身必须被Git跟踪
