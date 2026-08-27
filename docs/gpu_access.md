# GPU 实例访问信息

> 本文档固化 AutoDL GPU 实例的连接信息，每次新会话先读本文件，不再询问用户。
> ⚠️ 本文档**不含密码**。SSH 免密登录已配置（id_rsa + id_ed25519），无需密码。
> 如免密失效，密码请从密码管理器获取，不要写入本文档。

## 连接信息

| 项目 | 值 |
|------|-----|
| Host | connect.westb.seetacloud.com |
| Port | 49530 |
| User | root |
| 认证方式 | SSH 公钥免密（id_rsa + id_ed25519） |
| 密码 | 不记录于此，见密码管理器 |

## SSH 免密配置

当前已配置两把公钥，双通道免密：

| 密钥类型 | 指纹 (SHA256) | 备注 |
|----------|---------------|------|
| id_rsa (RSA 4096) | rdjsjq8QEJs3xrumMwtw67mTNIc2n+SquLLhNHdoxiM | Mac 本地 `~/.ssh/id_rsa` |
| id_ed25519 | PGaMq...（全账号控制台已添加） | Mac 本地 `~/.ssh/id_ed25519`，备注 "ed25519 backup" |

控制台级别的公钥（全账号生效，新开实例自动继承）：
- AutoDL 控制台 → 设置 → SSH免密登录 → 已添加 ed25519 backup

## 实例路径

| 项目 | 路径 |
|------|------|
| 项目根目录 | /root/autodl-tmp/music-ai-dataset/ |
| 数据盘 | /root/autodl-tmp/（关机不丢，释放才丢） |
| 系统盘 | /root/（保存镜像才进镜像） |
| CLAP 模型 | /root/autodl-tmp/models/clap_fusion/630k-audioset-fusion-best.pt |
| MERT 模型 | /root/autodl-tmp/models/MERT-v1-95M/ |
| Conda 环境 | labelstudio-env |

## 常用命令

### 连接实例
```bash
# 免密登录（推荐）
ssh -p 49530 root@connect.westb.seetacloud.com
```

### 同步数据到 GPU
```bash
# 同步 manifest + QC报告 + 母版
rsync -avz -e "ssh -p 49530" \
  data/00_raw_collect/audio_manifest.csv \
  data/00.5_cleaned/reports/qc_gate_report.csv \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/music-ai-dataset/data/00_raw_collect/

rsync -avz -e "ssh -p 49530" \
  data/01_preprocess/processed_master/ \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/music-ai-dataset/data/01_preprocess/processed_master/
```

### 同步结果从 GPU 回本地
```bash
rsync -avz -e "ssh -p 49530" \
  root@connect.westb.seetacloud.com:/root/autodl-tmp/music-ai-dataset/data/02_preannotation/l2_embedding/ \
  data/02_preannotation/l2_embedding/
```

### 远程可靠启动任务（setsid + nohup + /dev/null）
```bash
ssh -p 49530 root@connect.westb.seetacloud.com << 'EOF'
cd /root/autodl-tmp/music-ai-dataset
source /root/miniconda3/etc/profile.d/conda.sh
conda activate labelstudio-env
setsid nohup python3 scripts/xxx.py \
  --arg value \
  > logs/task_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
echo "PID: $!"
EOF
```

**关键：`setsid` + `< /dev/null` 让进程完全脱离 SSH 会话，断连不丢失。**

## 代码同步规范

| 规则 | 说明 |
|------|------|
| Mac 是代码中心仓 | 改码、commit、push 都在 Mac 完成 |
| GPU 只 pull 代码 | GPU 上执行 `git pull origin main`，不要在 GPU 上直接改代码 |
| GPU 数据不进 Git | GPU 上 `data/` 目录是运行时产物，通过 rsync 同步，不做 Git 操作 |
| 禁止 `git reset --hard` | 除非明确确认要放弃 GPU 上的所有本地修改和数据产物 |

## 实例状态检查

```bash
# 检查实例是否在线 + GPU状态
ssh -p 49530 root@connect.westb.seetacloud.com "hostname; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
```

## 注意事项

1. **数据盘 vs 系统盘**：代码和数据放 `/root/autodl-tmp/`（数据盘，关机不丢）；不要放 `/root/`（系统盘，释放实例就没了）
2. **实例关机**：AutoDL 按小时计费，不用时关机省钱；关机后数据盘保留，开机继续用
3. **端口变化**：每次开机端口可能变，以 AutoDL 控制台显示的为准
4. **manifest 是数据血缘核心**：`data/00_raw_collect/audio_manifest.csv` 必须被 Git 跟踪，绝不能加 `.gitignore`
