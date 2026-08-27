#!/usr/bin/env bash
# idle_watchdog.sh — AutoDL 闲置监测（纯告警版，v2）
#
# ⚠️ 2026-08-28 变更：移除自动关机能力。
#   原因：AutoDL 容器内 /usr/bin/shutdown 实际是 `kill supervisord`，
#   既关不掉容器（平台拦截，三次实证），又可能干扰运行中的任务（有害动作）。
#   闲置处置改由人工/控制台完成：AutoDL 控制台「定时关机」或手动关机。
#
# 现在的用途：仅记录闲置状态到日志，供事后核查"实例哪段时间在空转计费"。
#   判定闲置 30 分钟后写一条 WARN 日志，之后不再重复告警（每 30 分钟一条）。
#
# "使用中"判定（任一命中即重置闲置计数）：
#   1. GPU 利用率 > 5%
#   2. 存在特征提取/训练相关的 python 任务进程
#      （关键词：feat_ / feature_extract / clap / mert / train / extract / smoke，
#       排除 jupyter/tensorboard/本看门狗自身）
#   3. 有活跃交互会话（who 输出非空，detached screen 不算）
#
# 部署（可选，SSH 进实例后）：
#   screen -dmS watchdog bash /root/autodl-tmp/idle_watchdog.sh
# 查看日志：
#   tail -f /root/autodl-tmp/idle_watchdog.log
#
# 说明：
#   - 脚本统一放在 /root/autodl-tmp/idle_watchdog.sh（数据盘，重启不丢）。
#   - 本版本无任何关机/杀进程副作用，可安全常驻。
#   - 不要再配置任何开机自启去拉起旧版脚本（旧版含 shutdown，已废弃）。

set -u

INTERVAL=300          # 检测间隔（秒）= 5 分钟
IDLE_THRESHOLD=6      # 连续闲置次数阈值 → 6 × 5 = 30 分钟
GPU_IDLE_PCT=5        # GPU 利用率低于此值视为闲置
LOG=/root/autodl-tmp/idle_watchdog.log

count=0
warned=0
while true; do
  sleep "$INTERVAL"
  ts=$(date '+%F %T')

  # --- 1. GPU 利用率（无卡模式 nvidia-smi 无输出，视为 0） ---
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
  gpu_util=${gpu_util:-0}

  # --- 2. 任务进程：按关键词匹配所有特征提取/训练相关的 python 进程 ---
  #     覆盖任意 conda 环境（labelstudio-env / mert-env / clap-env / 裸 python 等）
  #     排除 jupyter / tensorboard / 本看门狗自身
  task_n=$(ps aux | grep -E '[p]ython.*(feat_|feature_extract|clap|mert|train|extract|smoke)' \
           | grep -vE 'jupyter|tensorboard|idle_watchdog' | wc -l | tr -d '[:space:]')

  # --- 3. 活跃交互会话（who 列出 pts 登录；detached screen 不计入） ---
  sess_n=$(who 2>/dev/null | wc -l | tr -d '[:space:]')

  if [ "$gpu_util" -gt "$GPU_IDLE_PCT" ] || [ "$task_n" -gt 0 ] || [ "$sess_n" -gt 0 ]; then
    echo "[$ts] 使用中 GPU=${gpu_util}% task=${task_n} ssh=${sess_n} -> 重置计数" >> "$LOG"
    count=0
    warned=0
  else
    count=$((count + 1))
    if [ "$count" -ge "$IDLE_THRESHOLD" ]; then
      if [ "$warned" -eq 0 ]; then
        echo "[$ts] WARN: 实例闲置超 $((IDLE_THRESHOLD * INTERVAL / 60)) 分钟仍在计费（GPU=${gpu_util}% task=${task_n} ssh=${sess_n}）。请到 AutoDL 控制台关机。本脚本不再执行任何关机动作。" >> "$LOG"
        warned=1
      fi
      # 告警一次后保持 warned=1，直到重新检测到使用才复位
    else
      echo "[$ts] 闲置 ${count}/${IDLE_THRESHOLD} (GPU=${gpu_util}% task=${task_n} ssh=${sess_n})" >> "$LOG"
    fi
  fi
done
