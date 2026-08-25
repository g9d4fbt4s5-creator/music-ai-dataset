#!/usr/bin/env bash
# idle_watchdog.sh — AutoDL 闲置看门狗
#
# 目的：连续 30 分钟（6 次 × 5 分钟）未检测到任何"使用"迹象，
#       自动执行 /usr/bin/shutdown 关机，避免忘记关机导致持续计费。
#
# "使用中"判定（任一命中即重置闲置计数）：
#   1. GPU 利用率 > 5%
#   2. 存在特征提取/训练相关的 python 任务进程
#      （按关键词匹配：feat_ / feature_extract / clap / mert / train / extract
#       排除 jupyter/tensorboard/本看门狗自身）
#   3. 有活跃交互会话（who 输出非空，detached screen 不算）
#
# 部署（SSH 进实例后）：
#   cp ~/music-ai-dataset/scripts/utils/idle_watchdog.sh /root/autodl-tmp/
#   screen -dmS watchdog bash /root/autodl-tmp/idle_watchdog.sh
# 查看日志：
#   tail -f /root/autodl-tmp/idle_watchdog.log
# 查看运行状态：
#   screen -ls   # 应有 watchdog 会话
#
# 说明：
#   - 脚本统一放在 /root/autodl-tmp/idle_watchdog.sh（数据盘，重启不丢）。
#   - 日志写数据盘 /root/autodl-tmp/idle_watchdog.log，关机不丢，开机后可查上次关机原因。
#   - AutoDL 公有云无原生"闲置检测关机"（仅私有云有），本脚本补足该能力。
#   - /usr/bin/shutdown 为 AutoDL 官方文档推荐的实例内关机命令，无需 API。
#   - 实例重启后 screen 会话丢失，需重新部署或配置开机自启（见文末）。

set -u

INTERVAL=300          # 检测间隔（秒）= 5 分钟
IDLE_THRESHOLD=6      # 连续闲置次数阈值 → 6 × 5 = 30 分钟
GPU_IDLE_PCT=5        # GPU 利用率低于此值视为闲置
LOG=/root/autodl-tmp/idle_watchdog.log

count=0
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
  else
    count=$((count + 1))
    echo "[$ts] 闲置 ${count}/${IDLE_THRESHOLD} (GPU=${gpu_util}% task=${task_n} ssh=${sess_n})" >> "$LOG"
    if [ "$count" -ge "$IDLE_THRESHOLD" ]; then
      echo "[$ts] !!! 闲置达 ${IDLE_THRESHOLD} 次（$((IDLE_THRESHOLD * INTERVAL / 60)) 分钟），执行 /usr/bin/shutdown" >> "$LOG"
      /usr/bin/shutdown
      exit 0
    fi
  fi
done

# ===== 开机自启（推荐）=====
# AutoDL 容器重启后 screen 会话丢失。以下两种方式可实现开机自动拉起看门狗，
# 优先用方式一（最可靠），方式二作为兜底。
#
# --- 方式一：crontab @reboot（推荐）---
#   crontab -e
#   添加一行：
#   @reboot screen -dmS watchdog bash /root/autodl-tmp/idle_watchdog.sh
#
# --- 方式二：AutoDL 控制台「自定义启动命令」---
#   创建/重启实例时，在「自定义启动命令」填：
#   bash /root/autodl-tmp/idle_watchdog.sh &
#
# 注意：.bashrc 仅交互登录触发，非交互开机不会执行，不适合用于开机自启。
