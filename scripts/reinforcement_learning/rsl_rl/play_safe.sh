#!/usr/bin/env bash
# Pre-flight check + launch play_moe.py safely (avoid OOM).
#
# Usage:
#   ./scripts/reinforcement_learning/rsl_rl/play_safe.sh [extra args to play_moe.py]
#
# 默认: --task=MoE-Scan-Teacher-Deeprobotics-M20-v0 --num_envs 1 --joystick
# 你可以传额外参数:
#   ./play_safe.sh --task=MoE-Gap-Teacher-Deeprobotics-M20-v0 --keyboard
#   ./play_safe.sh --vis-scan-obs

set -e

cd "$(dirname "$0")/../../.."   # repo root

# ---- 1. Pre-flight check ----
echo "=== Pre-flight ==="

# 内存 (LC_ALL=C 强制英文输出, 避免中文 locale 全角冒号导致 awk 失配)
mem_avail_gb=$(LC_ALL=C free -g | awk '/^Mem:/ {print $7}')
mem_total_gb=$(LC_ALL=C free -g | awk '/^Mem:/ {print $2}')
echo "  Memory: ${mem_avail_gb}G free / ${mem_total_gb}G total"
if [ "${mem_avail_gb:-0}" -lt 20 ]; then
    echo "  ⚠️  Free memory < 20GB. Consider closing other apps."
fi

# GPU
if command -v nvidia-smi &>/dev/null; then
    gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "  GPU0: ${gpu_used}MiB used / ${gpu_total}MiB total"
fi

# 残留进程 (包括 Omniverse Hub - 是 OOM 主因)
zombie_pids=$(pgrep -f 'isaac-sim|kit.*python|isaaclab|/pkg/hub-' 2>/dev/null | grep -v $$ || true)
if [ -n "$zombie_pids" ]; then
    echo "  ⚠️  Found existing Isaac/Kit/Hub processes:"
    ps -p $zombie_pids -o pid,user,vsz,rss,cmd 2>/dev/null | sed 's/^/    /'
    echo "  自动 kill ..."
    kill -9 $zombie_pids 2>/dev/null
    sleep 1
fi
# 清 /tmp 残留 hub 锁文件
rm -f /tmp/hub-ouge-*.config.json /tmp/hub-ouge-*.lock 2>/dev/null

# 磁盘
disk_avail=$(df -BG ~ | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "  Disk: ${disk_avail}G free in \$HOME"
if [ "${disk_avail:-0}" -lt 10 ]; then
    echo "  ⚠️  Disk < 10GB free. Cache writes may fail."
fi

# 上次 OOM 检查
oom_recent=$(sudo -n dmesg --since '10 min ago' 2>/dev/null | grep -c 'Out of memory: Killed' 2>/dev/null || echo 0)
if [ "$oom_recent" -gt 0 ]; then
    echo "  ⚠️  $oom_recent recent OOM kill(s) in last 10 min"
fi

echo

# ---- 2. Launch ----
DEFAULT_ARGS=("--task=MoE-Scan-Teacher-Deeprobotics-M20-v0" "--num_envs" "1" "--joystick")

# 如果用户传了 --task 或 --keyboard 等, 用用户参数, 否则默认
if [ $# -eq 0 ]; then
    ARGS=("${DEFAULT_ARGS[@]}")
else
    # 用户传了参数 — 但保证 --num_envs 默认 1
    ARGS=("$@")
    # 如果用户没传 --num_envs, 加上 1
    if ! echo "$@" | grep -q -- '--num_envs'; then
        ARGS=("--num_envs" "1" "${ARGS[@]}")
    fi
fi

echo "=== Launch ==="
echo "  python scripts/reinforcement_learning/rsl_rl/play_moe.py ${ARGS[*]}"
echo

exec python scripts/reinforcement_learning/rsl_rl/play_moe.py "${ARGS[@]}"
