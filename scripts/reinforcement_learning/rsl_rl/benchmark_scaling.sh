#!/usr/bin/env bash
# Run benchmark_scaling.py for 1/4/8 GPUs sequentially and compare results.
#
# Pre-reqs:
#   conda activate env_isaaclab
#   GPUs idle (no zombie procs)
#
# Tunables via env vars:
#   TASK         (default: MoE-Scan-Teacher-Deeprobotics-M20-v0)
#   NUM_ENVS     (default: 2000) — per-GPU env count
#   MAX_ITER     (default: 50)   — keep small to keep total bench < 30 min
#   GPUS         (default: "1 4 8")
#   OUTDIR       (default: /tmp/bench_$(date +%s))

set -e

TASK="${TASK:-MoE-Scan-Teacher-Deeprobotics-M20-v0}"
NUM_ENVS="${NUM_ENVS:-2000}"
MAX_ITER="${MAX_ITER:-50}"
GPUS="${GPUS:-1 4 8}"
OUTDIR="${OUTDIR:-/tmp/bench_$(date +%s)}"

mkdir -p "$OUTDIR"
echo "=== Benchmark scaling ==="
echo "  task=$TASK  num_envs/gpu=$NUM_ENVS  max_iter=$MAX_ITER"
echo "  outdir=$OUTDIR  gpus=[$GPUS]"
echo

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PY="$SCRIPT_DIR/benchmark_scaling.py"

for NGPU in $GPUS; do
    OUT="$OUTDIR/bench_${NGPU}gpu.json"
    LOG="$OUTDIR/bench_${NGPU}gpu.log"
    echo "--- $NGPU GPU(s) ---"
    if [ "$NGPU" -eq 1 ]; then
        python "$PY" \
            --task "$TASK" --num_envs "$NUM_ENVS" --max_iter "$MAX_ITER" \
            --output "$OUT" 2>&1 | tee "$LOG"
    else
        torchrun --nproc_per_node="$NGPU" --master_port=$((20000 + RANDOM % 10000)) "$PY" \
            --task "$TASK" --num_envs "$NUM_ENVS" --max_iter "$MAX_ITER" \
            --output "$OUT" 2>&1 | tee "$LOG"
    fi
    echo "  → $OUT"
    # let GPU memory clear between runs
    sleep 5
done

echo
echo "=== Comparison ==="
python "$SCRIPT_DIR/benchmark_compare.py" "$OUTDIR"/bench_*.json
echo
echo "Raw JSON: ls $OUTDIR/"
