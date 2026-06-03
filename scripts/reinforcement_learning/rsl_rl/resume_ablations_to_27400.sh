#!/usr/bin/env bash
# Resume every ablation variant + the MLP baseline from their iter-15000/20000
# checkpoints to iteration 27400, so the whole comparison is iso-iteration with
# the deployed full SplitMoE (model_27400). Runs SEQUENTIALLY (each job uses all
# 8 GPUs x 3000 envs). NO code changes — pure CLI flags on the existing
# eval_train.py. learn(N) is incremental, so --train_iters = 27400 - start + 1.
#
# Usage on 102:  cd ~/rl_training && bash scripts/reinforcement_learning/rsl_rl/resume_ablations_to_27400.sh
set -u
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root (~/rl_training)

PY=/home/ubuntu/miniconda3/envs/env_isaaclab/bin/torchrun
SCRIPT=scripts/reinforcement_learning/rsl_rl/eval_train.py
BASE=logs/moe_training
LOGDIR=logs/resume27400_logs
mkdir -p "$LOGDIR"
MOE_TASK=Rough-MoE-Teacher-Deeprobotics-M20-v0
MLP_TASK=Rough-MlpBaseline-Teacher-Deeprobotics-M20-v0

# variant | task | ablation flag | load_run (dated dir) | start_ckpt iter | train_iters(->27400)
# front-loaded: meaningful comparisons first, the always-collapsed A3/B2 last.
JOBS=(
  "A1|$MOE_TASK|A1|split_moe_teacher_parallel_abl_A1/2026-05-27_11-42-44|14999|12402"
  "A2|$MOE_TASK|A2|split_moe_teacher_parallel_abl_A2/2026-05-26_09-45-07|14999|12402"
  "B1|$MOE_TASK|B1|split_moe_teacher_parallel_abl_B1/2026-05-23_15-38-08|14999|12402"
  "MLP|$MLP_TASK|full|mlp_baseline_teacher_parallel/2026-05-29_06-37-18|19999|7402"
  "A3|$MOE_TASK|A3|split_moe_teacher_parallel_abl_A3/2026-05-22_14-30-22|14999|12402"
  "B2|$MOE_TASK|B2|split_moe_teacher_parallel_abl_B2/2026-05-25_07-37-15|14999|12402"
)

run_one() {
  local tag="$1" task="$2" abl="$3" subdir="$4" start="$5" iters="$6"
  local run_dir="$BASE/$subdir"
  local load_run; load_run="$(basename "$subdir")"
  local ckpt="$run_dir/model_${start}.pt"
  local log="$LOGDIR/${tag}_resume_$(basename "$subdir").log"
  echo "============================================================"
  echo "[$(date '+%F %T')] RESUME $tag  ablation=$abl  task=$task"
  echo "  from: $ckpt  (iter $start)  -> +$iters  => iter $((start+iters-1))"
  echo "  log:  $log"
  echo "============================================================"
  if [ ! -f "$ckpt" ]; then echo "[ERR] missing checkpoint $ckpt — SKIP $tag"; return 1; fi
  PER_RANK_TERRAIN=1 PER_RANK_NO_ILLEGAL_CONTACT=1 \
  "$PY" --standalone --nproc_per_node=8 "$SCRIPT" \
      --task "$task" --ablation "$abl" \
      --num_envs 3000 --distributed --max_init_terrain_level 1 \
      --logger wandb \
      --resume --load_run "$load_run" --checkpoint "$ckpt" \
      --train_iters "$iters" \
      2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] $tag finished rc=$rc"
  return $rc
}

ONLY="${1:-}"   # optional: run a single tag (e.g. A1) then stop, for pipeline verification
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag task abl subdir start iters <<< "$job"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ]; then continue; fi
  run_one "$tag" "$task" "$abl" "$subdir" "$start" "$iters"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "[STOP] $tag returned rc=$rc — halting queue."; exit "$rc"; fi
done
echo "[$(date '+%F %T')] ALL DONE — every variant at iter 27400."
