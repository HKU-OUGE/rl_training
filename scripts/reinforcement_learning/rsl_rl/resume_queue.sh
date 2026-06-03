#!/usr/bin/env bash
# Idempotent resume-to-27400 queue for the iso-iteration ablation comparison.
# Safe to re-run after a crash: for each variant it finds the LATEST checkpoint
# across all its run dirs and resumes from there to iter 27400; a variant that
# already has model_27400.pt is skipped. learn(N) is incremental, so
# train_iters = TARGET - latest + 1. Sequential (each job uses all 8 GPUs).
# This SUPERSEDES resume_ablations_to_27400.sh (which was non-idempotent).
#
# Usage on 102:  cd ~/rl_training && bash scripts/reinforcement_learning/rsl_rl/resume_queue.sh
# Single variant: bash .../resume_queue.sh A2
set -u
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root (~/rl_training)

PY=/home/ubuntu/miniconda3/envs/env_isaaclab/bin/torchrun
SCRIPT=scripts/reinforcement_learning/rsl_rl/eval_train.py
BASE=logs/moe_training
LOGDIR=logs/resume27400_logs
mkdir -p "$LOGDIR"
MOE_TASK=Rough-MoE-Teacher-Deeprobotics-M20-v0
MLP_TASK=Rough-MlpBaseline-Teacher-Deeprobotics-M20-v0
TARGET=27400

# tag | task | ablation flag | experiment_name (dir under logs/moe_training)
JOBS=(
  "A1|$MOE_TASK|A1|split_moe_teacher_parallel_abl_A1"
  "A2|$MOE_TASK|A2|split_moe_teacher_parallel_abl_A2"
  "B1|$MOE_TASK|B1|split_moe_teacher_parallel_abl_B1"
  "MLP|$MLP_TASK|full|mlp_baseline_teacher_parallel"
  "A3|$MOE_TASK|A3|split_moe_teacher_parallel_abl_A3"
  "B2|$MOE_TASK|B2|split_moe_teacher_parallel_abl_B2"
)

latest_ckpt_iter() {  # echo "<iter> <path>" of the highest model_N.pt under $BASE/$1/*/
  local exp="$BASE/$1" best=-1 bestp=""
  for f in "$exp"/*/model_*.pt; do
    [ -e "$f" ] || continue
    local n; n=$(basename "$f"); n=${n#model_}; n=${n%.pt}
    case "$n" in (*[!0-9]*) continue;; esac
    if [ "$n" -gt "$best" ]; then best="$n"; bestp="$f"; fi
  done
  echo "$best $bestp"
}

run_one() {
  local tag="$1" task="$2" abl="$3" exp="$4"
  read -r latest ckpt < <(latest_ckpt_iter "$exp")
  if [ "$latest" -lt 0 ]; then echo "[ERR] $tag: no checkpoint under $BASE/$exp — SKIP"; return 1; fi
  if [ "$latest" -ge "$TARGET" ]; then echo "[SKIP] $tag already at iter $latest (>=$TARGET) — done."; return 0; fi
  local load_run; load_run="$(basename "$(dirname "$ckpt")")"
  local iters=$(( TARGET - latest + 1 ))
  local log="$LOGDIR/${tag}_resume.log"
  echo "============================================================"
  echo "[$(date '+%F %T')] RESUME $tag  ablation=$abl  task=$task"
  echo "  from: $ckpt  (iter $latest)  -> +$iters  => iter $TARGET"
  echo "  log:  $log"
  echo "============================================================"
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

ONLY="${1:-}"
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag task abl exp <<< "$job"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ]; then continue; fi
  run_one "$tag" "$task" "$abl" "$exp"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "[STOP] $tag returned rc=$rc — halting queue (monitor will retry)."; exit "$rc"; fi
done
echo "[$(date '+%F %T')] ALL DONE — every variant at iter $TARGET."
