#!/usr/bin/env bash
# v2.6 comparison: 8 variants × 4 levels = 32 cells, 4 cells/GPU (all at once).
# Pinned checkpoints. Output -> logs/moe_eval/v26/<variant>_<level>_s<seed>.
set -u
cd ~/rl_training
PY=/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python
SEED=${1:-42}
NENV=${2:-250}
LEVELS=(easy med hard extreme)
mkdir -p /tmp/v26_logs
rm -f /tmp/v26_done

declare -A EXTRA
EXTRA[full]="--ablation full --load_run 2026-05-18_02-52-49 --checkpoint model_27400.pt"
EXTRA[A1]="--ablation A1 --checkpoint model_14999.pt"
EXTRA[A2]="--ablation A2 --checkpoint model_14999.pt"
EXTRA[A3]="--ablation A3 --checkpoint model_14999.pt"
EXTRA[B1]="--ablation B1 --checkpoint model_14999.pt"
EXTRA[B2]="--ablation B2 --checkpoint model_14999.pt"
EXTRA[locomoe]="--ablation locomoe --checkpoint model_14999.pt"
EXTRA[mlp_baseline]="--ablation mlp_baseline --checkpoint model_19999.pt"

VARIANTS=(full A1 A2 A3 B1 B2 locomoe mlp_baseline)
start=$(date +%s)
i=0
for v in "${VARIANTS[@]}"; do
  for l in "${LEVELS[@]}"; do
    gpu=$((i % 8))
    label="${v}_${l}_s${SEED}"
    out="logs/moe_eval/v26/${label}"
    echo "[$(date +%H:%M:%S)] LAUNCH ${label} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u scripts/reinforcement_learning/rsl_rl/eval_course.py \
      --level $l ${EXTRA[$v]} --num_envs $NENV --num_steps 2500 --seed $SEED \
      --output_dir "$out" > "/tmp/v26_logs/${label}.log" 2>&1 &
    i=$((i+1))
    sleep 1.5
  done
done
wait
echo "[$(date +%H:%M:%S)] V26 DONE total=$(($(date +%s)-start))s cells=${i}"
touch /tmp/v26_done
