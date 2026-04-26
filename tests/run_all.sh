#!/bin/bash
# Run all moe_terrain tests in sequence and report.
# Usage: bash tests/run_all.sh

set -u
cd "$(dirname "$0")/.."

PY="${PYTHON:-/home/ouge/miniconda3/envs/env_isaaclab/bin/python}"
TESTS_DIR="tests"

TESTS=(
    "test_mirror_fb.py"
    "test_mirror_fb_integration.py"
    "test_moe_terrain_components.py"
    "test_actor_critic_forward.py"
    "test_sym_loss_training.py"
)

PASS=0
FAIL=0
declare -a FAILED_NAMES=()

for t in "${TESTS[@]}"; do
    echo
    echo "================================================================"
    echo "RUNNING: $t"
    echo "================================================================"
    if $PY "$TESTS_DIR/$t" > /tmp/_test_out.log 2>&1; then
        # Print just the summary block and per-test markers
        grep -E "^  \[[✓✗]\]|passed|failed" /tmp/_test_out.log | tail -25
        echo "  → PASSED"
        PASS=$((PASS+1))
    else
        cat /tmp/_test_out.log | tail -30
        echo "  → FAILED"
        FAIL=$((FAIL+1))
        FAILED_NAMES+=("$t")
    fi
done

echo
echo "================================================================"
echo "OVERALL: ${PASS}/${#TESTS[@]} test files passed"
if [ $FAIL -gt 0 ]; then
    echo "Failed: ${FAILED_NAMES[*]}"
    exit 1
fi
exit 0
