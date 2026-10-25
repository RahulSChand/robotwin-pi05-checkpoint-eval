#!/usr/bin/env bash
set -euo pipefail

. /workspace/robotwin_eval/env.sh

ROOT=/workspace/robotwin_eval
PYTHON="$LEROBOT_ROOT/.venv/bin/python"
SWEEP="$ROOT/run_checkpoint_sweep.py"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# Interleave checkpoints so each GPU gets a representative mix of early and
# late policies. Each worker streams one checkpoint at a time and removes it
# after evaluation; the preloaded 5k checkpoint is retained.
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$SWEEP" \
  --steps 5000 15000 25000 35000 45000 55000 \
  --episodes 30 --episode-length 400 --batch-size 1 --no-compile-model --max-episodes-rendered 0 \
  --observation-refresh-interval 50 \
  >"$LOG_DIR/sweep_gpu0.log" 2>&1 &
gpu0_pid=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SWEEP" \
  --steps 10000 20000 30000 40000 50000 \
  --episodes 30 --episode-length 400 --batch-size 1 --no-compile-model --max-episodes-rendered 0 \
  --observation-refresh-interval 50 \
  >"$LOG_DIR/sweep_gpu1.log" 2>&1 &
gpu1_pid=$!

status=0
wait "$gpu0_pid" || status=$?
wait "$gpu1_pid" || status=$?
exit "$status"
