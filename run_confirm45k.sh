#!/usr/bin/env bash
set -euo pipefail

. /workspace/robotwin_eval/env.sh

ROOT=/workspace/robotwin_eval
POLICY="$ROOT/checkpoints/checkpoints/step-45000"

if [ ! -f "$POLICY/model.safetensors" ]; then
  "$LEROBOT_ROOT/.venv/bin/hf" download Myungkyu/pi05_robotwin_vanilla_b128_60k \
    --include 'checkpoints/step-45000/*' \
    --local-dir "$ROOT/checkpoints"
fi

common=(
  --steps 45000
  --episodes 30
  --episode-length 400
  --batch-size 1
  --no-compile-model
  --max-episodes-rendered 0
  --observation-refresh-interval 50
  --keep-checkpoints
)

CUDA_VISIBLE_DEVICES=0 "$LEROBOT_ROOT/.venv/bin/python" "$ROOT/run_checkpoint_sweep.py" \
  "${common[@]}" --seed 2000 --results-name pick_dual_bottles_clean_confirm_45k_shard0 &
gpu0_pid=$!

CUDA_VISIBLE_DEVICES=1 "$LEROBOT_ROOT/.venv/bin/python" "$ROOT/run_checkpoint_sweep.py" \
  "${common[@]}" --seed 2030 --results-name pick_dual_bottles_clean_confirm_45k_shard1 &
gpu1_pid=$!

status=0
wait "$gpu0_pid" || status=$?
wait "$gpu1_pid" || status=$?
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

exec "$LEROBOT_ROOT/.venv/bin/python" "$ROOT/aggregate_confirm45k.py"

