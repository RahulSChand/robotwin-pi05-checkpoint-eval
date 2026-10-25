#!/usr/bin/env bash
set -euo pipefail

. /workspace/robotwin_eval/env.sh

POLICY=/workspace/robotwin_eval/checkpoints/checkpoints/step-5000
OUTPUT=/workspace/robotwin_eval/outputs/smoke_step5000_5ep
RENAME_MAP='{"observation.images.head_camera":"observation.images.cam_high","observation.images.left_camera":"observation.images.cam_left_wrist","observation.images.right_camera":"observation.images.cam_right_wrist"}'

# The required gated tokenizer files are cached locally. Keep processor loading
# offline so this instance does not need a Hugging Face credential.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -f "$POLICY/model.safetensors" ]; then
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  "$LEROBOT_ROOT/.venv/bin/hf" download Myungkyu/pi05_robotwin_vanilla_b128_60k \
    --include 'checkpoints/step-5000/*' \
    --local-dir /workspace/robotwin_eval/checkpoints
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

cd "$ROBOTWIN_ROOT"
exec "$LEROBOT_ROOT/.venv/bin/lerobot-eval" \
  --policy.path="$POLICY" \
  --policy.compile_model=false \
  --env.type=robotwin \
  --env.task=pick_dual_bottles \
  --env.episode_length=400 \
  --env.observation_refresh_interval=50 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --seed=1000 \
  --output_dir="$OUTPUT" \
  --rename_map="$RENAME_MAP"
