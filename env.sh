#!/usr/bin/env bash

export ROBOTWIN_EVAL_ROOT=/workspace/robotwin_eval
export ROBOTWIN_ROOT="$ROBOTWIN_EVAL_ROOT/RoboTwin"
export LEROBOT_ROOT="$ROBOTWIN_EVAL_ROOT/lerobot"
export PATH="$LEROBOT_ROOT/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH="$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.0
export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1

