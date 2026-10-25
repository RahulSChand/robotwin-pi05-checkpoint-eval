# RoboTwin π0.5 checkpoint evaluation

Reproducible evaluation of the LeRobot π0.5 checkpoints in
[`Myungkyu/pi05_robotwin_vanilla_b128_60k`](https://huggingface.co/Myungkyu/pi05_robotwin_vanilla_b128_60k)
on RoboTwin's clean `pick_dual_bottles` task.

Published results: [`Chand0320/robotwin-pi05-checkpoint-eval-results`](https://huggingface.co/datasets/Chand0320/robotwin-pi05-checkpoint-eval-results)

## Pinned sources

| Component | Revision |
|---|---|
| Evaluation-enabled LeRobot fork | `RahulSChand/lerobot@5502f49040f91dff8d115d49ad294b8619ff53e5` |
| LeRobot upstream base | `d3793b7e36522321c1c455f0dc4826db3793dceb` |
| RoboTwin | `0aeea2d669c0f8516f4d5785f0aa33ba812c14b4` |
| CuRobo | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` (`v0.7.8`) |
| PyTorch3D | `75ebeeaea0908c5527e7b1e305fbc7681382db47` |

The evaluator fork adds:

- RoboTwin native evaluation mode and official per-task action limits;
- open-loop observation reuse aligned with the policy's execution horizon;
- removal of a redundant terminal environment reset;
- configurable diagnostic video count;
- per-episode seeds, success flags, and episode lengths in `eval_info.json`.

## Machine requirements

- Native Linux with NVIDIA Vulkan/EGL support
- Python 3.12
- CUDA toolkit 12.8
- Approximately 40 GB free disk for repositories, assets, and two streamed checkpoints
- One or more NVIDIA GPUs; the measured run used 2× A100 40 GB

The exact tested package inventory is recorded in [`setup_manifest.json`](setup_manifest.json).

## Installation

The simplest starting point is the pinned benchmark Dockerfile from the evaluator fork:

```bash
git clone https://github.com/RahulSChand/lerobot.git
cd lerobot
git checkout 5502f49040f91dff8d115d49ad294b8619ff53e5
docker build -f docker/Dockerfile.benchmark.robotwin -t lerobot-robotwin-eval .
```

For the native Vast.ai setup used for the published results, clone this runner repository as `/workspace/robotwin_eval`, clone the two pinned source repositories inside it, and install the runtime:

```bash
git clone https://github.com/RahulSChand/robotwin-pi05-checkpoint-eval.git \
  /workspace/robotwin_eval
cd /workspace/robotwin_eval

git clone https://github.com/RahulSChand/lerobot.git
git -C lerobot checkout 5502f49040f91dff8d115d49ad294b8619ff53e5

git clone https://github.com/RoboTwin-Platform/RoboTwin.git
git -C RoboTwin checkout 0aeea2d669c0f8516f4d5785f0aa33ba812c14b4

cd lerobot
uv sync --frozen --extra pi --extra evaluation
uv pip install --python .venv/bin/python -r ../requirements-robotwin-runtime.txt
uv pip install --python .venv/bin/python 'numpy==1.26.4'
uv pip install --python .venv/bin/python --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'

cd ../RoboTwin/envs
git clone https://github.com/NVlabs/curobo.git
git -C curobo checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
TORCH_CUDA_ARCH_LIST=8.0 uv pip install --python ../../lerobot/.venv/bin/python \
  -e curobo --no-build-isolation
```

The π0.5 tokenizer is gated. Authenticate once and cache its lightweight files
without downloading the PaliGemma model weights:

```bash
hf auth login
HF_HOME=/workspace/.hf_home hf download google/paligemma-3b-pt-224 \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.msgpack'
```

Download RoboTwin's public `embodiments.zip` and `objects.zip` assets from
[`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0),
extract them under `RoboTwin/assets`, then run:

```bash
cd /workspace/robotwin_eval/RoboTwin
../lerobot/.venv/bin/python script/update_embodiment_config_path.py
```

RoboTwin's installer also applies small compatibility patches to `mplib==0.2.1`
and `sapien==3.0.0b1`; the pinned
[`Dockerfile.benchmark.robotwin`](https://github.com/RahulSChand/lerobot/blob/5502f49040f91dff8d115d49ad294b8619ff53e5/docker/Dockerfile.benchmark.robotwin)
is the authoritative reference for those patches and asset steps.

Load the runtime environment:

```bash
. /workspace/robotwin_eval/env.sh
```

## Evaluation protocol

- Task: `pick_dual_bottles`
- Environment: `demo_clean`
- Official task limit: 400 policy actions
- π0.5 execution horizon: 50 actions
- Observation refresh interval: 50 actions
- Batch size: 1 environment per GPU
- Inference: eager BF16 (`torch.compile` was slower in measured tests)
- Diagnostic videos: disabled for bulk evaluation
- Curve seeds: 1000–1029 at every checkpoint
- Curve checkpoints: 5k through 55k, every 5k
- Confirmation: step 45k on 60 fresh seeds, 2000–2059

The observation reuse setting still executes every action and every success check;
it only skips camera/state capture when the policy is consuming an already-predicted
open-loop action chunk. A paired two-seed comparison matched success flags and
episode lengths exactly and reduced runtime from 312.1 seconds to 69.4 seconds.

## Run

Smoke test:

```bash
./run_smoke_5episodes.sh
```

Two-GPU 5k–55k curve, 30 episodes per checkpoint:

```bash
./run_dual_gpu_sweep.sh
```

Validate and plot:

```bash
lerobot/.venv/bin/python validate_and_plot.py \
  --final-results results/pick_dual_bottles_clean_curve
```

Independent two-GPU 45k confirmation on 60 fresh episodes:

```bash
./run_confirm45k.sh
```

For resilient long-running execution, copy the files in `supervisor/` to
`/etc/supervisor/conf.d/`, then run `supervisorctl reread && supervisorctl update`.

## Outputs

Each checkpoint produces:

- `run_manifest.json` with protocol and provenance;
- `eval_info.json` with aggregate and per-episode records;
- `summary.json` with success rate, confidence interval, and mean episode length;
- one row in `checkpoint_success_curve.csv`.

The published result set intentionally excludes bulky evaluation logs, downloaded
weights, and the incomplete 60k attempt.
