# RoboTwin π0.5 Checkpoint-vs-Success-Rate Evaluation

## Objective

Measure how RoboTwin success rate changes during co-trained π0.5 fine-tuning by evaluating the same task at checkpoints from 5k through 60k optimizer steps.

Primary task:

```text
pick_dual_bottles
```

This is a useful middle-difficulty probe because it explicitly requires both arms, has a relatively short average trajectory, and is neither a clear floor nor ceiling task. The official RoboTwin co-trained π0.5 result is approximately 72% on clean scenes and 57% on randomized scenes. These values are only a difficulty reference; the public checkpoint reproduction may produce different results.

For the first experiment, evaluate only the clean environment and hold every evaluation setting and scene seed fixed across checkpoints.

## Important checkpoint-format distinction

The checkpoints in this experiment come from:

```text
Myungkyu/pi05_robotwin_vanilla_b128_60k
```

They are LeRobot/PyTorch checkpoints containing `model.safetensors` and LeRobot policy processor files. They are not the task-specific JAX/Orbax checkpoints released by OpenPI-PACE.

Therefore:

- Use Hugging Face LeRobot to load and run the policy.
- Use RoboTwin as the simulator.
- Do not use `niejunnan25/Openpi-PACE` or `niejunnan25/PACE` for this initial experiment.
- Do not convert the checkpoints to the OpenPI/PACE format.

## Repositories

Clone these two repositories:

```bash
git clone https://github.com/huggingface/lerobot.git
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git
```

| Repository | Purpose |
|---|---|
| `huggingface/lerobot` | Loads the π0.5 safetensors checkpoint and provides `lerobot-eval` |
| `RoboTwin-Platform/RoboTwin` | Provides the simulator, task environments, assets, resets, and success checks |

Follow the current official installation instructions for both repositories. RoboTwin should run on native Linux with an NVIDIA GPU and Vulkan support; Python 3.10 and CUDA 12.1 are recommended for its simulator environment. Confirm that a single RoboTwin environment can reset and render before testing the policy.

The checkpoint was trained with LeRobot 0.5.2 at source revision `09808183`. Start with current LeRobot because its current tree contains the RoboTwin environment adapter. If checkpoint deserialization fails because of an API change, compare against that training revision while preserving or porting the current `src/lerobot/envs/robotwin.py` integration.

## Checkpoint schedule

Evaluate:

```text
5k, 10k, 15k, 20k, 25k, 30k,
35k, 40k, 45k, 50k, 55k, 60k
```

The 5k–55k checkpoints live under `checkpoints/step-<N>` in the Hugging Face repository. The final 60k checkpoint lives at the repository root rather than under `checkpoints/step-60000`.

Every intermediate directory is independently loadable and contains files similar to:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
SHA256SUMS
```

The checkpoints are approximately 9.4 GB each. Downloading all checkpoints requires over 100 GB, so downloading and evaluating one checkpoint at a time is reasonable.

## Downloading checkpoints

Example for step 5k:

```bash
hf download Myungkyu/pi05_robotwin_vanilla_b128_60k \
  --include "checkpoints/step-5000/*" \
  --local-dir /data/pi05-robotwin
```

The local policy directory should then be:

```text
/data/pi05-robotwin/checkpoints/step-5000
```

Repeat by changing `step-5000` to the desired intermediate step.

Download the final 60k checkpoint without downloading every intermediate:

```bash
hf download Myungkyu/pi05_robotwin_vanilla_b128_60k \
  --exclude "checkpoints/*" \
  --local-dir /data/pi05-robotwin/final-60000
```

Before evaluation, verify that the selected directory contains `config.json`, `model.safetensors`, and both preprocessor and postprocessor assets. Do not replace its normalization files with task-specific statistics.

## Policy interface

The checkpoint expects:

```text
observation.state:                    14 dimensions
observation.images.cam_high:          RGB, 3 × 240 × 320
observation.images.cam_left_wrist:    RGB, 3 × 240 × 320
observation.images.cam_right_wrist:   RGB, 3 × 240 × 320
task:                                 natural-language instruction
action:                               14 dimensions
action chunk:                         50 steps
```

The checkpoint includes its own preprocessing, quantile normalization, and action postprocessing. Loading only `model.safetensors` is insufficient; load the whole policy directory through LeRobot.

## Initial smoke test

First run five clean episodes using step 5k:

```bash
lerobot-eval \
  --policy.path=/data/pi05-robotwin/checkpoints/step-5000 \
  --env.type=robotwin \
  --env.task=pick_dual_bottles \
  --eval.batch_size=1 \
  --eval.n_episodes=5
```

Confirm all of the following before launching the sweep:

- The three camera tensors reach the policy under the expected feature names.
- The policy receives a nonempty task instruction.
- The policy returns a `50 × 14` action chunk.
- Joint and gripper values are denormalized by the checkpoint postprocessor.
- Left and right arm dimensions are not swapped.
- Rollout videos have correct RGB channel ordering.
- The simulator completes episodes and reports success/failure without exceptions.

Do not manually add a second normalization or denormalization step.

## Full evaluation protocol

For each checkpoint, run 100 clean episodes:

```bash
lerobot-eval \
  --policy.path=/data/pi05-robotwin/checkpoints/step-5000 \
  --env.type=robotwin \
  --env.task=pick_dual_bottles \
  --eval.batch_size=1 \
  --eval.n_episodes=100
```

Change only the checkpoint path between runs. Keep constant:

- Task: `pick_dual_bottles`
- Environment condition: clean
- Evaluation seed and initial scene list
- Number of episodes: 100
- Action execution/replanning setting
- Number of inference steps
- Camera names and order
- Image preprocessing
- Simulator commit and assets
- LeRobot commit
- RoboTwin commit
- GPU precision and inference dtype

If `lerobot-eval` exposes a seed argument, set it explicitly and reuse it for every checkpoint. Record the complete resolved configuration in each result directory.

The current LeRobot RoboTwin adapter uses `task_config = "demo_clean"` internally. That is correct for this initial curve. Do not edit the RoboTwin task implementations.

## Randomized evaluation

Randomized evaluation is optional and should happen only after the clean curve works.

The current LeRobot RoboTwin adapter hardcodes `demo_clean` in:

```text
lerobot/src/lerobot/envs/robotwin.py
```

To support randomized evaluation, change the LeRobot environment configuration so `task_config` is a user-facing option whose allowed values include `demo_clean` and `demo_randomized`. Make this change in LeRobot's RoboTwin adapter, not in the RoboTwin task code. Verify that the selected condition is included in saved result metadata, and keep clean and randomized outputs separate.

## Results to save

For each checkpoint, save at minimum:

```text
checkpoint_step
checkpoint_path
task_name
environment_condition
evaluation_seed
number_of_episodes
number_of_successes
success_rate
mean_episode_length
simulator_commit
lerobot_commit
checkpoint SHA256, if available
resolved evaluation configuration
```

Also retain per-episode records containing scene seed, success/failure, and episode length. Save videos for all smoke-test episodes and a smaller diagnostic sample for full runs.

Create a final CSV such as:

```csv
checkpoint_step,condition,episodes,successes,success_rate
5000,clean,100,...,...
10000,clean,100,...,...
15000,clean,100,...,...
20000,clean,100,...,...
25000,clean,100,...,...
30000,clean,100,...,...
35000,clean,100,...,...
40000,clean,100,...,...
45000,clean,100,...,...
50000,clean,100,...,...
55000,clean,100,...,...
60000,clean,100,...,...
```

Plot checkpoint step on the x-axis and success rate on the y-axis. Add a binomial 95% confidence interval to each point. Around 50% success with 100 episodes, uncertainty remains roughly ±10 percentage points, so small adjacent-checkpoint changes should not be treated as meaningful.

## Recommended execution order

1. Install LeRobot and RoboTwin.
2. Download RoboTwin assets.
3. Verify a simulator-only reset/render smoke test.
4. Download the 5k checkpoint.
5. Run five `pick_dual_bottles` episodes.
6. Inspect observations, actions, videos, and success checks.
7. Run 100 episodes at 5k.
8. Download and evaluate 10k through 60k using the identical protocol.
9. Aggregate results and plot the clean SR curve.
10. Only then add randomized evaluation if needed.

## Expected interpretation

The co-trained model used 60,000 optimizer steps with effective global batch size 128 over 2,500 clean RoboTwin demonstrations, approximately 552,287 timesteps. This corresponds to about 14 effective passes through the joint dataset. The goal is to determine whether `pick_dual_bottles` improves steadily, saturates early, or becomes non-monotonic during those 60,000 updates.

The official co-trained π0.5 per-task result is not guaranteed to match this public reproduction. Report the measured curve from these exact weights rather than substituting official leaderboard numbers.
