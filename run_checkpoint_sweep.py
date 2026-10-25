#!/usr/bin/env python3
"""Stream and evaluate native LeRobot π0.5 RoboTwin checkpoints one at a time."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

from huggingface_hub import snapshot_download

REPO_ID = "Myungkyu/pi05_robotwin_vanilla_b128_60k"
DEFAULT_STEPS = tuple(range(5_000, 60_001, 5_000))
REQUIRED = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor.json",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    "train_config.json",
    "SHA256SUMS",
)
RENAME_MAP = {
    "observation.images.head_camera": "observation.images.cam_high",
    "observation.images.left_camera": "observation.images.cam_left_wrist",
    "observation.images.right_camera": "observation.images.cam_right_wrist",
}


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_checkpoint(policy_dir: pathlib.Path) -> str:
    missing = [name for name in REQUIRED if not (policy_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing: {missing}")
    expected = {}
    for line in (policy_dir / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split(maxsplit=1)
        expected[name.lstrip("* ")] = checksum
    for name, checksum in expected.items():
        actual = digest(policy_dir / name)
        if actual != checksum:
            raise ValueError(
                f"SHA-256 mismatch for {policy_dir / name}: {actual} != {checksum}"
            )
    return expected["model.safetensors"]


def download_checkpoint(step: int, cache: pathlib.Path) -> pathlib.Path:
    destination = cache / f"step-{step:06d}"
    if step < 60_000:
        prefix = f"checkpoints/step-{step}"
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{prefix}/*"],
            local_dir=destination,
        )
        return destination / prefix
    snapshot_download(
        repo_id=REPO_ID,
        # The root checkpoint's SHA256SUMS also covers its README, unlike the
        # intermediate checkpoint folders. Download it so integrity verification
        # can validate every entry listed by the repository.
        allow_patterns=[*REQUIRED, "README.md"],
        local_dir=destination,
    )
    return destination


def git_commit(path: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    fields = [
        "checkpoint_step",
        "condition",
        "episodes",
        "successes",
        "success_rate",
        "ci95_low",
        "ci95_high",
        "mean_episode_length",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_csv_row(path: pathlib.Path, row: dict) -> None:
    """Atomically merge a checkpoint row when multiple GPU workers share results."""
    lock_path = path.with_suffix(".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows_by_step = {}
        if path.is_file():
            with path.open(newline="") as stream:
                rows_by_step = {
                    int(item["checkpoint_step"]): item
                    for item in csv.DictReader(stream)
                }
        rows_by_step[int(row["checkpoint_step"])] = row
        write_csv(path, [rows_by_step[key] for key in sorted(rows_by_step)])
        fcntl.flock(lock, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--max-episodes-rendered", type=int, default=0)
    parser.add_argument("--observation-refresh-interval", type=int, default=50)
    parser.add_argument(
        "--compile-model", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument(
        "--root", type=pathlib.Path, default=pathlib.Path("/workspace/robotwin_eval")
    )
    parser.add_argument("--results-name", default="pick_dual_bottles_clean_curve")
    args = parser.parse_args()

    if (
        args.episodes < 1
        or args.episode_length < 1
        or args.batch_size < 1
        or args.observation_refresh_interval < 1
    ):
        raise ValueError(
            "episodes, episode-length, batch-size, and observation-refresh-interval must be positive"
        )
    if args.max_episodes_rendered < 0:
        raise ValueError("max-episodes-rendered must be non-negative")
    if any(step not in DEFAULT_STEPS for step in args.steps):
        raise ValueError(f"steps must be selected from {DEFAULT_STEPS}")

    root = args.root.resolve()
    lerobot = root / "lerobot"
    robotwin = root / "RoboTwin"
    cache = root / "sweep_cache"
    results = root / "results" / args.results_name
    cache.mkdir(parents=True, exist_ok=True)
    (root / ".cache" / "torchinductor").mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "checkpoint_success_curve.csv"

    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(robotwin),
        VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
        CUDA_HOME="/usr/local/cuda-12.8",
        HF_HOME="/workspace/.hf_home",
        HF_HUB_DISABLE_XET="1",
        # Checkpoints are downloaded by this parent process before launch. The
        # evaluator itself stays offline so the cached gated PaliGemma
        # tokenizer is used without requiring a credential on the instance.
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        TORCHINDUCTOR_COMPILE_THREADS="4",
        TORCHINDUCTOR_CACHE_DIR=str(root / ".cache" / "torchinductor"),
    )
    environment["PATH"] = (
        f"{lerobot / '.venv/bin'}:/usr/local/cuda-12.8/bin:{environment['PATH']}"
    )
    lerobot_commit = git_commit(lerobot)
    robotwin_commit = git_commit(robotwin)

    for step in args.steps:
        output = results / f"step-{step:06d}"
        eval_info = output / "eval_info.json"
        if eval_info.is_file():
            existing = json.loads((output / "run_manifest.json").read_text())
            requested = {
                "evaluation_seed": args.seed,
                "number_of_episodes": args.episodes,
                "episode_length": args.episode_length,
                "batch_size": args.batch_size,
                "compile_mode": args.compile_mode,
                "max_episodes_rendered": args.max_episodes_rendered,
                "observation_refresh_interval": args.observation_refresh_interval,
                "compile_model": args.compile_model,
            }
            if all(existing.get(key) == value for key, value in requested.items()):
                print(f"step={step} already_complete=true", flush=True)
                continue
            raise FileExistsError(f"Existing result has a different protocol: {output}")
        preloaded = root / "checkpoints" / "checkpoints" / f"step-{step}"
        policy_dir = (
            preloaded
            if step < 60_000 and preloaded.is_dir()
            else download_checkpoint(step, cache)
        )
        checkpoint_sha256 = verify_checkpoint(policy_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint_step": step,
            "checkpoint_path": str(policy_dir),
            "checkpoint_sha256": checkpoint_sha256,
            "task_name": "pick_dual_bottles",
            "environment_condition": "demo_clean",
            "evaluation_seed": args.seed,
            "number_of_episodes": args.episodes,
            "episode_length": args.episode_length,
            "batch_size": args.batch_size,
            "compile_mode": args.compile_mode,
            "max_episodes_rendered": args.max_episodes_rendered,
            "observation_refresh_interval": args.observation_refresh_interval,
            "compile_model": args.compile_model,
            "lerobot_commit": lerobot_commit,
            "robotwin_commit": robotwin_commit,
            "checkpoint_repo": REPO_ID,
            "rename_map": RENAME_MAP,
            "started_at": time.time(),
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        command = [
            str(lerobot / ".venv/bin/lerobot-eval"),
            f"--policy.path={policy_dir}",
            f"--policy.compile_model={'true' if args.compile_model else 'false'}",
            f"--policy.compile_mode={args.compile_mode}",
            "--env.type=robotwin",
            "--env.task=pick_dual_bottles",
            f"--env.episode_length={args.episode_length}",
            f"--env.observation_refresh_interval={args.observation_refresh_interval}",
            f"--eval.batch_size={args.batch_size}",
            "--eval.use_async_envs=true",
            f"--eval.max_episodes_rendered={args.max_episodes_rendered}",
            f"--eval.n_episodes={args.episodes}",
            f"--seed={args.seed}",
            f"--output_dir={output}",
            f"--rename_map={json.dumps(RENAME_MAP, separators=(',', ':'))}",
        ]
        with (output / "evaluation.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=robotwin,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)

        info = json.loads(eval_info.read_text())
        task = info["per_task"][0]
        per_episode = task["metrics"].get("per_episode")
        if not per_episode or len(per_episode) != args.episodes:
            raise RuntimeError(
                "eval_info.json is missing complete per-episode metadata"
            )
        overall = info["overall"]
        ci_low, ci_high = overall["pc_success_ci95"]
        summary = {
            **manifest,
            "number_of_successes": int(overall["n_success"]),
            "success_rate": float(overall["pc_success"]) / 100.0,
            "mean_episode_length": sum(
                int(row["episode_length"]) for row in per_episode
            )
            / len(per_episode),
            "ci95_percent": [float(ci_low), float(ci_high)],
            "completed_at": time.time(),
            "per_episode": per_episode,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        csv_row = {
            "checkpoint_step": step,
            "condition": "clean",
            "episodes": args.episodes,
            "successes": summary["number_of_successes"],
            "success_rate": summary["success_rate"],
            "ci95_low": float(ci_low) / 100.0,
            "ci95_high": float(ci_high) / 100.0,
            "mean_episode_length": summary["mean_episode_length"],
        }
        merge_csv_row(csv_path, csv_row)
        print(
            f"step={step} successes={summary['number_of_successes']}/{args.episodes} "
            f"success_rate={summary['success_rate']:.4f}",
            flush=True,
        )
        if not args.keep_checkpoints and policy_dir.is_relative_to(cache):
            shutil.rmtree(cache / f"step-{step:06d}")


if __name__ == "__main__":
    main()
