#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXPECTED_STEPS = list(range(5_000, 55_001, 5_000))


def validate_smoke(output: Path) -> None:
    info_path = output / "eval_info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Smoke test did not produce {info_path}")
    info = json.loads(info_path.read_text())
    episodes = info["per_task"][0]["metrics"].get("per_episode", [])
    if len(episodes) != 5:
        raise RuntimeError(f"Expected 5 smoke episodes, found {len(episodes)}")
    seeds = [int(row["seed"]) for row in episodes]
    if seeds != list(range(1000, 1005)):
        raise RuntimeError(f"Unexpected smoke seeds: {seeds}")
    videos = list((output / "videos").glob("**/eval_episode_*.mp4"))
    if len(videos) != 5 or any(path.stat().st_size == 0 for path in videos):
        raise RuntimeError(f"Expected 5 nonempty smoke videos, found {len(videos)}")
    print(
        f"smoke_valid=true successes={info['overall']['n_success']}/5 seeds={seeds}",
        flush=True,
    )


def validate_and_plot(results: Path) -> None:
    csv_path = results / "checkpoint_success_curve.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["checkpoint_step"]) for row in rows]
    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"Expected checkpoint rows {EXPECTED_STEPS}, found {steps}")
    if any(int(row["episodes"]) != 30 for row in rows):
        raise RuntimeError("Every checkpoint must contain exactly 30 episodes")
    for step in EXPECTED_STEPS:
        summary = results / f"step-{step:06d}" / "summary.json"
        if not summary.is_file():
            raise RuntimeError(f"Missing {summary}")
        data = json.loads(summary.read_text())
        if len(data.get("per_episode", [])) != 30:
            raise RuntimeError(f"Incomplete per-episode metadata in {summary}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = [float(row["success_rate"]) for row in rows]
    lows = [float(row["ci95_low"]) for row in rows]
    highs = [float(row["ci95_high"]) for row in rows]
    errors = [
        [rate - low for rate, low in zip(rates, lows)],
        [high - rate for rate, high in zip(rates, highs)],
    ]

    fig, ax = plt.subplots(figsize=(9, 5.25), constrained_layout=True)
    ax.errorbar(steps, rates, yerr=errors, fmt="o-", capsize=4, linewidth=1.8)
    ax.set(
        title="RoboTwin pick_dual_bottles — clean",
        xlabel="Optimizer step",
        ylabel="Success rate",
    )
    ax.set_xticks(steps)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.savefig(results / "checkpoint_success_curve.png", dpi=180)
    fig.savefig(results / "checkpoint_success_curve.pdf")
    print(
        f"final_results_valid=true rows={len(rows)} plot={results / 'checkpoint_success_curve.png'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-smoke", type=Path)
    group.add_argument("--final-results", type=Path)
    args = parser.parse_args()
    if args.validate_smoke:
        validate_smoke(args.validate_smoke)
    else:
        validate_and_plot(args.final_results)


if __name__ == "__main__":
    main()
