#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path("/workspace/robotwin_eval/results")
INPUTS = [
    ROOT
    / "pick_dual_bottles_clean_confirm_45k_shard0"
    / "step-045000"
    / "summary.json",
    ROOT
    / "pick_dual_bottles_clean_confirm_45k_shard1"
    / "step-045000"
    / "summary.json",
]
OUTPUT = ROOT / "pick_dual_bottles_clean_confirm_45k_60ep"


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes * episodes))
        / denominator
    )
    return [center - radius, center + radius]


summaries = [json.loads(path.read_text()) for path in INPUTS]
episodes = [episode for summary in summaries for episode in summary["per_episode"]]
if len(episodes) != 60 or len({int(row["seed"]) for row in episodes}) != 60:
    raise RuntimeError("Expected 60 unique confirmation episodes")
episodes.sort(key=lambda row: int(row["seed"]))
successes = sum(bool(row["success"]) for row in episodes)
rate = successes / len(episodes)
ci_low, ci_high = wilson(successes, len(episodes))

combined = {
    **{key: value for key, value in summaries[0].items() if key != "per_episode"},
    "evaluation_seed_ranges": [[2000, 2029], [2030, 2059]],
    "number_of_episodes": len(episodes),
    "number_of_successes": successes,
    "success_rate": rate,
    "mean_episode_length": sum(int(row["episode_length"]) for row in episodes)
    / len(episodes),
    "ci95_percent": [100 * ci_low, 100 * ci_high],
    "per_episode": episodes,
}
OUTPUT.mkdir(parents=True, exist_ok=False)
(OUTPUT / "summary.json").write_text(
    json.dumps(combined, indent=2, sort_keys=True) + "\n"
)
with (OUTPUT / "result.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
        [
            "checkpoint_step",
            "condition",
            "episodes",
            "successes",
            "success_rate",
            "ci95_low",
            "ci95_high",
        ]
    )
    writer.writerow([45000, "clean", len(episodes), successes, rate, ci_low, ci_high])
print(
    json.dumps(
        {
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": rate,
            "ci95": [ci_low, ci_high],
        }
    )
)
