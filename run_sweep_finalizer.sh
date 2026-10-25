#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/robotwin_eval
while supervisorctl status robotwin-sweep | grep -q RUNNING; do
  sleep 60
done

"$ROOT/lerobot/.venv/bin/python" "$ROOT/validate_and_plot.py" \
  --final-results "$ROOT/results/pick_dual_bottles_clean_curve"

