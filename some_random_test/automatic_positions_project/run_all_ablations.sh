#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: $0 POSITIONAL_GATE_CHECKPOINT" >&2; exit 2; fi
for input in normalized_position token_embedding early_residual early_residual_plus_position; do
  python -m automatic_positions_project.train_router_probe --expert-checkpoint "$1" --router-input "$input" --output-dir "automatic_positions_project/results/$input"
done
