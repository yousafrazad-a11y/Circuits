#!/usr/bin/env python3
"""Report hierarchy-effective structure for a learned-position checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from automatic_positions_project.train_joint_positions import build_joint_circuit, load_gate_state
from comparison_experiments.position_aware_node_pruning.utils import analyze_and_finalize_circuit
from comparison_experiments.structural_metrics import node_structural_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt2")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    num_positions = int(payload["args"]["num_positions"])
    circuit = build_joint_circuit(args.model, num_positions, torch.device("cpu"))
    load_gate_state(circuit, payload["gate_state"])
    circuit.set_final_circuit_mode(True)

    analysis = analyze_and_finalize_circuit(circuit, verbose=False)
    analysis["section_names"] = tuple(f"learned_position_{i}" for i in range(num_positions))
    report = node_structural_report(circuit, analysis)
    report["checkpoint"] = str(args.checkpoint.resolve())

    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
