"""Launch one isolated 1000-epoch, 2x-LR, 2x-global-pressure IOI run."""

from __future__ import annotations

import argparse
from pathlib import Path


def configure_shared(shared, output: Path, args) -> None:
    shared.NUM_EPOCHS = args.epochs
    shared.LEARNING_RATE = args.learning_rate
    shared.GLOBAL_SPARSITY_MULTIPLIER = args.global_multiplier
    shared.choose_batch_size = lambda device: 64
    shared.OUTPUT_DIRECTORY = output
    shared.OUTPUT_PATH = output / "ioi_abba_gates.pt"
    shared.METRICS_PATH = output / "ioi_abba_training_metrics.jsonl"
    shared.CHECKPOINT_DIRECTORY = output / "checkpoints"
    shared.INITIAL_GATE_CHECKPOINT = args.initial_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("global", "position"), required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=6e-2)
    parser.add_argument("--global-multiplier", type=float, default=2.0)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--run-name", default="rerun_1000ep_lr2x_pressure2x")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    from comparison_experiments.position_aware_node_pruning import ioi as shared

    if args.method == "global":
        from comparison_experiments.non_position_node_pruning import ioi as global_run
        global_run.configure_global_baseline()
        output = root / "non_position_node_pruning/outputs" / args.run_name
    else:
        output = root / "position_aware_node_pruning/outputs" / args.run_name

    configure_shared(shared, output, args)
    print(
        "RERUN_CONFIG "
        f"method={args.method} epochs={shared.NUM_EPOCHS} "
        f"learning_rate={shared.LEARNING_RATE} "
        f"global_sparsity_multiplier={shared.GLOBAL_SPARSITY_MULTIPLIER} "
        "component_lambdas=unchanged batch_size=64 "
        f"initial_checkpoint={shared.INITIAL_GATE_CHECKPOINT} "
        f"output={output}",
        flush=True,
    )
    shared.main()


if __name__ == "__main__":
    main()
