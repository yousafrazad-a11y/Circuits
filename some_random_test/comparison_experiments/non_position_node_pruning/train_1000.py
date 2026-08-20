"""Fresh 1,000-epoch global-mask IOI run with validation checkpoints."""

from __future__ import annotations

from pathlib import Path

from comparison_experiments.position_aware_node_pruning import ioi as shared_trainer

from .ioi import configure_global_baseline


RUN_DIRECTORY = (
    Path(__file__).resolve().parent / "outputs" / "continuous_1000_run_01"
)


def configure_run() -> None:
    configure_global_baseline()
    shared_trainer.NUM_EPOCHS = 1000
    shared_trainer.VALIDATION_INTERVAL = 10
    shared_trainer.OUTPUT_DIRECTORY = RUN_DIRECTORY
    shared_trainer.OUTPUT_PATH = RUN_DIRECTORY / "ioi_abba_global_gates_final.pt"
    shared_trainer.METRICS_PATH = RUN_DIRECTORY / "training_metrics.jsonl"
    shared_trainer.CHECKPOINT_DIRECTORY = RUN_DIRECTORY / "checkpoints"


def main() -> None:
    configure_run()
    existing_checkpoints = list(
        shared_trainer.CHECKPOINT_DIRECTORY.glob("epoch_*.pt")
    )
    if existing_checkpoints:
        raise RuntimeError(
            f"Fresh-run directory already contains {len(existing_checkpoints)} "
            "checkpoints; choose a new run directory instead of overwriting them."
        )
    print(
        "FRESH GLOBAL BASELINE: 1000 epochs; continuous log-alpha checkpoint "
        "every 10 epochs",
        flush=True,
    )
    shared_trainer.main()


if __name__ == "__main__":
    main()
