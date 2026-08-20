"""Run the IOI node-pruning baseline with one mask shared by every token.

This entry point intentionally reuses the position-aware training pipeline and
changes only the mask-position cardinality.  The dataset still constructs the
same eight logical sections to guarantee clean/corrupted token alignment, but
all section IDs presented to the circuit model are zero.  Consequently every
token uses the same component gates, matching the original global-mask method.
"""

from __future__ import annotations

from pathlib import Path

import torch

from comparison_experiments.position_aware_node_pruning import ioi as shared_trainer
from comparison_experiments.position_aware_node_pruning import utils as shared_utils
from comparison_experiments.position_aware_node_pruning.dataset.ioi import IOIDataset as AlignedIOIDataset


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"


class GlobalMaskIOIDataset(AlignedIOIDataset):
    """Keep aligned IOI data while mapping every token to one global mask."""

    def __getitem__(self, index: int) -> dict:
        item = dict(super().__getitem__(index))
        item["section_ids"] = torch.zeros_like(item["section_ids"])
        return item


def configure_global_baseline() -> None:
    """Set the shared trainer to a single, globally reused gate row."""

    shared_trainer.NUM_SECTIONS = 1
    shared_trainer.choose_batch_size = lambda device: 64
    shared_trainer.IOIDataset = GlobalMaskIOIDataset
    shared_trainer.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    shared_trainer.OUTPUT_PATH = OUTPUT_DIRECTORY / "ioi_abba_global_gates.pt"
    shared_trainer.METRICS_PATH = (
        OUTPUT_DIRECTORY / "ioi_abba_training_metrics.jsonl"
    )
    shared_trainer.CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"

    # Make final reports and checkpoint metadata accurately name the one mask.
    shared_utils.SECTION_NAMES = ("global",)


def main() -> None:
    configure_global_baseline()
    print(
        "CONTROLLED BASELINE: one component mask shared across every token",
        flush=True,
    )
    shared_trainer.main()


if __name__ == "__main__":
    main()

