"""IOI dataset generation and evaluation with logical position sections."""

from .ioi import (
    IOIDataset,
    SECTION_NAMES,
    filter_dataset_by_model_correctness,
    load_or_generate_ioi_data,
    run_evaluation,
)

__all__ = [
    "IOIDataset",
    "SECTION_NAMES",
    "filter_dataset_by_model_correctness",
    "load_or_generate_ioi_data",
    "run_evaluation",
]

