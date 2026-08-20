"""Position-aware prunable model components."""

from .gpt2_circuit import PrunableGPT2LMHeadModel, PruningConfig

__all__ = ["PrunableGPT2LMHeadModel", "PruningConfig"]

