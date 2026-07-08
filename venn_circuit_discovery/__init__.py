"""venn_circuit_discovery -- Automated Multi-Task Circuit Discovery for Llama.

A clean, modular library for discovering the *intersection* (AND) or *union*
(OR) of the circuits underlying two distinct corrupted tasks in a single
training run. It combines:

* L0-regularised **Venn-Gates** (core / A-only / B-only) with a Straight-Through
  differentiable OR.
* **Homoscedastic Uncertainty** weighting to auto-balance the two task losses.
* **Dual PID Controllers** with dynamic linking to schedule the sparsity
  pressure of each region.

Public API:
    >>> from venn_circuit_discovery import VennCircuitDiscoverer, VennBatch
"""

from .gates import HardConcreteGate, VennConcreteGate
from .loss import HomoscedasticUncertaintyLoss, kl_divergence_loss, margin_loss
from .models import (
    LlamaVennCircuit,
    VennDecoderLayer,
    VennForwardOutput,
    VennPruningConfig,
)
from .scheduler import (
    DualVennScheduler,
    DualVennSchedulerConfig,
    LogicMode,
    PIDController,
)
from .trainer import StepMetrics, TrainerConfig, VennBatch, VennTrainer
from .api import VennCircuitDiscoverer, VennHyperparameters

__all__ = [
    "HardConcreteGate",
    "VennConcreteGate",
    "HomoscedasticUncertaintyLoss",
    "kl_divergence_loss",
    "margin_loss",
    "LlamaVennCircuit",
    "VennDecoderLayer",
    "VennForwardOutput",
    "VennPruningConfig",
    "DualVennScheduler",
    "DualVennSchedulerConfig",
    "LogicMode",
    "PIDController",
    "StepMetrics",
    "TrainerConfig",
    "VennBatch",
    "VennTrainer",
    "VennCircuitDiscoverer",
    "VennHyperparameters",
]

__version__ = "0.1.0"
