"""Hard-concrete gates used by the position-aware circuit model."""

from collections.abc import Sequence
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


GateShape = Union[int, Sequence[int]]


class HardConcreteGate(nn.Module):
    """A stochastic binary gate trained with a straight-through estimator."""

    def __init__(
        self,
        shape: GateShape,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        init_min: float = 2.5,
        init_max: float = 3.5,
    ):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError(f"Gate shape must be positive, received {shape}.")

        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("zeta", torch.tensor(zeta))
        self.final_mode = False
        self.log_alpha = nn.Parameter(torch.empty(shape))
        self.init_weights(init_min, init_max)

    def init_weights(self, init_min: float, init_max: float) -> None:
        with torch.no_grad():
            self.log_alpha.uniform_(init_min, init_max)

    def _soft_gate(self, sample: bool) -> torch.Tensor:
        if sample:
            uniform = torch.rand_like(self.log_alpha).clamp(1e-8, 1.0 - 1e-8)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            sigmoid_input = (logistic_noise + self.log_alpha) / self.beta
        else:
            sigmoid_input = self.log_alpha
        stretched = torch.sigmoid(sigmoid_input) * (self.zeta - self.gamma) + self.gamma
        return F.hardtanh(stretched, min_val=0.0, max_val=1.0)

    def forward(self) -> torch.Tensor:
        gate_soft = self._soft_gate(sample=self.training)
        gate_hard = (gate_soft > 0.5).to(gate_soft.dtype)
        if self.training:
            return (gate_hard - gate_soft).detach() + gate_soft
        return gate_hard

    def num_gates(self) -> int:
        return self.log_alpha.numel()

    def get_sparsity_loss(self) -> torch.Tensor:
        """Expected open-gate density, averaged over sections and components."""
        probability_open = torch.sigmoid(
            self.log_alpha - self.beta * torch.log(-self.gamma / self.zeta)
        )
        return probability_open.mean()

    def get_sparsity_rate(self) -> float:
        return 1.0 - self.get_sparsity_loss().item()

    def hard_mask(self) -> torch.Tensor:
        """Return the exact deterministic mask used during evaluation."""
        with torch.no_grad():
            return (self._soft_gate(sample=False) > 0.5)

    def get_num_active(self) -> int:
        return int(self.hard_mask().sum().item())

    def set_final_mode(self, mode: bool = True) -> None:
        self.final_mode = mode

    def get_mask_statistics(self) -> dict:
        with torch.no_grad():
            deterministic_gate = self._soft_gate(sample=False)
            return {
                "mean_gate": deterministic_gate.mean().item(),
                "std_gate": deterministic_gate.std().item()
                if deterministic_gate.numel() > 1
                else 0.0,
                "min_gate": deterministic_gate.min().item(),
                "max_gate": deterministic_gate.max().item(),
                "sparsity_rate": self.get_sparsity_rate(),
                "num_active": self.get_num_active(),
                "num_total": deterministic_gate.numel(),
                "expected_density": self.get_sparsity_loss().item(),
            }
