"""L0-regularized gates for multi-task Venn circuit discovery.

This module provides two building blocks:

* :class:`HardConcreteGate` -- a single Hard-Concrete gate (Louizos et al., 2018)
  with a Straight-Through Estimator (STE) for stable binary training. This is a
  cleaned-up, fully-typed port of the gate used in the reference codebase.
* :class:`VennConcreteGate` -- three coupled Hard-Concrete gates
  (``g_core``, ``g_a_only``, ``g_b_only``) combined through a differentiable OR
  to yield two effective masks (``mask_a`` and ``mask_b``) that share a common
  ``core`` while allowing task-specific extensions.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteGate(nn.Module):
    """A single Hard-Concrete gate with a Straight-Through Estimator.

    Each element of the gate produces a value in ``[0, 1]``. During training the
    gate samples binary values with reparameterised Concrete noise but passes the
    (continuous) soft value through for gradients (STE). At evaluation time the
    gate is deterministic.

    Args:
        size: Number of independent gate logits.
        beta: Concrete temperature (``2/3`` in the original paper).
        gamma: Lower stretch bound (negative, enables exact zeros).
        zeta: Upper stretch bound (``> 1``, enables exact ones).
        init_min: Lower bound for the uniform ``log_alpha`` initialisation.
        init_max: Upper bound for the uniform ``log_alpha`` initialisation.
    """

    def __init__(
        self,
        size: int,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        init_min: float = 2.5,
        init_max: float = 3.5,
    ) -> None:
        super().__init__()

        # Distribution constants are buffers so they move with `.to(device)`.
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("zeta", torch.tensor(zeta))

        self.log_alpha = nn.Parameter(torch.empty(size))
        with torch.no_grad():
            self.log_alpha.uniform_(init_min, init_max)

    def forward(self) -> torch.Tensor:
        """Return gate values in ``[0, 1]`` with shape ``(size,)``."""
        if self.training:
            # Reparameterised Concrete sample.
            u = torch.rand_like(self.log_alpha).clamp_(1e-8, 1.0 - 1e-8)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
        else:
            # Deterministic expectation (noise removed).
            s = torch.sigmoid(self.log_alpha)

        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        gate_soft = F.hardtanh(s_stretched, min_val=0.0, max_val=1.0)

        # STE: forward pass is binary, backward pass flows through the soft value.
        gate_hard = (gate_soft > 0.5).float()
        return (gate_hard - gate_soft).detach() + gate_soft

    def expected_l0(self) -> torch.Tensor:
        """Expected L0 density: mean probability that a gate is open.

        Using the *mean* (density) rather than the *sum* (count) keeps the
        sparsity penalty scale-invariant to the number of gates.
        """
        p_open = torch.sigmoid(self.log_alpha - self.beta * torch.log(-self.gamma / self.zeta))
        return p_open.mean()

    def num_gates(self) -> int:
        """Number of independent gate logits."""
        return int(self.log_alpha.numel())


class VennConcreteGate(nn.Module):
    """Three coupled Hard-Concrete gates forming a differentiable Venn diagram.

    The gate models the relationship between two tasks (A and B) via three
    disjoint regions:

    * ``g_core``   -- structure shared by both tasks (A AND B).
    * ``g_a_only`` -- structure required only by task A.
    * ``g_b_only`` -- structure required only by task B.

    The effective per-task masks are the differentiable unions::

        mask_a = OR(g_core, g_a_only)
        mask_b = OR(g_core, g_b_only)

    A mask value of ``1`` means "keep the clean signal" (the unit is inside the
    circuit); ``0`` means "replace with the corrupted signal" (ablated).
    """

    def __init__(self, size: int, **gate_kwargs) -> None:
        super().__init__()
        self.size = size
        self.g_core = HardConcreteGate(size, **gate_kwargs)
        self.g_a_only = HardConcreteGate(size, **gate_kwargs)
        self.g_b_only = HardConcreteGate(size, **gate_kwargs)

    @staticmethod
    def differentiable_or(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Straight-Through differentiable OR of two gate tensors.

        Forward pass uses the hard boolean OR of the rounded gates; the backward
        pass flows through the smooth probabilistic OR ``g1 + g2 - g1 * g2``.
        """
        hard_or = torch.clamp(g1.round() + g2.round(), 0.0, 1.0)
        soft_or = g1 + g2 - (g1 * g2)
        return (hard_or - soft_or).detach() + soft_or

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mask_a, mask_b)``, each of shape ``(size,)``."""
        core = self.g_core()
        a_only = self.g_a_only()
        b_only = self.g_b_only()
        mask_a = self.differentiable_or(core, a_only)
        mask_b = self.differentiable_or(core, b_only)
        return mask_a, mask_b

    def sparsity(self) -> Dict[str, torch.Tensor]:
        """Expected-L0 density for each of the three regions.

        Keys ``core`` / ``a_only`` / ``b_only`` are consumed by the trainer to
        build the Venn sparsity loss with region-specific lambdas.
        """
        return {
            "core": self.g_core.expected_l0(),
            "a_only": self.g_a_only.expected_l0(),
            "b_only": self.g_b_only.expected_l0(),
        }

    def num_gates(self) -> int:
        return self.size
