"""Training loop for multi-task Venn circuit discovery.

The trainer wires together the four-stream model, the homoscedastic uncertainty
loss and the dual PID scheduler, and performs the exact optimisation described
in the methodology:

1. Compute golden logits (clean pass).
2. Compute logits A and logits B (four-stream Venn pass).
3. Faithfulness (KL) and correctness (margin) for both tasks.
4. Scale the raw task losses with the homoscedastic module.
5. Fetch dynamic lambdas from the dual scheduler.
6. Venn sparsity loss: ``lambda_a * L_a_only + lambda_b * L_b_only + lambda_core * L_core``.
7. Backpropagate the total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch

from .loss import HomoscedasticUncertaintyLoss, kl_divergence_loss, margin_loss
from .models import LlamaVennCircuit
from .scheduler import DualVennScheduler


@dataclass
class VennBatch:
    """A single training batch for two corrupted tasks.

    ``answer_positions`` indexes the token whose logits are read for the margin
    loss; ``kl_position_mask`` (optional) selects the positions that count toward
    the faithfulness KL (defaults to all positions).
    """

    clean_input_ids: torch.Tensor
    corr_a_input_ids: torch.Tensor
    corr_b_input_ids: torch.Tensor

    answer_positions: torch.Tensor          # (B,)
    target_a: torch.Tensor                   # (B,)
    distractor_a: torch.Tensor               # (B,)
    target_b: torch.Tensor                   # (B,)
    distractor_b: torch.Tensor               # (B,)

    attention_mask: Optional[torch.Tensor] = None      # (B, T)
    kl_position_mask: Optional[torch.Tensor] = None     # (B, T)

    def to(self, device: torch.device) -> "VennBatch":
        moved = {
            f: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for f, v in self.__dict__.items()
        }
        return VennBatch(**moved)


@dataclass
class TrainerConfig:
    """Optimisation hyper-parameters for :class:`VennTrainer`."""

    gate_lr: float = 0.05
    uncertainty_lr: float = 0.01
    margin: float = 4.0
    grad_clip: float = 1.0
    log_every: int = 10


@dataclass
class StepMetrics:
    """Per-step scalar metrics (all Python floats for cheap logging)."""

    step: int
    kl_a: float
    kl_b: float
    margin_a: float
    margin_b: float
    scaled_a: float
    scaled_b: float
    sparsity: float
    total: float
    lambda_a: float
    lambda_b: float
    lambda_core: float


class VennTrainer:
    """Drives the Venn circuit-discovery optimisation over a data iterable."""

    def __init__(
        self,
        model: LlamaVennCircuit,
        scheduler: DualVennScheduler,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.config = config or TrainerConfig()
        self.device = device or model.device

        self.uncertainty = HomoscedasticUncertaintyLoss().to(self.device)

        gate_params = [g for gate in model.iter_venn_gates().values() for g in gate.parameters()]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": gate_params, "lr": self.config.gate_lr},
                {"params": self.uncertainty.parameters(), "lr": self.config.uncertainty_lr},
            ]
        )
        self.history: List[StepMetrics] = []

    # ------------------------------------------------------------------

    def _task_losses(
        self,
        logits: torch.Tensor,
        golden: torch.Tensor,
        target: torch.Tensor,
        distractor: torch.Tensor,
        batch: VennBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(kl, margin)`` for one task's circuit logits."""
        kl = kl_divergence_loss(logits, golden, batch.kl_position_mask)
        mrg = margin_loss(
            logits, target, distractor, batch.answer_positions, margin=self.config.margin
        )
        return kl, mrg

    def step(self, batch: VennBatch, step_idx: int) -> StepMetrics:
        """Run one optimisation step and return its metrics."""
        self.model.train()
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        # 1. Golden (clean) logits -- the shared faithfulness target.
        golden = self.model.golden_logits(batch.clean_input_ids, batch.attention_mask)

        # 2. Four-stream Venn pass -> logits for each task.
        out = self.model.venn_forward(
            clean_input_ids=batch.clean_input_ids,
            corr_a_input_ids=batch.corr_a_input_ids,
            corr_b_input_ids=batch.corr_b_input_ids,
            attention_mask=batch.attention_mask,
        )

        # 3. Faithfulness + correctness for each task.
        kl_a, margin_a = self._task_losses(
            out.logits_a, golden, batch.target_a, batch.distractor_a, batch
        )
        kl_b, margin_b = self._task_losses(
            out.logits_b, golden, batch.target_b, batch.distractor_b, batch
        )

        raw_a = kl_a + margin_a
        raw_b = kl_b + margin_b

        # 4. Homoscedastic uncertainty scaling.
        scaled_a, scaled_b = self.uncertainty(raw_a, raw_b)

        # 5. Dynamic lambdas from the dual PID scheduler (driven by measured KL).
        lambdas = self.scheduler.step(kl_a.item(), kl_b.item())

        # 6. Venn sparsity loss with region-specific dynamic lambdas.
        regions = self.model.venn_sparsity()
        sparsity = (
            lambdas["lambda_a"] * regions["a_only"]
            + lambdas["lambda_b"] * regions["b_only"]
            + lambdas["lambda_core"] * regions["core"]
        )

        # 7. Total objective and backprop.
        total = scaled_a + scaled_b + sparsity
        total.backward()

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for grp in self.optimizer.param_groups for p in grp["params"]],
                self.config.grad_clip,
            )
        self.optimizer.step()

        metrics = StepMetrics(
            step=step_idx,
            kl_a=kl_a.item(),
            kl_b=kl_b.item(),
            margin_a=margin_a.item(),
            margin_b=margin_b.item(),
            scaled_a=scaled_a.item(),
            scaled_b=scaled_b.item(),
            sparsity=float(sparsity.item()),
            total=float(total.item()),
            **lambdas,
        )
        self.history.append(metrics)
        return metrics

    def train(self, data: Iterable[VennBatch], epochs: int = 1) -> List[StepMetrics]:
        """Iterate over ``data`` for ``epochs`` epochs, returning all metrics."""
        step_idx = 0
        for epoch in range(epochs):
            for batch in data:
                metrics = self.step(batch, step_idx)
                if self.config.log_every and step_idx % self.config.log_every == 0:
                    self._log(epoch, metrics)
                step_idx += 1
        return self.history

    def _log(self, epoch: int, m: StepMetrics) -> None:
        print(
            f"[ep {epoch} | step {m.step}] "
            f"KL_a={m.kl_a:.4f} KL_b={m.kl_b:.4f} "
            f"margin_a={m.margin_a:.3f} margin_b={m.margin_b:.3f} | "
            f"λa={m.lambda_a:.3f} λb={m.lambda_b:.3f} λcore={m.lambda_core:.3f} | "
            f"sparsity={m.sparsity:.4f} total={m.total:.4f}"
        )
