"""One mathematically shared metric implementation for all circuit backends."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class CircuitMetricAccumulator:
    """Accumulate IOI metrics from answer-position logits.

    Inputs have shape ``[batch, vocab]``. KL is D_KL(full || circuit),
    matching PyTorch's ``kl_div`` argument order.
    """

    correct: float = 0.0
    pairwise_correct: float = 0.0
    logit_diff_sum: float = 0.0
    full_logit_diff_sum: float = 0.0
    kl_sum: float = 0.0
    exact: float = 0.0
    samples: int = 0
    circuit_predictions: list[int] = field(default_factory=list)
    full_predictions: list[int] = field(default_factory=list)

    @torch.no_grad()
    def update(self, circuit_logits, full_logits, correct_ids, wrong_ids) -> None:
        if circuit_logits.ndim == 1:
            circuit_logits = circuit_logits.unsqueeze(0)
        if full_logits.ndim == 1:
            full_logits = full_logits.unsqueeze(0)
        correct_ids = correct_ids.reshape(-1).to(circuit_logits.device)
        wrong_ids = wrong_ids.reshape(-1).to(circuit_logits.device)
        if circuit_logits.shape != full_logits.shape:
            raise ValueError("Circuit and full-model logits must have equal shapes.")
        if circuit_logits.shape[0] != correct_ids.numel():
            raise ValueError("One correct/wrong token ID is required per example.")

        rows = torch.arange(circuit_logits.shape[0], device=circuit_logits.device)
        circuit_diff = circuit_logits[rows, correct_ids] - circuit_logits[rows, wrong_ids]
        full_diff = full_logits[rows, correct_ids] - full_logits[rows, wrong_ids]
        circuit_pred = circuit_logits.argmax(dim=-1)
        full_pred = full_logits.argmax(dim=-1)
        self.correct += (circuit_pred == correct_ids).sum().item()
        self.pairwise_correct += (circuit_diff >= 0).sum().item()
        self.logit_diff_sum += circuit_diff.sum().item()
        self.full_logit_diff_sum += full_diff.sum().item()
        self.kl_sum += F.kl_div(
            F.log_softmax(circuit_logits, dim=-1),
            F.log_softmax(full_logits, dim=-1),
            reduction="sum", log_target=True,
        ).item()
        self.exact += (circuit_pred == full_pred).sum().item()
        self.samples += circuit_logits.shape[0]
        self.circuit_predictions.extend(circuit_pred.detach().cpu().tolist())
        self.full_predictions.extend(full_pred.detach().cpu().tolist())

    def compute(self) -> dict[str, float | int]:
        count = max(self.samples, 1)
        circuit_diff = self.logit_diff_sum / count
        full_diff = self.full_logit_diff_sum / count
        return {
            "n_examples": self.samples,
            "accuracy": self.correct / count,
            "pairwise_accuracy": self.pairwise_correct / count,
            "logit_diff": circuit_diff,
            "full_model_logit_diff": full_diff,
            "soft_faithfulness": circuit_diff / full_diff if abs(full_diff) > 1e-12 else 0.0,
            "kl_div": self.kl_sum / count,
            "exact_match": self.exact / count,
        }
