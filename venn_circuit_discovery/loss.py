"""Task losses and Homoscedastic Uncertainty weighting.

Two ingredients are provided:

* Raw per-task objectives:
    - :func:`kl_divergence_loss` (faithfulness -- match the golden logits).
    - :func:`margin_loss`        (correctness -- keep target above distractor).
* :class:`HomoscedasticUncertaintyLoss` -- learns to balance the two tasks by
  treating each task loss as the negative log-likelihood of a Gaussian with a
  learnable observation noise ``sigma`` (Kendall et al., 2018).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence_loss(
    student_logits: torch.Tensor,
    golden_logits: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token KL divergence ``KL(student || golden)`` used as faithfulness.

    Args:
        student_logits: Logits from a gated (circuit) stream, ``(B, T, V)``.
        golden_logits: Logits from the clean stream (the target distribution),
            ``(B, T, V)``. Should be detached by the caller.
        position_mask: Optional ``(B, T)`` boolean/float mask selecting the
            positions that contribute to the loss (e.g. answer positions).

    Returns:
        A scalar tensor: mean KL over the selected tokens.
    """
    log_p_student = F.log_softmax(student_logits.float(), dim=-1)
    log_p_golden = F.log_softmax(golden_logits.float(), dim=-1)

    # Per-token KL: sum over vocabulary.
    kl_per_token = F.kl_div(
        log_p_student,
        log_p_golden,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)  # (B, T)

    if position_mask is not None:
        mask = position_mask.to(kl_per_token.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (kl_per_token * mask).sum() / denom
    return kl_per_token.mean()


def margin_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    distractor_tokens: torch.Tensor,
    answer_positions: torch.Tensor,
    margin: float = 4.0,
) -> torch.Tensor:
    """Hinge margin loss keeping the target logit above the distractor logit.

    Args:
        logits: ``(B, T, V)`` logits from the circuit stream.
        target_tokens: ``(B,)`` id of the correct next token.
        distractor_tokens: ``(B,)`` id of the competing token.
        answer_positions: ``(B,)`` position (index into ``T``) at which the
            prediction is read out.
        margin: Desired logit gap between target and distractor.

    Returns:
        Scalar hinge loss ``mean(relu(margin - (logit_target - logit_distractor)))``.
    """
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    logit_at_pos = logits[batch_idx, answer_positions].float()  # (B, V)
    logit_target = logit_at_pos[batch_idx, target_tokens]
    logit_distractor = logit_at_pos[batch_idx, distractor_tokens]
    return F.relu(margin - (logit_target - logit_distractor)).mean()


class HomoscedasticUncertaintyLoss(nn.Module):
    """Homoscedastic uncertainty weighting for the two tasks.

    Registers two learnable log-variances (``log_sigma_a`` / ``log_sigma_b``,
    both initialised to ``0.0``). Given raw task losses it returns the scaled,
    automatically-balanced losses::

        sigma        = exp(log_sigma)
        scaled_loss  = raw_loss / (2 * sigma ** 2) + log_sigma

    The ``+ log_sigma`` term regularises the noise so it cannot grow without
    bound to trivially minimise the loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_sigma_a = nn.Parameter(torch.zeros(()))
        self.log_sigma_b = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        raw_loss_a: torch.Tensor,
        raw_loss_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scaled_loss_a, scaled_loss_b)``."""
        sigma_a = torch.exp(self.log_sigma_a)
        sigma_b = torch.exp(self.log_sigma_b)

        scaled_loss_a = raw_loss_a / (2 * sigma_a**2) + self.log_sigma_a
        scaled_loss_b = raw_loss_b / (2 * sigma_b**2) + self.log_sigma_b
        return scaled_loss_a, scaled_loss_b
