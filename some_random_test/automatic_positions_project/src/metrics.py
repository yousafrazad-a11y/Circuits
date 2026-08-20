"""Router-specific diagnostics that complement the shared circuit metrics."""

from __future__ import annotations

import torch


@torch.no_grad()
def routing_diagnostics(
    probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_section_ids: torch.Tensor | None = None,
) -> dict:
    valid = attention_mask.bool()
    selected = probabilities.argmax(dim=-1)
    valid_probabilities = probabilities[valid]
    valid_selected = selected[valid]
    count = max(int(valid.sum().item()), 1)

    entropy = -(
        valid_probabilities.clamp_min(1e-12)
        * valid_probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    usage = torch.bincount(
        valid_selected, minlength=probabilities.shape[-1]
    ).to(torch.float32) / count
    result = {
        "assignment_entropy": float(entropy.mean().item()),
        "expert_usage": usage.cpu().tolist(),
        "experts_used": int((usage > 0).sum().item()),
    }
    if hidden_section_ids is not None:
        # Expert labels are arbitrary. Report raw agreement only as an easy
        # diagnostic; the saved assignments permit permutation-aware analysis.
        result["raw_section_agreement"] = float(
            (valid_selected == hidden_section_ids[valid]).float().mean().item()
        )
    return result
