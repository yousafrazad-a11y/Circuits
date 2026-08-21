"""Router-specific diagnostics that complement the shared circuit metrics."""

from __future__ import annotations

import torch


@torch.no_grad()
def routing_diagnostics(
    probabilities: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_section_ids: torch.Tensor | None = None,
    selected_assignments: torch.Tensor | None = None,
    hidden_section_count: int | None = None,
) -> dict:
    """Return soft confidence and hard assignment statistics.

    Probabilities must be the pre-hardening distribution; passing one-hot
    weights would make entropy identically zero. Selected assignments may
    contain either integer expert IDs or one-hot circuit routing weights.
    """
    valid = attention_mask.bool()
    if selected_assignments is None:
        selected = probabilities.argmax(dim=-1)
    elif selected_assignments.ndim == probabilities.ndim:
        selected = selected_assignments.argmax(dim=-1)
    else:
        selected = selected_assignments.long()
    valid_probabilities = probabilities[valid]
    valid_selected = selected[valid]
    count = max(int(valid.sum().item()), 1)

    entropy = -(
        valid_probabilities.clamp_min(1e-12)
        * valid_probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    expert_counts = torch.bincount(
        valid_selected, minlength=probabilities.shape[-1]
    )
    usage = expert_counts.to(torch.float32) / count
    top_two = valid_probabilities.topk(k=2, dim=-1).values
    result = {
        "assignment_entropy": float(entropy.mean().item()),
        "mean_max_probability": float(valid_probabilities.max(dim=-1).values.mean().item()),
        "mean_top2_margin": float((top_two[:, 0] - top_two[:, 1]).mean().item()),
        "expert_usage": usage.cpu().tolist(),
        "expert_counts": expert_counts.cpu().tolist(),
        "experts_used": int((usage > 0).sum().item()),
        "valid_token_count": count,
    }
    if hidden_section_ids is not None:
        result["raw_section_agreement"] = float(
            (valid_selected == hidden_section_ids[valid]).float().mean().item()
        )
        expert_count = probabilities.shape[-1]
        section_count = hidden_section_count or int(hidden_section_ids[valid].max().item() + 1)
        pairs = hidden_section_ids[valid].long() * expert_count + valid_selected
        result["section_to_expert_counts"] = torch.bincount(
            pairs, minlength=section_count * expert_count
        ).view(section_count, expert_count).cpu().tolist()
    return result
