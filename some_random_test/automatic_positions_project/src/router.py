"""Small token routers for selecting frozen position-specific circuit masks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ROUTER_INPUTS = (
    "normalized_position",
    "token_embedding",
    "early_residual",
    "early_residual_plus_position",
)


class TokenRouter(nn.Module):
    """Map per-token features to a distribution over frozen mask experts."""

    def __init__(self, input_size: int, hidden_size: int, num_experts: int):
        super().__init__()
        normalization = nn.Identity() if input_size == 1 else nn.LayerNorm(input_size)
        self.network = nn.Sequential(
            normalization,
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_experts),
        )

    def forward(
        self,
        features: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.network(features)
        if hard:
            # Straight-through hard assignment: discrete forward, soft gradient.
            weights = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        else:
            weights = F.softmax(logits / temperature, dim=-1)
        return weights, logits


def normalized_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return positions in [0, 1], normalized independently per prompt."""
    mask = attention_mask.to(torch.float32)
    indices = torch.arange(mask.shape[1], device=mask.device, dtype=mask.dtype)
    denominators = (mask.sum(dim=1, keepdim=True) - 1.0).clamp_min(1.0)
    positions = indices.unsqueeze(0) / denominators
    return (positions * mask).unsqueeze(-1)


@torch.no_grad()
def extract_frozen_features(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_kind: str,
    residual_layer: int = 2,
) -> torch.Tensor:
    """Extract router evidence without allowing gradients into GPT-2."""
    if input_kind not in ROUTER_INPUTS:
        raise ValueError(f"Unknown router input {input_kind!r}.")

    position = normalized_positions(attention_mask)
    if input_kind == "normalized_position":
        return position
    if input_kind == "token_embedding":
        return model.transformer.wte(input_ids).detach()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if residual_layer < 0 or residual_layer >= len(outputs.hidden_states):
        raise ValueError(
            f"residual_layer={residual_layer} is outside the available "
            f"0..{len(outputs.hidden_states) - 1} range."
        )
    residual = outputs.hidden_states[residual_layer].detach()
    if input_kind == "early_residual":
        return residual
    return torch.cat((residual, position.to(residual.dtype)), dim=-1)


def router_input_size(input_kind: str, model_hidden_size: int) -> int:
    if input_kind == "normalized_position":
        return 1
    if input_kind in {"token_embedding", "early_residual"}:
        return model_hidden_size
    if input_kind == "early_residual_plus_position":
        return model_hidden_size + 1
    raise ValueError(f"Unknown router input {input_kind!r}.")
