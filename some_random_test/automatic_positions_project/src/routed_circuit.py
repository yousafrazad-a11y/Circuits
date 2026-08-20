"""Isolated adapter that soft-routes the existing section-aware node circuit."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch

from comparison_experiments.position_aware_node_pruning.models.l0 import HardConcreteGate
import comparison_experiments.position_aware_node_pruning.models.gpt2_circuit as circuit_module


_ROUTING_WEIGHTS: ContextVar[torch.Tensor | None] = ContextVar(
    "automatic_position_routing_weights", default=None
)
_ORIGINAL_SECTION_GATE = circuit_module._section_gate


def _routed_section_gate(
    gate: HardConcreteGate,
    section_ids: torch.LongTensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = _ROUTING_WEIGHTS.get()
    if weights is None:
        return _ORIGINAL_SECTION_GATE(gate, section_ids, dtype)

    experts = gate().to(dtype)
    if experts.shape[0] != weights.shape[-1]:
        raise ValueError(
            f"Router has {weights.shape[-1]} experts but gate has "
            f"{experts.shape[0]} section rows."
        )
    # [batch, token, expert] x [expert, components...] ->
    # [batch, token, components...]. This is differentiable in the router.
    return torch.tensordot(weights.to(dtype), experts, dims=([-1], [0]))


def install_routing_adapter() -> None:
    """Install once; ordinary integer-section calls retain their exact behavior."""
    if circuit_module._section_gate is not _routed_section_gate:
        circuit_module._section_gate = _routed_section_gate


@contextmanager
def routed_by(weights: torch.Tensor) -> Iterator[None]:
    """Make all gates in one circuit forward use the supplied token routing."""
    token = _ROUTING_WEIGHTS.set(weights)
    try:
        yield
    finally:
        _ROUTING_WEIGHTS.reset(token)


def routed_forward(model, batch: dict, routing_weights: torch.Tensor):
    """Run the unchanged circuit with a differentiable mixture of expert rows."""
    install_routing_adapter()
    # The base implementation validates an integer [batch, token] tensor. The
    # adapter ignores its values while routing_weights is active.
    shape_proxy = torch.zeros_like(batch["input_ids"], dtype=torch.long)
    with routed_by(routing_weights):
        return model(
            input_ids=batch["input_ids"],
            corrupted_input_ids=batch["corrupted_input_ids"],
            attention_mask=batch["attention_mask"],
            corrupted_attention_mask=batch["corrupted_attention_mask"],
            section_ids=shape_proxy,
            return_dict=True,
        )
