"""Finalization and reporting for logical-section node circuits."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .dataset.ioi import SECTION_NAMES
from .models.l0 import HardConcreteGate


def disable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def _mask(gate: Optional[HardConcreteGate], fallback_shape: tuple[int, ...]) -> torch.Tensor:
    if gate is None:
        return torch.ones(fallback_shape, dtype=torch.bool)
    return gate.hard_mask().detach().cpu()


def _close(gate: Optional[HardConcreteGate], index) -> None:
    if gate is not None:
        gate.log_alpha[index] = -1e6


def _estimate_union_compression(model, layer_reports: list[dict]) -> dict:
    """Estimate extractable parameters using the union of active sections.

    This is a structural proxy.  The current PyTorch model remains dense and still
    executes both streams; no physical tensors are rewritten by finalization.
    """
    config = model.config
    hidden_size = config.hidden_size
    intermediate_size = config.n_inner or 4 * hidden_size
    total_attention = (
        hidden_size * 3 * hidden_size
        + 3 * hidden_size
        + hidden_size * hidden_size
        + hidden_size
    )
    total_mlp = (
        hidden_size * intermediate_size
        + intermediate_size
        + intermediate_size * hidden_size
        + hidden_size
    )
    total_prunable = len(layer_reports) * (total_attention + total_mlp)
    active_prunable = 0

    for report in layer_reports:
        if not report["layer_active_anywhere"]:
            continue
        if report["attention_block_active_anywhere"]:
            active_dimensions = report["attention_neurons_union"]
            active_prunable += (
                hidden_size * 3 * active_dimensions
                + 3 * active_dimensions
                + active_dimensions * hidden_size
                + hidden_size
            )
        if report["mlp_block_active_anywhere"]:
            active_hidden = report["mlp_hidden_union"]
            active_output = report["mlp_output_union"]
            active_prunable += (
                hidden_size * active_hidden
                + active_hidden
                + active_hidden * active_output
                + active_output
            )

    total_model = sum(parameter.numel() for parameter in model.parameters())
    fixed = total_model - total_prunable
    effective = fixed + active_prunable
    return {
        "total_prunable_params": total_prunable,
        "active_prunable_params_union_proxy": active_prunable,
        "prunable_compression_proxy": (
            total_prunable / active_prunable if active_prunable else float("inf")
        ),
        "total_model_params": total_model,
        "effective_model_params_proxy": effective,
        "effective_compression_proxy": (
            total_model / effective if effective else float("inf")
        ),
    }


def analyze_and_finalize_circuit(model: nn.Module, verbose: bool = True) -> dict:
    """Harden gates and enforce hierarchy independently in every logical section."""
    model.eval()
    model.set_final_circuit_mode(True)

    section_count = model.pruning_config.num_sections
    config = model.config
    layer_count = len(model.transformer.h)
    head_count = config.n_head
    head_dimension = config.hidden_size // head_count
    hidden_size = config.hidden_size
    intermediate_size = config.n_inner or 4 * hidden_size

    with torch.no_grad():
        for layer_index, block in enumerate(model.transformer.h):
            layer_gate = (
                model.layer_gates[layer_index]
                if model.layer_gates is not None
                else None
            )
            layer_open = _mask(layer_gate, (section_count,))

            for section in range(section_count):
                if not layer_open[section]:
                    _close(block.attention_block_gate, section)
                    _close(block.mlp_block_gate, section)
                    _close(block.attn.head_gates, section)
                    _close(block.attn.neuron_gates, section)
                    _close(block.mlp.hidden_gates, section)
                    _close(block.mlp.output_gates, section)

            attention_block_open = _mask(
                block.attention_block_gate, (section_count,)
            )
            mlp_block_open = _mask(block.mlp_block_gate, (section_count,))
            for section in range(section_count):
                if not attention_block_open[section]:
                    _close(block.attn.head_gates, section)
                    _close(block.attn.neuron_gates, section)
                if not mlp_block_open[section]:
                    _close(block.mlp.hidden_gates, section)
                    _close(block.mlp.output_gates, section)

            if block.attn.head_gates is not None and block.attn.neuron_gates is not None:
                head_open = block.attn.head_gates.hard_mask()
                neuron_open = block.attn.neuron_gates.hard_mask().view(
                    section_count, head_count, head_dimension
                )
                block.attn.neuron_gates.log_alpha[~head_open] = -1e6
                neuron_open = block.attn.neuron_gates.hard_mask().view(
                    section_count, head_count, head_dimension
                )
                heads_without_neurons = ~neuron_open.any(dim=-1)
                block.attn.head_gates.log_alpha[heads_without_neurons] = -1e6

            if block.attention_block_gate is not None:
                if block.attn.head_gates is not None:
                    empty_attention_sections = ~block.attn.head_gates.hard_mask().any(dim=-1)
                elif block.attn.neuron_gates is not None:
                    empty_attention_sections = ~block.attn.neuron_gates.hard_mask().view(
                        section_count, -1
                    ).any(dim=-1)
                else:
                    empty_attention_sections = torch.zeros(
                        section_count, dtype=torch.bool, device=block.attention_block_gate.log_alpha.device
                    )
                block.attention_block_gate.log_alpha[empty_attention_sections] = -1e6

            if block.mlp_block_gate is not None:
                hidden_empty = (
                    ~block.mlp.hidden_gates.hard_mask().any(dim=-1)
                    if block.mlp.hidden_gates is not None
                    else torch.zeros(section_count, dtype=torch.bool)
                )
                output_empty = (
                    ~block.mlp.output_gates.hard_mask().any(dim=-1)
                    if block.mlp.output_gates is not None
                    else torch.zeros(section_count, dtype=torch.bool)
                )
                empty_mlp_sections = hidden_empty.to(block.mlp_block_gate.log_alpha.device) | output_empty.to(
                    block.mlp_block_gate.log_alpha.device
                )
                block.mlp_block_gate.log_alpha[empty_mlp_sections] = -1e6
                _close(block.mlp.hidden_gates, empty_mlp_sections)
                _close(block.mlp.output_gates, empty_mlp_sections)

            if layer_gate is not None:
                attention_closed = (
                    ~block.attention_block_gate.hard_mask()
                    if block.attention_block_gate is not None
                    else torch.zeros(section_count, dtype=torch.bool)
                )
                mlp_closed = (
                    ~block.mlp_block_gate.hard_mask()
                    if block.mlp_block_gate is not None
                    else torch.zeros(section_count, dtype=torch.bool)
                )
                empty_layer_sections = attention_closed.to(layer_gate.log_alpha.device) & mlp_closed.to(
                    layer_gate.log_alpha.device
                )
                layer_gate.log_alpha[empty_layer_sections] = -1e6

    category_totals = {
        "embedding": [0, 0],
        "layers": [0, 0],
        "attention_blocks": [0, 0],
        "mlp_blocks": [0, 0],
        "attention_heads": [0, 0],
        "attention_neurons": [0, 0],
        "mlp_hidden": [0, 0],
        "mlp_output": [0, 0],
    }
    section_reports = [
        {key: 0 for key in category_totals} for _ in range(section_count)
    ]
    layer_reports: list[dict] = []

    embedding_mask = _mask(model.embedding_gate, (section_count,))
    if model.embedding_gate is not None:
        category_totals["embedding"] = [section_count, int(embedding_mask.sum())]
        for section in range(section_count):
            section_reports[section]["embedding"] = int(embedding_mask[section])

    for layer_index, block in enumerate(model.transformer.h):
        layer_mask = _mask(
            model.layer_gates[layer_index] if model.layer_gates is not None else None,
            (section_count,),
        )
        attention_block_mask = _mask(
            block.attention_block_gate, (section_count,)
        )
        mlp_block_mask = _mask(block.mlp_block_gate, (section_count,))
        head_mask = _mask(block.attn.head_gates, (section_count, head_count))
        attention_neuron_mask = _mask(
            block.attn.neuron_gates, (section_count, head_count, head_dimension)
        )
        mlp_hidden_mask = _mask(
            block.mlp.hidden_gates, (section_count, intermediate_size)
        )
        mlp_output_mask = _mask(
            block.mlp.output_gates, (section_count, hidden_size)
        )

        masks_by_category = {
            "layers": (layer_mask, model.layer_gates is not None),
            "attention_blocks": (
                attention_block_mask,
                block.attention_block_gate is not None,
            ),
            "mlp_blocks": (mlp_block_mask, block.mlp_block_gate is not None),
            "attention_heads": (head_mask, block.attn.head_gates is not None),
            "attention_neurons": (
                attention_neuron_mask,
                block.attn.neuron_gates is not None,
            ),
            "mlp_hidden": (mlp_hidden_mask, block.mlp.hidden_gates is not None),
            "mlp_output": (mlp_output_mask, block.mlp.output_gates is not None),
        }
        for category, (mask, is_configured) in masks_by_category.items():
            if not is_configured:
                continue
            category_totals[category][0] += mask.numel()
            category_totals[category][1] += int(mask.sum())
            flattened = mask.reshape(section_count, -1)
            for section in range(section_count):
                section_reports[section][category] += int(flattened[section].sum())

        layer_reports.append(
            {
                "layer": layer_index,
                "layer_active_anywhere": bool(layer_mask.any()),
                "attention_block_active_anywhere": bool(attention_block_mask.any()),
                "mlp_block_active_anywhere": bool(mlp_block_mask.any()),
                "attention_heads_union": int(head_mask.any(dim=0).sum()),
                "attention_neurons_union": int(
                    attention_neuron_mask.reshape(section_count, -1).any(dim=0).sum()
                ),
                "mlp_hidden_union": int(mlp_hidden_mask.any(dim=0).sum()),
                "mlp_output_union": int(mlp_output_mask.any(dim=0).sum()),
            }
        )

    compression = _estimate_union_compression(model, layer_reports)
    if verbose:
        print("\nPOSITION-AWARE CIRCUIT BY LOGICAL SECTION")
        for section, report in enumerate(section_reports):
            section_name = (
                SECTION_NAMES[section]
                if section < len(SECTION_NAMES)
                else f"section_{section}"
            )
            print(
                f"  {section:>2} {section_name:<22} "
                f"heads={report['attention_heads']:>4} "
                f"attn_dims={report['attention_neurons']:>5} "
                f"mlp_hidden={report['mlp_hidden']:>6} "
                f"mlp_out={report['mlp_output']:>5}"
            )

        print("\nPOSITION-SPECIFIC GATE SLOTS")
        for category, (total, active) in category_totals.items():
            if total:
                print(
                    f"  {category:<22} {active:>8,}/{total:<8,} "
                    f"({active / total:>7.2%} active)"
                )

        print("\nUNION OF PHYSICAL COMPONENTS ACROSS SECTIONS")
        for report in layer_reports:
            print(
                f"  layer {report['layer']:>2}: "
                f"heads={report['attention_heads_union']:>2}/{head_count}, "
                f"attn_dims={report['attention_neurons_union']:>3}/{hidden_size}, "
                f"mlp_hidden={report['mlp_hidden_union']:>4}/{intermediate_size}, "
                f"mlp_out={report['mlp_output_union']:>3}/{hidden_size}"
            )
        print(f"\nCompression proxy: {compression}")
        print(
            "Note: masks are applied dynamically; reported compression is a union-based "
            "extraction estimate, not an actual reduction of the current dense model."
        )

    return {
        "section_names": SECTION_NAMES,
        "section_reports": section_reports,
        "category_totals": category_totals,
        "layer_reports": layer_reports,
        "compression_proxy": compression,
    }


def save_gate_state(model: nn.Module, output_path: str) -> None:
    """Save only learned gate tensors rather than duplicating all GPT-2 weights."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    gate_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "gate" in name
    }
    torch.save(
        {
            "pruning_config": vars(model.pruning_config),
            "gate_state": gate_state,
            "section_names": SECTION_NAMES,
        },
        destination,
    )
