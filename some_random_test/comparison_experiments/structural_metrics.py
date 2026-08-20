"""Comparable, explicitly typed structural metrics for circuit methods."""

from __future__ import annotations

from collections import Counter

import torch


def _counts(total: float, active: float) -> dict[str, float | int]:
    closed = total - active
    integral = float(total).is_integer() and float(active).is_integer()
    cast = int if integral else float
    return {
        "total": cast(total),
        "active": cast(active),
        "closed": cast(closed),
        "percent_active": 100.0 * active / total if total else 0.0,
        "percent_pruned": 100.0 * closed / total if total else 0.0,
    }


def _effective_high_level_masks(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hierarchy-finalized [section, layer, component] masks."""
    sections = model.pruning_config.num_sections
    heads = model.config.n_head
    head_rows = []
    mlp_rows = []
    for block in model.transformer.h:
        if block.attn.head_gates is None:
            head = torch.ones(sections, heads, dtype=torch.bool)
        else:
            head = block.attn.head_gates.hard_mask().detach().cpu().bool()
        if block.mlp_block_gate is None:
            mlp = torch.ones(sections, dtype=torch.bool)
        else:
            mlp = block.mlp_block_gate.hard_mask().detach().cpu().bool()
        head_rows.append(head)
        mlp_rows.append(mlp)
    return torch.stack(head_rows, dim=1), torch.stack(mlp_rows, dim=1)


def _peap_compatible_full_edge_count(sections: int, layers: int, heads: int) -> int:
    """Full abstract graph size under the argo dense-survivor convention.

    This follows ``circuit_pruning-argo/edge_pruning/utils.py::count_dense_edges``:
    every surviving upstream component is densely connected to every valid
    downstream component.  The two PEAP adaptations are (1) collapsing Q/K/V
    to one abstract source-to-head edge and (2) adding causal attention
    transport edges between logical sections.
    """
    components = layers * (heads + 1)
    # Per section: residual input -> components, earlier layers -> later
    # components, and same-layer heads -> MLP. Only the final section is read
    # by the next-token output.
    causal_component_edges = 0
    for source_layer in range(layers):
        # Each head and the MLP feed every head/MLP in later layers.
        causal_component_edges += (heads + 1) * (layers - source_layer - 1) * (heads + 1)
        # Same-layer head outputs feed that layer's MLP.
        causal_component_edges += heads
    local_edges = sections * (components + causal_component_edges) + components
    transport_edges = layers * heads * sections * (sections + 1) // 2
    return local_edges + transport_edges


def _high_level_edge_report(model, section_names) -> dict:
    """Count node-induced edges on PEAP's abstract head/MLP graph.

    Q/K/V are collapsed to one abstract source-to-head connection, as in
    ``eval_utils.get_edge_score``. Residual input/output endpoints are always
    open. Position-aware heads additionally have causal attention-transport
    edges for every source/destination section pair; such an edge is active
    only when that head is open at both positions.
    """
    head_open, mlp_open = _effective_high_level_masks(model)
    sections, layers, heads = head_open.shape

    local_total = 0
    local_active = 0
    per_section = {}
    for section in range(sections):
        nodes = []
        # Each layer contributes simultaneous head outputs followed by its MLP.
        for layer in range(layers):
            nodes.extend(("h", layer, head) for head in range(heads))
            nodes.append(("m", layer, None))

        def active(node):
            kind, layer, index = node
            return bool(head_open[section, layer, index]) if kind == "h" else bool(mlp_open[section, layer])

        section_total = len(nodes)  # residual input -> every component
        section_active = sum(active(node) for node in nodes)
        # A head may feed its same-layer MLP; otherwise only earlier layers feed
        # later components. Heads in one layer do not directly feed one another.
        for source in nodes:
            for target in nodes:
                source_kind, source_layer, _ = source
                target_kind, target_layer, _ = target
                allowed = source_layer < target_layer or (
                    source_kind == "h" and target_kind == "m" and source_layer == target_layer
                )
                if allowed:
                    section_total += 1
                    section_active += int(active(source) and active(target))
        # Only the final logical position is read by the IOI next-token output.
        if section == sections - 1:
            section_total += len(nodes)
            section_active += sum(active(node) for node in nodes)
        local_total += section_total
        local_active += section_active
        label = section_names[section] if section < len(section_names) else f"section_{section}"
        per_section[label] = _counts(section_total, section_active)

    # AttnIn(position s) -> AttnOut(position t), s <= t. This includes the
    # within-position head transport edge and all causal cross-position edges.
    transport_total = layers * heads * sections * (sections + 1) // 2
    transport_active = 0
    cross_total = 0
    cross_active = 0
    for source_section in range(sections):
        for destination_section in range(source_section, sections):
            count = int((head_open[source_section] & head_open[destination_section]).sum())
            transport_active += count
            if source_section != destination_section:
                cross_total += layers * heads
                cross_active += count

    total = local_total + transport_total
    active = local_active + transport_active
    return {
        "unit": "peap_abstract_directed_edge_qkv_collapsed",
        "derivation": "circuit_pruning-argo dense-survivor edge counting, adapted to PEAP positions and QKV collapse",
        "all_high_level_edges": _counts(total, active),
        "residual_and_component_edges": _counts(local_total, local_active),
        "attention_transport_edges": _counts(transport_total, transport_active),
        "cross_position_attention_edges": _counts(cross_total, cross_active),
        "residual_policy": "Residual endpoints are fixed open; an incident edge closes when its gated head or MLP endpoint closes.",
        "output_policy": "Only the final logical section connects to the next-token output.",
        "by_source_section_for_non_crossing_edges": per_section,
    }


def node_structural_report(model, analysis: dict) -> dict:
    """Convert hierarchy-finalized node analysis into percentages.

    Every count is an effective count: closed layers close both blocks, closed
    blocks close their children, and closed heads close their attention neurons.
    Categories are not summed because a head and its neurons are nested units.
    """
    section_count = model.pruning_config.num_sections
    by_category = {
        name: _counts(total, active)
        for name, (total, active) in analysis["category_totals"].items()
        if total
    }
    per_section = {}
    names = analysis["section_names"]
    for section_index, active_counts in enumerate(analysis["section_reports"]):
        name = names[section_index] if section_index < len(names) else f"section_{section_index}"
        per_section[name] = {
            category: _counts(total // section_count, active_counts[category])
            for category, (total, _) in analysis["category_totals"].items()
            if total
        }

    cfg = model.config
    layers = len(model.transformer.h)
    head_count = cfg.n_head
    hidden = cfg.hidden_size
    intermediate = cfg.n_inner or 4 * hidden
    layer_reports = analysis["layer_reports"]
    physical_union = {
        "layers": _counts(layers, sum(r["layer_active_anywhere"] for r in layer_reports)),
        "attention_blocks": _counts(layers, sum(r["attention_block_active_anywhere"] for r in layer_reports)),
        "mlp_blocks": _counts(layers, sum(r["mlp_block_active_anywhere"] for r in layer_reports)),
        "attention_heads": _counts(layers * head_count, sum(r["attention_heads_union"] for r in layer_reports)),
        "attention_neurons": _counts(layers * hidden, sum(r["attention_neurons_union"] for r in layer_reports)),
        "mlp_hidden": _counts(layers * intermediate, sum(r["mlp_hidden_union"] for r in layer_reports)),
        "mlp_output": _counts(layers * hidden, sum(r["mlp_output_union"] for r in layer_reports)),
    }
    proxy = analysis["compression_proxy"]
    # The position-aware model owns more scalar gate parameters than the global
    # model. Exclude all gates so both proxies use the same dense GPT-2 base.
    gate_parameters = sum(total for total, _ in analysis["category_totals"].values())
    base_model_parameters = proxy["total_model_params"] - gate_parameters
    fixed_parameters = base_model_parameters - proxy["total_prunable_params"]
    effective_base_parameters = (
        fixed_parameters + proxy["active_prunable_params_union_proxy"]
    )
    parameter_proxy = {
        "prunable_parameters": _counts(
            proxy["total_prunable_params"],
            proxy["active_prunable_params_union_proxy"],
        ),
        "whole_model_effective_parameters": _counts(
            base_model_parameters, effective_base_parameters
        ),
        "excluded_gate_parameters": gate_parameters,
        "note": (
            "Union-based extraction proxy over base GPT-2 parameters; learned gate "
            "parameters are excluded and the evaluated PyTorch model remains dense."
        ),
    }
    return {
        "unit": "hierarchical_node_gate",
        "num_position_sections": section_count,
        "effective_gate_slots_by_granularity": by_category,
        "effective_gate_slots_by_section": per_section,
        "physical_component_union_across_sections": physical_union,
        "parameter_pruning_proxy": parameter_proxy,
        "high_level_edge_pruning": _high_level_edge_report(model, names),
        "aggregation_warning": "Granularities are nested and must not be added together.",
    }


def peap_structural_report(graph, concrete_active: float, concrete_full: float, full_nodes: float) -> dict:
    """Report PEAP in its native unit: directed computation edges."""
    edge_types = Counter()
    crossing = Counter()
    for downstream, upstream in graph.edges:
        edge_types[f"{upstream.node_type}_to_{downstream.node_type}"] += 1
        is_crossing = (
            downstream.node_type == "h"
            and "result" in downstream.get_input_name()
            and downstream.span_idx != upstream.span_idx
        )
        crossing["crossing_attention" if is_crossing else "same_position_or_component"] += 1
    node_types = Counter(node.node_type for node in graph.nodes)
    sample_node = next(iter(graph.nodes), None)
    if sample_node is not None and hasattr(sample_node, "model_cfg") and hasattr(sample_node, "exp"):
        layers = int(sample_node.model_cfg.n_layers)
        heads = int(sample_node.model_cfg.n_heads)
        sections = len(sample_node.exp.spans) - 1
        abstract_full = _peap_compatible_full_edge_count(sections, layers, heads)
        abstract_selected = len(graph.edges)
        argo_comparison = {
            "unit": "peap_abstract_directed_edge_qkv_collapsed",
            "num_position_sections": sections,
            "selected_edges_against_full_graph": _counts(abstract_full, abstract_selected),
            "derivation": (
                "circuit_pruning-argo dense-survivor edge counting adapted by "
                "collapsing Q/K/V and adding causal logical-position transport"
            ),
        }
    else:
        argo_comparison = None
    return {
        "unit": "directed_computation_edge",
        "concrete_token_level_edges": _counts(concrete_full, concrete_active),
        "full_graph_mean_nodes": float(full_nodes),
        "selected_abstract_graph": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "nodes_by_type": dict(sorted(node_types.items())),
            "edges_by_upstream_to_downstream_type": dict(sorted(edge_types.items())),
            "edges_by_position_relation": dict(sorted(crossing.items())),
        },
        "argo_style_peap_compatible_comparison": argo_comparison,
        "comparison_warning": (
            "PEAP prunes edges, not heads/MLPs/neurons. Concrete edge sparsity is "
            "comparable behaviorally but is not the same structural unit as node sparsity."
        ),
    }
