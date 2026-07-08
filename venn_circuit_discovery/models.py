"""Llama patching for multi-task Venn circuit discovery.

The wrapper :class:`LlamaVennCircuit` intercepts every attention and MLP module
of a HuggingFace Llama model (validated for the 1B / 8B / 32B variants) and runs
**four** activation streams through the network in a single forward pass:

* ``stream_a``   -- the gated circuit for task A (starts from the clean input).
* ``stream_b``   -- the gated circuit for task B (starts from the clean input).
* ``corr_a``     -- the pure corrupted-A reference (never gated).
* ``corr_b``     -- the pure corrupted-B reference (never gated).

At every gated location a :class:`VennConcreteGate` produces two masks. ``mask_a``
mixes ``stream_a`` with ``corr_a`` and ``mask_b`` mixes ``stream_b`` with
``corr_b`` (mask ``1`` == keep clean/circuit, ``0`` == ablate to corrupted). The
final LM head is applied to ``stream_a`` and ``stream_b`` to yield ``logits_a``
and ``logits_b``.

Only the shared clean stream and the Venn gates carry gradients; the corrupted
reference streams are detached at the input so no gradient leaks through them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from .gates import VennConcreteGate

# Base sparsity weights per gate granularity. The dynamic per-task lambdas from
# the scheduler multiply *on top* of these fixed structural weights.
_PRUNING_FACTOR = 5.0


@dataclass
class VennPruningConfig:
    """Which granularities to gate, and their fixed base sparsity weights."""

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.01 * _PRUNING_FACTOR

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.002 * _PRUNING_FACTOR

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 0.005 * _PRUNING_FACTOR

    prune_mlp_output: bool = True
    lambda_mlp_output: float = 0.005 * _PRUNING_FACTOR

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.02 * _PRUNING_FACTOR

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.02 * _PRUNING_FACTOR

    prune_full_layers: bool = True
    lambda_full_layers: float = 0.05 * _PRUNING_FACTOR


def _venn_mix(
    gate: VennConcreteGate,
    stream_a: torch.Tensor,
    stream_b: torch.Tensor,
    corr_a: torch.Tensor,
    corr_b: torch.Tensor,
    view_shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mix each circuit stream with its corrupted reference using Venn masks.

    ``mask_a``/``mask_b`` are broadcast to ``view_shape`` and applied as
    ``mask * clean + (1 - mask) * corrupted``.
    """
    mask_a, mask_b = gate()
    ma = mask_a.to(stream_a.dtype).view(view_shape)
    mb = mask_b.to(stream_b.dtype).view(view_shape)
    mixed_a = ma * stream_a + (1.0 - ma) * corr_a
    mixed_b = mb * stream_b + (1.0 - mb) * corr_b
    return mixed_a, mixed_b


# ======================================================================
# Attention
# ======================================================================


class VennAttention(nn.Module):
    """Wraps ``LlamaAttention`` with per-head and per-neuron Venn gates."""

    def __init__(
        self,
        original: LlamaAttention,
        config: LlamaConfig,
        pruning: VennPruningConfig,
    ) -> None:
        super().__init__()
        self.attn = original
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(original, "head_dim", config.hidden_size // self.num_heads)

        self.head_gates = (
            VennConcreteGate(self.num_heads) if pruning.prune_attention_heads else None
        )
        self.neuron_gates = (
            VennConcreteGate(self.num_heads * self.head_dim)
            if pruning.prune_attention_neurons
            else None
        )

    def _pre_oproj(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run the original attention but capture the output *before* ``o_proj``.

        Temporarily swapping ``o_proj`` for an identity guarantees we reuse the
        exact SDPA/flash kernels of the base model while still gating per head.
        """
        saved = self.attn.o_proj
        self.attn.o_proj = nn.Identity()
        try:
            out, _ = self.attn(hidden_states, **kwargs)
        finally:
            self.attn.o_proj = saved
        return out

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        corr_a: torch.Tensor,
        corr_b: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = a.shape[:2]
        hshape = (bsz, seq_len, self.num_heads, self.head_dim)

        a_h = self._pre_oproj(a, **kwargs).view(hshape)
        b_h = self._pre_oproj(b, **kwargs).view(hshape)
        ca_h = self._pre_oproj(corr_a, **kwargs).view(hshape)
        cb_h = self._pre_oproj(corr_b, **kwargs).view(hshape)

        if self.head_gates is not None:
            a_h, b_h = _venn_mix(self.head_gates, a_h, b_h, ca_h, cb_h, (1, 1, self.num_heads, 1))

        if self.neuron_gates is not None:
            a_h, b_h = _venn_mix(
                self.neuron_gates, a_h, b_h, ca_h, cb_h, (1, 1, self.num_heads, self.head_dim)
            )

        # Flatten heads back to hidden and apply the (shared, frozen) output proj.
        flat = (bsz, seq_len, self.num_heads * self.head_dim)
        o = self.attn.o_proj
        return (
            o(a_h.reshape(flat)),
            o(b_h.reshape(flat)),
            o(ca_h.reshape(flat)),
            o(cb_h.reshape(flat)),
        )

    def venn_gates(self) -> Dict[str, VennConcreteGate]:
        gates = {}
        if self.head_gates is not None:
            gates["attention_heads"] = self.head_gates
        if self.neuron_gates is not None:
            gates["attention_neurons"] = self.neuron_gates
        return gates


# ======================================================================
# MLP (SwiGLU)
# ======================================================================


class VennMLP(nn.Module):
    """Wraps ``LlamaMLP`` (SwiGLU) with intermediate and output Venn gates."""

    def __init__(
        self,
        original: LlamaMLP,
        config: LlamaConfig,
        pruning: VennPruningConfig,
    ) -> None:
        super().__init__()
        self.mlp = original
        self.hidden_gates = (
            VennConcreteGate(config.intermediate_size) if pruning.prune_mlp_hidden else None
        )
        self.output_gates = (
            VennConcreteGate(config.hidden_size) if pruning.prune_mlp_output else None
        )

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation: ``silu(gate_proj(x)) * up_proj(x)``."""
        m = self.mlp
        return m.act_fn(m.gate_proj(x)) * m.up_proj(x)

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        corr_a: torch.Tensor,
        corr_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        act_a = self._activation(a)
        act_b = self._activation(b)
        act_ca = self._activation(corr_a)
        act_cb = self._activation(corr_b)

        if self.hidden_gates is not None:
            act_a, act_b = _venn_mix(self.hidden_gates, act_a, act_b, act_ca, act_cb, (1, 1, -1))

        down = self.mlp.down_proj
        out_a, out_b = down(act_a), down(act_b)
        out_ca, out_cb = down(act_ca), down(act_cb)

        if self.output_gates is not None:
            out_a, out_b = _venn_mix(self.output_gates, out_a, out_b, out_ca, out_cb, (1, 1, -1))

        return out_a, out_b, out_ca, out_cb

    def venn_gates(self) -> Dict[str, VennConcreteGate]:
        gates = {}
        if self.hidden_gates is not None:
            gates["mlp_hidden"] = self.hidden_gates
        if self.output_gates is not None:
            gates["mlp_output"] = self.output_gates
        return gates


# ======================================================================
# Decoder layer
# ======================================================================


class VennDecoderLayer(nn.Module):
    """Wraps ``LlamaDecoderLayer`` and threads all four streams through it."""

    def __init__(
        self,
        original: LlamaDecoderLayer,
        config: LlamaConfig,
        pruning: VennPruningConfig,
    ) -> None:
        super().__init__()
        self.input_layernorm = original.input_layernorm
        self.post_attention_layernorm = original.post_attention_layernorm

        self.attn = VennAttention(original.self_attn, config, pruning)
        self.mlp = VennMLP(original.mlp, config, pruning)

        self.attn_block_gate = VennConcreteGate(1) if pruning.prune_attention_blocks else None
        self.mlp_block_gate = VennConcreteGate(1) if pruning.prune_mlp_blocks else None

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        corr_a: torch.Tensor,
        corr_b: torch.Tensor,
        **attn_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # --- Attention sub-block (pre-norm residual) ---
        ln = self.input_layernorm
        att_a, att_b, att_ca, att_cb = self.attn(
            ln(a), ln(b), ln(corr_a), ln(corr_b), **attn_kwargs
        )

        if self.attn_block_gate is not None:
            att_a, att_b = _venn_mix(self.attn_block_gate, att_a, att_b, att_ca, att_cb, (1, 1, 1))

        a, b = a + att_a, b + att_b
        corr_a, corr_b = corr_a + att_ca, corr_b + att_cb

        # --- MLP sub-block (pre-norm residual) ---
        pn = self.post_attention_layernorm
        mlp_a, mlp_b, mlp_ca, mlp_cb = self.mlp(pn(a), pn(b), pn(corr_a), pn(corr_b))

        if self.mlp_block_gate is not None:
            mlp_a, mlp_b = _venn_mix(self.mlp_block_gate, mlp_a, mlp_b, mlp_ca, mlp_cb, (1, 1, 1))

        a, b = a + mlp_a, b + mlp_b
        corr_a, corr_b = corr_a + mlp_ca, corr_b + mlp_cb
        return a, b, corr_a, corr_b

    def clean_forward(self, hidden: torch.Tensor, **attn_kwargs) -> torch.Tensor:
        """Ungated single-stream pass using the frozen base sub-modules.

        Numerically equivalent to the original ``LlamaDecoderLayer`` and used to
        produce the golden (clean) logits without routing through the gates.
        """
        residual = hidden
        attn_out, _ = self.attn.attn(self.input_layernorm(hidden), **attn_kwargs)
        hidden = residual + attn_out

        residual = hidden
        hidden = residual + self.mlp.mlp(self.post_attention_layernorm(hidden))
        return hidden

    def venn_gates(self) -> Dict[str, VennConcreteGate]:
        gates: Dict[str, VennConcreteGate] = {}
        gates.update(self.attn.venn_gates())
        gates.update(self.mlp.venn_gates())
        if self.attn_block_gate is not None:
            gates["attention_blocks"] = self.attn_block_gate
        if self.mlp_block_gate is not None:
            gates["mlp_blocks"] = self.mlp_block_gate
        return gates


# ======================================================================
# Full model
# ======================================================================


@dataclass
class VennForwardOutput:
    """Container for the dual-task logits produced by a Venn forward pass."""

    logits_a: torch.Tensor
    logits_b: torch.Tensor


class LlamaVennCircuit(LlamaForCausalLM):
    """``LlamaForCausalLM`` extended with four-stream Venn circuit discovery."""

    @classmethod
    def from_pretrained_with_venn(
        cls,
        model_name: str,
        pruning_config: Optional[VennPruningConfig] = None,
        **kwargs,
    ) -> "LlamaVennCircuit":
        """Load a pretrained Llama (1B/8B/32B) and attach Venn gates.

        The base weights are frozen; only the Venn gate logits are trainable.
        """
        pruning_config = pruning_config or VennPruningConfig()
        model = cls.from_pretrained(model_name, **kwargs)

        model.model.layers = nn.ModuleList(
            [VennDecoderLayer(layer, model.config, pruning_config) for layer in model.model.layers]
        )

        model.layer_gates = (
            nn.ModuleList([VennConcreteGate(1) for _ in model.model.layers])
            if pruning_config.prune_full_layers
            else None
        )

        model.pruning_config = pruning_config

        # Freeze every base parameter; keep only VennConcreteGate logits trainable.
        for param in model.parameters():
            param.requires_grad_(False)
        for gate in model.iter_venn_gates().values():
            for p in gate.parameters():
                p.requires_grad_(True)

        return model

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _build_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.rotary_emb(hidden_states, position_ids)

    def _build_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if hasattr(self.model, "_update_causal_mask"):
            return self.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, None, False
            )
        # Fallback additive causal mask for older transformers releases.
        seq_len = inputs_embeds.size(1)
        device = inputs_embeds.device
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )
        return causal[None, None].to(inputs_embeds.dtype)

    def venn_forward(
        self,
        clean_input_ids: torch.LongTensor,
        corr_a_input_ids: torch.LongTensor,
        corr_b_input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> VennForwardOutput:
        """Run the four-stream pass and return ``logits_a`` / ``logits_b``."""
        embed = self.model.embed_tokens
        clean_embeds = embed(clean_input_ids)
        # Corrupted references never receive gradients.
        corr_a_embeds = embed(corr_a_input_ids).detach()
        corr_b_embeds = embed(corr_b_input_ids).detach()

        # Both circuit streams start from the (shared) clean activations.
        a = clean_embeds
        b = clean_embeds.clone()
        corr_a = corr_a_embeds.clone()
        corr_b = corr_b_embeds.clone()

        bsz, seq_len = clean_embeds.shape[:2]
        device = clean_embeds.device
        cache_position = torch.arange(seq_len, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self._build_position_embeddings(a, position_ids)
        causal_mask = self._build_causal_mask(attention_mask, clean_embeds, cache_position)

        attn_kwargs = dict(
            attention_mask=causal_mask,
            position_embeddings=position_embeddings,
            past_key_value=None,
            cache_position=cache_position,
        )

        for i, layer in enumerate(self.model.layers):
            a, b, corr_a, corr_b = layer(a, b, corr_a, corr_b, **attn_kwargs)

            if self.layer_gates is not None:
                a, b = _venn_mix(self.layer_gates[i], a, b, corr_a, corr_b, (1, 1, 1))

        norm = self.model.norm
        logits_a = self.lm_head(norm(a))
        logits_b = self.lm_head(norm(b))
        return VennForwardOutput(logits_a=logits_a, logits_b=logits_b)

    @torch.no_grad()
    def golden_logits(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Clean-model logits (the faithfulness target).

        Runs a single ungated stream through the frozen base sub-modules, so the
        result matches the unpatched Llama exactly and never touches the gates.
        """
        hidden = self.model.embed_tokens(input_ids)
        seq_len = hidden.size(1)
        device = hidden.device

        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)
        attn_kwargs = dict(
            attention_mask=self._build_causal_mask(attention_mask, hidden, cache_position),
            position_embeddings=self._build_position_embeddings(hidden, position_ids),
            past_key_value=None,
            cache_position=cache_position,
        )

        for layer in self.model.layers:
            hidden = layer.clean_forward(hidden, **attn_kwargs)
        return self.lm_head(self.model.norm(hidden))

    # ------------------------------------------------------------------
    # Gate bookkeeping
    # ------------------------------------------------------------------

    def iter_venn_gates(self) -> Dict[str, VennConcreteGate]:
        """Return every Venn gate keyed by a unique dotted name."""
        gates: Dict[str, VennConcreteGate] = {}
        if getattr(self, "layer_gates", None) is not None:
            for i, gate in enumerate(self.layer_gates):
                gates[f"layer.{i}.full_layers"] = gate
        for i, layer in enumerate(self.model.layers):
            for name, gate in layer.venn_gates().items():
                gates[f"layer.{i}.{name}"] = gate
        return gates

    def _base_lambda(self, gate_name: str) -> float:
        cfg = self.pruning_config
        # gate_name looks like 'layer.<i>.<group>'.
        group = gate_name.split(".")[-1]
        return {
            "attention_heads": cfg.lambda_attention_heads,
            "attention_neurons": cfg.lambda_attention_neurons,
            "mlp_hidden": cfg.lambda_mlp_hidden,
            "mlp_output": cfg.lambda_mlp_output,
            "attention_blocks": cfg.lambda_attention_blocks,
            "mlp_blocks": cfg.lambda_mlp_blocks,
            "full_layers": cfg.lambda_full_layers,
        }[group]

    def venn_sparsity(self) -> Dict[str, torch.Tensor]:
        """Aggregate base-lambda-weighted expected-L0 for each Venn region.

        Returns a dict with keys ``core`` / ``a_only`` / ``b_only`` -- scalar
        tensors that the trainer multiplies by the dynamic scheduler lambdas.
        """
        totals: Dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros((), device=self.device)
        )
        for name, gate in self.iter_venn_gates().items():
            weight = self._base_lambda(name)
            for region, value in gate.sparsity().items():
                totals[region] = totals[region] + weight * value
        return dict(totals)

    def gate_group_sizes(self) -> Dict[str, int]:
        sizes: Dict[str, int] = defaultdict(int)
        for name, gate in self.iter_venn_gates().items():
            sizes[name.split(".")[-1]] += gate.num_gates()
        return dict(sizes)
