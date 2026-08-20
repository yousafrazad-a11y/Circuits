"""GPT-2 with hard-concrete node gates shared within logical prompt sections."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP

from .l0 import HardConcreteGate


@dataclass
class PruningConfig:
    num_sections: int = 8
    gate_init_min: float = 2.5
    gate_init_max: float = 3.5
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.1

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.8
    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.15

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0
    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.5
    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.5

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0
    prune_embedding: bool = False
    lambda_embedding: float = 25.0

    def __post_init__(self) -> None:
        if self.num_sections <= 0:
            raise ValueError("num_sections must be positive.")


def _new_gate(pruning_config: PruningConfig, *component_shape: int) -> HardConcreteGate:
    return HardConcreteGate(
        (pruning_config.num_sections, *component_shape),
        init_min=pruning_config.gate_init_min,
        init_max=pruning_config.gate_init_max,
    )


def _section_gate(
    gate: HardConcreteGate,
    section_ids: torch.LongTensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gather one row of a section gate for every token in a batch."""
    if section_ids.dtype != torch.long:
        section_ids = section_ids.long()
    return gate()[section_ids].to(dtype)


class PrunableGPT2Attention(nn.Module):
    def __init__(
        self,
        original_attention: GPT2Attention,
        gpt_config: GPT2Config,
        pruning_config: PruningConfig,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.num_heads = gpt_config.num_attention_heads
        self.head_dim = gpt_config.hidden_size // self.num_heads

        self.head_gates = (
            _new_gate(pruning_config, self.num_heads)
            if pruning_config.prune_attention_heads
            else None
        )
        self.neuron_gates = (
            _new_gate(pruning_config, self.num_heads, self.head_dim)
            if pruning_config.prune_attention_neurons
            else None
        )

    def _forward_pre_projection(self, hidden_states: torch.Tensor, **kwargs):
        attention = self.original_attention
        saved_projection = attention.c_proj
        saved_dropout = attention.resid_dropout
        attention.c_proj = nn.Identity()
        attention.resid_dropout = nn.Identity()
        try:
            pre_projection, attention_weights = attention(hidden_states, **kwargs)
        finally:
            attention.c_proj = saved_projection
            attention.resid_dropout = saved_dropout
        return pre_projection, attention_weights

    def forward(
        self,
        clean_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
        section_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if corrupted_states is None:
            return self.original_attention(clean_states, **kwargs)
        if section_ids is None:
            raise ValueError("section_ids are required for a position-aware pruning run.")

        clean_pre, clean_weights = self._forward_pre_projection(clean_states, **kwargs)
        corrupted_kwargs = dict(kwargs)
        corrupted_kwargs["past_key_values"] = None
        corrupted_pre, _ = self._forward_pre_projection(
            corrupted_states, **corrupted_kwargs
        )

        batch_size, sequence_length = clean_pre.shape[:2]
        clean_heads = clean_pre.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        corrupted_heads = corrupted_pre.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )

        gated_heads = clean_heads
        if self.head_gates is not None:
            gate = _section_gate(self.head_gates, section_ids, clean_heads.dtype)
            gated_heads = gate.unsqueeze(-1) * gated_heads + (
                1.0 - gate.unsqueeze(-1)
            ) * corrupted_heads
        if self.neuron_gates is not None:
            gate = _section_gate(self.neuron_gates, section_ids, clean_heads.dtype)
            gated_heads = gate * gated_heads + (1.0 - gate) * corrupted_heads

        gated_flat = gated_heads.reshape(batch_size, sequence_length, -1).contiguous()
        corrupted_flat = corrupted_heads.reshape(
            batch_size, sequence_length, -1
        ).contiguous()
        attention = self.original_attention
        gated_output = attention.resid_dropout(attention.c_proj(gated_flat))
        corrupted_output = attention.resid_dropout(attention.c_proj(corrupted_flat))
        return (gated_output, clean_weights), corrupted_output


class PrunableGPT2MLP(nn.Module):
    def __init__(
        self,
        original_mlp: GPT2MLP,
        gpt_config: GPT2Config,
        pruning_config: PruningConfig,
    ):
        super().__init__()
        self.original_mlp = original_mlp
        intermediate_size = gpt_config.n_inner or 4 * gpt_config.hidden_size
        self.hidden_gates = (
            _new_gate(pruning_config, intermediate_size)
            if pruning_config.prune_mlp_hidden
            else None
        )
        self.output_gates = (
            _new_gate(pruning_config, gpt_config.hidden_size)
            if pruning_config.prune_mlp_output
            else None
        )

    def forward(
        self,
        clean_states: torch.Tensor,
        corrupted_states: Optional[torch.Tensor] = None,
        section_ids: Optional[torch.LongTensor] = None,
    ):
        mlp = self.original_mlp
        if corrupted_states is None:
            return mlp(clean_states)
        if section_ids is None:
            raise ValueError("section_ids are required for a position-aware pruning run.")

        clean_activation = mlp.act(mlp.c_fc(clean_states))
        corrupted_activation = mlp.act(mlp.c_fc(corrupted_states))
        gated_activation = clean_activation
        if self.hidden_gates is not None:
            gate = _section_gate(
                self.hidden_gates, section_ids, clean_activation.dtype
            )
            gated_activation = gate * clean_activation + (
                1.0 - gate
            ) * corrupted_activation

        gated_output = mlp.dropout(mlp.c_proj(gated_activation))
        corrupted_output = mlp.dropout(mlp.c_proj(corrupted_activation))
        if self.output_gates is not None:
            gate = _section_gate(self.output_gates, section_ids, gated_output.dtype)
            gated_output = gate * gated_output + (1.0 - gate) * corrupted_output
        return gated_output, corrupted_output


class PrunableGPT2Block(nn.Module):
    def __init__(
        self,
        original_block: GPT2Block,
        gpt_config: GPT2Config,
        pruning_config: PruningConfig,
    ):
        super().__init__()
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2
        self.attn = PrunableGPT2Attention(
            original_block.attn, gpt_config, pruning_config
        )
        self.mlp = PrunableGPT2MLP(original_block.mlp, gpt_config, pruning_config)
        self.attention_block_gate = (
            _new_gate(pruning_config)
            if pruning_config.prune_attention_blocks
            else None
        )
        self.mlp_block_gate = (
            _new_gate(pruning_config) if pruning_config.prune_mlp_blocks else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        corrupted_states: Optional[torch.Tensor] = None,
        section_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        attention_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            **kwargs,
        }

        if corrupted_states is None:
            residual = hidden_states
            attention_output, _ = self.attn(self.ln_1(hidden_states), **attention_kwargs)
            hidden_states = residual + attention_output
            residual = hidden_states
            return residual + self.mlp(self.ln_2(hidden_states))
        if section_ids is None:
            raise ValueError("section_ids are required for a position-aware pruning run.")

        clean_states = hidden_states
        attention_outputs, corrupted_attention_output = self.attn(
            self.ln_1(clean_states),
            self.ln_1(corrupted_states),
            section_ids=section_ids,
            **attention_kwargs,
        )
        attention_output = attention_outputs[0]
        if self.attention_block_gate is not None:
            gate = _section_gate(
                self.attention_block_gate, section_ids, attention_output.dtype
            ).unsqueeze(-1)
            attention_output = gate * attention_output + (
                1.0 - gate
            ) * corrupted_attention_output

        clean_states = clean_states + attention_output
        corrupted_states = corrupted_states + corrupted_attention_output

        mlp_output, corrupted_mlp_output = self.mlp(
            self.ln_2(clean_states),
            self.ln_2(corrupted_states),
            section_ids=section_ids,
        )
        if self.mlp_block_gate is not None:
            gate = _section_gate(
                self.mlp_block_gate, section_ids, mlp_output.dtype
            ).unsqueeze(-1)
            mlp_output = gate * mlp_output + (1.0 - gate) * corrupted_mlp_output

        return (
            clean_states + mlp_output,
            corrupted_states + corrupted_mlp_output,
            attention_outputs,
        )


class PrunableGPT2LMHeadModel(GPT2LMHeadModel):
    """GPT-2 whose node masks are indexed by logical prompt section."""

    supports_position_aware_pruning = True

    @classmethod
    def from_pretrained_with_pruning(
        cls, model_name: str, pruning_config: PruningConfig, **kwargs
    ) -> "PrunableGPT2LMHeadModel":
        model = cls.from_pretrained(model_name, **kwargs)
        model.embedding_gate = (
            _new_gate(pruning_config) if pruning_config.prune_embedding else None
        )
        model.transformer.h = nn.ModuleList(
            [
                PrunableGPT2Block(block, model.config, pruning_config)
                for block in model.transformer.h
            ]
        )
        model.layer_gates = (
            nn.ModuleList(
                [_new_gate(pruning_config) for _ in range(len(model.transformer.h))]
            )
            if pruning_config.prune_full_layers
            else None
        )
        model.pruning_config = pruning_config
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        corrupted_input_ids: Optional[torch.LongTensor] = None,
        section_ids: Optional[torch.LongTensor] = None,
        corrupted_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        corrupted_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        pruning_run = (
            corrupted_input_ids is not None or corrupted_inputs_embeds is not None
        )
        if not pruning_run:
            return super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                **kwargs,
            )
        if section_ids is None:
            raise ValueError("section_ids must be supplied with corrupted inputs.")
        if corrupted_attention_mask is not None and attention_mask is not None:
            if not torch.equal(corrupted_attention_mask, attention_mask):
                raise ValueError(
                    "Clean and corrupted attention masks must match section by section."
                )

        transformer = self.transformer
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            raise ValueError("Specify input_ids or inputs_embeds for the clean stream.")

        if section_ids.shape != input_shape:
            raise ValueError(
                f"section_ids shape {tuple(section_ids.shape)} does not match "
                f"input shape {tuple(input_shape)}."
            )
        if section_ids.min().item() < 0 or section_ids.max().item() >= self.pruning_config.num_sections:
            raise ValueError("section_ids contain an out-of-range logical section.")

        if inputs_embeds is None:
            inputs_embeds = transformer.wte(input_ids)
        if corrupted_inputs_embeds is None:
            if corrupted_input_ids is None:
                raise ValueError("Specify corrupted_input_ids or corrupted_inputs_embeds.")
            corrupted_inputs_embeds = transformer.wte(corrupted_input_ids)
        if corrupted_inputs_embeds.shape != inputs_embeds.shape:
            raise ValueError("Clean and corrupted embeddings must have identical shapes.")
        corrupted_inputs_embeds = corrupted_inputs_embeds.detach()

        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0)
        position_embeddings = transformer.wpe(position_ids).to(inputs_embeds.device)
        clean_states = inputs_embeds + position_embeddings
        corrupted_states = corrupted_inputs_embeds + position_embeddings

        if self.embedding_gate is not None:
            gate = _section_gate(
                self.embedding_gate, section_ids, clean_states.dtype
            ).unsqueeze(-1)
            clean_states = gate * clean_states + (1.0 - gate) * corrupted_states

        if token_type_ids is not None:
            token_type_embeddings = transformer.wte(
                token_type_ids.view(-1, input_shape[-1])
            )
            clean_states = clean_states + token_type_embeddings
            corrupted_states = corrupted_states + token_type_embeddings

        clean_states = transformer.drop(clean_states)
        corrupted_states = transformer.drop(corrupted_states)
        output_shape = (-1,) + input_shape[1:] + (clean_states.size(-1),)

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for layer_index, block in enumerate(transformer.h):
            if output_hidden_states:
                all_hidden_states += (clean_states,)
            clean_states, corrupted_states, attention_outputs = block(
                clean_states,
                attention_mask=causal_mask,
                use_cache=False,
                corrupted_states=corrupted_states,
                section_ids=section_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            if self.layer_gates is not None:
                gate = _section_gate(
                    self.layer_gates[layer_index], section_ids, clean_states.dtype
                ).unsqueeze(-1)
                clean_states = gate * clean_states + (1.0 - gate) * corrupted_states
            if output_attentions and attention_outputs is not None:
                all_attentions += (attention_outputs[1],)

        clean_states = transformer.ln_f(clean_states).view(output_shape)
        if output_hidden_states:
            all_hidden_states += (clean_states,)
        logits = self.lm_head(clean_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits, None, all_hidden_states, all_attentions, None)
            return ((loss,) + output) if loss is not None else output
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def set_final_circuit_mode(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, HardConcreteGate):
                module.set_final_mode(enabled)

    def gate_group_sizes(self) -> Dict[str, int]:
        sizes: defaultdict[str, int] = defaultdict(int)
        if self.embedding_gate is not None:
            sizes["embedding"] += self.embedding_gate.num_gates()
        if self.layer_gates is not None:
            for gate in self.layer_gates:
                sizes["full_layers"] += gate.num_gates()
        for block in self.transformer.h:
            for key, gate in (
                ("attention_blocks", block.attention_block_gate),
                ("mlp_blocks", block.mlp_block_gate),
                ("attention_heads", block.attn.head_gates),
                ("attention_neurons", block.attn.neuron_gates),
                ("mlp_hidden", block.mlp.hidden_gates),
                ("mlp_output", block.mlp.output_gates),
            ):
                if gate is not None:
                    sizes[key] += gate.num_gates()
        return dict(sizes)

    def get_sparsity_loss(self, step: int = 0) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)
        term_count = 0
        warmup = min(
            1.0,
            step / self.pruning_config.sparsity_warmup_steps
            if self.pruning_config.sparsity_warmup_steps > 0
            else 1.0,
        )

        def add(
            key: str,
            gate: Optional[HardConcreteGate],
            coefficient: float,
            layer_index: Optional[int] = None,
        ) -> None:
            nonlocal total_loss, term_count
            if gate is None:
                return
            depth_multiplier = 1.0
            if layer_index is not None:
                layer_count = len(self.transformer.h)
                early_layer_fraction = (layer_count - 1 - layer_index) / max(
                    1, layer_count - 1
                )
                depth_multiplier += (
                    self.pruning_config.depth_penalty_scaling * early_layer_fraction
                )
            value = (
                coefficient
                * warmup
                * depth_multiplier
                * gate.get_sparsity_loss()
            )
            losses[key] = losses.get(key, torch.tensor(0.0, device=self.device)) + value
            total_loss = total_loss + value
            term_count += 1

        add("embedding", self.embedding_gate, self.pruning_config.lambda_embedding)
        if self.layer_gates is not None:
            for index, gate in enumerate(self.layer_gates):
                add(
                    "full_layers",
                    gate,
                    self.pruning_config.lambda_full_layers,
                    index,
                )
        for index, block in enumerate(self.transformer.h):
            add(
                "attention_blocks",
                block.attention_block_gate,
                self.pruning_config.lambda_attention_blocks,
                index,
            )
            add(
                "mlp_blocks",
                block.mlp_block_gate,
                self.pruning_config.lambda_mlp_blocks,
                index,
            )
            add(
                "attention_heads",
                block.attn.head_gates,
                self.pruning_config.lambda_attention_heads,
                index,
            )
            add(
                "attention_neurons",
                block.attn.neuron_gates,
                self.pruning_config.lambda_attention_neurons,
                index,
            )
            add(
                "mlp_hidden",
                block.mlp.hidden_gates,
                self.pruning_config.lambda_mlp_hidden,
                index,
            )
            add(
                "mlp_output",
                block.mlp.output_gates,
                self.pruning_config.lambda_mlp_output,
                index,
            )
        losses["total_sparsity"] = total_loss / max(term_count, 1)
        return losses
