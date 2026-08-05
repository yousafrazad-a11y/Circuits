import os
import sys
import dataclasses
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, LlamaForCausalLM
from typing import Dict, Any, Optional

# Add the local circuit_pruning-argo to path to import from it
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from models.l0 import HardConcreteGate

class CircuitPruningManager:
    """
    A single-access point wrapper for circuit pruning capabilities.
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = None
        self.baseline_model = None

    def _get_default_config(self) -> PruningConfig:
        """
        Full-depth pruning at fine granularity: attention heads, attention
        neurons, MLP hidden and MLP output gates enabled. Coarse levels
        (attention blocks, MLP blocks, full layers) are DISABLED - their
        scalar per-layer gates destabilize training (runs either collapse
        to a fully-pruned model or barely prune at all).
        Default lambda ratios kept; use --lambda_sparsity to scale together.
        """
        config = PruningConfig()
        config.prune_attention_heads = True
        config.prune_mlp_hidden = True
        config.prune_mlp_output = True
        config.prune_attention_neurons = True
        config.prune_attention_blocks = False
        config.prune_mlp_blocks = False
        config.prune_full_layers = False
        # lambdas and other params keep their default PruningConfig values
        return config

    def initialize_model(self, config: Optional[PruningConfig] = None):
        """Initialize both the prunable model and the baseline model."""
        if config is None:
            config = self._get_default_config()
            
        print("Initializing Prunable Llama...")
        self.model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
            self.model_name, pruning_config=config, torch_dtype=torch.bfloat16
        ).to(self.device)
        
        print("Initializing Baseline Llama (for KL divergence)...")
        self.baseline_model = LlamaForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.baseline_model.eval()
        for param in self.baseline_model.parameters():
            param.requires_grad = False
            
        # Only gates require grad in the prunable model
        GATE_PATTERNS = ('_gates.', '_gate.', 'layer_gates.', 'log_alpha')
        for name, param in self.model.named_parameters():
            is_gate = any(p in name for p in GATE_PATTERNS)
            param.requires_grad = is_gate
            if is_gate:
                param.data = param.data.float()

    def set_global_sparsity_lambda(self, value: float):
        """
        Set the sparsity lambda globally across all granularity levels.

        Anchored on lambda_attention_heads: every level's lambda is scaled by
        the same factor (value / current lambda_attention_heads), preserving
        the relative weighting between levels (e.g. heads 0.05 : full layers 0.25).
        Example: with default config, value=0.10 doubles every level's lambda.
        """
        if self.model is None:
            raise ValueError("Model not initialized.")

        cfg = self.model.pruning_config
        scale = value / cfg.lambda_attention_heads
        for f in dataclasses.fields(cfg):
            if f.name.startswith("lambda_"):
                setattr(cfg, f.name, getattr(cfg, f.name) * scale)
        print(f"Global sparsity lambda set (x{scale:.3g} on all levels): "
              f"attention_heads={cfg.lambda_attention_heads:.4g}")

    def train_masks(self, dataloader, epochs: int = 10, lr: float = 0.05, config: Optional[PruningConfig] = None):
        """
        Train the pruning masks on the provided dataloader.
        Expects dataloader to yield batches with:
        - 'input_ids'
        - 'corrupted_input_ids'
        - 'attention_mask'
        """
        if self.model is None:
            self.initialize_model(config)
            
        optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model.train()
        
        total_steps = 0
        for epoch in range(epochs):
            epoch_kl = 0
            epoch_sparsity = 0
            
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Get baseline logits
                with torch.no_grad():
                    baseline_out = self.baseline_model(
                        input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        use_cache=False
                    )
                    baseline_logits = baseline_out.logits.detach()
                
                # Get circuit logits
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    corrupted_input_ids=batch['corrupted_input_ids'],
                    attention_mask=batch['attention_mask'],
                    use_cache=False
                )
                
                # Compute KL divergence over all unpadded tokens
                mask = batch['attention_mask'].bool()
                
                circuit_logits = outputs.logits[mask]
                golden_logits = baseline_logits[mask]
                
                kl_loss = F.kl_div(
                    F.log_softmax(circuit_logits.float(), dim=-1),
                    F.log_softmax(golden_logits.float(), dim=-1),
                    reduction='batchmean',
                    log_target=True
                )
                
                # Sparsity loss
                sparsity_losses = self.model.get_sparsity_loss(step=total_steps)
                sparsity_loss = sparsity_losses['total_sparsity']
                
                loss = kl_loss + sparsity_loss
                loss.backward()
                optimizer.step()
                
                total_steps += 1
                epoch_kl += kl_loss.item()
                epoch_sparsity += sparsity_loss.item()
                
                # Clamp log_alpha (frozen gates are re-pinned instead)
                with torch.no_grad():
                    for module in self.model.modules():
                        if isinstance(module, HardConcreteGate) and hasattr(module, 'log_alpha'):
                            if hasattr(module, 'frozen_off'):
                                keep = ~module.frozen_off
                                module.log_alpha.data[keep] = module.log_alpha.data[keep].clamp(-5.0, 5.0)
                                # Guarantee frozen gates stay off despite weight decay / optimizer drift
                                module.log_alpha.data[module.frozen_off] = -1e6
                            else:
                                module.log_alpha.clamp_(-5.0, 5.0)
                            
            print(f"Epoch {epoch+1}/{epochs} | KL Loss: {epoch_kl/len(dataloader):.4f} | Sparsity Loss: {epoch_sparsity/len(dataloader):.4f}")

    def save_checkpoint(self, save_path: str):
        """Save the continuous float log_alpha states for resuming training."""
        if self.model is None:
            raise ValueError("Model not initialized.")
            
        checkpoint_state = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, HardConcreteGate) and hasattr(module, 'log_alpha'):
                    checkpoint_state[name] = module.log_alpha.data.cpu()
                    
        torch.save(checkpoint_state, save_path)
        print(f"Training checkpoint saved to {save_path}")
        
    def load_checkpoint(self, load_path: str, config: Optional[PruningConfig] = None):
        """Load continuous float log_alpha states to resume training."""
        if self.model is None:
            self.initialize_model(config)
            
        checkpoint_state = torch.load(load_path, weights_only=True)
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, HardConcreteGate) and hasattr(module, 'log_alpha'):
                    if name in checkpoint_state:
                        module.log_alpha.data = checkpoint_state[name].to(self.device)
        print(f"Training checkpoint loaded from {load_path}")

    def save_masks(self, save_path: str):
        """Save the trained masks to a file."""
        if self.model is None:
            raise ValueError("Model not initialized.")
            
        self.model.eval()
        mask_state = {}
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, HardConcreteGate) and hasattr(module, 'log_alpha'):
                    # Stretched sigmoid to get boolean masks
                    s = torch.sigmoid(module.log_alpha)
                    s_stretched = s * 1.2 - 0.1
                    mask_state[name] = (s_stretched > 0.5).bool().cpu()
                    
        torch.save(mask_state, save_path)
        print(f"Masks saved to {save_path}")

    def load_masks(self, load_path: str, config: Optional[PruningConfig] = None):
        """Load trained masks from a file and apply them to the model."""
        if self.model is None:
            self.initialize_model(config)
            
        mask_state = torch.load(load_path, weights_only=True)
        self.use_model(enable_masks=True) # Ensure it's in final circuit mode
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, HardConcreteGate) and hasattr(module, 'log_alpha'):
                    if name in mask_state:
                        module.log_alpha.data = torch.where(
                            mask_state[name].to(self.device),
                            torch.tensor(5.0, device=self.device),
                            torch.tensor(-1e6, device=self.device)
                        )
                    else:
                        module.log_alpha.data.fill_(-1e6)
        print(f"Masks loaded from {load_path}")

    def load_masks_for_finetuning(self, load_path: str, config: Optional[PruningConfig] = None):
        """
        Load a binary mask to start phase-2 finetuning from a pure binary state.

        Gates that are ON in the mask start at log_alpha=+5 and remain trainable.
        Gates that are OFF are frozen off with a hard guarantee:
          1. log_alpha pinned to -1e6 (sigmoid and its derivative underflow to
             exactly 0.0, so neither noise nor gradient can reopen the gate),
          2. gradients at frozen positions are zeroed via a hook,
          3. train_masks re-pins frozen positions to -1e6 at every clamp,
             defeating weight-decay / optimizer drift.
        """
        if self.model is None:
            self.initialize_model(config)

        mask_state = torch.load(load_path, weights_only=True)
        self.use_model(enable_masks=False)  # stay in continuous training mode

        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, HardConcreteGate) and hasattr(module, 'log_alpha'):
                    if name in mask_state:
                        on = mask_state[name].to(self.device)
                    else:
                        on = torch.zeros_like(module.log_alpha, dtype=torch.bool)
                    module.log_alpha.data = torch.where(
                        on,
                        torch.full_like(module.log_alpha.data, 5.0),
                        torch.full_like(module.log_alpha.data, -1e6)
                    )
                    frozen = ~on
                    module.frozen_off = frozen
                    module.log_alpha.register_hook(
                        lambda grad, m=frozen: grad.masked_fill(m, 0)
                    )
        print(f"Binary mask loaded from {load_path} (off-gates frozen for finetuning)")

    def use_model(self, enable_masks: bool = True):
        """
        Enable or disable hard masks.
        If True, the model uses hard 0/1 masks.
        If False, the model goes back to continuous mode (for training).
        """
        if self.model is None:
            self.initialize_model()
            
        self.model.set_final_circuit_mode(enable_masks)

    def evaluate_kl_divergence(self, dataloader) -> float:
        """
        Evaluate KL divergence on the dataset with the currently applied masks.
        """
        if self.model is None or self.baseline_model is None:
            raise ValueError("Models not initialized.")
            
        self.model.eval()
        self.baseline_model.eval()
        
        total_kl = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                baseline_out = self.baseline_model(
                    input_ids=batch['input_ids'], 
                    attention_mask=batch['attention_mask'],
                    use_cache=False
                )
                baseline_logits = baseline_out.logits
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    corrupted_input_ids=batch['corrupted_input_ids'],
                    attention_mask=batch['attention_mask'],
                    use_cache=False
                )
                
                mask = batch['attention_mask'].bool()
                circuit_logits = outputs.logits[mask]
                golden_logits = baseline_logits[mask]
                
                kl = F.kl_div(
                    F.log_softmax(circuit_logits.float(), dim=-1),
                    F.log_softmax(golden_logits.float(), dim=-1),
                    reduction='sum',
                    log_target=True
                )
                
                total_kl += kl.item()
                total_tokens += mask.sum().item()
                
        avg_kl = total_kl / max(total_tokens, 1)
        print(f"Evaluated KL Divergence: {avg_kl:.4f}")
        return avg_kl
