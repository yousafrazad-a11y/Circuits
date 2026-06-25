"""
Custom 6-Hop Circuit Discovery Experiment for Llama Models.

Adapted from circuit_pruning-argo/ioi_llama.py to work with custom JSONL dataset.
"""

import sys
import os

# --- MODIFICATION: Add original codebase to path so we can reuse models and utils ---
sys.path.append("/home/exouser/pruning/circuit_pruning-argo")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
import time
import argparse
import json

# --- MODIFICATION: Import from custom dataset loader instead of dataset.ioi_llama ---
from custom_dataset import Custom6HopDatasetLlama, load_jsonl_dataset, filter_dataset_by_model_correctness, custom_run_evaluation
# --- ORIGINAL IMPORTS ---
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from utils import disable_dropout, analyze_and_finalize_circuit

# ==============================================================================
# PRUNING CONFIGURATION (from original code)
# ==============================================================================
from dataclasses import dataclass

PRUNING_FACTOR = 0.12

@dataclass
class LlamaPruningConfig(PruningConfig):
    """PruningConfig with defaults tuned for Llama 3.2-1B scale."""
    init_value: float = 0.5
    sparsity_warmup_steps: int = 1000
    depth_penalty_scaling: float = 0.0

    prune_attention_heads: bool = True
    lambda_attention_heads: float = 0.8 * PRUNING_FACTOR

    prune_mlp_hidden: bool = True
    lambda_mlp_hidden: float = 1.0 * PRUNING_FACTOR

    prune_mlp_output: bool = True
    lambda_mlp_output: float = 1.0 * PRUNING_FACTOR

    prune_attention_neurons: bool = True
    lambda_attention_neurons: float = 0.15 * PRUNING_FACTOR

    prune_attention_blocks: bool = True
    lambda_attention_blocks: float = 0.5 * PRUNING_FACTOR

    prune_mlp_blocks: bool = True
    lambda_mlp_blocks: float = 0.5 * PRUNING_FACTOR

    prune_full_layers: bool = False
    lambda_full_layers: float = 0.0 * PRUNING_FACTOR

    prune_embedding: bool = False
    lambda_embedding: float = 1.0 * PRUNING_FACTOR


# ==============================================================================
# CHECKPOINTING (from original code)
# ==============================================================================
def _save_checkpoint(circuit_model, optimizer, epoch, total_steps,
                     best_val_accuracy, val_results, path, gate_patterns):
    """Save only gate parameters (not the full 1.2B model) plus optimizer state."""
    gate_state = {}
    for name, param in circuit_model.named_parameters():
        if any(p in name for p in gate_patterns):
            gate_state[name] = param.data.cpu()

    checkpoint = {
        'epoch': epoch,
        'total_steps': total_steps,
        'gate_state_dict': gate_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy,
        'val_results': val_results,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path} (epoch {epoch+1})")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Custom 6-Hop Circuit Discovery for Llama")
    parser.add_argument('--dry-run', action='store_true', help='Quick test with minimal data')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='HuggingFace model name/path')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--save-dir', type=str, default='checkpoints_llama_custom',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh even if checkpoints exist')
    args = parser.parse_args()

    # --- MODIFICATION: Hardcoded HF token as requested by user ---
    hf_token = "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"

    # --- Configuration (Mostly from original code) ---
    MODEL_NAME = args.model
    NUM_EPOCHS = 2 if args.dry_run else args.epochs
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = 128 # --- MODIFICATION: Increased max length for 6-hop prompts ---
    ACCURACY_BUDGET = 0.05
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    SAVE_DIR = args.save_dir

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Dry run: {args.dry_run}")
    print(f"Save dir: {SAVE_DIR}")

    pruning_config = LlamaPruningConfig()

    # --- Model and Tokenizer Setup (from original code) ---
    print("\n--- Loading tokenizer and models ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {"token": hf_token, "torch_dtype": torch.bfloat16}

    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config, **model_kwargs
    ).to(DEVICE).eval()

    full_model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, **model_kwargs
    ).to(DEVICE).eval()
    for param in full_model.parameters():
        param.requires_grad = False

    print("\n--- Disabling all dropout layers in the circuit model ---")
    disable_dropout(circuit_model)

    print("Freezing base model weights and unfreezing gate parameters...")
    total_params = 0
    trainable_params = 0
    GATE_PATTERNS = ('_gates.', '_gate.', 'embedding_gate.', 'layer_gates.')

    for name, param in circuit_model.named_parameters():
        total_params += param.numel()
        is_pruning_gate = any(p in name for p in GATE_PATTERNS)
        if not is_pruning_gate:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()
            trainable_params += param.numel()

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable gate parameters: {trainable_params} ({trainable_params/total_params*100:.4f}%)")

    # --- Dataset Setup ---
    print("\nSetting up custom JSONL dataset...")
    
    # --- MODIFICATION: Load from JSONL and split manually ---
    dataset_path = "/home/exouser/pruning/datasets/dataset.jsonl"
    all_data = load_jsonl_dataset(dataset_path)
    
    if args.dry_run:
        train_data = all_data[:10]
        val_data = all_data[10:15]
        test_data = all_data[15:20]
    else:
        # We have 1001 samples. We'll use 600 train, 200 val, 201 test.
        train_data = all_data[:600]
        val_data = all_data[600:800]
        test_data = all_data[800:]

    print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.")

    # Filter datasets by model correctness (from original code)
    print("\n--- Filtering datasets based on Base Model correctness ---")
    val_data = filter_dataset_by_model_correctness(
        val_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE
    )
    test_data = filter_dataset_by_model_correctness(
        test_data, full_model, tokenizer, DEVICE, batch_size=BATCH_SIZE
    )

    # Create Dataset objects (using our new custom class)
    train_dataset = Custom6HopDatasetLlama(train_data, tokenizer, max_length=MAX_SEQ_LEN)
    val_dataset = Custom6HopDatasetLlama(val_data, tokenizer, max_length=MAX_SEQ_LEN)
    test_dataset = Custom6HopDatasetLlama(test_data, tokenizer, max_length=MAX_SEQ_LEN)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # --- Baseline Evaluation (from original code) ---
    print("\n--- Baseline evaluation on full model ---")
    baseline_results = custom_run_evaluation(
        model_to_eval=full_model,
        model_name="Baseline Full Model",
        full_model=None,
        dataloader=test_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )
    base_accuracy = baseline_results.get("base_clean_acc", 0.0)

    # --- Initial Circuit Model Evaluation (from original code) ---
    print("\n--- Initial evaluation of the Circuit Discovery Model ---")
    circuit_model.eval()
    initial_results = custom_run_evaluation(
        model_to_eval=circuit_model,
        model_name="Initial Circuit Model",
        full_model=full_model,
        dataloader=val_dataloader,
        device=DEVICE,
        tokenizer=tokenizer,
    )

    # --- Training Setup (from original code) ---
    gate_params = [p for p in circuit_model.parameters() if p.requires_grad]
    optimizer = AdamW(gate_params, lr=LEARNING_RATE)

    print(f"\n--- Starting training to find custom 6-hop circuit ---")
    
    start_epoch = 0
    total_steps = 0
    best_val_accuracy = 0.0

    if not args.no_resume:
        resume_path = None
        if os.path.exists(SAVE_DIR):
            candidates = []
            for f in os.listdir(SAVE_DIR):
                if f.endswith('.pt'):
                    fpath = os.path.join(SAVE_DIR, f)
                    candidates.append((os.path.getmtime(fpath), fpath, f))
            if candidates:
                candidates.sort(reverse=True)
                resume_path = candidates[0][1]
                print(f"\n--- Found checkpoint: {candidates[0][2]} ---")

        if resume_path and os.path.exists(resume_path):
            print(f"--- Auto-resuming from: {resume_path} ---")
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            gate_state = checkpoint['gate_state_dict']
            model_state = circuit_model.state_dict()
            model_state.update(gate_state)
            circuit_model.load_state_dict(model_state)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            total_steps = checkpoint.get('total_steps', 0)
            best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            print(f"  Resumed from epoch {start_epoch}, step {total_steps}")
        else:
            print("\n--- No checkpoint found, starting fresh ---")

    # Pre-cache full model outputs (speedup)
    print("\nPre-caching full model outputs for training data...")
    cached_train_logits = {}
    full_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Caching full model")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)
            out = full_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )
            cached_train_logits[batch_idx] = out.logits.detach()
    print(f"Cached {len(cached_train_logits)} batches of full model outputs.")

    circuit_model.train()
    epoch_pbar = tqdm(range(start_epoch, NUM_EPOCHS), desc="Training Progress", initial=start_epoch, total=NUM_EPOCHS)

    # --- Training Loop (from original code) ---
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        epoch_loss = epoch_kl_loss = epoch_sparsity_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(DEVICE)

            # Dual-stream forward pass
            circuit_outputs = circuit_model(
                input_ids=batch['input_ids'],
                corrupted_input_ids=batch['corrupted_input_ids'],
                attention_mask=batch['attention_mask'],
                use_cache=False,
            )

            target_logits = cached_train_logits[batch_idx]
            batch_size_curr = circuit_outputs.logits.size(0)
            total_kl = 0

            for i in range(batch_size_curr):
                t_start = batch['T_Start'][i].item() - 1
                t_end = batch['T_End'][i].item() - 1
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, int(valid_length))

                if t_start < end_pos:
                    circuit_logits_slice = circuit_outputs.logits[i, t_start:end_pos].float()
                    target_logits_slice = target_logits[i, t_start:end_pos].float()

                    kl = F.kl_div(
                        F.log_softmax(circuit_logits_slice, dim=-1),
                        F.log_softmax(target_logits_slice, dim=-1),
                        reduction='sum',
                        log_target=True,
                    )
                    total_kl += kl

            # Task loss
            pos_good = batch['T_Start'] - 1
            pos_bad = batch['D_Start'] - 1
            token_good = batch['target_tokens'][:, 0]
            token_bad = batch['distractor_tokens'][:, 0]
            batch_indices = torch.arange(batch_size_curr, device=DEVICE)

            logit_good = circuit_outputs.logits[batch_indices, pos_good, token_good].float()
            logit_bad = circuit_outputs.logits[batch_indices, pos_bad, token_bad].float()

            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()

            kl_loss = total_kl / batch_size_curr
            sparsity_loss = circuit_model.get_sparsity_loss(step=total_steps)['total_sparsity']

            loss = kl_loss * 1.5 + sparsity_loss + task_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            total_steps += 1

        epoch_duration = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(len(train_dataloader), 1)
        avg_sparsity = epoch_sparsity_loss / max(len(train_dataloader), 1)

        epoch_pbar.set_postfix({'L': f"{avg_loss:.3f}", 'Sp': f"{avg_sparsity:.3f}", 'Time': f"{epoch_duration:.2f}s"})

        if (epoch + 1) % 10 == 0:
            circuit_model.eval()
            val_results = custom_run_evaluation(
                model_to_eval=circuit_model, model_name=f"Val Ep {epoch+1}",
                full_model=full_model, dataloader=val_dataloader,
                device=DEVICE, tokenizer=tokenizer,
            )

            val_acc = val_results.get('pruned_clean_acc', 0.0)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                _save_checkpoint(
                    circuit_model, optimizer, epoch, total_steps, best_val_accuracy, val_results,
                    os.path.join(SAVE_DIR, 'best_checkpoint.pt'), GATE_PATTERNS,
                )
                print(f"  >> New best val accuracy: {best_val_accuracy:.4f} (saved)")
            circuit_model.train()

        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                circuit_model, optimizer, epoch, total_steps, best_val_accuracy, None,
                os.path.join(SAVE_DIR, f'checkpoint_ep{epoch+1}.pt'), GATE_PATTERNS,
            )

    if start_epoch < NUM_EPOCHS:
        _save_checkpoint(
            circuit_model, optimizer, epoch, total_steps, best_val_accuracy, None,
            os.path.join(SAVE_DIR, 'final_checkpoint.pt'), GATE_PATTERNS,
        )

    # --- Final Analysis and Pruning (from original code) ---
    print("\n--- Pre-finalization evaluation on test set ---")
    circuit_model.eval()
    pre_final_results = custom_run_evaluation(
        model_to_eval=circuit_model, model_name="Pre-Finalization (Test Set)",
        full_model=full_model, dataloader=test_dataloader,
        device=DEVICE, tokenizer=tokenizer,
    )

    print("\n--- Analyzing and finalizing circuit ---")
    pruning_details = analyze_and_finalize_circuit(circuit_model)

    print("\n--- Final evaluation on test set ---")
    circuit_model.eval()
    final_results = custom_run_evaluation(
        model_to_eval=circuit_model, model_name="Final Pruned Circuit",
        full_model=full_model, dataloader=test_dataloader,
        device=DEVICE, tokenizer=tokenizer,
    )

    # Save results to file
    results_dict = {
        "baseline_results": baseline_results,
        "pre_final_results": pre_final_results,
        "final_results": final_results,
        "pruning_details": pruning_details
    }
    results_path = os.path.join(SAVE_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"\nSaved evaluation results to {results_path}")
