#!/usr/bin/env python3
"""
Train an IOI pruning mask with ONLY per-head attention gates + per-layer scalar
MLP-block gates (matching the granularity of our category-task masks), then run
the 2x2 decomposition evaluation (OPEN / FULL / HEADS_ONLY / MLP_ONLY).

Reuses the circuit_pruning-argo code (models/llama_circuit.py gates,
dataset/ioi_llama.py data) with the training loss from ioi_llama.py:
    loss = 1.5 * KL(circuit || full model @ target pos) + sparsity + task_margin

Hyperparameters (mirroring our category-task run: frozen_fruits_300ep_l005):
    300 epochs, lambda 0.05 for both heads and MLP blocks, lr 3e-2 (script
    default), sparsity warmup 1000 steps, batch 32, 800 train samples.

Outputs:
    task_generalization/masks/ioi_heads_mlp_mask.pt       (binarized bools)
    task_generalization/masks/ioi_heads_mlp_checkpoint.pt (continuous log_alphas)
    task_generalization/results_ioi_2x2.json
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuit_pruning-argo")
sys.path.insert(0, REPO_DIR)

from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from models.l0 import HardConcreteGate
from dataset.ioi_llama import (
    generate_ioi_data_llama,
    IOIDatasetLlama,
    filter_dataset_by_model_correctness,
)
from utils import disable_dropout

MODEL_NAME = "meta-llama/Llama-3.2-1B"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_PATH = os.path.join(BASE_DIR, "masks", "ioi_heads_mlp_mask.pt")
CKPT_PATH = os.path.join(BASE_DIR, "masks", "ioi_heads_mlp_checkpoint.pt")
OUT_PATH = os.path.join(BASE_DIR, "results_ioi_2x2.json")

token_file = os.path.join(REPO_DIR, "hf_tokken.txt")
hf_token = None
if os.path.exists(token_file):
    with open(token_file) as f:
        hf_token = f.read().strip() or None


def build_pruning_config():
    """Only head gates + scalar MLP-block gates are prunable; nothing else exists."""
    return PruningConfig(
        init_value=0.5,
        sparsity_warmup_steps=1000,
        depth_penalty_scaling=0.0,
        prune_attention_heads=True,
        lambda_attention_heads=0.05,
        prune_attention_neurons=False,
        prune_mlp_hidden=False,
        prune_mlp_output=False,
        prune_attention_blocks=False,
        prune_mlp_blocks=True,
        lambda_mlp_blocks=0.05,
        prune_full_layers=False,
    )


@torch.no_grad()
def evaluate(model, dataloader, device, dual_stream=True):
    """IOI metric: accuracy = logit(IO) >= logit(S) at position T_Start-1."""
    model.eval()
    correct, total_ld, n = 0, 0.0, 0
    for batch in dataloader:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)
        kwargs = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if dual_stream:
            kwargs["corrupted_input_ids"] = batch["corrupted_input_ids"]
        logits = model(**kwargs).logits
        pos = batch["T_Start"] - 1
        rows = torch.arange(logits.size(0), device=device)
        t = logits[rows, pos, batch["target_tokens"][:, 0]].float()
        d = logits[rows, pos, batch["distractor_tokens"][:, 0]].float()
        correct += (t >= d).sum().item()
        total_ld += (t - d).sum().item()
        n += logits.size(0)
    return {"accuracy": correct / n, "logit_diff": total_ld / n, "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--train-samples", type=int, default=800)
    parser.add_argument("--bench", action="store_true", help="time 30 steps and exit")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(os.path.join(BASE_DIR, "masks"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading models...")
    pruning_config = build_pruning_config()
    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config, token=hf_token, torch_dtype=torch.bfloat16
    ).to(device).eval()
    full_model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, token=hf_token, torch_dtype=torch.bfloat16
    ).to(device).eval()
    for p in full_model.parameters():
        p.requires_grad = False

    disable_dropout(circuit_model)
    print("Gate groups:", circuit_model.gate_group_sizes())

    # Freeze base weights, unfreeze gates (float32 for stability)
    n_train = 0
    for name, param in circuit_model.named_parameters():
        if any(p in name for p in ("head_gates.", "mlp_block_gate.")):
            param.requires_grad = True
            param.data = param.data.float()
            n_train += param.numel()
        else:
            param.requires_grad = False
    print(f"Trainable gate params: {n_train}")

    # ---- data ----
    print("\nGenerating data...")
    train_data = generate_ioi_data_llama(num_samples=args.train_samples, tokenizer=tokenizer, seed=42)
    val_data = generate_ioi_data_llama(num_samples=200, tokenizer=tokenizer, seed=456)
    val_data = filter_dataset_by_model_correctness(val_data, full_model, tokenizer, device,
                                                   batch_size=args.batch_size)
    train_dataset = IOIDatasetLlama(train_data, tokenizer, max_length=64)
    val_dataset = IOIDatasetLlama(val_data, tokenizer, max_length=64)

    class IndexedDataset(torch.utils.data.Dataset):
        """Adds the dataset index so cached KL targets survive shuffling."""
        def __init__(self, ds): self.ds = ds
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            item["idx"] = i
            return item

    train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- pre-cache full-model logits at the target position (KL target) ----
    # ioi_llama.py computes KL over positions [T_Start-1 : T_End-1]; with
    # single-token names that is exactly one position (T_Start-1).
    print("Caching full-model logits at target positions...")
    cache_loader = DataLoader(IndexedDataset(train_dataset), batch_size=args.batch_size, shuffle=False)
    cached = torch.empty(len(train_dataset), circuit_model.config.vocab_size,
                         dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        for batch in cache_loader:
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
            out = full_model(input_ids=batch["input_ids"],
                             attention_mask=batch["attention_mask"], use_cache=False)
            pos = batch["T_Start"] - 1
            rows = torch.arange(out.logits.size(0), device=device)
            cached[batch["idx"]] = out.logits[rows, pos, :].to(torch.bfloat16)
    print(f"Cached logits for {len(train_dataset)} train samples.")

    optimizer = AdamW([p for p in circuit_model.parameters() if p.requires_grad], lr=args.lr)

    # ---- training ----
    circuit_model.train()
    total_steps = 0
    steps_per_epoch = len(train_loader)
    t0 = time.time()
    best_val_acc = -1.0
    best_gates = None

    for epoch in range(args.epochs):
        ep_loss = ep_kl = ep_sp = ep_task = 0.0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            out = circuit_model(
                input_ids=batch["input_ids"],
                corrupted_input_ids=batch["corrupted_input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            pos = batch["T_Start"] - 1
            rows = torch.arange(out.logits.size(0), device=device)
            circ = out.logits[rows, pos, :].float()
            tgt = cached[batch["idx"]].float()

            kl = F.kl_div(F.log_softmax(circ, dim=-1), F.log_softmax(tgt, dim=-1),
                          log_target=True, reduction="batchmean")

            t_logit = circ[rows, batch["target_tokens"][:, 0]]
            d_logit = circ[rows, batch["distractor_tokens"][:, 0]]
            task_loss = F.relu(4.0 - (t_logit - d_logit)).mean()

            sparsity = circuit_model.get_sparsity_loss(step=total_steps)["total_sparsity"]

            loss = 1.5 * kl + sparsity + task_loss
            loss.backward()
            optimizer.step()

            ep_loss += loss.item(); ep_kl += kl.item()
            ep_sp += sparsity.item(); ep_task += task_loss.item()
            total_steps += 1

            if args.bench and total_steps >= 30:
                dt = time.time() - t0
                print(f"\nBENCH: 30 steps in {dt:.1f}s -> {dt/30*1000:.0f} ms/step; "
                      f"300 epochs ({300*steps_per_epoch} steps) ~= "
                      f"{dt/30*300*steps_per_epoch/60:.1f} min")
                return

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            circuit_model.eval()
            val = evaluate(circuit_model, val_loader, device)
            circuit_model.train()
            marker = ""
            if val["accuracy"] > best_val_acc:
                best_val_acc = val["accuracy"]
                best_gates = {n: p.data.clone() for n, p in circuit_model.named_parameters()
                              if p.requires_grad}
                marker = "  (new best)"
            print(f"ep {epoch+1:4d} | loss {ep_loss/steps_per_epoch:.3f} | "
                  f"kl {ep_kl/steps_per_epoch:.3f} | sparsity {ep_sp/steps_per_epoch:.4f} | "
                  f"task {ep_task/steps_per_epoch:.3f} | "
                  f"val acc {val['accuracy']:.4f} ld {val['logit_diff']:.3f}{marker} | "
                  f"{(time.time()-t0)/60:.1f} min")

    train_minutes = (time.time() - t0) / 60
    print(f"\nTraining done in {train_minutes:.1f} min. Best val acc: {best_val_acc:.4f}")

    # ---- save continuous checkpoint + binarized mask (final gates) ----
    gate_state = {n: p.data.cpu() for n, p in circuit_model.named_parameters()
                  if any(k in n for k in ("head_gates.", "mlp_block_gate."))}
    torch.save({
        "gate_state_dict": gate_state,
        "best_val_accuracy": best_val_acc,
        "config": vars(pruning_config),
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "train_samples": args.train_samples, "train_minutes": train_minutes,
    }, CKPT_PATH)
    print(f"Continuous checkpoint saved: {CKPT_PATH}")

    mask = {n: (v > 0) for n, v in gate_state.items()}
    torch.save(mask, MASK_PATH)
    print(f"Binarized mask saved: {MASK_PATH}")

    # ---- mask anatomy ----
    heads_on = sum(int(v.sum()) for k, v in mask.items() if "head_gates" in k)
    mlp_on = sum(int(v.sum()) for k, v in mask.items() if "mlp_block_gate" in k)
    per_layer = {}
    for i in range(16):
        h = int(mask[f"model.layers.{i}.attn.head_gates"].sum())
        m = bool(mask[f"model.layers.{i}.mlp_block_gate"].all())
        per_layer[str(i)] = {"heads_on": h, "mlp_on": m}
    print(f"\nMask anatomy: heads ON {heads_on}/512, MLP blocks ON {mlp_on}/16")
    for i in range(16):
        print(f"  layer {i:2d}: heads {per_layer[str(i)]['heads_on']:2d}/32, "
              f"mlp {'ON' if per_layer[str(i)]['mlp_on'] else 'OFF'}")

    # ================= 2x2 EVALUATION =================
    print("\n" + "=" * 60 + "\n2x2 DECOMPOSITION EVALUATION\n" + "=" * 60)

    # Same eval set as mlp_ablation_ioi.py: seed 123, 900 candidates, filtered, first 400
    eval_data = generate_ioi_data_llama(num_samples=900, tokenizer=tokenizer, seed=123)
    eval_data = filter_dataset_by_model_correctness(eval_data, full_model, tokenizer, device,
                                                    batch_size=args.batch_size)[:400]
    eval_dataset = IOIDatasetLlama(eval_data, tokenizer, max_length=64)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Eval set size: {len(eval_data)}")

    final_gates = {n: p.data.clone() for n, p in circuit_model.named_parameters()
                   if p.requires_grad}
    head_names = [n for n in final_gates if "head_gates" in n]
    mlp_names = [n for n in final_gates if "mlp_block_gate" in n]

    def apply_gates(head_mode, mlp_mode):
        """head_mode/mlp_mode: 'trained' or 'open'."""
        with torch.no_grad():
            for n, p in circuit_model.named_parameters():
                if not p.requires_grad:
                    continue
                if "head_gates" in n:
                    p.data.copy_(final_gates[n] if head_mode == "trained"
                                 else torch.full_like(p.data, 1e6))
                elif "mlp_block_gate" in n:
                    p.data.copy_(final_gates[n] if mlp_mode == "trained"
                                 else torch.full_like(p.data, 1e6))

    results = {"mask_anatomy": {
        "heads_on": heads_on, "heads_total": 512,
        "mlp_on": mlp_on, "mlp_total": 16,
        "per_layer": per_layer,
        "best_val_accuracy": best_val_acc,
        "train_minutes": train_minutes,
        "hyperparameters": {
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "train_samples": args.train_samples,
            "lambda_attention_heads": 0.05, "lambda_mlp_blocks": 0.05,
            "sparsity_warmup_steps": 1000, "kl_weight": 1.5, "task_margin": 4.0,
        },
    }}

    for name, hm, mm in [("OPEN", "open", "open"),
                         ("FULL", "trained", "trained"),
                         ("HEADS_ONLY", "trained", "open"),
                         ("MLP_ONLY", "open", "trained")]:
        apply_gates(hm, mm)
        r = evaluate(circuit_model, eval_loader, device)
        results[name] = r
        print(f"{name:<12} acc {r['accuracy']:.4f}  logit_diff {r['logit_diff']:.4f}  (n={r['n']})")

    print("\n" + "=" * 44)
    print(f"{'Condition':<14}{'Accuracy':>10}{'LogitDiff':>12}")
    print("-" * 44)
    for name in ("OPEN", "FULL", "HEADS_ONLY", "MLP_ONLY"):
        r = results[name]
        print(f"{name:<14}{r['accuracy']:>10.4f}{r['logit_diff']:>12.4f}")
    print("=" * 44)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
