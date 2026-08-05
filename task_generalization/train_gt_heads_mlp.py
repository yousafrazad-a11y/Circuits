#!/usr/bin/env python3
"""
Train a GT (greater-than) pruning mask with ONLY per-head attention gates +
per-layer scalar MLP-block gates, then run the 2x2 decomposition evaluation.

STRETCH counterpart of train_ioi_heads_mlp.py. GT adaptations for Llama (same
as mlp_ablation_gt.py):
  - two-digit tokens mapped from f"{i:02d}" WITHOUT leading space (all single
    tokens under the Llama tokenizer; " NN" is not),
  - generated samples get a 'prefix' alias for GTDataset,
  - train data is filtered by base-model correctness (as gt.py does), since
    unfiltered GT accuracy of Llama-3.2-1B is only ~54%.

Loss (following gt.py, but same weighting form as the IOI trainer):
    loss = 1.5 * KL_digits(circuit || full model) + sparsity
where KL is over the renormalized 100 two-digit token distribution at the last
prompt token. Hyperparameters match the IOI run: 300 epochs, lambda 0.05 for
heads and MLP blocks, lr 3e-2, warmup 1000 steps, batch 32.

Outputs:
    task_generalization/masks/gt_heads_mlp_mask.pt       (binarized bools)
    task_generalization/masks/gt_heads_mlp_checkpoint.pt (continuous log_alphas)
    task_generalization/results_gt_2x2.json
"""

import os
import sys
import json
import time
import random
import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuit_pruning-argo")
sys.path.insert(0, REPO_DIR)

from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from dataset.gt_gpt2 import (
    generate_gt_sample_pair,
    GTDataset,
    run_evaluation,
    filter_dataset_by_model_correctness,
)
from utils import disable_dropout

MODEL_NAME = "meta-llama/Llama-3.2-1B"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_PATH = os.path.join(BASE_DIR, "masks", "gt_heads_mlp_mask.pt")
CKPT_PATH = os.path.join(BASE_DIR, "masks", "gt_heads_mlp_checkpoint.pt")
OUT_PATH = os.path.join(BASE_DIR, "results_gt_2x2.json")

token_file = os.path.join(REPO_DIR, "hf_tokken.txt")
hf_token = None
if os.path.exists(token_file):
    with open(token_file) as f:
        hf_token = f.read().strip() or None


def build_pruning_config():
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


def create_two_digit_mapping_llama(tokenizer):
    mapping = {}
    for i in range(100):
        enc = tokenizer.encode(f"{i:02d}", add_special_tokens=False)
        assert len(enc) == 1
        mapping[i] = enc[0]
    return mapping


def gen_gt(n, seed):
    random.seed(seed)
    out = []
    for _ in range(n):
        s = generate_gt_sample_pair()
        s["prefix"] = s["clean_prompt"]
        out.append(s)
    return out


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, ds): self.ds = ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        item = self.ds[i]
        item["idx"] = i
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--train-samples", type=int, default=800)
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

    for name, param in circuit_model.named_parameters():
        if any(p in name for p in ("head_gates.", "mlp_block_gate.")):
            param.requires_grad = True
            param.data = param.data.float()
        else:
            param.requires_grad = False
    n_train = sum(p.numel() for p in circuit_model.parameters() if p.requires_grad)
    print(f"Trainable gate params: {n_train}")

    two_digit_tokens = create_two_digit_mapping_llama(tokenizer)
    sorted_tokens = sorted(two_digit_tokens.items())
    digit_token_ids = torch.tensor([t for _, t in sorted_tokens], device=device)

    # ---- data: generate + filter by base-model correctness ----
    print("\nGenerating + filtering train data (base is only ~54% on GT)...")
    train_raw = gen_gt(args.train_samples * 2, seed=42)
    train_data = filter_dataset_by_model_correctness(
        train_raw, full_model, tokenizer, device, two_digit_tokens,
        batch_size=args.batch_size)[:args.train_samples]
    val_data = filter_dataset_by_model_correctness(
        gen_gt(400, seed=456), full_model, tokenizer, device, two_digit_tokens,
        batch_size=args.batch_size)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = GTDataset(train_data, tokenizer, max_length=32)
    val_dataset = GTDataset(val_data, tokenizer, max_length=32)
    train_loader = DataLoader(IndexedDataset(train_dataset), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- cache full-model digit logits at the last token position ----
    print("Caching full-model digit logits...")
    cache_loader = DataLoader(IndexedDataset(train_dataset), batch_size=args.batch_size, shuffle=False)
    cached = torch.empty(len(train_dataset), 100, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        for batch in cache_loader:
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)
            out = full_model(input_ids=batch["clean_input_ids"],
                             attention_mask=batch["clean_attention_mask"], use_cache=False)
            last = out.logits[torch.arange(out.logits.size(0), device=device),
                              batch["last_token_idx"], :]
            digits = torch.gather(last, 1, digit_token_ids.unsqueeze(0).expand(last.size(0), -1))
            cached[batch["idx"]] = digits.to(torch.bfloat16)

    optimizer = AdamW([p for p in circuit_model.parameters() if p.requires_grad], lr=args.lr)

    # ---- training ----
    circuit_model.train()
    total_steps = 0
    steps_per_epoch = len(train_loader)
    t0 = time.time()
    best_val_acc = -1.0

    for epoch in range(args.epochs):
        ep_loss = ep_kl = ep_sp = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            out = circuit_model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corrupted_input_ids"],
                attention_mask=batch["clean_attention_mask"],
                use_cache=False,
            )
            last = out.logits[torch.arange(out.logits.size(0), device=device),
                              batch["last_token_idx"], :].float()
            digits = torch.gather(last, 1, digit_token_ids.unsqueeze(0).expand(last.size(0), -1))
            tgt = cached[batch["idx"]].float()

            kl = F.kl_div(F.log_softmax(digits, dim=-1), F.log_softmax(tgt, dim=-1),
                          log_target=True, reduction="batchmean")
            sparsity = circuit_model.get_sparsity_loss(step=total_steps)["total_sparsity"]
            loss = 1.5 * kl + sparsity
            loss.backward()
            optimizer.step()

            ep_loss += loss.item(); ep_kl += kl.item(); ep_sp += sparsity.item()
            total_steps += 1

        if (epoch + 1) % 25 == 0 or epoch == args.epochs - 1:
            circuit_model.eval()
            val = run_evaluation(circuit_model, "val", None, val_loader, device,
                                 two_digit_tokens, verbose=False)
            circuit_model.train()
            marker = ""
            if val["accuracy"] > best_val_acc:
                best_val_acc = val["accuracy"]
                marker = "  (new best)"
            print(f"ep {epoch+1:4d} | loss {ep_loss/steps_per_epoch:.3f} | "
                  f"kl {ep_kl/steps_per_epoch:.3f} | sparsity {ep_sp/steps_per_epoch:.4f} | "
                  f"val acc {val['accuracy']:.4f} pd {val['prob_diff']:.4f}{marker} | "
                  f"{(time.time()-t0)/60:.1f} min")

    train_minutes = (time.time() - t0) / 60
    print(f"\nTraining done in {train_minutes:.1f} min. Best val acc: {best_val_acc:.4f}")

    # ---- save ----
    gate_state = {n: p.data.cpu() for n, p in circuit_model.named_parameters()
                  if any(k in n for k in ("head_gates.", "mlp_block_gate."))}
    torch.save({
        "gate_state_dict": gate_state, "best_val_accuracy": best_val_acc,
        "config": vars(pruning_config), "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "train_samples": len(train_data),
        "train_minutes": train_minutes,
    }, CKPT_PATH)
    mask = {n: (v > 0) for n, v in gate_state.items()}
    torch.save(mask, MASK_PATH)
    print(f"Saved: {CKPT_PATH}\nSaved: {MASK_PATH}")

    heads_on = sum(int(v.sum()) for k, v in mask.items() if "head_gates" in k)
    mlp_on = sum(int(v.sum()) for k, v in mask.items() if "mlp_block_gate" in k)
    per_layer = {str(i): {
        "heads_on": int(mask[f"model.layers.{i}.attn.head_gates"].sum()),
        "mlp_on": bool(mask[f"model.layers.{i}.mlp_block_gate"].all()),
    } for i in range(16)}
    print(f"\nMask anatomy: heads ON {heads_on}/512, MLP blocks ON {mlp_on}/16")
    for i in range(16):
        pl = per_layer[str(i)]
        print(f"  layer {i:2d}: heads {pl['heads_on']:2d}/32, mlp {'ON' if pl['mlp_on'] else 'OFF'}")

    # ================= 2x2 EVALUATION =================
    print("\n" + "=" * 60 + "\n2x2 DECOMPOSITION EVALUATION (GT)\n" + "=" * 60)
    # Same eval set as mlp_ablation_gt.py: seed 123, 1200 candidates, filtered, first 400
    eval_data = filter_dataset_by_model_correctness(
        gen_gt(1200, seed=123), full_model, tokenizer, device, two_digit_tokens,
        batch_size=args.batch_size)[:400]
    eval_loader = DataLoader(GTDataset(eval_data, tokenizer, max_length=32),
                             batch_size=args.batch_size, shuffle=False)
    print(f"Eval set size: {len(eval_data)}")

    final_gates = {n: p.data.clone() for n, p in circuit_model.named_parameters()
                   if p.requires_grad}

    def apply_gates(head_mode, mlp_mode):
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
        "heads_on": heads_on, "heads_total": 512, "mlp_on": mlp_on, "mlp_total": 16,
        "per_layer": per_layer, "best_val_accuracy": best_val_acc,
        "train_minutes": train_minutes,
        "hyperparameters": {
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "train_samples": len(train_data), "lambda_attention_heads": 0.05,
            "lambda_mlp_blocks": 0.05, "sparsity_warmup_steps": 1000, "kl_weight": 1.5,
        },
    }}

    for name, hm, mm in [("OPEN", "open", "open"),
                         ("FULL", "trained", "trained"),
                         ("HEADS_ONLY", "trained", "open"),
                         ("MLP_ONLY", "open", "trained")]:
        apply_gates(hm, mm)
        r = run_evaluation(circuit_model, name, None, eval_loader, device,
                           two_digit_tokens, verbose=False)
        results[name] = r
        print(f"{name:<12} acc {r['accuracy']:.4f}  prob_diff {r['prob_diff']:.4f}  "
              f"sharpness {r['cutoff_sharpness']:.4f}")

    print("\n" + "=" * 52)
    print(f"{'Condition':<14}{'Accuracy':>10}{'ProbDiff':>12}{'Sharpness':>12}")
    print("-" * 52)
    for name in ("OPEN", "FULL", "HEADS_ONLY", "MLP_ONLY"):
        r = results[name]
        print(f"{name:<14}{r['accuracy']:>10.4f}{r['prob_diff']:>12.4f}{r['cutoff_sharpness']:>12.4f}")
    print("=" * 52)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
