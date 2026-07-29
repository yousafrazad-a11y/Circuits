#!/usr/bin/env python3
"""
MLP-block corruption ablation on IOI with Llama-3.2-1B.

Tests whether corrupting a layer's MLP output (replacing it with the
corrupted-stream activation via the mlp_block_gate, all attention gates open)
degrades IOI accuracy the way it degrades our category-word tracking task.

Conditions:
  BASE          - plain baseline LlamaForCausalLM (sanity, ~1.0 on filtered set)
  OPEN          - prunable model, every gate open, dual stream (should ~= BASE)
  FRUITS_PATTERN- corrupt the 3 MLP layers OFF in the frozen fruits mask (1,6,7)
  RANDOM_k      - corrupt k random MLP layers, seeds 0,1,2 (k in {1,3,8})

Eval set: IOI examples filtered so the BASE model answers correctly
(logit(IO) > logit(S)), so we measure damage only on examples the full
model gets right.

Results -> /home/exouser/pruning/task_generalization/results_ioi_mlp_ablation.json
"""

import os
import sys
import json
import torch
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

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MASK_PATH = "/home/exouser/pruning/intersection_experiments_4/masks/frozen_fruits_300ep_l005_mask.pt"
OUT_PATH = "/home/exouser/pruning/task_generalization/results_ioi_mlp_ablation.json"

NUM_GENERATE = 900        # candidates before correctness filtering
NUM_EVAL = 400            # eval-set size after filtering
GEN_SEED = 123
BATCH_SIZE = 32
NUM_LAYERS = 16

# Read optional HF token (model is in local HF cache, so usually unnecessary)
token_file = os.path.join(REPO_DIR, "hf_tokken.txt")
hf_token = None
if os.path.exists(token_file):
    with open(token_file) as f:
        hf_token = f.read().strip() or None


def load_fruits_off_layers(path: str):
    """Return sorted list of layer indices whose mlp_block_gate is OFF in the mask."""
    mask = torch.load(path, map_location="cpu", weights_only=True)
    off = []
    for k, v in mask.items():
        if k.endswith("mlp_block_gate") and not bool(v.all()):
            # key like 'model.layers.6.mlp_block_gate'
            off.append(int(k.split(".")[2]))
    return sorted(off)


def set_gates(model, mlp_off_layers):
    """Open every HardConcreteGate, then close mlp_block_gate on mlp_off_layers.

    Gate output is hard 0/1 in eval mode (threshold log_alpha > 0), so +/-1e6
    forces fully open / fully closed.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HardConcreteGate):
                module.log_alpha.fill_(1e6)
        for i in mlp_off_layers:
            model.model.layers[i].mlp_block_gate.log_alpha.fill_(-1e6)


@torch.no_grad()
def evaluate(model, dataloader, device, dual_stream: bool):
    """Same metric as dataset/ioi_llama.py run_evaluation:
    at position T_Start-1, accuracy = logit(IO target token) >= logit(S distractor token).
    """
    model.eval()
    correct = 0
    total_logit_diff = 0.0
    n = 0
    for batch in dataloader:
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)

        kwargs = dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if dual_stream:
            kwargs["corrupted_input_ids"] = batch["corrupted_input_ids"]
        logits = model(**kwargs).logits

        pos = batch["T_Start"] - 1                       # last position before target
        target = batch["target_tokens"][:, 0]            # single-token names
        distractor = batch["distractor_tokens"][:, 0]

        rows = torch.arange(logits.size(0), device=device)
        t_logit = logits[rows, pos, target].float()
        d_logit = logits[rows, pos, distractor].float()

        correct += (t_logit >= d_logit).sum().item()
        total_logit_diff += (t_logit - d_logit).sum().item()
        n += logits.size(0)

    return {"accuracy": correct / n, "logit_diff": total_logit_diff / n, "n": n}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- tokenizer + models ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading baseline model...")
    baseline = LlamaForCausalLM.from_pretrained(
        MODEL_NAME, token=hf_token, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print("Loading prunable model...")
    prunable = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL_NAME, PruningConfig(), token=hf_token, torch_dtype=torch.bfloat16
    ).to(device).eval()

    # ---- eval set: IOI examples the base model gets right ----
    print(f"Generating {NUM_GENERATE} IOI candidates (seed={GEN_SEED})...")
    candidates = generate_ioi_data_llama(num_samples=NUM_GENERATE, tokenizer=tokenizer, seed=GEN_SEED)
    filtered = filter_dataset_by_model_correctness(candidates, baseline, tokenizer, device,
                                                   batch_size=BATCH_SIZE)
    filtered = filtered[:NUM_EVAL]
    print(f"Eval set size: {len(filtered)}")

    dataset = IOIDatasetLlama(filtered, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- fruits mask ----
    fruits_off = load_fruits_off_layers(MASK_PATH)
    print(f"Fruits pattern OFF layers: {fruits_off}")

    results = {}

    # BASE: plain baseline
    print("\n=== BASE (plain baseline) ===")
    results["BASE"] = evaluate(baseline, loader, device, dual_stream=False)
    print(results["BASE"])

    # OPEN: prunable, all gates open, dual stream
    print("\n=== OPEN (all gates open, dual stream) ===")
    set_gates(prunable, mlp_off_layers=[])
    results["OPEN"] = evaluate(prunable, loader, device, dual_stream=True)
    print(results["OPEN"])

    # FRUITS_PATTERN: corrupt the 3 fruits-OFF MLP layers
    print(f"\n=== FRUITS_PATTERN (MLP corrupt layers {fruits_off}) ===")
    set_gates(prunable, mlp_off_layers=fruits_off)
    results["FRUITS_PATTERN"] = evaluate(prunable, loader, device, dual_stream=True)
    results["FRUITS_PATTERN"]["off_layers"] = fruits_off
    print(results["FRUITS_PATTERN"])

    # RANDOM_k for k in {1,3,8}, seeds 0,1,2
    for k in (1, 3, 8):
        key = f"RANDOM_{k}"
        results[key] = {"seeds": {}}
        accs, lds = [], []
        for seed in (0, 1, 2):
            gen = torch.Generator().manual_seed(seed)
            layers = torch.randperm(NUM_LAYERS, generator=gen)[:k].tolist()
            set_gates(prunable, mlp_off_layers=layers)
            r = evaluate(prunable, loader, device, dual_stream=True)
            r["off_layers"] = layers
            results[key]["seeds"][str(seed)] = r
            accs.append(r["accuracy"])
            lds.append(r["logit_diff"])
            print(f"\n=== {key} seed={seed} layers={layers} ===")
            print(r)
        results[key]["accuracy"] = sum(accs) / len(accs)
        results[key]["logit_diff"] = sum(lds) / len(lds)
        results[key]["n"] = len(filtered)

    # ---- report ----
    print("\n" + "=" * 72)
    print(f"{'Condition':<28}{'Accuracy':>10}{'LogitDiff':>12}{'N':>6}   Layers off")
    print("-" * 72)
    for cond in ("BASE", "OPEN", "FRUITS_PATTERN"):
        r = results[cond]
        layers = r.get("off_layers", "-")
        print(f"{cond:<28}{r['accuracy']:>10.4f}{r['logit_diff']:>12.4f}{r['n']:>6}   {layers}")
    for k in (1, 3, 8):
        key = f"RANDOM_{k}"
        r = results[key]
        per_seed = " ".join(f"{r['seeds'][s]['accuracy']:.3f}" for s in ("0", "1", "2"))
        print(f"{key + ' (mean of 3 seeds)':<28}{r['accuracy']:>10.4f}{r['logit_diff']:>12.4f}{r['n']:>6}   [{per_seed}]")
    print("=" * 72)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
