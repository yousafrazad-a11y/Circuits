#!/usr/bin/env python3
"""
MLP-block corruption ablation on GT (greater-than) with Llama-3.2-1B.

Same design as mlp_ablation_ioi.py, using dataset/gt_gpt2.py with the Llama
tokenizer. Adaptations for Llama tokenization:
  - Two-digit number tokens are mapped from f"{i:02d}" WITHOUT the leading
    space: Llama splits " 42" into ['Ġ', '42'], so the repo's " NN" single-token
    assert fails for every number, but the bare "NN" (what the model actually
    predicts after a prompt ending in "... to 11") is a single token for all
    00-99.
  - generate_gt_sample_pair() output gets a 'prefix' key (alias of
    'clean_prompt') because GTDataset.__getitem__ reads item['prefix'].

Conditions identical to the IOI script (BASE / OPEN / FRUITS_PATTERN / RANDOM_k).
Metric (from dataset/gt_gpt2.py run_evaluation): renormalized probs over the
100 two-digit tokens; accuracy = P(>YY, window 10) > P(<=YY, window 10).

Results -> /home/exouser/pruning/task_generalization/results_gt_mlp_ablation.json
"""

import os
import sys
import json
import random
import torch
from torch.utils.data import DataLoader

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "circuit_pruning-argo")
sys.path.insert(0, REPO_DIR)

from transformers import AutoTokenizer, LlamaForCausalLM
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from models.l0 import HardConcreteGate
from dataset.gt_gpt2 import (
    generate_gt_sample_pair,
    GTDataset,
    run_evaluation,
    filter_dataset_by_model_correctness,
)

MODEL_NAME = "meta-llama/Llama-3.2-1B"
MASK_PATH = "/home/exouser/pruning/exp4_heads_mlp_main/masks/frozen_fruits_300ep_l005_mask.pt"
OUT_PATH = "/home/exouser/pruning/task_generalization/results_gt_mlp_ablation.json"

NUM_GENERATE = 1200
NUM_EVAL = 400
GEN_SEED = 123
BATCH_SIZE = 32
NUM_LAYERS = 16

token_file = os.path.join(REPO_DIR, "hf_tokken.txt")
hf_token = None
if os.path.exists(token_file):
    with open(token_file) as f:
        hf_token = f.read().strip() or None


def create_two_digit_mapping_llama(tokenizer):
    """Bare two-digit tokens (no leading space) — all single tokens for Llama-3."""
    mapping = {}
    for i in range(100):
        enc = tokenizer.encode(f"{i:02d}", add_special_tokens=False)
        assert len(enc) == 1, f"{i:02d} not a single token: {enc}"
        mapping[i] = enc[0]
    return mapping


def load_fruits_off_layers(path: str):
    mask = torch.load(path, map_location="cpu", weights_only=True)
    return sorted(int(k.split(".")[2]) for k, v in mask.items()
                  if k.endswith("mlp_block_gate") and not bool(v.all()))


def set_gates(model, mlp_off_layers):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, HardConcreteGate):
                module.log_alpha.fill_(1e6)
        for i in mlp_off_layers:
            model.model.layers[i].mlp_block_gate.log_alpha.fill_(-1e6)


class CleanOnlyWrapper(torch.nn.Module):
    """Drops corrupted_input_ids so gt run_evaluation works on the plain baseline."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

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

    two_digit_tokens = create_two_digit_mapping_llama(tokenizer)

    # ---- eval set: GT examples the base model gets right ----
    random.seed(GEN_SEED)
    candidates = []
    for _ in range(NUM_GENERATE):
        s = generate_gt_sample_pair()
        s["prefix"] = s["clean_prompt"]  # GTDataset reads item['prefix']
        candidates.append(s)

    filtered = filter_dataset_by_model_correctness(
        candidates, baseline, tokenizer, device, two_digit_tokens, batch_size=BATCH_SIZE
    )
    filtered = filtered[:NUM_EVAL]
    print(f"Eval set size: {len(filtered)}")
    if len(filtered) < 50:
        print("Too few correct samples — base model likely cannot do GT. Aborting.")
        return

    dataset = GTDataset(filtered, tokenizer, max_length=32)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    fruits_off = load_fruits_off_layers(MASK_PATH)
    print(f"Fruits pattern OFF layers: {fruits_off}")

    results = {}

    def run(name, model, loader_):
        r = run_evaluation(model, name, None, loader_, device, two_digit_tokens, verbose=False)
        r["n"] = len(filtered)
        print(f"=== {name} ===\n{r}")
        return r

    results["BASE"] = run("BASE (plain baseline)", CleanOnlyWrapper(baseline), loader)

    set_gates(prunable, mlp_off_layers=[])
    results["OPEN"] = run("OPEN (all gates open)", prunable, loader)

    set_gates(prunable, mlp_off_layers=fruits_off)
    results["FRUITS_PATTERN"] = run(f"FRUITS_PATTERN {fruits_off}", prunable, loader)
    results["FRUITS_PATTERN"]["off_layers"] = fruits_off

    for k in (1, 3, 8):
        key = f"RANDOM_{k}"
        results[key] = {"seeds": {}}
        accs, pds = [], []
        for seed in (0, 1, 2):
            gen = torch.Generator().manual_seed(seed)
            layers = torch.randperm(NUM_LAYERS, generator=gen)[:k].tolist()
            set_gates(prunable, mlp_off_layers=layers)
            r = run(f"{key} seed={seed} layers={layers}", prunable, loader)
            r["off_layers"] = layers
            results[key]["seeds"][str(seed)] = r
            accs.append(r["accuracy"])
            pds.append(r["prob_diff"])
        results[key]["accuracy"] = sum(accs) / len(accs)
        results[key]["prob_diff"] = sum(pds) / len(pds)
        results[key]["n"] = len(filtered)

    print("\n" + "=" * 78)
    print(f"{'Condition':<28}{'Accuracy':>10}{'ProbDiff':>12}{'Sharpness':>12}   Layers off")
    print("-" * 78)
    for cond in ("BASE", "OPEN", "FRUITS_PATTERN"):
        r = results[cond]
        print(f"{cond:<28}{r['accuracy']:>10.4f}{r['prob_diff']:>12.4f}{r['cutoff_sharpness']:>12.4f}   {r.get('off_layers', '-')}")
    for k in (1, 3, 8):
        key = f"RANDOM_{k}"
        r = results[key]
        per_seed = " ".join(f"{r['seeds'][s]['accuracy']:.3f}" for s in ("0", "1", "2"))
        print(f"{key + ' (mean of 3 seeds)':<28}{r['accuracy']:>10.4f}{r['prob_diff']:>12.4f}{'-':>12}   [{per_seed}]")
    print("=" * 78)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
