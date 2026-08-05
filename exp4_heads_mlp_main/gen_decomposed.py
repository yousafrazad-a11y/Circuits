"""Decomposed-mask generation probe: is category-specific information in the MLP blocks?

Takes the frozen fruits circuit and builds three masks:
  1. fruits_full       - the frozen fruits mask as-is
  2. fruits_heads_only - fruits head gates ON pattern, ALL MLP blocks ON
  3. fruits_mlp_only   - ALL heads ON, fruits MLP block gates ON pattern

Then generates on random prompts from fruits (control), the other trained
categories, and heldout categories. If heads carry the general sequence-tracking
logic and MLPs carry category specifics, fruits_heads_only should still track
sequences on other categories while fruits_mlp_only should not.

Run from repo root:
    venv/bin/python exp4_heads_mlp_main/gen_decomposed.py
"""
import os
import sys
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import CategoryDataset
from gen_probes import generate

BASE_MASK = "exp4_heads_mlp_main/masks/frozen_fruits_300ep_l005_mask.pt"
DATASETS = ["fruits_2", "animals_2", "colors_2", "metals_2", "vehicles_2",
            "clothing_2", "sports_2"]
N_EXAMPLES = 5
GEN_TOKENS = 12
OUT_PATH = "exp4_heads_mlp_main/results/decomposed_generation.txt"


def build_variants():
    mask = torch.load(BASE_MASK, weights_only=True)
    heads_only, mlp_only = {}, {}
    for k, v in mask.items():
        if 'head_gates' in k:
            heads_only[k] = v.clone()
            mlp_only[k] = torch.ones_like(v)
        elif 'mlp_block_gate' in k:
            heads_only[k] = torch.ones_like(v)
            mlp_only[k] = v.clone()
    h_path = "exp4_heads_mlp_main/masks/fruits_heads_only.pt"
    m_path = "exp4_heads_mlp_main/masks/fruits_mlp_only.pt"
    torch.save(heads_only, h_path)
    torch.save(mlp_only, m_path)
    return {"fruits_full": BASE_MASK, "fruits_heads_only": h_path, "fruits_mlp_only": m_path}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(123)

    samples = {}
    for ds_name in DATASETS:
        ds = CategoryDataset(f"exp4_heads_mlp_main/datasets/{ds_name}.jsonl")
        idxs = random.sample(range(len(ds)), N_EXAMPLES)
        samples[ds_name] = [ds.data[i] for i in idxs]

    masks = build_variants()
    with open(OUT_PATH, 'w') as f:
        for mask_name, mask_path in masks.items():
            print(f"=== {mask_name} ===")
            manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
            config = manager._get_default_config()
            config.prune_mlp_blocks = True
            manager.initialize_model(config)
            manager.load_masks(mask_path)
            manager.use_model(enable_masks=True)
            model = manager.model
            tok = manager.tokenizer

            f.write(f"\n{'='*80}\nMASK: {mask_name}\n{'='*80}\n")
            for ds_name in DATASETS:
                f.write(f"\n--- PROMPTS FROM: {ds_name} ---\n")
                for i, ex in enumerate(samples[ds_name]):
                    gen = generate(model, tok, ex['clean_prompt'], ex['corr_prompt'], device, GEN_TOKENS)
                    f.write(f"\n[{i+1}] PROMPT: {ex['clean_prompt']}\n")
                    f.write(f"    TARGET: {ex['target']}\n")
                    f.write(f"    GENERATION: {gen}\n")
                f.flush()
                print(f"  done {ds_name}")
            del manager, model
            torch.cuda.empty_cache()

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
