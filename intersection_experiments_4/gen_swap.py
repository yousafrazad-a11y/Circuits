"""Cross-assembly leak test: do MLP blocks carry category vocabulary?

Builds hybrid masks (head gates from category A, MLP block gates from category B)
and checks whether B-category words leak into generations on other categories'
prompts. E.g. fruits_heads + colors_MLPs on a fruits prompt leaking "red"/"blue"
is direct evidence that MLP blocks carry category-specific vocabulary.

Run from repo root:
    venv/bin/python intersection_experiments_4/gen_swap.py
"""
import os
import sys
import json
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import CategoryDataset
from gen_probes import generate

FRUITS = "intersection_experiments_4/masks/frozen_fruits_300ep_l005_mask.pt"
COLORS = "intersection_experiments_4/masks/frozen_colors_300ep_l005_mask.pt"
DATASETS = ["fruits_2", "animals_2", "vehicles_2", "colors_2"]
N_EXAMPLES = 5
GEN_TOKENS = 20
OUT_PATH = "intersection_experiments_4/results/swap_generation.txt"


def build_hybrid(head_mask_path, mlp_mask_path, out_path):
    hm = torch.load(head_mask_path, weights_only=True)
    mm = torch.load(mlp_mask_path, weights_only=True)
    hybrid = {}
    for k in hm:
        if 'head_gates' in k:
            hybrid[k] = hm[k].clone()
        elif 'mlp_block_gate' in k:
            hybrid[k] = mm[k].clone()
    torch.save(hybrid, out_path)
    return out_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(7)

    with open("intersection_experiments_4/categories.json") as f:
        categories = json.load(f)
    color_words = set(w.lower() for w in categories["colors_2"])
    fruit_words = set(w.lower() for w in categories["fruits_2"])

    masks = {
        "fruitsH_colorsM": build_hybrid(FRUITS, COLORS, "intersection_experiments_4/masks/swap_fruitsH_colorsM.pt"),
        "fruitsH_fruitsM (control)": FRUITS,
        "colorsH_colorsM (control)": COLORS,
        "colorsH_fruitsM": build_hybrid(COLORS, FRUITS, "intersection_experiments_4/masks/swap_colorsH_fruitsM.pt"),
    }

    samples = {}
    for ds_name in DATASETS:
        ds = CategoryDataset(f"intersection_experiments_4/datasets/{ds_name}.jsonl")
        idxs = random.sample(range(len(ds)), N_EXAMPLES)
        samples[ds_name] = [ds.data[i] for i in idxs]

    with open(OUT_PATH, 'w') as f:
        for mask_name, mask_path in masks.items():
            print(f"=== {mask_name} ===")
            manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
            config = manager._get_default_config()
            config.prune_mlp_blocks = True
            manager.initialize_model(config)
            manager.load_masks(mask_path)
            manager.use_model(enable_masks=True)
            model, tok = manager.model, manager.tokenizer

            f.write(f"\n{'='*80}\nMASK: {mask_name}\n{'='*80}\n")
            for ds_name in DATASETS:
                color_hits = 0
                fruit_hits = 0
                f.write(f"\n--- PROMPTS FROM: {ds_name} ---\n")
                for i, ex in enumerate(samples[ds_name]):
                    gen = generate(model, tok, ex['clean_prompt'], ex['corr_prompt'], device, GEN_TOKENS)
                    gen_words = set(w.strip(".,;:!?\"'").lower() for w in gen.split())
                    c_hits = gen_words & color_words
                    f_hits = gen_words & fruit_words
                    color_hits += len(c_hits)
                    fruit_hits += len(f_hits)
                    leak = ""
                    if ds_name != "colors_2" and c_hits:
                        leak = f"  <<< COLOR LEAK: {sorted(c_hits)}"
                    if ds_name != "fruits_2" and f_hits:
                        leak += f"  <<< FRUIT LEAK: {sorted(f_hits)}"
                    f.write(f"\n[{i+1}] PROMPT: {ex['clean_prompt']}\n")
                    f.write(f"    TARGET: {ex['target']}\n")
                    f.write(f"    GENERATION: {gen}{leak}\n")
                f.write(f"\n  >> {ds_name}: color-word hits={color_hits}, fruit-word hits={fruit_hits}\n")
                f.flush()
                print(f"  done {ds_name} (color hits {color_hits}, fruit hits {fruit_hits})")
            del manager, model
            torch.cuda.empty_cache()

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
