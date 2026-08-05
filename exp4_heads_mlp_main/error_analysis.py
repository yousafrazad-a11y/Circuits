"""Error-type analysis: what do the missing disjoint heads contribute?

Runs the heads-only intersection mask (47 shared heads + all MLPs clean) on the
5 trained test sets and classifies every first-token error:
  correct | in_category (right category, wrong word) | off_category | empty
Compare with individual heads-only masks (e.g. fruits_heads_only) to see which
error type the disjoint heads fix.

Run from repo root:
    venv/bin/python exp4_heads_mlp_main/error_analysis.py
"""
import os
import sys
import json
import re
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import CategoryDataset
from gen_probes import generate

MASKS = {
    "heads_intersection (47)": "exp4_heads_mlp_main/masks/intersection_frozen_heads_only.pt",
    "fruits_heads_only (94)": "exp4_heads_mlp_main/masks/fruits_heads_only.pt",
}
DATASETS = ["fruits_2", "animals_2", "colors_2", "metals_2", "vehicles_2"]
N_EXAMPLES = 20
GEN_TOKENS = 3
OUT_PATH = "exp4_heads_mlp_main/results/error_analysis.txt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open("exp4_heads_mlp_main/categories.json") as f:
        categories = json.load(f)

    with open(OUT_PATH, 'w') as f:
        for mask_name, mask_path in MASKS.items():
            print(f"=== {mask_name} ===")
            manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
            config = manager._get_default_config()
            config.prune_mlp_blocks = True
            manager.initialize_model(config)
            manager.load_masks(mask_path)
            manager.use_model(enable_masks=True)
            model, tok = manager.model, manager.tokenizer

            f.write(f"\nMASK: {mask_name}\n")
            for ds_name in DATASETS:
                ds = CategoryDataset(f"exp4_heads_mlp_main/datasets/{ds_name}.jsonl")
                cat_words = set(w.lower() for w in categories[ds_name])
                stats = {"correct": 0, "in_category": 0, "off_category": 0, "empty": 0}
                examples = []
                for i in range(N_EXAMPLES):
                    ex = ds.data[i]
                    gen = generate(model, tok, ex['clean_prompt'], ex['corr_prompt'], device, GEN_TOKENS)
                    first = re.sub(r'[^a-zA-Z]', '', gen.strip().split()[0] if gen.strip() else "").lower()
                    if first == ex['target'].lower():
                        stats["correct"] += 1
                    elif not first:
                        stats["empty"] += 1
                    elif first in cat_words:
                        stats["in_category"] += 1
                        examples.append((ex['clean_prompt'], ex['target'], first))
                    else:
                        stats["off_category"] += 1
                        examples.append((ex['clean_prompt'], ex['target'], first))
                f.write(f"  {ds_name}: " + " ".join(f"{k}={v}" for k, v in stats.items()) + "\n")
                for p, t, g in examples[:4]:
                    f.write(f"      err: target={t} got={g} | {p}\n")
                f.flush()
                print(f"  {ds_name}: {stats}")
            del manager, model
            torch.cuda.empty_cache()

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
