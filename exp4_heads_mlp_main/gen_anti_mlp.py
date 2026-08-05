"""Anti-MLP probe: category's heads + INVERTED category MLP gates.

For each frozen category mask, build a mask with the category's own head gates
but its MLP block gates inverted: the MLP blocks the category selected are
turned OFF (corrupted), all blocks it pruned are turned ON (clean).
Then generate on 10 random prompts from the category's own test set.

If the selected MLP blocks carry something the category needs, performance
should collapse vs the full mask / heads-only mask. If the choice is arbitrary
(solution degeneracy), it should keep working.

Run from repo root:
    venv/bin/python exp4_heads_mlp_main/gen_anti_mlp.py
"""
import os
import sys
import json
import random
import re
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import CategoryDataset
from gen_probes import generate

CATS = ["fruits", "animals", "colors", "metals", "vehicles"]
N_EXAMPLES = 10
GEN_TOKENS = 12
OUT_PATH = "exp4_heads_mlp_main/results/anti_mlp_generation.txt"


def build_anti_mlp(mask_path, out_path):
    mask = torch.load(mask_path, weights_only=True)
    anti = {}
    for k, v in mask.items():
        if 'head_gates' in k:
            anti[k] = v.clone()          # keep the category's head selection
        elif 'mlp_block_gate' in k:
            anti[k] = ~v                 # invert the category's MLP selection
    torch.save(anti, out_path)
    return out_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(99)

    with open("exp4_heads_mlp_main/categories.json") as f:
        categories = json.load(f)

    with open(OUT_PATH, 'w') as f:
        for cat in CATS:
            src = f"exp4_heads_mlp_main/masks/frozen_{cat}_300ep_l005_mask.pt"
            anti_path = build_anti_mlp(src, f"exp4_heads_mlp_main/masks/anti_mlp_{cat}.pt")
            print(f"=== {cat}: heads from {cat}, MLPs inverted ===")

            manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
            config = manager._get_default_config()
            config.prune_mlp_blocks = True
            manager.initialize_model(config)
            manager.load_masks(anti_path)
            manager.use_model(enable_masks=True)
            model, tok = manager.model, manager.tokenizer

            ds_name = f"{cat}_2"
            ds = CategoryDataset(f"exp4_heads_mlp_main/datasets/{ds_name}.jsonl")
            idxs = random.sample(range(len(ds)), N_EXAMPLES)
            cat_words = set(w.lower() for w in categories[ds_name])

            f.write(f"\n{'='*80}\nMASK: {cat} heads + INVERTED {cat} MLPs | PROMPTS: {ds_name}\n{'='*80}\n")
            correct = 0
            for i, idx in enumerate(idxs):
                ex = ds.data[idx]
                gen = generate(model, tok, ex['clean_prompt'], ex['corr_prompt'], device, GEN_TOKENS)
                first = re.sub(r'[^a-zA-Z]', '', gen.strip().split()[0] if gen.strip() else "").lower()
                ok = first == ex['target'].lower()
                correct += ok
                f.write(f"\n[{i+1}] PROMPT: {ex['clean_prompt']}\n")
                f.write(f"    TARGET: {ex['target']}  |  FIRST TOKEN: '{first}'  {'CORRECT' if ok else 'wrong'}\n")
                f.write(f"    GENERATION: {gen}\n")
            f.write(f"\n  >> {cat}: first-token correct {correct}/{N_EXAMPLES}\n")
            f.flush()
            print(f"  {cat}: {correct}/{N_EXAMPLES} correct")
            del manager, model
            torch.cuda.empty_cache()

    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
