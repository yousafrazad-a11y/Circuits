"""Generation probe: for each mask, sample N random examples from each of the
10 '_2' datasets and greedily generate tokens with the pruned (circuit) model.
Saves prompts, targets and generations to a text file for qualitative inspection.

Run from repo root:
    venv/bin/python intersection_experiments_2/gen_probes.py
"""
import os
import sys
import json
import random
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import CategoryDataset

MASKS = {
    "intersection_frozen_300_300": "intersection_experiments_2/masks/intersection_frozen_300_300.pt",
    "intersection_600": "intersection_experiments_2/masks/intersection_600.pt",
    "anti_intersection_frozen_300_300": "intersection_experiments_2/masks/anti_intersection_frozen_300_300.pt",
    "anti_intersection_600": "intersection_experiments_2/masks/anti_intersection_600.pt",
}
DATASETS = ["fruits_2", "animals_2", "colors_2", "metals_2", "vehicles_2",
            "clothing_2", "furniture_2", "instruments_2", "professions_2", "sports_2"]
TRAINED = {"fruits_2", "animals_2", "colors_2", "metals_2", "vehicles_2"}
N_EXAMPLES = 10
GEN_TOKENS = 20
OUT_PATH = "intersection_experiments_2/results/generation_probes.txt"


def generate(model, tokenizer, prompt, corr_prompt, device, n_tokens):
    tokenizer.padding_side = 'left'
    enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    corr = tokenizer(corr_prompt, return_tensors='pt', add_special_tokens=True)
    ids = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    corr_ids = corr['input_ids'].to(device)
    # left-pad corrupted stream to same length as clean
    if corr_ids.size(1) < ids.size(1):
        pad = torch.full((1, ids.size(1) - corr_ids.size(1)), tokenizer.pad_token_id,
                         dtype=corr_ids.dtype, device=device)
        corr_ids = torch.cat([pad, corr_ids], dim=1)
    elif ids.size(1) < corr_ids.size(1):
        pad = torch.full((1, corr_ids.size(1) - ids.size(1)), tokenizer.pad_token_id,
                         dtype=ids.dtype, device=device)
        ids = torch.cat([pad, ids], dim=1)
        mask = torch.cat([torch.zeros_like(pad), mask], dim=1)

    out_tokens = []
    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(input_ids=ids, attention_mask=mask,
                        corrupted_input_ids=corr_ids, use_cache=False)
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
            out_tokens.append(nxt[0].item())
            ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=-1)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=device)], dim=-1)
            corr_ids = torch.cat([corr_ids, torch.full((1, 1), tokenizer.pad_token_id,
                                                       dtype=corr_ids.dtype, device=device)], dim=-1)
    return tokenizer.decode(out_tokens, skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)

    # Pre-sample examples (same examples for every mask)
    samples = {}
    for ds_name in DATASETS:
        ds = CategoryDataset(f"intersection_experiments_2/datasets/{ds_name}.jsonl")
        idxs = random.sample(range(len(ds)), N_EXAMPLES)
        samples[ds_name] = [ds.data[i] for i in idxs]

    with open(OUT_PATH, 'w') as f:
        for mask_name, mask_path in MASKS.items():
            print(f"=== {mask_name} ===")
            manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
            tok = manager.tokenizer
            mask_state = torch.load(mask_path, weights_only=True)
            config = manager._get_default_config()
            if any('mlp_block_gate' in k for k in mask_state):
                config.prune_mlp_blocks = True
            manager.initialize_model(config)
            manager.load_masks(mask_path)
            manager.use_model(enable_masks=True)
            model = manager.model

            f.write(f"\n{'='*80}\nMASK: {mask_name}\n{'='*80}\n")
            for ds_name in DATASETS:
                split = "trained" if ds_name in TRAINED else "heldout"
                f.write(f"\n--- DATASET: {ds_name} ({split}) ---\n")
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
