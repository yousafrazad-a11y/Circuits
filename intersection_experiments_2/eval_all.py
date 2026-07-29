"""Batch evaluation: evaluate one mask on all 10 '_2' test datasets in a single
model load. Reuses the evaluation logic from evaluate_mask.py (same metrics,
same CSV columns). Run from repo root:

    venv/bin/python intersection_experiments_2/eval_all.py \
        --mask intersection_experiments_2/masks/intersection_600.pt \
        --output eval_intersection_600.csv
"""
import os
import sys
import json
import csv
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import CategoryDataset, collate_fn, evaluate_accuracies, get_active_heads

DATASETS = ["fruits_2", "animals_2", "colors_2", "metals_2", "vehicles_2",
            "clothing_2", "furniture_2", "instruments_2", "professions_2", "sports_2"]
TRAINED = {"fruits_2", "animals_2", "colors_2", "metals_2", "vehicles_2"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate one mask on all 10 '_2' datasets.")
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="CSV file name in results dir.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open("intersection_experiments_2/categories.json") as f:
        categories = json.load(f)

    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)

    print(f"Loading mask from {args.mask}...")
    mask_state = torch.load(args.mask, weights_only=True)
    config = manager._get_default_config()
    if any('mlp_block_gate' in k for k in mask_state):
        config.prune_mlp_blocks = True
        print("Mask contains mlp_block_gate entries -> enabling MLP block pruning.")
    manager.initialize_model(config)
    manager.load_masks(args.mask)
    active_heads, total_heads = get_active_heads(args.mask)
    print(f"Active Heads in mask: {active_heads}/{total_heads}")

    os.makedirs("intersection_experiments_2/results", exist_ok=True)
    output_path = os.path.join("intersection_experiments_2/results", args.output)
    if not output_path.endswith('.csv'):
        output_path += '.csv'
    keys = ["mask_path", "dataset", "split", "base_prob_acc", "base_gen_acc",
            "circ_prob_acc", "circ_gen_acc", "kl_divergence", "active_heads", "total_heads"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for ds_name in DATASETS:
            ds_path = f"intersection_experiments_2/datasets/{ds_name}.jsonl"
            ds = CategoryDataset(ds_path)
            dl = DataLoader(ds, batch_size=8, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, manager.tokenizer))
            cat_set = categories[ds_name]

            print(f"\n--- EVALUATING {ds_name.upper()} ---")
            base_prob, base_gen = evaluate_accuracies(manager.baseline_model, dl, manager.tokenizer, cat_set, device)
            manager.use_model(enable_masks=True)
            circ_prob, circ_gen = evaluate_accuracies(manager.model, dl, manager.tokenizer, cat_set, device)
            kl = manager.evaluate_kl_divergence(dl)

            split = "trained" if ds_name in TRAINED else "heldout"
            print(f"{ds_name} ({split}) | Base P/G: {base_prob:.4f}/{base_gen:.4f} | "
                  f"Circ P/G: {circ_prob:.4f}/{circ_gen:.4f} | KL: {kl:.4f}")
            writer.writerow({
                "mask_path": args.mask, "dataset": ds_name, "split": split,
                "base_prob_acc": f"{base_prob:.4f}", "base_gen_acc": f"{base_gen:.4f}",
                "circ_prob_acc": f"{circ_prob:.4f}", "circ_gen_acc": f"{circ_gen:.4f}",
                "kl_divergence": f"{kl:.4f}",
                "active_heads": active_heads, "total_heads": total_heads,
            })
            f.flush()

    print(f"\nAll results written to {output_path}")


if __name__ == "__main__":
    main()
