"""Batch evaluation: evaluate one mask on all three A/B/C test sets in a single
model load. Run from repo root:

    venv/bin/python exp5_abc_retrieval/eval_all.py \
        --mask exp5_abc_retrieval/masks/combined_300_mask.pt \
        --output eval_combined.csv

Omit --mask for base-model evaluation.
"""
import os
import sys
import csv
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager
from evaluate_mask import MemoryDataset, collate_fn, evaluate_accuracies, get_active_heads

DATASETS = ["A_test.jsonl", "B_test.jsonl", "C_test.jsonl"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate one mask on A/B/C test sets.")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output", type=str, required=True, help="CSV file name in results dir.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)

    active_heads, total_heads = (None, None)
    if args.mask:
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
    else:
        manager.initialize_model(manager._get_default_config())

    os.makedirs("exp5_abc_retrieval/results", exist_ok=True)
    output_path = os.path.join("exp5_abc_retrieval/results", args.output)
    if not output_path.endswith('.csv'):
        output_path += '.csv'
    keys = ["mask_path", "dataset", "base_prob_acc", "base_gen_acc", "base_logit_diff",
            "circ_prob_acc", "circ_gen_acc", "circ_logit_diff", "kl_divergence",
            "active_heads", "total_heads"]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for ds_file in DATASETS:
            ds = MemoryDataset(f"exp5_abc_retrieval/datasets/{ds_file}")
            dl = DataLoader(ds, batch_size=8, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, manager.tokenizer))
            ds_name = ds_file.replace(".jsonl", "")

            print(f"\n--- EVALUATING {ds_name.upper()} ---")
            base_prob, base_gen, base_ld = evaluate_accuracies(manager.baseline_model, dl, manager.tokenizer, device)

            circ_prob = circ_gen = circ_ld = kl = None
            if args.mask:
                manager.use_model(enable_masks=True)
                circ_prob, circ_gen, circ_ld = evaluate_accuracies(manager.model, dl, manager.tokenizer, device)
                kl = manager.evaluate_kl_divergence(dl)

            print(f"{ds_name} | Base P/G/LD: {base_prob:.4f}/{base_gen:.4f}/{base_ld:.3f}"
                  + (f" | Circ P/G/LD: {circ_prob:.4f}/{circ_gen:.4f}/{circ_ld:.3f} | KL: {kl:.4f}" if args.mask else ""))
            writer.writerow({
                "mask_path": args.mask or "BASE", "dataset": ds_name,
                "base_prob_acc": f"{base_prob:.4f}", "base_gen_acc": f"{base_gen:.4f}",
                "base_logit_diff": f"{base_ld:.4f}",
                "circ_prob_acc": f"{circ_prob:.4f}" if circ_prob is not None else "",
                "circ_gen_acc": f"{circ_gen:.4f}" if circ_gen is not None else "",
                "circ_logit_diff": f"{circ_ld:.4f}" if circ_ld is not None else "",
                "kl_divergence": f"{kl:.4f}" if kl is not None else "",
                "active_heads": active_heads if active_heads is not None else "",
                "total_heads": total_heads if total_heads is not None else "",
            })
            f.flush()

    print(f"\nAll results written to {output_path}")


if __name__ == "__main__":
    main()
