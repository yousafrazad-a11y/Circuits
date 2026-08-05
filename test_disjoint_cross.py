import os
import sys
import json
import random
import argparse
import torch

sys.path.insert(0, os.path.abspath("."))
from test_disjoint import collate_fn, prob_predictions_batch, generate_batch

BASE_CATS = ["fruits", "animals", "colors", "metals", "vehicles"]
ALL_CATS = BASE_CATS + ["instruments", "sports", "professions", "clothing", "furniture"]

def main():
    parser = argparse.ArgumentParser(description="Test each category's disjoint (anti-intersection) mask on a random OTHER dataset.")
    parser.add_argument("--exp_dir", type=str, default="exp2_heads_mlp_overpruned")
    parser.add_argument("--split", type=str, default="2", help="Dataset version to sample test examples from.")
    parser.add_argument("--n", type=int, default=10, help="Examples per test.")
    parser.add_argument("--gen_tokens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = args.out or os.path.join(args.exp_dir, "results", "disjoint_cross_test.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(os.path.join(args.exp_dir, "categories.json")) as f:
        categories = json.load(f)

    # Random assignment: each base category -> random other category's dataset
    rng = random.Random(args.seed)
    assignment = {}
    for c in BASE_CATS:
        choices = [x for x in ALL_CATS if x != c]
        assignment[c] = rng.choice(choices)
    print("Assignment (mask -> test dataset):", assignment)

    sys.path.insert(0, os.path.abspath(args.exp_dir))
    from pruning_manager import CircuitPruningManager

    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    config = manager._get_default_config()
    config.prune_mlp_blocks = True  # disjoint masks contain mlp_block_gate entries
    manager.initialize_model(config)
    tokenizer = manager.tokenizer

    lines = []
    lines.append(f"Cross test: each disjoint mask on a random OTHER dataset ({args.split} split), seed {args.seed}")
    lines.append(f"Assignment: {assignment}")
    lines.append("")

    for c in BASE_CATS:
        target_cat = assignment[c]
        dataset_key = f"{target_cat}_{args.split}"
        mask_path = os.path.join(args.exp_dir, "masks", "disjoint", f"l01_frozen_{c}_300ep_mask_minus_intersection.pt")

        with open(os.path.join(args.exp_dir, "datasets", f"{dataset_key}.jsonl")) as f:
            data = [json.loads(l) for l in f]
        examples = rng.sample(data, args.n)

        manager.load_masks(mask_path)
        manager.use_model(enable_masks=True)
        heads = sum(int(v.sum()) for k, v in torch.load(mask_path, weights_only=True).items() if 'head_gates' in k)
        mlps = sum(int(v.sum()) for k, v in torch.load(mask_path, weights_only=True).items() if 'mlp_block_gate' in k)

        batch = collate_fn(examples, tokenizer)
        cat_set = categories[dataset_key]
        preds = prob_predictions_batch(manager.model, batch, tokenizer, cat_set, device)
        circ_gens = generate_batch(manager.model, batch, tokenizer, args.gen_tokens, device, use_corrupted=True)
        base_gens = generate_batch(manager.baseline_model, batch, tokenizer, args.gen_tokens, device, use_corrupted=False)

        lines.append("=" * 78)
        lines.append(f"MASK: {c} minus intersection ({heads} heads, {mlps} MLP blocks)  ->  TEST: {dataset_key}")
        lines.append("=" * 78)
        n_ok = 0
        for i, ex in enumerate(examples, 1):
            ok = preds[i-1].lower() == ex['target'].lower()
            n_ok += ok
            lines.append(f"\n[{i}] Prompt: {ex['clean_prompt']}")
            lines.append(f"    Target: {ex['target']}")
            lines.append(f"    Circuit prob-pred: {preds[i-1]}  ({'OK' if ok else 'WRONG'})")
            lines.append(f"    Circuit gen: {circ_gens[i-1]!r}")
            lines.append(f"    Base gen:    {base_gens[i-1]!r}")
        lines.append(f"\nprob-pred on these {args.n} examples: {n_ok}/{args.n}")
        lines.append("")
        print(f"{c} -> {dataset_key}: {n_ok}/{args.n}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
