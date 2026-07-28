import os
import sys
import json
import glob
import random
import argparse
import torch

CATEGORIES = ["fruits", "animals", "colors", "metals", "vehicles"]

def collate_fn(batch, tokenizer):
    """Same left-padding logic as evaluate_mask.py (clean/corrupted padded to equal length)."""
    tokenizer.padding_side = 'left'
    clean_enc = tokenizer([b['clean_prompt'] for b in batch], padding=True, return_tensors='pt', add_special_tokens=True)
    corr_enc = tokenizer([b['corr_prompt'] for b in batch], padding=True, return_tensors='pt', add_special_tokens=True)
    max_len = max(clean_enc['input_ids'].size(1), corr_enc['input_ids'].size(1))

    def pad_left(t, val):
        pad = max_len - t.size(1)
        return torch.cat([torch.full((t.size(0), pad), val, dtype=t.dtype), t], dim=1) if pad > 0 else t

    return {
        'input_ids': pad_left(clean_enc['input_ids'], tokenizer.pad_token_id),
        'attention_mask': pad_left(clean_enc['attention_mask'], 0),
        'corrupted_input_ids': pad_left(corr_enc['input_ids'], tokenizer.pad_token_id),
    }

def prob_predictions_batch(model, batch, tokenizer, cat_set, device):
    """Per-example argmax of last-position logits restricted to the category tokens."""
    cat_tokens = {w: tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in cat_set}
    all_toks = list(cat_tokens.values())
    tok_to_word = {v: k for k, v in cat_tokens.items()}
    with torch.no_grad():
        out = model(input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    corrupted_input_ids=batch['corrupted_input_ids'].to(device),
                    use_cache=False)
    best = torch.argmax(out.logits[:, -1, :][:, all_toks], dim=-1)
    return [tok_to_word[all_toks[i.item()]] for i in best]

def generate_batch(model, batch, tokenizer, n_tokens, device, use_corrupted):
    """Per-example greedy generation of n_tokens. Returns list of decoded strings."""
    curr_input = batch['input_ids'].to(device)
    curr_mask = batch['attention_mask'].to(device)
    curr_corr = batch['corrupted_input_ids'].to(device) if use_corrupted else None
    bsz = curr_input.size(0)
    gen_ids = [[] for _ in range(bsz)]
    with torch.no_grad():
        for _ in range(n_tokens):
            if curr_corr is not None:
                out = model(input_ids=curr_input, attention_mask=curr_mask, corrupted_input_ids=curr_corr, use_cache=False)
            else:
                out = model(input_ids=curr_input, attention_mask=curr_mask, use_cache=False)
            next_toks = torch.argmax(out.logits[:, -1, :], dim=-1)
            for i in range(bsz):
                gen_ids[i].append(next_toks[i].item())
            curr_input = torch.cat([curr_input, next_toks.unsqueeze(-1)], dim=-1)
            curr_mask = torch.cat([curr_mask, torch.ones((bsz, 1), device=device, dtype=curr_mask.dtype)], dim=-1)
            if curr_corr is not None:
                curr_corr = torch.cat([curr_corr, torch.full((bsz, 1), tokenizer.pad_token_id, device=device)], dim=-1)
    return [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen_ids]

def main():
    parser = argparse.ArgumentParser(description="Sample N examples per category, generate with disjoint-part masks, save for inspection.")
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Experiment folder, e.g. intersection_experiments_2 or intersection_experiments_2_full")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory containing the *_minus_intersection.pt masks (default: <exp_dir>/masks/disjoint)")
    parser.add_argument("--n", type=int, default=10, help="Number of random examples per dataset.")
    parser.add_argument("--gen_tokens", type=int, default=6, help="Number of tokens to generate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Output text file (default: <exp_dir>/results/disjoint_test_outputs.txt)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_dir = args.mask_dir or os.path.join(args.exp_dir, "masks", "disjoint")
    out_path = args.out or os.path.join(args.exp_dir, "results", "disjoint_test_outputs.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(os.path.join(args.exp_dir, "categories.json")) as f:
        categories = json.load(f)

    sys.path.insert(0, os.path.abspath(args.exp_dir))
    from pruning_manager import CircuitPruningManager

    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    manager.initialize_model()
    tokenizer = manager.tokenizer

    lines = []
    for cat in CATEGORIES:
        dataset_key = f"{cat}_1"
        with open(os.path.join(args.exp_dir, "datasets", f"{dataset_key}.jsonl")) as f:
            data = [json.loads(l) for l in f]

        mask_matches = glob.glob(os.path.join(mask_dir, f"*{cat}*_minus_intersection.pt"))
        if len(mask_matches) != 1:
            raise ValueError(f"Expected 1 disjoint mask for '{cat}' in {mask_dir}, found {mask_matches}")
        mask_path = mask_matches[0]

        manager.load_masks(mask_path)
        manager.use_model(enable_masks=True)

        heads = sum(int(v.sum()) for k, v in torch.load(mask_path, weights_only=True).items() if 'head_gates' in k)

        rng = random.Random(args.seed)
        examples = rng.sample(data, args.n)
        batch = collate_fn(examples, tokenizer)
        cat_set = categories[dataset_key]

        preds = prob_predictions_batch(manager.model, batch, tokenizer, cat_set, device)
        circ_gens = generate_batch(manager.model, batch, tokenizer, args.gen_tokens, device, use_corrupted=True)
        base_gens = generate_batch(manager.baseline_model, batch, tokenizer, args.gen_tokens, device, use_corrupted=False)

        lines.append("=" * 70)
        lines.append(f"DATASET: {dataset_key} | MASK: {os.path.basename(mask_path)} ({heads}/512 heads)")
        lines.append("=" * 70)

        n_ok = 0
        for i, ex in enumerate(examples, 1):
            ok = preds[i-1].lower() == ex['target'].lower()
            n_ok += ok
            lines.append(f"\n[{i}] Prompt: {ex['clean_prompt']}")
            lines.append(f"    Target: {ex['target']}")
            lines.append(f"    Circuit prob-pred: {preds[i-1]}  ({'OK' if ok else 'WRONG'})")
            lines.append(f"    Circuit gen: {circ_gens[i-1]!r}")
            lines.append(f"    Base gen:    {base_gens[i-1]!r}")
            print(f"{cat} [{i}/{args.n}] pred={preds[i-1]} target={ex['target']} {'OK' if ok else 'WRONG'}")

        lines.append(f"\nprob-pred on these {args.n} examples: {n_ok}/{args.n}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
