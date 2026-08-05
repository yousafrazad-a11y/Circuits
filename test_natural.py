import os
import sys
import json
import argparse
import torch

def collate_fn(batch, tokenizer):
    """Left-pad clean + corrupted prompts to equal length (same as evaluate_mask.py)."""
    tokenizer.padding_side = 'left'
    clean_enc = tokenizer([b['prompt'] for b in batch], padding=True, return_tensors='pt', add_special_tokens=True)
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
    parser = argparse.ArgumentParser(description="Generate continuations for natural-language tracking prompts with full model + circuit masks.")
    parser.add_argument("--exp_dir", type=str, default="exp3_dual_init_union")
    parser.add_argument("--data", type=str, default=None, help="JSONL with prompt/corr_prompt/target fields (default: <exp_dir>/natural_tracking_test.jsonl)")
    parser.add_argument("--masks", type=str, nargs='+', default=["union_ns_dn_fruits_300", "anti_union_ns_dn_fruits_300"],
                        help="Mask base names in <exp_dir>/masks/ to evaluate alongside the full model.")
    parser.add_argument("--gen_tokens", type=int, default=25)
    parser.add_argument("--out", type=str, default=None, help="Default: <exp_dir>/results/natural_tracking_generations.txt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = args.data or os.path.join(args.exp_dir, "natural_tracking_test.jsonl")
    out_path = args.out or os.path.join(args.exp_dir, "results", "natural_tracking_generations.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(data_path) as f:
        examples = [json.loads(l) for l in f]

    sys.path.insert(0, os.path.abspath(args.exp_dir))
    from pruning_manager import CircuitPruningManager

    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    manager.initialize_model()
    tokenizer = manager.tokenizer
    batch = collate_fn(examples, tokenizer)

    # Full (unpruned) model
    print("Generating with full model...")
    gens = {"full_model": generate_batch(manager.baseline_model, batch, tokenizer, args.gen_tokens, device, use_corrupted=False)}

    # Each circuit mask
    for m in args.masks:
        mask_path = os.path.join(args.exp_dir, "masks", f"{m}_mask.pt")
        print(f"Generating with {m}...")
        manager.load_masks(mask_path)
        manager.use_model(enable_masks=True)
        gens[m] = generate_batch(manager.model, batch, tokenizer, args.gen_tokens, device, use_corrupted=True)

    cols = ["full_model"] + args.masks
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append("=" * 78)
        lines.append(f"[{ex['id']}] ({ex['difficulty']}) chain: {' -> '.join(ex['chain'])}")
        lines.append(f"PROMPT: {ex['prompt']}")
        lines.append(f"EXPECTED: {ex['target']}")
        for c in cols:
            lines.append(f"  {c}: {gens[c][i-1]!r}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
