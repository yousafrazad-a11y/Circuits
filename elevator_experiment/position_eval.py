"""Per-position failure analysis for a pruned checkpoint.

For every test pair:
  1. full model greedily generates the gold trace from the clean prompt
  2. for EACH token position t of the gold trace, the pruned model
     (dual-stream, gates active) sees clean prompt + gold[:t] and its argmax
     is compared with gold[t] — teacher-forced, so errors cannot compound

Reports accuracy per token position across all pairs and the histogram of
first-failure positions. Positions are structurally aligned across examples
(same answer format), so per-position numbers are meaningful.

Example:
  ../venv/bin/python position_eval.py --checkpoint masks/all_components_l4_lr2.5.pt
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from train_circuit import chat_text, load_and_lock_masks

ROOT = Path(__file__).resolve().parent
MAX_NEW = 80


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=str(ROOT / "datasets" / "test.jsonl"))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    heads_only = ((ckpt.get("meta") or {}).get("args") or {}).get("heads_only", False)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    stop_ids = {tok.eos_token_id,
                tok.convert_tokens_to_ids("<|eot_id|>")} - {None}

    cfg = PruningConfig(prune_mlp_hidden=False, prune_mlp_output=False,
                        prune_attention_neurons=False,
                        prune_attention_blocks=False,
                        prune_mlp_blocks=False) if heads_only else PruningConfig()

    device = "cuda"
    pruned = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model, cfg, torch_dtype=torch.bfloat16).to(device)
    load_and_lock_masks(pruned, args.checkpoint)
    pruned.eval()
    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    full.eval()

    pairs = [json.loads(l) for l in open(args.data)]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    pos_correct = defaultdict(int)
    pos_total = defaultdict(int)
    first_fail = defaultdict(int)
    pos_token = {}  # position -> representative token string
    clean_lens = []

    for i, p in enumerate(pairs):
        clean_ids = tok(chat_text(tok, p["clean"]["prompt"]),
                        add_special_tokens=True)["input_ids"]
        corrupt_ids = tok(chat_text(tok, p["corrupt"]["prompt"]),
                          add_special_tokens=True)["input_ids"]
        clean_lens.append(len(clean_ids))

        # gold trace from the full model
        out = full.generate(torch.tensor([clean_ids], device=device),
                            max_new_tokens=MAX_NEW, do_sample=False,
                            eos_token_id=list(stop_ids),
                            pad_token_id=tok.eos_token_id)
        gold = out[0, len(clean_ids):].tolist()

        # teacher-forced per-position scoring, batched over positions:
        # one dual-stream forward on prompt+gold gives argmax at every cut
        seq_c = clean_ids + gold
        seq_x = corrupt_ids + gold
        out = pruned(input_ids=torch.tensor([seq_c], device=device),
                     corrupted_input_ids=torch.tensor([seq_x], device=device),
                     use_cache=False)
        preds = out.logits[0, len(clean_ids) - 1: len(clean_ids) - 1 + len(gold)]
        preds = preds.argmax(-1).tolist()

        failed = False
        for t, (pred, g) in enumerate(zip(preds, gold)):
            if g in stop_ids:
                continue
            pos_total[t] += 1
            pos_token.setdefault(t, tok.decode([g]))
            if pred == g:
                pos_correct[t] += 1
            elif not failed:
                first_fail[t] += 1
                failed = True
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(pairs)} scored", flush=True)

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"clean prompt token length: min={min(clean_lens)} "
          f"max={max(clean_lens)} (corrupt same length: "
          f"{min(clean_lens) == max(clean_lens)})\n")
    print(f"{'pos':>4} {'gold token':<14} {'acc':>7}   n")
    for t in sorted(pos_total):
        print(f"{t:>4} {pos_token[t]!r:<14} "
              f"{pos_correct[t]/pos_total[t]:>7.4f}   {pos_total[t]}")
    print(f"\nfirst-failure position histogram:")
    for t in sorted(first_fail):
        print(f"  pos {t:>3} ({pos_token.get(t)!r}): {first_fail[t]} pairs")
    n_fail = sum(first_fail.values())
    print(f"  (no failure: {len(pairs) - n_fail} pairs)")


if __name__ == "__main__":
    main()
