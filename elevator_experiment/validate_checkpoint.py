"""Validate a saved mask checkpoint with gates ACTIVE.

Unlike the old evaluate() in train_circuit.py (single-stream, which bypasses
gates entirely), this runs the dual-stream forward with the checkpoint's
masks applied: closed heads swap in the corrupted-stream values, exactly the
semantics the gates were trained under. Gates are deterministic 0/1 in eval
mode.

Defaults to the newest masks/heads_only*.pt checkpoint and all 21 test
divisions. Prints overall accuracy plus a per-division breakdown.

Example:
  ../venv/bin/python validate_checkpoint.py
  ../venv/bin/python validate_checkpoint.py --checkpoint masks/heads_only.ep10.pt --max-examples 2000
"""
import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from train_circuit import (collate, gate_modules, load_and_lock_masks,
                           load_examples)

ROOT = Path(__file__).resolve().parent


def newest_checkpoint():
    cands = glob.glob(str(ROOT / "masks" / "heads_only*.pt")) or \
        glob.glob(str(ROOT / "masks" / "*.pt"))
    return max(cands, key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="mask .pt (default: newest masks/heads_only*.pt)")
    ap.add_argument("--data", nargs="+",
                    default=sorted(glob.glob(
                        str(ROOT / "datasets" / "divisions" / "test_*.jsonl"))))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=0,
                    help="cap total examples (0 = all)")
    args = ap.parse_args()

    ckpt_path = args.checkpoint or newest_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta_args = (ckpt.get("meta") or {}).get("args") or {}
    heads_only = meta_args.get("heads_only", False)
    epoch = (ckpt.get("meta") or {}).get("epoch", "?")
    print(f"checkpoint: {ckpt_path} (epoch {epoch}, heads_only={heads_only})")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token

    if heads_only:
        pruning_config = PruningConfig(
            prune_attention_heads=True,
            prune_mlp_hidden=False,
            prune_mlp_output=False,
            prune_attention_neurons=False,
            prune_attention_blocks=False,
            prune_mlp_blocks=False)
    else:
        pruning_config = PruningConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model, pruning_config, torch_dtype=torch.bfloat16).to(device)
    load_and_lock_masks(model, ckpt_path)
    model.eval()

    with torch.no_grad():
        gates = gate_modules(model)
        n_open = sum(int((g() > 0.5).sum()) for g in gates.values())
        n_all = sum(g.log_alpha.numel() for g in gates.values())
    print(f"gates open: {n_open}/{n_all} ({100*n_open/n_all:.2f}%)")

    examples = load_examples(tok, args.data, args.max_examples)
    print(f"evaluating {len(examples)} example pairs from "
          f"{len(args.data)} file(s) ...", flush=True)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    correct = total = 0
    per_div = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            out = model(input_ids=batch["clean_ids"].to(device),
                        corrupted_input_ids=batch["corrupt_ids"].to(device),
                        attention_mask=batch["mask"].to(device),
                        use_cache=False)
            idx = torch.arange(len(batch["pos"]), device=device)
            pred = out.logits[idx, batch["pos"].to(device)].argmax(-1)
            tgt = batch["target"].to(device)
            hit = (pred == tgt).cpu()
            correct += int(hit.sum())
            total += len(batch["pos"])
            for r, h in zip(batch["_rows"], hit.tolist()):
                per_div[r["source"]][0] += int(h)
                per_div[r["source"]][1] += 1
            if (bi + 1) % 100 == 0:
                print(f"  {total}/{len(examples)} acc={correct/total:.4f}",
                      flush=True)

    print(f"\nOVERALL: {correct}/{total} = {correct/max(total,1):.4f}\n")
    print(f"{'division':<34} {'acc':>7}   n")
    for name in sorted(per_div):
        c, n = per_div[name]
        print(f"{name:<34} {c/n:>7.4f}   {n}")


if __name__ == "__main__":
    main()
