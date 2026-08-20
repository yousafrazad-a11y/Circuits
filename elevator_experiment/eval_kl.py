"""Test-set KL + accuracy for a mask checkpoint (deterministic gates).

For each example in the given division file(s): dual-stream forward with the
mask applied (eval mode = deterministic gates), KL(full || pruned) at the
target position, plus argmax accuracy. Reports mean/median/p90/max KL.

  ../venv/bin/python eval_kl.py --checkpoint masks/sections/sec_03_fine.ep20.pt \
      --data datasets/divisions/test_03_step_1_word.jsonl
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from train_circuit import collate, load_and_lock_masks, load_examples

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta_args = (ckpt.get("meta") or {}).get("args") or {}
    if meta_args.get("heads_only"):
        cfg = PruningConfig(prune_mlp_hidden=False, prune_mlp_output=False,
                            prune_attention_neurons=False,
                            prune_attention_blocks=False,
                            prune_mlp_blocks=False)
    elif meta_args.get("heads_mlp_blocks"):
        cfg = PruningConfig(prune_mlp_hidden=False, prune_mlp_output=False,
                            prune_attention_neurons=False,
                            prune_attention_blocks=False,
                            prune_mlp_blocks=True, prune_full_layers=False)
    elif meta_args.get("fine"):
        cfg = PruningConfig(prune_attention_blocks=False,
                            prune_full_layers=False)
    else:
        cfg = PruningConfig()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    device = "cuda"
    pruned = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model, cfg, torch_dtype=torch.bfloat16).to(device)
    load_and_lock_masks(pruned, args.checkpoint)
    pruned.eval()
    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    full.eval()

    examples = load_examples(tok, args.data)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    kls, correct, total = [], 0, 0
    with torch.no_grad():
        for batch in loader:
            kw = dict(input_ids=batch["clean_ids"].to(device),
                      attention_mask=batch["mask"].to(device), use_cache=False)
            out_p = pruned(corrupted_input_ids=batch["corrupt_ids"].to(device),
                           **kw)
            out_f = full(**kw)
            idx = torch.arange(len(batch["pos"]), device=device)
            pos = batch["pos"].to(device)
            lp = F.log_softmax(out_p.logits[idx, pos].float(), dim=-1)
            lf = F.log_softmax(out_f.logits[idx, pos].float(), dim=-1)
            kl = F.kl_div(lp, lf, log_target=True,
                          reduction="none").sum(-1)
            kls.extend(kl.tolist())
            pred = out_p.logits[idx, pos].argmax(-1)
            correct += int((pred == batch["target"].to(device)).sum())
            total += len(batch["pos"])

    k = torch.tensor(kls)
    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data}  (n={total})")
    print(f"accuracy:  {correct}/{total} = {correct/total:.4f}")
    print(f"KL(full||pruned) at target: mean={k.mean():.4f} "
          f"median={k.median():.4f} p90={k.quantile(0.9):.4f} "
          f"max={k.max():.4f}")


if __name__ == "__main__":
    main()
