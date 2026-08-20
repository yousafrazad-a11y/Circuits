"""Generation-level eval: pruned model (mask checkpoint) vs full model.

For every pair in the test split, both models generate greedily from the
CLEAN task prompt (same chat template used everywhere else).

Pruned generation is dual-stream: the corrupted stream starts from the
corrupt prompt and receives the same generated tokens; closed gates swap in
corrupted activations — exactly the semantics the gates were trained under.
(Plain .generate() would bypass gates entirely.)

Reports:
  1. exact-match:     % of pruned generations identical to full model's
  2. final-answer:    % of pruned generations whose FINAL ANSWER floor is correct
  (plus full-model final-answer accuracy as a sanity check)

Example:
  ../venv/bin/python gen_eval.py --checkpoint masks/heads_only.ep10.pt
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from train_circuit import chat_text, load_and_lock_masks

ROOT = Path(__file__).resolve().parent
MAX_NEW = 80


def parse_final(text):
    m = re.search(r"FINAL ANSWER:\s*(\S+)", text)
    return m.group(1) if m else None


@torch.no_grad()
def pruned_generate(model, clean_ids, corrupt_ids, stop_ids, device):
    """Greedy dual-stream decode; returns list of generated token ids."""
    clean, corr = list(clean_ids), list(corrupt_ids)
    for _ in range(MAX_NEW):
        out = model(input_ids=torch.tensor([clean], device=device),
                    corrupted_input_ids=torch.tensor([corr], device=device),
                    use_cache=False)
        nxt = int(out.logits[0, -1].argmax())
        clean.append(nxt)
        corr.append(nxt)
        if nxt in stop_ids:
            break
    return clean[len(clean_ids):]


@torch.no_grad()
def full_generate(model, ids, stop_ids, device):
    out = model.generate(torch.tensor([ids], device=device),
                         max_new_tokens=MAX_NEW, do_sample=False,
                         eos_token_id=list(stop_ids),
                         pad_token_id=list(stop_ids)[0])
    return out[0, len(ids):].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="masks/heads_only.ep10.pt")
    ap.add_argument("--data", default=str(ROOT / "datasets" / "test.jsonl"))
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args()

    ckpt_path = str(ROOT / args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta_args = (ckpt.get("meta") or {}).get("args") or {}
    heads_only = meta_args.get("heads_only", False)
    print(f"checkpoint: {ckpt_path} (epoch {(ckpt.get('meta') or {}).get('epoch', '?')})")

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
    load_and_lock_masks(pruned, ckpt_path)
    pruned.eval()

    full = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    full.eval()

    pairs = [json.loads(l) for l in open(args.data)]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    n_exact = n_final_ok = n_full_ok = 0
    mismatches = []
    for i, p in enumerate(pairs):
        clean_ids = tok(chat_text(tok, p["clean"]["prompt"]),
                        add_special_tokens=True)["input_ids"]
        corrupt_ids = tok(chat_text(tok, p["corrupt"]["prompt"]),
                          add_special_tokens=True)["input_ids"]
        expected = parse_final(p["clean"]["answer"])

        p_ids = pruned_generate(pruned, clean_ids, corrupt_ids, stop_ids, device)
        f_ids = full_generate(full, clean_ids, stop_ids, device)
        p_text = tok.decode(p_ids, skip_special_tokens=True).strip()
        f_text = tok.decode(f_ids, skip_special_tokens=True).strip()

        exact = p_text == f_text
        p_final = parse_final(p_text)
        f_final = parse_final(f_text)
        n_exact += exact
        n_final_ok += (p_final == expected)
        n_full_ok += (f_final == expected)
        if not exact and len(mismatches) < 5:
            mismatches.append((p["pair_id"], f_text, p_text))
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(pairs)} exact={n_exact/(i+1):.4f} "
                  f"final={n_final_ok/(i+1):.4f} full={n_full_ok/(i+1):.4f}",
                  flush=True)

    n = len(pairs)
    print(f"\n=== {n} pairs, checkpoint {ckpt_path} ===")
    print(f"generation exact-match vs full model: {n_exact}/{n} = {n_exact/n:.4f}")
    print(f"pruned final-answer accuracy:         {n_final_ok}/{n} = {n_final_ok/n:.4f}")
    print(f"full-model final-answer accuracy:     {n_full_ok}/{n} = {n_full_ok/n:.4f}")

    for pid, f, pp in mismatches:
        print(f"\n--- mismatch pair {pid} ---\nfull:\n{f}\npruned:\n{pp}")


if __name__ == "__main__":
    main()
