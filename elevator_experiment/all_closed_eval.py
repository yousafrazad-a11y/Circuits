"""Null baseline: accuracy of every section with ALL gates forced closed.

Closed gate = corrupted-stream passthrough, so this answers: "does this
section's target depend on the clean/corrupt difference at all?" Sections
that score ~1.0 need no circuit; sections that drop are the ones pruning
must preserve capacity for.

  ../venv/bin/python all_closed_eval.py
"""
import glob
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from train_circuit import collate, gate_modules, load_examples

ROOT = Path(__file__).resolve().parent
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LOCK_VALUE = -1e6


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    device = "cuda"
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        MODEL, PruningConfig(), torch_dtype=torch.bfloat16).to(device)

    gates = gate_modules(model)
    with torch.no_grad():
        for g in gates.values():
            g.log_alpha.fill_(LOCK_VALUE)
    model.eval()
    n_all = sum(g.log_alpha.numel() for g in gates.values())
    print(f"forced ALL {n_all} gates closed (log_alpha={LOCK_VALUE})\n")

    files = sorted(glob.glob(str(ROOT / "datasets" / "divisions" / "test_*.jsonl")))
    print(f"{'division':<34} {'acc':>7}   n")
    overall_c = overall_n = 0
    with torch.no_grad():
        for path in files:
            examples = load_examples(tok, path)
            loader = DataLoader(examples, batch_size=32, shuffle=False,
                                collate_fn=lambda b: collate(b, tok.pad_token_id))
            correct = total = 0
            for batch in loader:
                out = model(input_ids=batch["clean_ids"].to(device),
                            corrupted_input_ids=batch["corrupt_ids"].to(device),
                            attention_mask=batch["mask"].to(device),
                            use_cache=False)
                idx = torch.arange(len(batch["pos"]), device=device)
                pred = out.logits[idx, batch["pos"].to(device)].argmax(-1)
                correct += int((pred == batch["target"].to(device)).sum())
                total += len(batch["pos"])
            overall_c += correct
            overall_n += total
            print(f"{Path(path).name:<34} {correct/total:>7.4f}   {total}",
                  flush=True)
    print(f"\nOVERALL {overall_c}/{overall_n} = {overall_c/overall_n:.4f}")


if __name__ == "__main__":
    main()
