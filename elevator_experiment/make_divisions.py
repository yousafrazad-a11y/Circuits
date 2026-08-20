"""Build trainable per-section divisions from the sectioned dataset.

One output file per of the 21 sections (7 lines x 3 parts), per split:

  datasets/divisions/{split}_{NN}_{section}_{part}.jsonl

A training example targets ONE token of the section; everything before it
is the prompt:

  model input = chat_template(task_prompt) + answer_prefix
  target      = the token text

Clean and corrupt are kept together per line (pairs are needed for
pruning). Tokens are assigned to parts by char span from a single
full-answer tokenization, so boundaries are exact (see
verify_boundaries.py). When a floor tokenizes to different lengths on the
two sides (only possible across the 9th/10th boundary), examples are
paired by token index up to the shorter side; dropped extras are counted
and reported.

Usage:  ../venv/bin/python make_divisions.py
"""
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"
OUT = DATA / "divisions"
TOK = "meta-llama/Llama-3.1-8B-Instruct"


def token_runs(answer, sections, tok):
    """Map every part to its run of tokens: list of (char_start, char_end)."""
    enc = tok(answer, add_special_tokens=False, return_offsets_mapping=True)
    spans = [(s, e) for s, e in enc["offset_mapping"]]
    runs = {}  # (section_label, part_label) -> [(s, e), ...]
    cursor = 0
    for sec in sections:
        for part in sec["parts"]:
            while cursor < len(spans) and spans[cursor][1] <= part["char_start"]:
                cursor += 1  # skip newline separators (belong to no part)
            run = []
            while cursor < len(spans) and spans[cursor][1] <= part["char_end"]:
                run.append(spans[cursor])
                cursor += 1
            assert run, f"no tokens in {sec['label']}/{part['label']}"
            runs[(sec["label"], part["label"])] = run
    assert cursor == len(spans)
    return runs


def section_order(sections):
    """Flat list of (sec_label, part_label) in generation order."""
    return [(s["label"], p["label"]) for s in sections for p in s["parts"]]


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    OUT.mkdir(exist_ok=True)
    dropped = Counter()

    for split in ("train", "test"):
        pairs = [json.loads(l) for l in open(DATA / f"{split}.jsonl")]
        order = section_order(pairs[0]["clean"]["sections"])
        assert len(order) == 21
        files = {i: open(OUT / f"{split}_{i:02d}_{sl}_{pl}.jsonl", "w")
                 for i, (sl, pl) in enumerate(order)}

        for pair in pairs:
            runs = {}
            for typ in ("clean", "corrupt"):
                side = pair[typ]
                runs[typ] = token_runs(side["answer"], side["sections"], tok)
            for i, key in enumerate(order):
                rc, rx = runs["clean"][key], runs["corrupt"][key]
                if len(rc) != len(rx):
                    dropped[key] += abs(len(rc) - len(rx))
                for j, (sc, sx) in enumerate(zip(rc, rx)):
                    ex = {"pair_id": pair["pair_id"],
                          "name": pair["name"],
                          "start_floor": pair["start_floor"],
                          "section_index": i,
                          "section": key[0], "part": key[1],
                          "token_index": j}
                    for typ, run, sp in (("clean", rc, sc), ("corrupt", rx, sx)):
                        side = pair[typ]
                        ex[typ] = {
                            "task_prompt": side["prompt"],
                            "answer_prefix": side["answer"][:sp[0]],
                            "target": side["answer"][sp[0]:sp[1]],
                            "n_section_tokens": len(run),
                        }
                    files[i].write(json.dumps(ex) + "\n")

        for i, (sl, pl) in enumerate(order):
            files[i].close()
            n = sum(1 for _ in open(OUT / f"{split}_{i:02d}_{sl}_{pl}.jsonl"))
            print(f"{split}_{i:02d}_{sl}_{pl}.jsonl: {n} example pairs")

    if dropped:
        print("dropped unpairable extra tokens (floor width mismatch):")
        for (sl, pl), n in dropped.items():
            print(f"  {sl}/{pl}: {n}")
    else:
        print("no token-count mismatches between clean/corrupt anywhere")


if __name__ == "__main__":
    main()
