"""Build the sectioned elevator dataset for circuit experiments.

Reads the verified pair selection (../datgen_multi/elevator_all3.jsonl) and
the three per-model raw files (all3_8b / all3_3b / all3_qwen7b). For every
pair side, EACH model's own answer is parsed independently into sections:

  section 0            "start"   Start: 7th
  section 1..n_moves   "step_k"  down -> 6th      (move k, floor after move k)
  section n_moves+1    "final"   FINAL ANSWER: 4th

A section carries exact character spans (char_start/char_end into the raw
answer string), so a program can locate section boundaries without any
tokenization assumptions. Every answer has the same number of sections in
the same order, so section index i is the same step across all examples.

Each section is further split into fine-grained "parts" — word / separator
/ floor (e.g. 'down' | ' ->' | ' 4th') — also with exact char spans. Every
part boundary is verified to be a Llama-3 token boundary (and every
newline a standalone token), so parts map to whole tokens exactly.

Assertions (hard-fails the build if violated):
  - all three models' answers parse into identical section layouts and are
    character-identical to the canonical answer,
  - sections join back to the exact answer text and spans are exact,
  - step directions/floors match ground truth from the move sequence,
  - every pair has the same section count on both sides.

Output: train.jsonl / test.jsonl (50/50 pair split, seed-shuffled), one
JSON line per pair with clean/corrupt nested.

Usage:  python build_dataset.py
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "datgen_multi"
PAIRS_FILE = SRC / "elevator_all3.jsonl"
RAW_FILES = {
    "llama-3.1-8b": SRC / "all3_8b-raw.jsonl",
    "llama-3.2-3b": SRC / "all3_3b-raw.jsonl",
    "qwen2.5-7b": SRC / "all3_qwen7b-raw.jsonl",
}
SEED = 42

ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
STEP = re.compile(r"^(up|down) -> (\d+)(?:st|nd|rd|th)$")
LINE = re.compile(r"^(Start|FINAL ANSWER|up|down)(:| ->)( \d+(?:st|nd|rd|th))$")


def split_parts(text, offset):
    """Split one line into fine-grained parts: word / separator / floor.

    Verified against the Llama-3 tokenizer: every part boundary is a token
    boundary, and every newline is a standalone token, so parts map to
    whole tokens exactly. 'floor' keeps number+suffix together (' 5th' =
    tokens ' ', '5', 'th' — also token-aligned if finer is ever needed).
    """
    m = LINE.fullmatch(text)
    assert m, f"line does not match expected pattern: {text!r}"
    parts = []
    for label, g in zip(("word", "sep", "floor"), m.groups()):
        parts.append({"label": label, "text": g,
                      "char_start": offset, "char_end": offset + len(g)})
        offset += len(g)
    return parts


def parse_sections(answer, moves, start_floor):
    """Split a verified answer into labeled sections with exact char spans."""
    lines = answer.split("\n")
    assert len(lines) == len(moves) + 2, \
        f"{len(lines)} lines for {len(moves)} moves"

    sections = []
    offset = 0
    for i, text in enumerate(lines):
        sec = {"index": i, "text": text,
               "char_start": offset, "char_end": offset + len(text)}
        if i == 0:
            m = re.fullmatch(r"Start: (\d+)(?:st|nd|rd|th)", text)
            assert m, f"bad start line: {text!r}"
            sec.update(label="start", floor=int(m.group(1)))
            assert sec["floor"] == start_floor
        elif i == len(lines) - 1:
            m = re.fullmatch(r"FINAL ANSWER: (\d+)(?:st|nd|rd|th)", text)
            assert m, f"bad final line: {text!r}"
            sec.update(label="final", floor=int(m.group(1)))
        else:
            m = STEP.fullmatch(text)
            assert m, f"bad step line: {text!r}"
            sec.update(label=f"step_{i}", move=m.group(1),
                       floor=int(m.group(2)))
            # ground-truth check: direction and resulting floor
            assert m.group(1) == moves[i - 1], f"step {i} direction"
            prev = sections[-1]["floor"]
            want = prev + (1 if m.group(1) == "up" else -1)
            assert sec["floor"] == want, f"step {i} floor"
        sec["parts"] = split_parts(text, offset)
        sections.append(sec)
        offset += len(text) + 1  # +1 for the '\n'

    assert sections[-1]["floor"] == sections[-2]["floor"], \
        "final answer must equal last step floor"
    return sections


def check_token_alignment(pairs, tokenizer_name="meta-llama/Llama-3.1-8B-Instruct"):
    """Prove every section/part boundary is a token boundary (Llama-3 tokenizer).

    Skipped (with a note) if transformers/torch are unavailable; the char
    spans are authoritative regardless.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not available; skipping token-alignment check")
        return
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    answers = {p[t]["answer"] for p in pairs for t in ("clean", "corrupt")}
    for ans in answers:
        enc = tok(ans, add_special_tokens=False, return_offsets_mapping=True)
        spans = enc["offset_mapping"]
        edges = {s for s, _ in spans} | {e for _, e in spans}
        assert all(ans[s:e] == "\n" for s, e in spans if "\n" in ans[s:e]), \
            "newline merged into another token"
        for p in pairs:
            for t in ("clean", "corrupt"):
                if p[t]["answer"] != ans:
                    continue
                for sec in p[t]["sections"]:
                    bounds = {sec["char_start"], sec["char_end"]}
                    for part in sec["parts"]:
                        bounds |= {part["char_start"], part["char_end"]}
                    assert bounds <= edges, \
                        f"token boundary mismatch in {p['pair_id']}/{t}"
    print(f"token-alignment check passed on {len(answers)} unique answers "
          f"({tokenizer_name})")


def main():
    # per-model raw answers indexed by (pair_id, type)
    raw = {}
    for model, path in RAW_FILES.items():
        raw[model] = {}
        for line in open(path):
            r = json.loads(line)
            raw[model][(r["pair_id"], r["type"])] = r["answer"]

    pairs = [json.loads(l) for l in open(PAIRS_FILE)]

    # Randomize which side is "clean" per pair (seeded). The generator's
    # dedup kept only one orientation (clean always starts 'down'); both
    # sides were generated and verified identically, so swapping the labels
    # per pair restores a 50/50 first-move balance in the clean stream.
    orient_rng = random.Random(SEED + 1)
    flipped = set()
    for p in pairs:
        if orient_rng.random() < 0.5:
            p["clean"], p["corrupt"] = p["corrupt"], p["clean"]
            flipped.add(p["pair_id"])

    n_sections = None
    for p in pairs:
        for typ in ("clean", "corrupt"):
            side = p[typ]
            canonical = side["answer"]
            # every model's own answer must parse to the SAME sections
            model_secs = []
            for model in RAW_FILES:
                # raw files are keyed by the ORIGINAL orientation
                rt = typ if p["pair_id"] not in flipped else \
                    ("corrupt" if typ == "clean" else "clean")
                ans = raw[model][(p["pair_id"], rt)]
                model_secs.append(
                    parse_sections(ans, side["moves"], p["start_floor"]))
                assert ans == canonical, \
                    f"{model} answer differs on {p['pair_id']}/{typ}"
            assert all(s == model_secs[0] for s in model_secs[1:]), \
                f"section mismatch across models on {p['pair_id']}/{typ}"
            side["sections"] = model_secs[0]
            if n_sections is None:
                n_sections = len(model_secs[0])
            assert len(model_secs[0]) == n_sections, "unequal section count"

    rng = random.Random(SEED)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    splits = {"train": order[: len(pairs) // 2],
              "test": order[len(pairs) // 2:]}

    check_token_alignment(pairs)

    for split, idxs in splits.items():
        out = ROOT / "datasets" / f"{split}.jsonl"
        with open(out, "w") as f:
            for i in idxs:
                f.write(json.dumps(pairs[i]) + "\n")
        print(f"wrote {len(idxs)} pairs -> {out}")

    print(f"all {len(pairs)} pairs x 2 sides x {len(RAW_FILES)} models "
          f"parse into exactly {n_sections} identical sections each")


if __name__ == "__main__":
    main()
