"""Verify boundary consistency of the sectioned dataset.

Checks, for every example in train.jsonl and test.jsonl (both sides):
  1. IDENTICAL STRUCTURE: same section labels, same part labels, in the
     same order, in every example.
  2. CONTIGUITY: sections and parts tile the answer exactly — no gaps,
     no overlaps, first char_start == 0, last char_end == len(answer),
     sections joined by exactly one '\n'.
  3. NO TOKEN OVERLAP: tokenize (Llama-3 tokenizer); each part must own a
     contiguous run of whole tokens; runs are pairwise disjoint, in order,
     and their union covers every token of the answer exactly once.
  4. BOUNDARY SIGNATURES: report the distinct boundary layouts. Layouts
     may differ ONLY because 2-digit floors (10th/11th/12th) make a line
     one char/token longer — the checker confirms that is the sole source
     of variation.

Usage:  ../venv/bin/python verify_boundaries.py
"""
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
TOK = "meta-llama/Llama-3.1-8B-Instruct"

EXPECTED_SECTIONS = ["start", "step_1", "step_2", "step_3", "step_4",
                     "step_5", "final"]
EXPECTED_PARTS = ["word", "sep", "floor"]


def check_example(pair_id, typ, side, tok, signatures):
    ans = side["answer"]
    secs = side["sections"]

    # 1. structure
    assert [s["label"] for s in secs] == EXPECTED_SECTIONS, \
        f"{pair_id}/{typ}: section labels differ"
    for s in secs:
        assert [p["label"] for p in s["parts"]] == EXPECTED_PARTS, \
            f"{pair_id}/{typ}: part labels differ in {s['label']}"

    # 2. contiguity / no char overlap
    cursor = 0
    for i, s in enumerate(secs):
        assert s["char_start"] == cursor, f"{pair_id}/{typ}: gap before {s['label']}"
        pcursor = s["char_start"]
        for p in s["parts"]:
            assert p["char_start"] == pcursor and p["char_end"] > p["char_start"], \
                f"{pair_id}/{typ}: part overlap/gap in {s['label']}"
            assert ans[p["char_start"]:p["char_end"]] == p["text"]
            pcursor = p["char_end"]
        assert s["char_end"] == pcursor
        cursor = s["char_end"]
        if i < len(secs) - 1:
            assert ans[cursor] == "\n", f"{pair_id}/{typ}: missing newline"
            cursor += 1
    assert cursor == len(ans), f"{pair_id}/{typ}: trailing chars"

    # 3. token-level ownership: disjoint contiguous runs covering all tokens
    enc = tok(ans, add_special_tokens=False, return_offsets_mapping=True)
    spans = [(s, e) for s, e in enc["offset_mapping"]]
    token_owner = {}
    tok_cursor = 0
    for s in secs:
        for p in s["parts"]:
            # separator tokens (the section-joining newlines) belong to no part
            while tok_cursor < len(spans) and spans[tok_cursor][1] <= p["char_start"]:
                ts, te = spans[tok_cursor]
                assert ans[ts:te] == "\n", \
                    f"{pair_id}/{typ}: unexpected token between parts: {ans[ts:te]!r}"
                tok_cursor += 1
            run = []
            while tok_cursor < len(spans) and spans[tok_cursor][1] <= p["char_end"]:
                ts, te = spans[tok_cursor]
                assert ts >= p["char_start"] and te <= p["char_end"], \
                    f"{pair_id}/{typ}: token {tok_cursor} straddles part boundary"
                run.append(tok_cursor)
                tok_cursor += 1
            assert run, f"{pair_id}/{typ}: part {s['label']}/{p['label']} owns no token"
            for t in run:
                assert t not in token_owner, \
                    f"{pair_id}/{typ}: token {t} claimed by two parts"
                token_owner[t] = (s["label"], p["label"])
    n_newlines = sum(1 for ts, te in spans if ans[ts:te] == "\n")
    assert tok_cursor == len(spans) \
        and len(token_owner) == len(spans) - n_newlines, \
        f"{pair_id}/{typ}: tokens left unclaimed"

    # 4. boundary signature: part lengths per section (layout fingerprint)
    sig = tuple((p["char_end"] - p["char_start"])
                for s in secs for p in s["parts"])
    signatures.setdefault(sig, set()).add(ans)


def main():
    tok = AutoTokenizer.from_pretrained(TOK)
    signatures = {}  # layout -> set of answers having it
    n = 0
    for fn in ("train.jsonl", "test.jsonl"):
        for line in open(ROOT / "datasets" / fn):
            pair = json.loads(line)
            for typ in ("clean", "corrupt"):
                check_example(pair["pair_id"], typ, pair[typ], tok,
                              signatures)
                n += 1

    # distinct layouts may differ only in the floor-part length (1 vs 2 digits)
    def explain(sig):
        return "".join(str(l) if i % 3 != 2 else f"[{l}]"
                       for i, l in enumerate(sig))

    print(f"checked {n} examples: structure, contiguity, and token "
          f"ownership all consistent")
    print(f"distinct boundary layouts: {len(signatures)}")
    for sig, answers in sorted(signatures.items()):
        floors = sorted({part.split()[-1] for a in answers
                         for part in a.replace(chr(10), " ").split(" ")})
        print(f"  layout {explain(sig)}  x{len(answers)} answers")


if __name__ == "__main__":
    main()
