#!/usr/bin/env python3
"""
gen_abc_dataset.py
==================
Matched A/B/C induction-retrieval dataset for verifying circuit intersection
with node pruning (hard-concrete gates) on Llama-1B / 7B.

Latent sample (shared 1:1:1 across formats):
    3 words + 3 value-codes + 3 key-codes (all mutually distinct),
    query index q, swap partner r (r != q).

Formats (same retrieval algorithm, different surface wrapper):
    A (colon KV):    "dog: ap, cat: bu, bird: cp, dog:"                -> " ap"
    B (sentence KV): "The code for dog is ap. ... The code for dog is" -> " ap"
    C (pure codes):  "cy ap cz bu df cp cy"                            -> " ap"

Corruption: swap the queried pair's value with the swap partner's value
(v_q <-> v_r) in all three renderings. Token-for-token, equal token count,
both candidate tokens present in both prompts, and the corrupted run is a
valid run with a different well-defined answer. The clean<->corrupt
contrast therefore isolates content, not activity — exactly what the
dual-stream gated-patching loss (and EAP-family methods) key on.

Output: 9 jsonl files {A,B,C}_{train,val,test}.jsonl, id-matched rows.

Row schema:
    id, dataset, clean_prompt, corrupt_prompt,
    clean_answer, corrupt_answer          (leading space, single token)
    ld_candidates                         ([clean_answer, corrupt_answer])
    query, pairs, values, c_keys, swap_indices

Pool design (verified, not guessed):
    * Codes are 2-letter lowercase NON-WORD bigrams. Digit-containing codes
      are impossible in principle: Llama-2 / Llama-3.x / Qwen tokenizers
      split every digit into its own token, so no letter+digit string is
      ever a single token.
    * Every code below is a single token in BOTH bare and leading-space
      form in Llama-3.2, Llama-2, Qwen2.5 AND GPT-2 (verified by
      enumeration + encode roundtrip). Bare form matters: the first word
      of A and the first key of C sit at prompt-initial position.
    * Blocklisted: real 2-letter words (Scrabble list), abbreviations,
      roman numerals (list/position circuitry), element symbols, US state
      codes, English onset clusters (word-fragment-like), tech abbrevs.
    * Words are concrete nouns, likewise verified single-token in both
      forms (Llama-3.2 ∩ Llama-2; also single-token spaced in Qwen2.5/GPT-2).

CPU-only. Optional dependency: `transformers`, used to re-verify pools on
your exact tokenizer and assert equal clean/corrupt token counts.
    pip install transformers
    huggingface-cli login     # for gated meta-llama/* tokenizers
or pass --no-tokenizer to skip validation (not recommended).

Usage:
    python gen_abc_dataset.py --out-dir ./abc_dataset \
        --tokenizer meta-llama/Llama-3.2-1B \
        --n 1500 --train 1000 --val 250 --test 250 --seed 0
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pools — verified single-token (bare AND leading-space) on Llama-3.2/Llama-2,
# portable to Qwen2.5/GPT-2. Do not "improve" by hand: run with --tokenizer
# to re-verify on your exact tokenizer.
# ---------------------------------------------------------------------------
WORDS = [
    "dog", "cat", "bird", "bear", "horse", "fish", "mouse", "cow", "hen",
    "ant", "ball", "box", "cup", "book", "pen", "chair", "table", "lamp",
    "door", "key", "ring", "hat", "bag", "bed", "plate", "fork", "clock",
    "mirror", "bucket", "chain", "wheel", "brick", "apple", "bread", "egg",
    "bean", "soup", "tree", "rock", "leaf", "star", "cloud", "river",
    "lake", "hill", "sand", "snow", "grass", "flower", "seed", "coat",
    "dress", "boot",
]

CODES = [
    "ap", "bu", "cp", "cy", "cz", "df", "di", "du", "dx", "eb", "ec", "ev",
    "ff", "fi", "fn", "fo", "fs", "gu", "gy", "hy", "ja", "je", "ju", "ke",
    "mp", "ns", "ob", "ol", "ot", "ov", "ro", "sv", "sy", "tu", "ty", "ub",
    "ul", "ur", "ve", "vo", "ze", "zo",
]


# ---------------------------------------------------------------------------
# Tokenizer utilities
# ---------------------------------------------------------------------------
def n_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def filter_pools(tokenizer, words, codes):
    """Keep only items that are a single token BOTH bare and with a leading
    space. Bare matters: the first word of A / first key of C sit at
    prompt-initial position (right after BOS), while every other occurrence
    — including the answer token — carries a leading space."""
    def ok(t):
        return n_tokens(tokenizer, t) == 1 and n_tokens(tokenizer, " " + t) == 1
    return [w for w in words if ok(w)], [c for c in codes if ok(c)]


# ---------------------------------------------------------------------------
# Latent sample + renderers
# ---------------------------------------------------------------------------
def make_sample(rng, words, codes):
    ws = rng.sample(words, 3)
    cs = rng.sample(codes, 6)          # 3 value-codes + 3 key-codes, all distinct
    q = rng.randrange(3)
    r = rng.choice([i for i in range(3) if i != q])
    return {"words": ws, "values": cs[:3], "keys": cs[3:], "q": q, "r": r}


def swapped(values, q, r):
    v = list(values)
    v[q], v[r] = v[r], v[q]
    return v


def render_A(words, values, q):
    body = ", ".join(f"{w}: {v}" for w, v in zip(words, values))
    return f"{body}, {words[q]}:"


def render_B(words, values, q):
    body = " ".join(f"The code for {w} is {v}." for w, v in zip(words, values))
    return f"{body} The code for {words[q]} is"


def render_C(keys, values, q):
    body = " ".join(f"{k} {v}" for k, v in zip(keys, values))
    return f"{body} {keys[q]}"


# ---------------------------------------------------------------------------
# Row construction (1 latent sample -> 3 matched rows) + validation
# ---------------------------------------------------------------------------
def build_rows(idx, s, tokenizer):
    w, v, k, q, r = s["words"], s["values"], s["keys"], s["q"], s["r"]
    vc = swapped(v, q, r)
    clean_ans, corrupt_ans = " " + v[q], " " + v[r]

    specs = [
        ("A", render_A(w, v, q),  render_A(w, vc, q),  w[q]),
        ("B", render_B(w, v, q),  render_B(w, vc, q),  w[q]),
        ("C", render_C(k, v, q),  render_C(k, vc, q),  k[q]),
    ]

    rows = []
    for name, clean_p, corrupt_p, query in specs:
        assert clean_p != corrupt_p
        if tokenizer is not None:
            # Hard requirements of the dual-stream gated forward:
            # equal-length clean/corrupt (shared attention mask) and
            # single-token answers (target_tokens[:, 0]).
            assert n_tokens(tokenizer, clean_p) == n_tokens(tokenizer, corrupt_p), \
                f"token-length mismatch: {clean_p!r} vs {corrupt_p!r}"
            assert n_tokens(tokenizer, clean_ans) == 1
            assert n_tokens(tokenizer, corrupt_ans) == 1
        rows.append({
            "id": f"{idx:06d}",
            "dataset": name,
            "clean_prompt": clean_p,
            "corrupt_prompt": corrupt_p,
            "clean_answer": clean_ans,
            "corrupt_answer": corrupt_ans,
            "ld_candidates": [clean_ans, corrupt_ans],
            "query": query,
            "pairs": {wi: vi for wi, vi in zip(w, v)},  # A/B bindings
            "values": v,                                 # C bindings: c_keys[i] -> values[i]
            "c_keys": k,
            "swap_indices": [q, r],
        })
    return rows


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="./abc_dataset")
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--no-tokenizer", action="store_true")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--train", type=int, default=1000)
    p.add_argument("--val", type=int, default=250)
    p.add_argument("--test", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.train + args.val + args.test != args.n:
        sys.exit("train + val + test must equal n")

    tokenizer = None
    words, codes = WORDS, CODES
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        except Exception as e:
            sys.exit(
                f"Could not load tokenizer '{args.tokenizer}': {e}\n"
                "Fix: pip install transformers && huggingface-cli login "
                "(accept the Llama license on the HF model page first), "
                "or rerun with --no-tokenizer to skip token-level validation.")
        words, codes = filter_pools(tokenizer, WORDS, CODES)
        print(f"[pools] {len(words)}/{len(WORDS)} words and "
              f"{len(codes)}/{len(CODES)} codes survive single-token filtering")
        if len(words) < 12 or len(codes) < 24:
            sys.exit("Pools too small after filtering — check the tokenizer.")
    else:
        print("[warn] --no-tokenizer: skipping token-level validation "
              "(single-token answers and equal-length pairs NOT guaranteed)")

    rng = random.Random(args.seed)
    samples, seen = [], set()
    while len(samples) < args.n:
        s = make_sample(rng, words, codes)
        key = (tuple(s["words"]), tuple(s["values"]), tuple(s["keys"]))
        if key in seen:
            continue
        seen.add(key)
        samples.append(s)

    split_of = {}
    for sp, lo, hi in [("train", 0, args.train),
                       ("val", args.train, args.train + args.val),
                       ("test", args.train + args.val, args.n)]:
        for i in range(lo, hi):
            split_of[i] = sp

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    handles = {(d, sp): open(out / f"{d}_{sp}.jsonl", "w")
               for d in "ABC" for sp in ("train", "val", "test")}

    for idx, s in enumerate(samples):
        sp = split_of[idx]
        for row in build_rows(idx, s, tokenizer):
            handles[(row["dataset"], sp)].write(json.dumps(row) + "\n")
    for h in handles.values():
        h.close()

    # ---- summary -----------------------------------------------------------
    print(f"\n[done] wrote 9 files to {out.resolve()}")
    for d in "ABC":
        for sp, cnt in (("train", args.train), ("val", args.val), ("test", args.test)):
            print(f"  {d}_{sp}.jsonl  ({cnt} rows)")
    print("\n[example train rows]")
    for row in build_rows(0, samples[0], tokenizer):
        print(f"  {row['dataset']}  clean:   {row['clean_prompt']!r} -> {row['clean_answer']!r}")
        print(f"  {row['dataset']}  corrupt: {row['corrupt_prompt']!r} -> {row['corrupt_answer']!r}")
    qpos = [s["q"] for s in samples]
    print(f"\n[balance] query-position counts: "
          f"{[qpos.count(i) for i in range(3)]} (expect ~{args.n//3} each)")


if __name__ == "__main__":
    main()