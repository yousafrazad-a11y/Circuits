#!/usr/bin/env python3
"""
gen_rule_datasets_7b_all_variants.py

One unified generator + base evaluator for every A/B candidate design we tried on
Llama-3.2-1B, now intended for a 7B-class Llama model.

Core suite:
  A   IOI with code names (retrieval + S-inhibition-like selection)
  C   all-code anchored KV lookup (the intended shared core)

B variants, in historical order:
  B_chain      two-hop code chain: query -> hop1 code -> hop2 value
  B2_calendar  computed successor query over day/month/number ordinal windows
  B3_explicit  B2 plus an explicit ordinal-successor sentence
  B1           bare reassignment, latest binding wins (determinism control)
  B4           three bindings for one key, latest binding wins
  B5           update sentence + "The code for X is" query (' now' attractor test)
  B5c          update sentence + bare colon query "X:"
  B5d          update sentence + "The code for X:" query
  B6           bare reassignment with latest binding adjacent to query
  B7           in-list update marker "then"
  B9           update sentence + attractor-consuming query "is now"
  B9f          B9 + trailing filler binding (best 1B candidate)

Historical 1B results used for comparison:
  A:       pairwise 0.996, top1 0.788
  C:       pairwise ~0.968, top1 ~0.816
  B9f:     pairwise 0.988, top1 0.796, stale 0.004
  B_chain: failed (~0.112)
  B2:      pairwise 0.556, correct 0.000, naive 0.136
  B3:      pairwise 0.588, correct 0.036, naive 0.220
  B1:      pairwise 0.600, correct 0.424, stale 0.280
  B4:      0.356
  B5:      pairwise 0.972, but top1 was ' now' on every row
  B5c/d:   stale route 0.680-0.852
  B6:      0.208, stale 0.460
  B7:      pairwise 0.528, correct 0.460
  B9:      pairwise 0.976, correct 0.652

Gate: pairwise >= 0.75 AND top-1 correct >= ~0.60.
Pairwise alone can lie (the B2 lesson).

Typical 7B sweep:
  python -u gen_rule_datasets_7b_all_variants.py --mode gen \
    --model_name meta-llama/Llama-2-7b-hf \
    --out_dir datasets_7b_all --n_train 1000 --n_val 250 --n_test 250

  python -u gen_rule_datasets_7b_all_variants.py --mode base_eval \
    --model_name meta-llama/Llama-2-7b-hf \
    --data_dir datasets_7b_all --batch_size 16 \
    --report_json base_eval_7b_all.json

After a B variant passes, export it under the standard A/B/C filenames:
  python -u gen_rule_datasets_7b_all_variants.py --mode make_suite \
    --suite_b B9f --out_dir datasets_7b_B9f

For a quick base-only sweep:
  ... --mode gen --n_train 0 --n_test 0 --n_val 250 --skip_verify

If by "7B" you mean Llama-3.1-8B rather than Llama-2-7B, just change:
  --model_name meta-llama/Llama-3.1-8B
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter

# Verified single-token codes (bare AND leading-space) across llama3.2/2,
# qwen2.5, and gpt2. Keep answers code-typed wherever possible.
CODES = ["ap","bu","cp","cy","cz","df","di","du","dx","eb","ec","ev","ff","fi",
         "fn","fo","fs","gu","gy","hy","ja","je","ju","ke","mp","ns","ob","ol",
         "ot","ov","ro","sv","sy","tu","ty","ub","ul","ur","ve","vo","ze","zo"]

PLACES = ["store", "park", "beach", "market"]
OBJECTS = ["ball", "book", "apple", "cup"]
TEMPLATES_A = [
    "{n1} and {n2} went to the {place}. {n1} gave the {obj} to",
    "While {n1} and {n2} were at the {place}, {n1} handed the {obj} to",
]

ORDINAL_SETS = {
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "number": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
}

N_PAIRS = 8
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- helpers ----
def fmt_pairs(pairs):
    return ", ".join(f"{k}: {v}" for k, v in pairs)


def stable_seed(base_seed, ds, split):
    """Independent deterministic stream per dataset/split.

    This is stronger than sharing one RNG across all variants: adding/removing a
    variant no longer shifts every later dataset. B1 can therefore remain a real
    determinism control across script versions.
    """
    s = f"{base_seed}|{ds}|{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little")


class Pool:
    def __init__(self, rng):
        self.c = rng.sample(CODES, 26)

    def codes(self, n):
        out, self.c = self.c[:n], self.c[n:]
        assert len(out) == n, "code pool exhausted"
        return out


def row_tuple(clean, corrupt, clean_answer, corrupt_answer, naive, meta):
    return clean, corrupt, clean_answer, corrupt_answer, naive, meta


def unique_candidates(clean_answer, corrupt_answer, naive):
    out = []
    for a in (clean_answer, corrupt_answer, naive):
        if a and a not in out:
            out.append(a)
    return out


# ------------------------------------------------------------- core tests ----
def test_C(rng, pool):
    """Anchored all-code KV lookup: 8 disjoint key/value pairs, query one key."""
    cs = pool.codes(16)
    keys, vals = cs[:8], cs[8:]
    pairs = list(zip(keys, vals))
    q = rng.randrange(N_PAIRS)
    r = rng.choice([i for i in range(N_PAIRS) if i != q])
    cp = pairs.copy()
    cp[q], cp[r] = (keys[q], vals[r]), (keys[r], vals[q])
    return row_tuple(
        f"{fmt_pairs(pairs)}, {keys[q]}:",
        f"{fmt_pairs(cp)}, {keys[q]}:",
        " " + vals[q], " " + vals[r], None,
        dict(variant="C", query=keys[q], pairs=[f"{k}:{v}" for k, v in pairs], swap=[q, r]),
    )


def test_A(rng, pool):
    """IOI with code names. Naive route = repeated code."""
    n1, n2 = pool.codes(2)
    tmpl = rng.choice(TEMPLATES_A)
    place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
    clean = tmpl.format(n1=n1, n2=n2, place=place, obj=obj)
    corrupt = tmpl.format(n1=n2, n2=n1, place=place, obj=obj)
    return row_tuple(
        clean, corrupt,
        " " + n2, " " + n1, " " + n1,
        dict(variant="A", query=n2, repeated=n1, place=place, obj=obj),
    )


# ------------------------------------------------------ compute-then-retrieve ----
def test_B_chain(rng, pool):
    """Original two-hop chain: query -> hop-1 key -> hop-2 value."""
    keys = pool.codes(8)
    q = rng.randrange(N_PAIRS)
    j = rng.choice([i for i in range(N_PAIRS) if i != q])
    k = rng.choice([i for i in range(N_PAIRS) if i not in (q, j)])
    filler = pool.codes(7)
    values, vi = [], iter(filler)
    for i in range(N_PAIRS):
        values.append(keys[j] if i == q else next(vi))
    pairs = list(zip(keys, values))

    cp = pairs.copy()
    cp[q] = (keys[q], keys[k])
    cp[k] = (keys[k], keys[j])

    return row_tuple(
        f"{fmt_pairs(pairs)}, {keys[q]}:",
        f"{fmt_pairs(cp)}, {keys[q]}:",
        " " + values[j],
        " " + keys[j],
        " " + keys[j],
        dict(variant="B_chain", query=keys[q], chain=[keys[q], keys[j], values[j]],
             corrupt_link=keys[k]),
    )


def _ordinal_example(rng, pool, explicit_statement):
    """Shared constructor for B2_calendar and B3_explicit.

    Codes are bound to a shuffled window of five ordinals. The query asks for the
    code attached to the ordinal after the queried code's ordinal. Corruption
    swaps the codes occupying two ordinal slots, keeping token lengths aligned.
    """
    set_name = rng.choice(list(ORDINAL_SETS))
    ordinal_pool = ORDINAL_SETS[set_name]
    start = rng.randrange(0, len(ordinal_pool) - 4)
    ordinals = ordinal_pool[start:start + 5]
    codes = pool.codes(5)

    q_idx = rng.randrange(0, 4)
    t_idx = q_idx + 1
    d_idx = rng.choice([i for i in range(5) if i not in (q_idx, t_idx)])

    clean_pairs = [(codes[i], ordinals[i]) for i in range(5)]
    corrupt_pairs = clean_pairs.copy()
    corrupt_pairs[t_idx] = (codes[d_idx], ordinals[t_idx])
    corrupt_pairs[d_idx] = (codes[t_idx], ordinals[d_idx])

    statement = ""
    if explicit_statement:
        statement = f" The {set_name} after {ordinals[q_idx]} is {ordinals[t_idx]}."
    query = f" The code for the {set_name} after {codes[q_idx]} is"

    variant = "B3_explicit" if explicit_statement else "B2_calendar"
    return row_tuple(
        f"{fmt_pairs(clean_pairs)}.{statement}{query}",
        f"{fmt_pairs(corrupt_pairs)}.{statement}{query}",
        " " + codes[t_idx],
        " " + codes[d_idx],
        " " + codes[q_idx],
        dict(variant=variant, ordinal_set=set_name, ordinals=ordinals,
             query_code=codes[q_idx], target_code=codes[t_idx],
             distractor_code=codes[d_idx], positions=[q_idx, t_idx, d_idx]),
    )


def test_B2_calendar(rng, pool):
    """Calendar successor: compute the next ordinal, then retrieve its code."""
    return _ordinal_example(rng, pool, explicit_statement=False)


def test_B3_explicit(rng, pool):
    """Calendar successor with the ordinal increment stated explicitly."""
    return _ordinal_example(rng, pool, explicit_statement=True)


# --------------------------------------------------------- reassignment family ----
def _dup_pairs(rng, pool):
    """Five-pair list; duplicate key appears twice with gap >= 2."""
    cs = pool.codes(9)
    keys, vals = cs[:4], cs[4:9]
    dup = keys[0]
    p = sorted(rng.sample(range(5), 2))
    while p[1] - p[0] < 2:
        p = sorted(rng.sample(range(5), 2))
    pairs = [None] * 5
    pairs[p[0]] = (dup, vals[0])      # stale/first binding
    pairs[p[1]] = (dup, vals[1])      # latest binding
    ri = 0
    for i in range(5):
        if pairs[i] is None:
            pairs[i] = (keys[1 + ri], vals[2 + ri])
            ri += 1
    cp = [(k, vals[1] if i == p[0] else vals[0] if i == p[1] else v)
          for i, (k, v) in enumerate(pairs)]
    return pairs, cp, dup, vals, p


def test_B1(rng, pool):
    """Bare reassignment, latest binding wins. Determinism control."""
    pairs, cp, dup, vals, p = _dup_pairs(rng, pool)
    return row_tuple(
        f"{fmt_pairs(pairs)}, {dup}:",
        f"{fmt_pairs(cp)}, {dup}:",
        " " + vals[1], " " + vals[0], " " + vals[0],
        dict(variant="B1", query=dup, pairs=[f"{k}:{v}" for k, v in pairs], dup_pos=p),
    )


def test_B4(rng, pool):
    """Three bindings for the same key; latest binding wins."""
    cs = pool.codes(10)
    keys, vals = cs[:4], cs[4:10]
    dup = keys[0]
    positions = sorted(rng.sample(range(6), 3))
    pairs = [None] * 6
    for pos, val in zip(positions, vals[:3]):
        pairs[pos] = (dup, val)
    ri = 0
    for i in range(6):
        if pairs[i] is None:
            pairs[i] = (keys[1 + ri], vals[3 + ri])
            ri += 1

    first_pos, latest_pos = positions[0], positions[-1]
    stale, latest = vals[0], vals[2]
    cp = pairs.copy()
    cp[first_pos] = (dup, latest)
    cp[latest_pos] = (dup, stale)

    return row_tuple(
        f"{fmt_pairs(pairs)}, {dup}:",
        f"{fmt_pairs(cp)}, {dup}:",
        " " + latest, " " + stale, " " + stale,
        dict(variant="B4", query=dup, pairs=[f"{k}:{v}" for k, v in pairs],
             dup_pos=positions),
    )


def test_B6(rng, pool):
    """Bare reassignment with the latest binding immediately before the query."""
    cs = pool.codes(9)
    keys, vals = cs[:4], cs[4:9]
    dup = keys[0]
    stale_pos = rng.randrange(0, 3)
    latest_pos = 4
    pairs = [None] * 5
    pairs[stale_pos] = (dup, vals[0])
    pairs[latest_pos] = (dup, vals[1])
    ri = 0
    for i in range(5):
        if pairs[i] is None:
            pairs[i] = (keys[1 + ri], vals[2 + ri])
            ri += 1
    cp = pairs.copy()
    cp[stale_pos] = (dup, vals[1])
    cp[latest_pos] = (dup, vals[0])

    return row_tuple(
        f"{fmt_pairs(pairs)}, {dup}:",
        f"{fmt_pairs(cp)}, {dup}:",
        " " + vals[1], " " + vals[0], " " + vals[0],
        dict(variant="B6", query=dup, pairs=[f"{k}:{v}" for k, v in pairs],
             dup_pos=[stale_pos, latest_pos]),
    )


def test_B7(rng, pool):
    """In-list update cue: 'then' marks the second binding."""
    pairs, cp, dup, vals, p = _dup_pairs(rng, pool)

    def render(prs):
        return ", ".join(("then " if i == p[1] else "") + f"{k}: {v}"
                         for i, (k, v) in enumerate(prs))

    return row_tuple(
        f"{render(pairs)}, {dup}:",
        f"{render(cp)}, {dup}:",
        " " + vals[1], " " + vals[0], " " + vals[0],
        dict(variant="B7", query=dup, pairs=[f"{k}:{v}" for k, v in pairs], dup_pos=p),
    )


def _sentence_update(rng, pool, query_style, trailing=False):
    """Update-sentence family used by B5, B5c, B5d, B9, and B9f."""
    cs = pool.codes(9)
    keys, vals = cs[:4], cs[4:9]
    dup = keys[0]
    stale, upd = vals[0], vals[1]
    pairs = [(dup, stale)] + [(keys[1 + i], vals[2 + i]) for i in range(2)]
    rng.shuffle(pairs)
    f_key, f_val = keys[3], vals[4]
    cpairs = [(k, upd if k == dup else v) for k, v in pairs]
    trail = f" The code for {f_key} is {f_val}." if trailing else ""

    if query_style == "is":
        query = f"The code for {dup} is"
    elif query_style == "bare_colon":
        query = f"{dup}:"
    elif query_style == "the_colon":
        query = f"The code for {dup}:"
    elif query_style == "now":
        query = f"The code for {dup} is now"
    else:
        raise ValueError(query_style)

    variant = {
        ("is", False): "B5",
        ("bare_colon", False): "B5c",
        ("the_colon", False): "B5d",
        ("now", False): "B9",
        ("now", True): "B9f",
    }[(query_style, trailing)]

    clean = f"{fmt_pairs(pairs)}. Then the code for {dup} changed to {upd}.{trail} {query}"
    corrupt = f"{fmt_pairs(cpairs)}. Then the code for {dup} changed to {stale}.{trail} {query}"
    return row_tuple(
        clean, corrupt,
        " " + upd, " " + stale, " " + stale,
        dict(variant=variant, query=dup, stale=stale, updated=upd,
             filler=(f_key, f_val) if trailing else None,
             pairs=[f"{k}:{v}" for k, v in pairs]),
    )


def test_B5(rng, pool):
    """Update sentence + 'The code for X is' query. Tests the ' now' attractor."""
    return _sentence_update(rng, pool, query_style="is", trailing=False)


def test_B5c(rng, pool):
    """Update sentence + bare colon query. Tests list-induction routing."""
    return _sentence_update(rng, pool, query_style="bare_colon", trailing=False)


def test_B5d(rng, pool):
    """Update sentence + 'The code for X:' query. Tests list-induction routing."""
    return _sentence_update(rng, pool, query_style="the_colon", trailing=False)


def test_B9(rng, pool):
    """Update sentence + attractor-consuming 'is now' query."""
    return _sentence_update(rng, pool, query_style="now", trailing=False)


def test_B9f(rng, pool):
    """B9 plus trailing filler binding. Best Llama-3.2-1B candidate."""
    return _sentence_update(rng, pool, query_style="now", trailing=True)


TESTS = {
    "A": test_A,
    "C": test_C,
    "B_chain": test_B_chain,
    "B2_calendar": test_B2_calendar,
    "B3_explicit": test_B3_explicit,
    "B1": test_B1,
    "B4": test_B4,
    "B5": test_B5,
    "B5c": test_B5c,
    "B5d": test_B5d,
    "B6": test_B6,
    "B7": test_B7,
    "B9": test_B9,
    "B9f": test_B9f,
}

ALL_DS = list(TESTS.keys())
B_VARIANTS = [ds for ds in ALL_DS if ds.startswith("B")]


def make_row(rng, ds, split, idx):
    pool = Pool(rng)
    clean, corrupt, ca, xa, naive, meta = TESTS[ds](rng, pool)
    return {
        "id": f"{ds}_{split}_{idx:05d}",
        "dataset": ds,
        "clean_prompt": clean,
        "corrupt_prompt": corrupt,
        "clean_answer": ca,
        "corrupt_answer": xa,
        "naive_answer": naive,
        "ld_candidates": unique_candidates(ca, xa, naive),
        **meta,
    }


def selected_datasets(args):
    ds = args.datasets if args.datasets else ALL_DS
    bad = [x for x in ds if x not in TESTS]
    if bad:
        raise SystemExit(f"unknown dataset(s): {bad}; choose from {ALL_DS}")
    return ds


def verify_rows(tok, rows, ds, split):
    """Check the two invariants train_abc.py silently relies on."""
    for r in rows[:50]:
        for a in r["ld_candidates"]:
            toks = tok.encode(a, add_special_tokens=False)
            assert len(toks) == 1, f"{ds}/{split}: multi-token answer {a!r} -> {toks}"
        c = tok.encode(r["clean_prompt"], add_special_tokens=True)
        x = tok.encode(r["corrupt_prompt"], add_special_tokens=True)
        assert len(c) == len(x), (
            f"{ds}/{split}: clean/corrupt token-length mismatch "
            f"({len(c)} vs {len(x)}) for row {r['id']}"
        )


def write_split(args, ds, split, n):
    rng = random.Random(stable_seed(args.seed, ds, split))
    rows = [make_row(rng, ds, split, i) for i in range(n)]
    return rows


def mode_gen(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tok = None
    if not args.skip_verify:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"[verify skipped: {type(e).__name__}: {e}]")

    counts = dict(train=args.n_train, val=args.n_val, test=args.n_test)
    for ds in selected_datasets(args):
        example = None
        for split in SPLITS:
            n = counts[split]
            if n <= 0:
                continue
            rows = write_split(args, ds, split, n)
            if tok is not None:
                verify_rows(tok, rows, ds, split)
            path = os.path.join(args.out_dir, f"{ds}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if example is None and rows:
                example = rows[0]
        if example:
            print(
                f"\n--- {ds} ---"
                f"\nCLEAN:   {example['clean_prompt']!r} -> {example['clean_answer']!r}"
                f"\nCORRUPT: {example['corrupt_prompt']!r} -> {example['corrupt_answer']!r}"
                f"\nNAIVE:   {example.get('naive_answer')!r}"
            )
    print(f"\nWrote selected datasets to {args.out_dir}/ using model {args.model_name}")


def mode_base_eval(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else \
             "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    model_kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except ImportError as e:
        if args.device_map == "auto" or "accelerate" not in str(e).lower():
            raise
        print("[accelerate missing; loading without low_cpu_mem_usage]")
        model_kwargs.pop("low_cpu_mem_usage", None)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.device_map != "auto":
        model = model.to(device)
    model.eval()
    input_device = next(model.parameters()).device
    print(f"device: {device} | input_device: {input_device} | model: {args.model_name}")

    tid_cache = {}
    def tid(s):
        if s not in tid_cache:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"answer is not single-token: {s!r} -> {ids}")
            tid_cache[s] = ids[0]
        return tid_cache[s]

    summary = {}
    for ds in selected_datasets(args):
        path = os.path.join(args.data_dir, f"{ds}_val.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}; run --mode gen first")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if args.max_rows:
            rows = rows[:args.max_rows]
        pair_ok = n = 0
        picks = {"correct": 0, "corrupt": 0, "naive": 0, "other": 0}
        others = Counter()

        with torch.inference_mode():
            for i in range(0, len(rows), args.batch_size):
                b = rows[i:i + args.batch_size]
                enc = tok(
                    [r["clean_prompt"] for r in b],
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to(input_device)
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
                am = logits.argmax(-1)
                t = torch.tensor([tid(r["clean_answer"]) for r in b], device=logits.device)
                d = torch.tensor([tid(r["corrupt_answer"]) for r in b], device=logits.device)
                pair_ok += (logits.gather(1, t[:, None]) > logits.gather(1, d[:, None])).sum().item()
                for j, r in enumerate(b):
                    pred = am[j].item()
                    target = t[j].item()
                    distractor = d[j].item()
                    naive = tid(r["naive_answer"]) if r.get("naive_answer") else None
                    if pred == target:
                        picks["correct"] += 1
                    elif pred == distractor:
                        picks["corrupt"] += 1
                    elif naive is not None and pred == naive:
                        picks["naive"] += 1
                    else:
                        picks["other"] += 1
                        others[pred] += 1
                n += len(b)

        pairwise = pair_ok / n
        correct = picks["correct"] / n
        gate = "PASS" if pairwise >= args.gate_pairwise and correct >= args.gate_correct else "FAIL"
        prof = " / ".join(f"{k} {v / n:.3f}" for k, v in picks.items())
        top = ", ".join(f"{tok.decode([tok_id])!r}x{c}" for tok_id, c in others.most_common(5))
        print(f"{ds:12s}: pairwise {pairwise:.3f} | picks: {prof} | gate {gate} | other top: {top} | n={n}")
        summary[ds] = {
            "n": n,
            "pairwise": pairwise,
            "picks": {k: v / n for k, v in picks.items()},
            "gate": gate,
            "other_top": [[tok.decode([tok_id]), c] for tok_id, c in others.most_common(10)],
        }

    print(f"\nGATE: pairwise >= {args.gate_pairwise:.2f} AND correct >= {args.gate_correct:.2f}.")
    print("For reassignment variants, the stale binding appears under 'corrupt' (and also 'naive').")
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": args.model_name,
                "data_dir": args.data_dir,
                "gate_pairwise": args.gate_pairwise,
                "gate_correct": args.gate_correct,
                "results": summary,
            }, f, indent=2, ensure_ascii=False)
        print(f"Wrote {args.report_json}")


def mode_make_suite(args):
    """Write A, selected-B, C under the standard names train_abc.py expects."""
    if args.suite_b not in B_VARIANTS:
        raise SystemExit(f"--suite_b must be one of {B_VARIANTS}")
    os.makedirs(args.out_dir, exist_ok=True)
    mapping = {"A": "A", args.suite_b: "B", "C": "C"}
    counts = dict(train=args.n_train, val=args.n_val, test=args.n_test)
    for source_ds, out_ds in mapping.items():
        for split in SPLITS:
            n = counts[split]
            if n <= 0:
                continue
            rows = write_split(args, source_ds, split, n)
            for r in rows:
                r["source_dataset"] = source_ds
                r["dataset"] = out_ds
                r["id"] = f"{out_ds}_{split}_{r['id'].rsplit('_', 1)[-1]}"
            path = os.path.join(args.out_dir, f"{out_ds}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote standard A/B/C suite using {args.suite_b} as B -> {args.out_dir}")
    print("Files: A_{train,val,test}.jsonl, B_{train,val,test}.jsonl, C_{train,val,test}.jsonl")


def mode_list():
    print("Available datasets:")
    for ds in ALL_DS:
        print(f"  {ds:12s} {TESTS[ds].__doc__.splitlines()[0] if TESTS[ds].__doc__ else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "base_eval", "make_suite", "list"], default="gen")
    ap.add_argument("--model_name", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--out_dir", default="datasets_7b_all")
    ap.add_argument("--data_dir", default="datasets_7b_all")
    ap.add_argument("--datasets", nargs="*", choices=ALL_DS, default=None,
                    help="Subset to generate/evaluate. Default: all variants.")
    ap.add_argument("--suite_b", default="B9f", choices=B_VARIANTS,
                    help="B variant exported as standard B_* files in make_suite mode.")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_rows", type=int, default=0, help="Optional cap for quick base_eval.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16 for base_eval.")
    ap.add_argument("--device_map", choices=["none", "auto"], default="none")
    ap.add_argument("--gate_pairwise", type=float, default=0.75)
    ap.add_argument("--gate_correct", type=float, default=0.60)
    ap.add_argument("--report_json", default="")
    args = ap.parse_args()

    if args.mode == "gen":
        mode_gen(args)
    elif args.mode == "base_eval":
        mode_base_eval(args)
    elif args.mode == "make_suite":
        mode_make_suite(args)
    else:
        mode_list()


if __name__ == "__main__":
    main()