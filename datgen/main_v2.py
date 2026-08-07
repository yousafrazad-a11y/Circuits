#!/usr/bin/env python3
"""
gen_rule_datasets_v2.py — natural-rule A/B/C suite (no demos, code outputs everywhere).

  C  all-code anchored KV lookup (the core: 1-hop induction)
       qz: ap, wv: bu, ro: cp, ..., qz:          ->  ap

  A  IOI with code names (core + S-inhibition family)
       ap and bu went to the store. ap gave the ball to   ->  bu
     corruption: roles swapped -> repeated code (= naive)

  B1 bare reassignment, 2 bindings (CONTROL; verified base: 0.424 correct / 0.280 stale,
     pairwise 0.600, reproduced on identical rows in two independent runs)
       qz: ap, wv: bu, ro: cp, qz: df, di: ev, qz:        ->  df

  B7 in-list update cue ("then"), both bindings stay literal matches
       qz: ap, wv: bu, ro: cp, then qz: df, qz:           ->  df

  B9 cued update, attractor-consuming query "is now"  [main candidate]
       qz: ap, wv: bu, ro: cp. Then the code for qz changed to df. The code for qz is now -> df
  B9f B9 + trailing filler binding (blocks "copy the last code" shortcut)
       ... changed to df. The code for ve is ub. The code for qz is now -> df

  corruption for B variants: swap stale and updated bindings -> stale = naive

  retired: B4/B6 (first-binding anchoring), B5c/B5d (colon query routes to
           list-induction -> retrieves the STALE binding, 0.68-0.85 stale),
           B5 "is"-query (0.972 association but ' now' attractor on 100% of rows)

GATE: pairwise >= 0.75 AND correct-pick >= ~0.6. Pairwise alone can lie.

  python main_v2.py --mode gen --out_dir datasets
  python main_v2.py --mode base_eval --data_dir datasets
"""

import argparse, json, os, random
from collections import Counter

# verified single-token (bare AND leading-space) across llama3.2/2, qwen2.5, gpt2
CODES = ["ap","bu","cp","cy","cz","df","di","du","dx","eb","ec","ev","ff","fi",
         "fn","fo","fs","gu","gy","hy","ja","je","ju","ke","mp","ns","ob","ol",
         "ot","ov","ro","sv","sy","tu","ty","ub","ul","ur","ve","vo","ze","zo"]

PLACES = ["store", "park", "beach", "market"]
OBJECTS = ["ball", "book", "apple", "cup"]
TEMPLATES_A = [
    "{n1} and {n2} went to the {place}. {n1} gave the {obj} to",
    "While {n1} and {n2} were at the {place}, {n1} handed the {obj} to",
]

N_PAIRS = 8


def fmt_pairs(pairs):
    return ", ".join(f"{k}: {v}" for k, v in pairs)


class Pool:
    def __init__(self, rng):
        self.c = rng.sample(CODES, 26)
    def codes(self, n):
        out, self.c = self.c[:n], self.c[n:]
        assert len(out) == n, "code pool exhausted"
        return out


# ---------------------------------------------------------------- tests ----
def test_C(rng, pool):
    cs = pool.codes(16)
    keys, vals = cs[:8], cs[8:]
    pairs = list(zip(keys, vals))
    q = rng.randrange(N_PAIRS)
    r = rng.choice([i for i in range(N_PAIRS) if i != q])
    cp = pairs.copy()
    cp[q], cp[r] = (keys[q], vals[r]), (keys[r], vals[q])
    return (f"{fmt_pairs(pairs)}, {keys[q]}:", f"{fmt_pairs(cp)}, {keys[q]}:",
            " " + vals[q], " " + vals[r], None,
            dict(query=keys[q], pairs=[f"{k}:{v}" for k, v in pairs], swap=[q, r]))


def test_A(rng, pool):
    n1, n2 = pool.codes(2)
    tmpl = rng.choice(TEMPLATES_A)
    place, obj = rng.choice(PLACES), rng.choice(OBJECTS)
    clean   = tmpl.format(n1=n1, n2=n2, place=place, obj=obj)
    corrupt = tmpl.format(n1=n2, n2=n1, place=place, obj=obj)
    return (clean, corrupt, " " + n2, " " + n1, " " + n1,
            dict(query=n2, repeated=n1, place=place, obj=obj))


def _dup_pairs(rng, pool):
    """5-slot list, dup key bound twice (stale first, latest second), gap >= 2."""
    cs = pool.codes(9)
    keys, vals = cs[:4], cs[4:9]
    dup = keys[0]
    p = sorted(rng.sample(range(5), 2))
    while p[1] - p[0] < 2:
        p = sorted(rng.sample(range(5), 2))
    pairs = [None] * 5
    pairs[p[0]] = (dup, vals[0])                # stale binding (naive / corrupt answer)
    pairs[p[1]] = (dup, vals[1])                # latest binding (clean answer)
    ri = 0
    for i in range(5):
        if pairs[i] is None:
            pairs[i] = (keys[1 + ri], vals[2 + ri]); ri += 1
    cp = [(k, vals[1] if i == p[0] else
               vals[0] if i == p[1] else v) for i, (k, v) in enumerate(pairs)]
    return pairs, cp, dup, vals, p


def test_B1(rng, pool):
    pairs, cp, dup, vals, p = _dup_pairs(rng, pool)
    return (f"{fmt_pairs(pairs)}, {dup}:", f"{fmt_pairs(cp)}, {dup}:",
            " " + vals[1], " " + vals[0], " " + vals[0],
            dict(query=dup, pairs=[f"{k}:{v}" for k, v in pairs], dup_pos=p))


def test_B7(rng, pool):
    """In-list update cue: 'then' marks the second binding; both stay literal matches."""
    pairs, cp, dup, vals, p = _dup_pairs(rng, pool)
    def render(prs):
        return ", ".join(("then " if i == p[1] else "") + f"{k}: {v}"
                         for i, (k, v) in enumerate(prs))
    return (f"{render(pairs)}, {dup}:", f"{render(cp)}, {dup}:",
            " " + vals[1], " " + vals[0], " " + vals[0],
            dict(query=dup, pairs=[f"{k}:{v}" for k, v in pairs], dup_pos=p))


def _cued_update(rng, pool, trailing):
    cs = pool.codes(9)
    keys, vals = cs[:4], cs[4:9]
    dup = keys[0]
    stale, upd = vals[0], vals[1]
    pairs = [(dup, stale)] + [(keys[1 + i], vals[2 + i]) for i in range(2)]
    rng.shuffle(pairs)                          # stale binding at a random list position
    f_key, f_val = keys[3], vals[4]             # trailing filler binding (after update)
    cpairs = [(k, upd if k == dup else v) for k, v in pairs]   # stale<->updated swapped
    trail = f" The code for {f_key} is {f_val}." if trailing else ""
    clean   = (f"{fmt_pairs(pairs)}. Then the code for {dup} changed to {upd}."
               f"{trail} The code for {dup} is now")
    corrupt = (f"{fmt_pairs(cpairs)}. Then the code for {dup} changed to {stale}."
               f"{trail} The code for {dup} is now")
    return (clean, corrupt, " " + upd, " " + stale, " " + stale,
            dict(query=dup, pairs=[f"{k}:{v}" for k, v in pairs]))


def test_B9(rng, pool):
    return _cued_update(rng, pool, trailing=False)


def test_B9f(rng, pool):
    return _cued_update(rng, pool, trailing=True)


TESTS = {"A": test_A, "B": test_B9f, "C": test_C}      # was test_A, test_B1, ...
ALL_DS = ["A", "B", "C"]


def make_row(rng, ds, split, idx):
    pool = Pool(rng)
    clean, corrupt, ca, xa, naive, meta = TESTS[ds](rng, pool)
    cand = [ca, xa] + ([naive] if naive and naive not in (ca, xa) else [])
    return {"id": f"{ds}_{split}_{idx:05d}", "dataset": ds,
            "clean_prompt": clean, "corrupt_prompt": corrupt,
            "clean_answer": ca, "corrupt_answer": xa,
            "naive_answer": naive, "ld_candidates": cand, **meta}


# ---------------------------------------------------------------- modes ----
def mode_gen(args):
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tok = None
    if not args.skip_verify:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            print(f"[verify skipped: {e}]")
    for ds in ALL_DS:
        for split, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
            rows = [make_row(rng, ds, split, i) for i in range(n)]
            if tok is not None:
                for r in rows[:50]:
                    for a in r["ld_candidates"]:
                        assert len(tok.encode(a, add_special_tokens=False)) == 1, f"multi-token {a!r}"
            with open(os.path.join(args.out_dir, f"{ds}_{split}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        ex = rows[0]
        print(f"\n--- {ds} ---\nCLEAN:   {ex['clean_prompt']!r} -> {ex['clean_answer']!r}"
              f"\nCORRUPT: {ex['corrupt_prompt']!r} -> {ex['corrupt_answer']!r}")
    print(f"\nWrote {args.out_dir}/{{{','.join(ALL_DS)}}}_{{train,val,test}}.jsonl")


def mode_base_eval(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16).to(device).eval()
    print(f"device: {device} | model: {args.model_name}")

    def tid(s):
        return tok.encode(s, add_special_tokens=False)[0]

    for ds in ALL_DS:
        rows = [json.loads(l) for l in open(os.path.join(args.data_dir, f"{ds}_val.jsonl"))]
        pair_ok = n = 0
        picks = {"correct": 0, "corrupt": 0, "naive": 0, "other": 0}
        others = Counter()
        with torch.no_grad():
            for i in range(0, len(rows), args.batch_size):
                b = rows[i:i + args.batch_size]
                enc = tok([r["clean_prompt"] for r in b], padding=True,
                          return_tensors="pt", add_special_tokens=True).to(device)
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
                am = logits.argmax(-1)
                t = torch.tensor([tid(r["clean_answer"]) for r in b], device=device)
                d = torch.tensor([tid(r["corrupt_answer"]) for r in b], device=device)
                pair_ok += (logits.gather(1, t[:, None]) > logits.gather(1, d[:, None])).sum().item()
                for j, r in enumerate(b):
                    a = am[j].item()
                    if a == t[j].item():   picks["correct"] += 1
                    elif a == d[j].item(): picks["corrupt"] += 1
                    elif r.get("naive_answer") and a == tid(r["naive_answer"]):
                                           picks["naive"] += 1
                    else:
                        picks["other"] += 1
                        others[a] += 1
                n += len(b)
        prof = " / ".join(f"{k} {v/n:.3f}" for k, v in picks.items())
        top = ", ".join(f"{tok.decode([t])!r}x{c}" for t, c in others.most_common(3))
        print(f"{ds}: pairwise {pair_ok/n:.3f} | picks: {prof} | other top: {top} | n={n}")
    print("\nGATE: pairwise >= 0.75 AND correct >= ~0.6."
          "\n      B1 is the determinism control (identical rows across versions)."
          "\n      For B variants the stale-binding picks appear under 'corrupt'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "base_eval"], default="gen")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--out_dir", default="datasets")
    ap.add_argument("--data_dir", default="datasets")
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_verify", action="store_true")
    args = ap.parse_args()
    mode_gen(args) if args.mode == "gen" else mode_base_eval(args)


if __name__ == "__main__":
    main()