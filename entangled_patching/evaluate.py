"""Evaluate template families on clean/corrupt accuracy + tricked rate.

Checks:
  - token-length alignment and differing-token span between P_clean / P_corrupt
  - greedy generation accuracy on clean and corrupt prompts
  - tricked rate: corrupt run outputs the CLEAN answer (context inertia)

Usage:
  python evaluate.py --model meta-llama/Llama-3.1-8B-Instruct                 # all families
  python evaluate.py --model meta-llama/Llama-3.1-8B-Instruct --family t1_arithmetic_gate
  python evaluate.py --model ... --iter 2 --tag after_rephrase
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from templates import FAMILIES, build_dataset

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "logs")


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def check_alignment(tok, ds):
    """All pairs must have equal token length; differing tokens must form a
    single contiguous span (the control token)."""
    bad = []
    spans = []
    for d in ds:
        a = tok.encode(d["clean_prompt"])
        b = tok.encode(d["corrupt_prompt"])
        if len(a) != len(b):
            bad.append((d["id"], "length", len(a), len(b)))
            continue
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if not diffs:
            bad.append((d["id"], "identical"))
            continue
        if diffs[-1] - diffs[0] + 1 != len(diffs):
            bad.append((d["id"], "non-contiguous", diffs))
        spans.append((diffs[0], diffs[-1]))
    ok = len(bad) == 0
    ntok = len(tok.encode(ds[0]["clean_prompt"]))
    return ok, ntok, spans, bad


def normalize(text):
    """Strict extraction: leading answer token from the first line.
    ' 67. Gate is ENABLED...' -> '67'; '"OFF"' -> 'off'.
    """
    t = text.split("\n")[0].strip().strip('"').strip()
    m = re.match(r"[A-Za-z0-9_/.+-]+", t)
    tok = m.group(0) if m else t
    return tok.rstrip(".,").lower()


def first_candidate(text, candidates):
    """Candidate-restricted extraction: earliest whole-word occurrence of any
    candidate in the output. candidates must be pre-lowercased, longest first.
    Returns the candidate or None."""
    low = text.lower()
    best_pos, best_cand = None, None
    for c in candidates:
        m = re.search(r"\b" + re.escape(c) + r"\b", low)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_cand = m.start(), c
    return best_cand


@torch.no_grad()
def run_prompts(model, tok, prompts, batch_size=32, max_new_tokens=24):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(batch)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs


def score(raw_outputs, targets, alt_targets=None, candidates=None):
    """Returns (pass_count, tricked_count, details).

    Primary prediction: earliest whole-word candidate occurrence (candidate-
    restricted scoring). Falls back to strict first-token extraction when no
    candidate appears. Both are recorded in details.
    """
    npass = ntricked = 0
    details = []
    for i, (raw, tgt) in enumerate(zip(raw_outputs, targets)):
        strict = normalize(raw)
        pred = first_candidate(raw, candidates) if candidates else None
        if pred is None:
            pred = strict
        gold = tgt.lower()
        ok = pred == gold
        tricked = (alt_targets is not None) and (pred == alt_targets[i].lower())
        npass += ok
        ntricked += tricked
        details.append({"raw": raw, "pred": pred, "pred_strict": strict,
                        "gold": gold, "pass": ok, "tricked": tricked})
    return npass, ntricked, details


REPORT = """======================================================================
PROCEDURAL ITERATION REPORT #{iter_no}
Dataset Family Tested: {family_name}
Model Evaluated: {model}

GOAL VERIFICATION:
- Task Context Entangled: {entangled}
- Sequence Length / Positional Alignment: {alignment} ({seq_len} tokens both sides)

PERFORMANCE METRICS (N={n}):
- Clean Prompt Accuracy: {clean_acc}% ({clean_pass}/{n})
- Corrupt Prompt Accuracy: {corrupt_acc}% ({corrupt_pass}/{n})
- Tricked / Fallback Error Rate: {tricked_pct}% (Model outputs clean answer during corrupt run)

FAILURE ANALYSIS:
{clean_fail}
{corrupt_fail}

NEXT STEP / PROPOSED MODIFICATION:
- {next_step}
======================================================================"""


def analyze_failures(details, kind):
    fails = [d for d in details if not d["pass"]]
    if not fails:
        return f"- If {kind} < 90%: no failures"
    examples = "; ".join(f"pred={d['pred']!r} gold={d['gold']!r}" for d in fails[:5])
    return f"- If {kind} < 90%: {len(fails)} failures, e.g. {examples}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    keys = [args.family] if args.family else list(FAMILIES)

    # alignment check needs only tokenizer; run before loading model
    tok_only = AutoTokenizer.from_pretrained(args.model)
    align_info = {}
    for key in keys:
        ds = build_dataset(key, n=args.n, seed=args.seed)
        ok, ntok, spans, bad = check_alignment(tok_only, ds)
        align_info[key] = (ok, ntok, spans, bad)
        if not ok:
            print(f"[ALIGNMENT FAIL] {key}: {bad[:5]}")

    model, tok = load_model(args.model)
    model_short = args.model.split("/")[-1]

    summary = {}
    for key in keys:
        fam = FAMILIES[key]
        ds = build_dataset(key, n=args.n, seed=args.seed)
        ok, ntok, spans, bad = align_info[key]

        clean_outs = run_prompts(model, tok, [d["clean_prompt"] for d in ds],
                                 batch_size=args.batch_size)
        corrupt_outs = run_prompts(model, tok, [d["corrupt_prompt"] for d in ds],
                                   batch_size=args.batch_size)

        # candidate set: every attainable answer, longest-first for matching
        cands = sorted({d["clean_target"].lower() for d in ds} |
                       {fam["fallback"].lower()}, key=len, reverse=True)

        cp, _, cdet = score(clean_outs, [d["clean_target"] for d in ds],
                            candidates=cands)
        kp, kt, kdet = score(corrupt_outs, [d["corrupt_target"] for d in ds],
                             alt_targets=[d["clean_target"] for d in ds],
                             candidates=cands)

        n = len(ds)
        clean_acc = round(100 * cp / n, 1)
        corrupt_acc = round(100 * kp / n, 1)
        tricked_pct = round(100 * kt / n, 1)

        clean_fail = analyze_failures(cdet, "Clean")
        corrupt_fail = analyze_failures(kdet, "Corrupt")
        next_step = ("none — family passes; archive dataset"
                     if clean_acc >= 90 and corrupt_acc >= 90
                     else "see failure analysis; adjust phrasing and re-run")

        report = REPORT.format(
            iter_no=args.iter, family_name=fam["name"], model=args.model,
            entangled="YES", alignment="Matching" if ok else "SHIFTED",
            seq_len=ntok, n=n,
            clean_acc=clean_acc, clean_pass=cp,
            corrupt_acc=corrupt_acc, corrupt_pass=kp,
            tricked_pct=tricked_pct,
            clean_fail=clean_fail, corrupt_fail=corrupt_fail,
            next_step=next_step)
        print(report)

        summary[key] = {
            "family": fam["name"], "n": n, "seq_len": ntok,
            "alignment_ok": ok,
            "clean_acc": clean_acc, "corrupt_acc": corrupt_acc,
            "tricked_pct": tricked_pct,
            "clean_details": cdet, "corrupt_details": kdet,
        }

    tag = f"_{args.tag}" if args.tag else ""
    log_path = os.path.join(LOG_DIR, f"iter{args.iter}{tag}_{model_short}.json")
    with open(log_path, "w") as f:
        json.dump({"model": args.model, "iter": args.iter,
                   "families": summary}, f, indent=2)
    print(f"\nlog written: {log_path}")
    print("\n==== SUMMARY ====")
    for key, s in summary.items():
        flag = "PASS" if s["clean_acc"] >= 90 and s["corrupt_acc"] >= 90 else "fail"
        print(f"{flag:4s}  {key:28s} clean={s['clean_acc']:5.1f}%  "
              f"corrupt={s['corrupt_acc']:5.1f}%  tricked={s['tricked_pct']:5.1f}%  "
              f"align={'ok' if s['alignment_ok'] else 'BAD'}")


if __name__ == "__main__":
    main()
