#!/usr/bin/env python3
"""
intersect_abc.py — the A∩B = C verification, instrumented.

Loads masks_abc/{A,B,C}{suffix}_mask.pt (children) and
masks_abc/combined{combined_suffix}_mask.pt (parent), then runs three blocks:

  1. SET STRUCTURE: sizes, Jaccard, containment, full 7-region Venn decomposition
     with head names. Region meanings:
       A^B^C        shared core found by all three
       A^B not C    shared extras (contamination if A^B performs on A or B)
       A^C not B    core heads that B's finetune closed (rent-paying issue)
       B^C not A    core heads that A's finetune closed
       A/B/C only   the dataset-specific extras (A: S-inhibition? B: tracking?)

  2. DIAGNOSTIC MATRIX: per dataset, per mask:
       pairwise   target logit > distractor logit (chance = 0.5; CAN LIE — B2 lesson)
       top1 tgt   argmax == target   (the honest "does the task" metric)
       top1 dis   argmax == distractor (for A and B the distractor IS the predicted
                  failure mode: repeated code / stale binding -> systematic failure)
       top1 oth   argmax == anything else (high = collapse, not a clean failure)
       margin     mean(target logit - distractor logit)
       KL         KL(base || masked) on the task distribution
     Controls: A^C, B^C (which parent broke containment), A∪B (must pass all),
     rand1-3   (random |A^B|-head subsets of the parent — what A^B must beat on C).

  3. AUTOMATED READING + JSON report (masks_abc/intersect{suffix}_report.json).

Clean-side eval only (that's where the signature lives).

Run:  python -u intersect_abc.py --suffix _v2lam8 --combined_suffix _v3 \
        2>&1 | tee logs/intersect.log
      (--combined_suffix defaults to --suffix when parent and children share names)
"""

import argparse, json, os, random, re
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM
from torch.utils.data import DataLoader
from train_abc import JsonlDataset, make_collate, build_model, gate_modules


# ------------------------------------------------------------- mask ops ----
def load_mask(path):
    return torch.load(path, weights_only=True)

def m_and(a, b):  return {k: a[k] & b[k] for k in a}
def m_or(a, b):   return {k: a[k] | b[k] for k in a}
def m_sub(a, b):  return {k: a[k] & ~b[k] for k in a}
def m_size(m):    return sum(v.sum().item() for v in m.values())

def head_names(mask):
    out = set()
    for name, v in mask.items():
        layer = re.search(r"layers\.(\d+)\.", name)
        layer = layer.group(1) if layer else name
        for h in v.nonzero().flatten().tolist():
            out.add(f"L{layer}.H{h}")
    return out

def random_mask_like(combined, k, rng):
    """k random ON heads drawn from the parent (combined) mask."""
    positions = [(key, h) for key, v in combined.items()
                 for h in v.nonzero().flatten().tolist()]
    chosen = set(rng.sample(positions, min(k, len(positions))))
    out = {}
    for key, v in combined.items():
        m = torch.zeros(v.numel(), dtype=torch.bool)
        for h in range(v.numel()):
            if (key, h) in chosen:
                m[h] = True
        out[key] = m.reshape(v.shape)
    return out

def apply_mask(model, mask):
    with torch.no_grad():
        for name, m in gate_modules(model):
            on = mask[name].to(m.log_alpha.device)
            m.log_alpha.data = torch.where(on, torch.full_like(m.log_alpha.data, 5.0),
                                           torch.full_like(m.log_alpha.data, -1e6))
    model.set_final_circuit_mode(True)


# ----------------------------------------------------------------- eval ----
@torch.no_grad()
def diagnose(model, baseline, dl, args, gated):
    """Clean-side diagnostic metrics for one mask on one dataset."""
    model.eval()
    n = pair = tt = td = to = 0
    margin = kl_sum = 0.0
    others = Counter()
    for batch in dl:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        bl = baseline(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"],
                      use_cache=False).logits[:, -1, :].float()
        if gated:
            gl = model(input_ids=batch["input_ids"],
                       corrupted_input_ids=batch["corrupted_input_ids"],
                       attention_mask=batch["attention_mask"],
                       use_cache=False).logits[:, -1, :].float()
        else:
            gl = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       use_cache=False).logits[:, -1, :].float()
        t, d = batch["target_ids"], batch["distractor_ids"]
        gt = gl.gather(1, t.unsqueeze(1)).squeeze(1)
        gd = gl.gather(1, d.unsqueeze(1)).squeeze(1)
        pair += (gt > gd).sum().item()
        margin += (gt - gd).sum().item()
        kl_sum += F.kl_div(F.log_softmax(gl, -1), F.log_softmax(bl, -1),
                           reduction="batchmean", log_target=True).item() * t.size(0)
        am = gl.argmax(-1)
        tt += (am == t).sum().item()
        td += (am == d).sum().item()
        oth = (am != t) & (am != d)
        to += oth.sum().item()
        for tok in am[oth].tolist():
            others[tok] += 1
        n += t.size(0)
    return dict(pairwise=pair / n, top1_target=tt / n, top1_distractor=td / n,
                top1_other=to / n, margin=margin / n, kl=kl_sum / n,
                other_top=others.most_common(2))


def reading(met):
    if met["top1_target"] >= 0.8:
        return f"PASS (does the task: top1 target {met['top1_target']:.3f})"
    if met["top1_distractor"] >= 0.5:
        return (f"SYSTEMATIC failure via the predicted naive route "
                f"(top1 distractor {met['top1_distractor']:.3f})")
    if met["top1_other"] >= 0.5:
        return (f"COLLAPSE — neither target nor distractor "
                f"(other {met['top1_other']:.3f})")
    return (f"MIXED — tgt {met['top1_target']:.3f} / dis {met['top1_distractor']:.3f} "
            f"/ oth {met['top1_other']:.3f}")


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--data_dir", default="datasets_v2")
    ap.add_argument("--out_dir", default="masks_abc")
    ap.add_argument("--suffix", default="_v2lam8",
                    help="suffix of the per-dataset A/B/C masks")
    ap.add_argument("--combined_suffix", default=None,
                    help="suffix of the combined parent mask (default: same as --suffix)")
    ap.add_argument("--datasets", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--prune_mlp_blocks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
    sfx = args.suffix
    csfx = args.combined_suffix if args.combined_suffix is not None else sfx
    print(f"device: {args.device} | suffix: '{sfx}' | combined suffix: '{csfx}'")

    masks = {d: load_mask(os.path.join(args.out_dir, f"{d}{sfx}_mask.pt"))
             for d in args.datasets}
    masks["combined"] = load_mask(os.path.join(args.out_dir, f"combined{csfx}_mask.pt"))
    A, B, C = masks["A"], masks["B"], masks["C"]
    AB  = m_and(A, B)
    AC  = m_and(A, C)
    BC  = m_and(B, C)
    ABC = m_and(AB, C)
    AUB = m_or(A, B)
    masks.update({"A^B": AB, "A^C": AC, "B^C": BC, "AUB": AUB})
    torch.save(AB, os.path.join(args.out_dir, f"ABint{sfx}_mask.pt"))

    rng = random.Random(args.seed)
    rand_masks = {f"rand{i+1}": random_mask_like(masks["combined"], m_size(AB), rng)
                  for i in range(args.n_random)}

    # ---------------- 1. set structure ----------------
    print("\n=== mask sizes (heads) ===")
    for k in ["combined", "A", "B", "C", "A^B", "A^C", "B^C", "AUB"]:
        print(f"  {k:9s} {m_size(masks[k])}")

    print("\n=== subset sanity (expect 0 outside combined) ===")
    n_bad = 0
    for d in args.datasets:
        bad = m_size(m_sub(masks[d], masks['combined']))
        n_bad += bad
        print(f"  {d} heads outside combined: {bad}")
    if n_bad:
        print(f"  WARNING: {n_bad} heads sit outside the loaded parent — the A/B/C "
              f"masks were NOT carved from combined{csfx}. Random-mask controls and "
              f"containment numbers below are then meaningless. Check --combined_suffix.")

    jab = m_size(AB) / m_size(AUB)
    print(f"\nJaccard(A,B)          = {jab:.3f}")
    print(f"C contained in A^B    = {m_size(m_and(C, AB))}/{m_size(C)} "
          f"({m_size(m_and(C, AB))/m_size(C):.1%})")
    print(f"C contained in A      = {m_size(m_and(C, A))}/{m_size(C)} "
          f"({m_size(m_and(C, A))/m_size(C):.1%})")
    print(f"C contained in B      = {m_size(m_and(C, B))}/{m_size(C)} "
          f"({m_size(m_and(C, B))/m_size(C):.1%})")
    print(f"A^B contained in C    = {m_size(m_and(AB, C))}/{m_size(AB)} "
          f"({m_size(m_and(AB, C))/m_size(AB):.1%})")
    # random-overlap nulls (hypergeometric expectation |X|.|Y|/|parent|)
    P = m_size(masks["combined"])
    print(f"\nrandom-overlap null (parent={P}): "
          f"A^B ~ {m_size(A)*m_size(B)/P:.1f} (obs {m_size(AB)}), "
          f"A^C ~ {m_size(A)*m_size(C)/P:.1f} (obs {m_size(AC)}), "
          f"B^C ~ {m_size(B)*m_size(C)/P:.1f} (obs {m_size(BC)})")

    BC_or = m_or(B, C)
    AC_or = m_or(A, C)
    AB_or = AUB
    regions = {
        "A^B^C  (shared core)":            ABC,
        "A^B\\C  (shared extras)":         m_sub(AB, C),
        "A^C\\B  (core closed by B!)":     m_sub(AC, B),
        "B^C\\A  (core closed by A!)":     m_sub(BC, A),
        "A only  (A extra: inhibition?)":  m_sub(A, BC_or),
        "B only  (B extra: tracking?)":    m_sub(B, AC_or),
        "C only  (C-specific)":            m_sub(C, AB_or),
    }
    print("\n=== Venn decomposition ===")
    for name, m in regions.items():
        hs = sorted(head_names(m))
        print(f"  {name:32s} {m_size(m):3d}  {hs if hs else ''}")

    # ---------------- 2. diagnostic matrix ----------------
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    collate = make_collate(tok)
    val_loaders = {d: DataLoader(JsonlDataset(os.path.join(args.data_dir, f"{d}_val.jsonl")),
                                 batch_size=args.batch_size, shuffle=False, collate_fn=collate)
                   for d in args.datasets}
    baseline = LlamaForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16).to(args.device).eval()
    model = build_model(args, 0.05, 0)      # lambda/warmup irrelevant for eval

    eval_order = ["combined", "A", "B", "C", "A^B", "A^C", "B^C", "AUB"] + list(rand_masks)
    all_masks = {**masks, **rand_masks}
    report = {"set_sizes": {k: m_size(m) for k, m in all_masks.items()},
              "suffix": sfx, "combined_suffix": csfx, "cells": {}}

    for ds in args.datasets:
        print(f"\n=== on {ds}_val ===")
        print(f"{'mask':10s} {'pairwise':>8s} {'top1tgt':>8s} {'top1dis':>8s} "
              f"{'top1oth':>8s} {'margin':>8s} {'KL':>7s}  other-top")
        cells = {}
        met = diagnose(baseline, baseline, val_loaders[ds], args, gated=False)
        cells["base"] = met
        print(f"{'base':10s} {met['pairwise']:8.3f} {met['top1_target']:8.3f} "
              f"{met['top1_distractor']:8.3f} {met['top1_other']:8.3f} "
              f"{met['margin']:8.3f} {met['kl']:7.3f}")
        for name in eval_order:
            apply_mask(model, all_masks[name])
            met = diagnose(model, baseline, val_loaders[ds], args, gated=True)
            cells[name] = met
            ot = ", ".join(f"{tok.decode([t])!r}x{c}" for t, c in met["other_top"])
            print(f"{name:10s} {met['pairwise']:8.3f} {met['top1_target']:8.3f} "
                  f"{met['top1_distractor']:8.3f} {met['top1_other']:8.3f} "
                  f"{met['margin']:8.3f} {met['kl']:7.3f}  {ot}")
        report["cells"][ds] = cells

    # ---------------- 3. automated reading ----------------
    print("\n=== automated reading ===")
    for ds in args.datasets:
        met = report["cells"][ds]["A^B"]
        print(f"  A^B on {ds}: {reading(met)}")
    for ds in args.datasets:
        if report["cells"][ds]["AUB"]["top1_target"] < 0.8:
            print(f"  WARNING: AUB fails {ds} — mask application or eval is broken, "
                  f"do not trust this matrix")
    if report["cells"]["C"]["A^B"]["top1_target"] < 0.8:
        ac_ok = report["cells"]["C"]["A^C"]["top1_target"] >= 0.8
        bc_ok = report["cells"]["C"]["B^C"]["top1_target"] >= 0.8
        if ac_ok and not bc_ok:
            print("  diagnosis: A^C passes C but B^C fails -> B's finetune closed core "
                  "heads (see region 'A^C\\B'); the rent-paying issue is in B")
        elif bc_ok and not ac_ok:
            print("  diagnosis: B^C passes C but A^C fails -> A's finetune closed core "
                  "heads (see region 'B^C\\A')")
        elif not ac_ok and not bc_ok:
            print("  diagnosis: both parents miss core heads -> genuinely different "
                  "circuits per dataset (multiple-circuits regime)")
    rnd = [report["cells"]["C"][r]["top1_target"] for r in rand_masks]
    print(f"  random-mask reference on C: top1 target "
          f"{['%.3f' % x for x in rnd]} (A^B must clearly beat this)")

    with open(os.path.join(args.out_dir, f"intersect{sfx}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport saved: {args.out_dir}/intersect{sfx}_report.json")


if __name__ == "__main__":
    main()