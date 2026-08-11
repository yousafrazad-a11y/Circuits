#!/usr/bin/env python3
"""
intersect_abc_8b.py — instrumented A∩B=C verification for the 8B runs.

Differences from the old 1B script:
  * defaults to meta-llama/Llama-3.1-8B
  * lower default batch size for a 40GB GPU
  * supports different suffixes for the combined parent and phase-2 children
  * prints random-overlap nulls, not just observed intersections
  * treats low A∪B-on-C as a containment diagnostic, not automatically broken eval
  * saves a JSON report with set sizes, Venn regions, random controls, and metrics

Expected files:
  OUT_DIR/combined{combined_suffix}_mask.pt
  OUT_DIR/A{suffix}_mask.pt
  OUT_DIR/B{suffix}_mask.pt
  OUT_DIR/C{suffix}_mask.pt

Typical run:
  python -u intersect_abc_8b.py \
    --model_name meta-llama/Llama-3.1-8B \
    --data_dir datasets/abc \
    --out_dir masks_abc_8b \
    --suffix _b3lam3 \
    --combined_suffix _b3lam3 \
    --batch_size 4 \
    2>&1 | tee logs/intersect_8b_b3lam3.log
"""

import argparse
import json
import os
import random
import re
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM

from train_abc import JsonlDataset, make_collate, build_model, gate_modules


# ------------------------------------------------------------- mask ops ----
def load_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, weights_only=True)


def m_and(a, b):
    return {k: a[k] & b[k] for k in a}


def m_or(a, b):
    return {k: a[k] | b[k] for k in a}


def m_sub(a, b):
    return {k: a[k] & ~b[k] for k in a}


def m_size(m):
    return sum(int(v.sum().item()) for v in m.values())


def head_names(mask):
    out = set()
    for name, v in mask.items():
        layer = re.search(r"layers\.(\d+)\.", name)
        layer = layer.group(1) if layer else name
        for h in v.nonzero().flatten().tolist():
            out.add(f"L{layer}.H{h}")
    return out


def random_mask_like(parent, k, rng):
    """k random ON heads drawn from the combined parent mask."""
    positions = [
        (key, h)
        for key, v in parent.items()
        for h in v.nonzero().flatten().tolist()
    ]
    if not positions:
        raise ValueError("parent mask has no active heads")
    chosen = set(rng.sample(positions, min(k, len(positions))))
    out = {}
    for key, v in parent.items():
        flat = torch.zeros(v.numel(), dtype=torch.bool)
        for h in range(v.numel()):
            flat[h] = (key, h) in chosen
        out[key] = flat.reshape(v.shape)
    return out


def apply_mask(model, mask):
    with torch.no_grad():
        for name, module in gate_modules(model):
            on = mask[name].to(module.log_alpha.device)
            module.log_alpha.data = torch.where(
                on,
                torch.full_like(module.log_alpha.data, 5.0),
                torch.full_like(module.log_alpha.data, -1e6),
            )
    model.set_final_circuit_mode(True)


# ----------------------------------------------------------------- eval ----
@torch.no_grad()
def diagnose(model, baseline, dl, args, gated):
    """Clean-side diagnostic metrics for one mask on one dataset."""
    model.eval()
    n = pair = top1_target = top1_distractor = top1_other = 0
    margin_sum = kl_sum = 0.0
    others = Counter()

    for batch in dl:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        base_logits = baseline(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, -1, :].float()

        if gated:
            logits = model(
                input_ids=batch["input_ids"],
                corrupted_input_ids=batch["corrupted_input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits[:, -1, :].float()
        else:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits[:, -1, :].float()

        target = batch["target_ids"]
        distractor = batch["distractor_ids"]
        target_logits = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        distractor_logits = logits.gather(1, distractor.unsqueeze(1)).squeeze(1)

        pair += (target_logits > distractor_logits).sum().item()
        margin_sum += (target_logits - distractor_logits).sum().item()
        kl_sum += F.kl_div(
            F.log_softmax(logits, -1),
            F.log_softmax(base_logits, -1),
            reduction="batchmean",
            log_target=True,
        ).item() * target.size(0)

        argmax = logits.argmax(-1)
        top1_target += (argmax == target).sum().item()
        top1_distractor += (argmax == distractor).sum().item()
        other = (argmax != target) & (argmax != distractor)
        top1_other += other.sum().item()
        for token_id in argmax[other].tolist():
            others[token_id] += 1
        n += target.size(0)

    return dict(
        pairwise=pair / n,
        top1_target=top1_target / n,
        top1_distractor=top1_distractor / n,
        top1_other=top1_other / n,
        margin=margin_sum / n,
        kl=kl_sum / n,
        other_top=others.most_common(3),
    )


def reading(metrics):
    if metrics["top1_target"] >= 0.8:
        return f"PASS (top1 target {metrics['top1_target']:.3f})"
    if metrics["top1_distractor"] >= 0.5:
        return (
            "SYSTEMATIC failure via the predicted naive route "
            f"(top1 distractor {metrics['top1_distractor']:.3f})"
        )
    if metrics["top1_other"] >= 0.5:
        return (
            "COLLAPSE — neither target nor distractor "
            f"(top1 other {metrics['top1_other']:.3f})"
        )
    return (
        f"MIXED — target {metrics['top1_target']:.3f}, "
        f"distractor {metrics['top1_distractor']:.3f}, "
        f"other {metrics['top1_other']:.3f}"
    )


def load_val_loader(args, collate, ds):
    path = os.path.join(args.data_dir, f"{ds}_val.jsonl")
    rows = JsonlDataset(path).rows
    if args.max_rows:
        rows = rows[:args.max_rows]
    return DataLoader(rows, batch_size=args.batch_size, shuffle=False, collate_fn=collate)


# ----------------------------------------------------------------- main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--data_dir", default="datasets/abc")
    parser.add_argument("--out_dir", default="masks_abc_8b")
    parser.add_argument("--suffix", default="_b3lam3",
                        help="Suffix used by A/B/C child masks.")
    parser.add_argument("--combined_suffix", default=None,
                        help="Suffix used by combined parent mask. Default: same as --suffix.")
    parser.add_argument("--datasets", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--n_random", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=0,
                        help="Optional validation-row cap for a quick smoke test.")
    parser.add_argument("--prune_mlp_blocks", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"

    if args.datasets != ["A", "B", "C"]:
        raise SystemExit("This intersection script expects --datasets A B C")

    suffix = args.suffix
    combined_suffix = args.combined_suffix if args.combined_suffix is not None else args.suffix
    print(f"device: {args.device}")
    print(f"child suffix: {suffix!r} | combined suffix: {combined_suffix!r}")
    print(f"model: {args.model_name}")

    # ---------------- load masks ----------------
    masks = {
        d: load_mask(os.path.join(args.out_dir, f"{d}{suffix}_mask.pt"))
        for d in ["A", "B", "C"]
    }
    masks["combined"] = load_mask(
        os.path.join(args.out_dir, f"combined{combined_suffix}_mask.pt")
    )

    A, B, C = masks["A"], masks["B"], masks["C"]
    parent = masks["combined"]
    AB = m_and(A, B)
    AC = m_and(A, C)
    BC = m_and(B, C)
    ABC = m_and(AB, C)
    AUB = m_or(A, B)
    masks.update({"A^B": AB, "A^C": AC, "B^C": BC, "AUB": AUB})

    if combined_suffix == suffix:
        ab_name = f"ABint{suffix}_mask.pt"
        report_name = f"intersect{suffix}_report.json"
    else:
        ab_name = f"ABint{suffix}_from_combined{combined_suffix}_mask.pt"
        report_name = f"intersect{suffix}_from_combined{combined_suffix}_report.json"
    torch.save(AB, os.path.join(args.out_dir, ab_name))

    rng = random.Random(args.seed)
    rand_masks = {
        f"rand{i + 1}": random_mask_like(parent, m_size(AB), rng)
        for i in range(args.n_random)
    }
    all_masks = {**masks, **rand_masks}

    # ---------------- 1. set structure ----------------
    print("\n=== mask sizes (heads) ===")
    for key in ["combined", "A", "B", "C", "A^B", "A^C", "B^C", "AUB"]:
        print(f"  {key:9s} {m_size(masks[key])}")

    print("\n=== subset sanity (expect 0 outside combined) ===")
    outside = {}
    for d in ["A", "B", "C"]:
        outside[d] = m_size(m_sub(masks[d], parent))
        print(f"  {d} heads outside combined: {outside[d]}")
    if any(outside.values()):
        print("  HARD WARNING: at least one child contains heads outside the parent.")
        print("  Check --suffix, --combined_suffix, and checkpoint lineage before trusting intersections.")

    parent_size = m_size(parent)
    nulls = {
        "A^B": (m_size(A) * m_size(B) / parent_size),
        "A^C": (m_size(A) * m_size(C) / parent_size),
        "B^C": (m_size(B) * m_size(C) / parent_size),
    }
    print("\n=== random-overlap nulls: E|X∩Y| = |X||Y|/P ===")
    for key, expected in nulls.items():
        observed = m_size(masks[key])
        print(f"  {key:4s}: observed {observed:3d} | random null {expected:5.1f} | delta {observed - expected:+5.1f}")

    union_ab = max(1, m_size(AUB))
    print(f"\nJaccard(A,B)       = {m_size(AB) / union_ab:.3f}")
    print(f"C contained in A^B = {m_size(m_and(C, AB))}/{m_size(C)} ({m_size(m_and(C, AB)) / m_size(C):.1%})")
    print(f"C contained in A   = {m_size(AC)}/{m_size(C)} ({m_size(AC) / m_size(C):.1%})")
    print(f"C contained in B   = {m_size(BC)}/{m_size(C)} ({m_size(BC) / m_size(C):.1%})")
    print(f"A^B contained in C = {m_size(ABC)}/{m_size(AB)} ({m_size(ABC) / max(1, m_size(AB)):.1%})")

    BC_or = m_or(B, C)
    AC_or = m_or(A, C)
    regions = {
        "A^B^C  (shared core)": ABC,
        "A^B\\C  (shared extras)": m_sub(AB, C),
        "A^C\\B  (core closed by B)": m_sub(AC, B),
        "B^C\\A  (core closed by A)": m_sub(BC, A),
        "A only  (A-specific)": m_sub(A, BC_or),
        "B only  (B-specific)": m_sub(B, AC_or),
        "C only  (C-specific)": m_sub(C, AUB),
    }

    print("\n=== Venn decomposition ===")
    region_report = {}
    for name, mask in regions.items():
        heads = sorted(head_names(mask))
        region_report[name] = {"size": m_size(mask), "heads": heads}
        print(f"  {name:30s} {m_size(mask):3d}  {heads if heads else ''}")

    # ---------------- 2. diagnostic matrix ----------------
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = make_collate(tok)
    val_loaders = {d: load_val_loader(args, collate, d) for d in ["A", "B", "C"]}

    baseline = LlamaForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    for p in baseline.parameters():
        p.requires_grad = False

    model = build_model(args, 0.05, 0)  # lambda/warmup are irrelevant for evaluation

    eval_order = ["combined", "A", "B", "C", "A^B", "A^C", "B^C", "AUB"] + list(rand_masks)
    report = {
        "model_name": args.model_name,
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "suffix": suffix,
        "combined_suffix": combined_suffix,
        "set_sizes": {k: m_size(m) for k, m in all_masks.items()},
        "random_overlap_nulls": nulls,
        "heads_outside_parent": outside,
        "venn_regions": region_report,
        "cells": {},
    }

    for ds in ["A", "B", "C"]:
        print(f"\n=== on {ds}_val ===")
        print(
            f"{'mask':10s} {'pairwise':>8s} {'top1tgt':>8s} {'top1dis':>8s} "
            f"{'top1oth':>8s} {'margin':>8s} {'KL':>7s}  other-top"
        )
        cells = {}
        met = diagnose(baseline, baseline, val_loaders[ds], args, gated=False)
        cells["base"] = met
        print(
            f"{'base':10s} {met['pairwise']:8.3f} {met['top1_target']:8.3f} "
            f"{met['top1_distractor']:8.3f} {met['top1_other']:8.3f} "
            f"{met['margin']:8.3f} {met['kl']:7.3f}"
        )
        for name in eval_order:
            apply_mask(model, all_masks[name])
            met = diagnose(model, baseline, val_loaders[ds], args, gated=True)
            cells[name] = met
            other_top = ", ".join(f"{tok.decode([token_id])!r}x{count}"
                                  for token_id, count in met["other_top"])
            print(
                f"{name:10s} {met['pairwise']:8.3f} {met['top1_target']:8.3f} "
                f"{met['top1_distractor']:8.3f} {met['top1_other']:8.3f} "
                f"{met['margin']:8.3f} {met['kl']:7.3f}  {other_top}"
            )
        report["cells"][ds] = cells

    # ---------------- 3. automated reading ----------------
    print("\n=== automated reading ===")
    for ds in ["A", "B", "C"]:
        met = report["cells"][ds]["A^B"]
        print(f"  A^B on {ds}: {reading(met)}")

    c_full = report["cells"]["C"]["C"]["top1_target"]
    print("\n=== functional containment on C ===")
    print(f"  C mask on C: {c_full:.3f}")
    for d in ["A", "B"]:
        score = report["cells"]["C"][d]["top1_target"]
        ratio = score / max(c_full, 1e-9)
        print(f"  {d} mask on C: {score:.3f} ({ratio:.1%} of C-mask performance)")

    for ds in ["A", "B"]:
        if report["cells"][ds]["AUB"]["top1_target"] < 0.8:
            print(f"  WARNING: AUB fails {ds}; mask application or eval may be broken.")

    aub_c = report["cells"]["C"]["AUB"]["top1_target"]
    c_only = region_report["C only  (C-specific)"]["size"]
    if aub_c < 0.8:
        if c_only > 0:
            print(
                f"  containment note: AUB scores {aub_c:.3f} on C and there are {c_only} C-only heads; "
                "this can be legitimate missing containment, not necessarily broken eval."
            )
        else:
            print(
                f"  WARNING: AUB scores only {aub_c:.3f} on C despite no C-only heads; "
                "inspect mask application/eval."
            )

    if report["cells"]["C"]["A^B"]["top1_target"] < 0.8:
        ac_ok = report["cells"]["C"]["A^C"]["top1_target"] >= 0.8
        bc_ok = report["cells"]["C"]["B^C"]["top1_target"] >= 0.8
        if ac_ok and not bc_ok:
            print("  diagnosis: A^C passes C but B^C fails -> B's finetune closed core heads.")
        elif bc_ok and not ac_ok:
            print("  diagnosis: B^C passes C but A^C fails -> A's finetune closed core heads.")
        elif not ac_ok and not bc_ok:
            print("  diagnosis: both parents miss core heads -> genuinely different circuits.")

    random_c = [report["cells"]["C"][name]["top1_target"] for name in rand_masks]
    print(f"  random-mask reference on C: {['%.3f' % x for x in random_c]}")
    print("  A^B must clearly beat these random same-sized parent subsets on C.")

    report["automated_reading"] = {
        f"A^B_on_{ds}": reading(report["cells"][ds]["A^B"]) for ds in ["A", "B", "C"]
    }
    report["functional_containment_on_C"] = {
        "C_mask_on_C": c_full,
        "A_mask_on_C": report["cells"]["C"]["A"]["top1_target"],
        "B_mask_on_C": report["cells"]["C"]["B"]["top1_target"],
        "AUB_on_C": aub_c,
    }
    report["random_reference_on_C"] = random_c

    report_path = os.path.join(args.out_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nA∩B mask saved: {os.path.join(args.out_dir, ab_name)}")
    print(f"report saved:  {report_path}")


if __name__ == "__main__":
    main()