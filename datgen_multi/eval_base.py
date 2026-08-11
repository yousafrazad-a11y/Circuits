"""Base-model evaluation for the multi-hop tracking dataset.

Modes:
  sweep  — generate samples in memory for each template variant and report
           clean/corrupt accuracy (the wording-iteration engine).
  file   — evaluate a saved JSONL dataset.
  probe  — behavioral proofs on generated samples (zero-swap & scramble).

Usage:
  python eval_base.py sweep --model meta-llama/Llama-3.2-1B
  python eval_base.py file  --model gpt2 --file datasets/multihop_test.jsonl
"""

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import task


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.eval().cuda()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def evaluate(model, tok, samples, template, batch_size=32, corrupt=False):
    """Returns (acc_cand, acc_vocab, details). For each sample the model sees the
    clean (or corrupt) prompt; we check the gold answer against (a) the restricted
    candidate set and (b) full-vocab argmax."""
    key = "corrupt" if corrupt else "clean"
    prompts, golds, cands = [], [], []
    for s in samples:
        swaps = s["corrupt_swaps"] if corrupt else s["swaps"]
        prompts.append(task.render(s, template, swaps=swaps))
        golds.append(task.answer_str(s, template, key))
        cands.append(task.candidate_strs(s, template))

    # tokenize candidate sets per sample
    gold_ids, cand_ids = [], []
    for g, cs in zip(golds, cands):
        gi = tok.encode(g, add_special_tokens=False)
        ci = [tok.encode(c, add_special_tokens=False) for c in cs]
        if len(gi) != 1 or any(len(c) != 1 for c in ci):
            raise ValueError(f"multi-token answer: {g!r} -> {gi}, {cs} -> {ci}")
        gold_ids.append(gi[0])
        cand_ids.append([c[0] for c in ci])

    n_ok_cand = n_ok_vocab = 0
    details = []
    for i in range(0, len(prompts), batch_size):
        bp = prompts[i:i + batch_size]
        enc = tok(bp, return_tensors="pt", padding=True).to("cuda")
        logits = model(**enc).logits[:, -1, :].float()
        am = logits.argmax(-1).tolist()
        for j in range(len(bp)):
            k = i + j
            lg = logits[j]
            best_cand = max(cand_ids[k], key=lambda c: lg[c].item())
            ok_c = best_cand == gold_ids[k]
            ok_v = am[j] == gold_ids[k]
            n_ok_cand += ok_c
            n_ok_vocab += ok_v
            details.append({
                "prompt": bp[j], "gold": golds[k],
                "pred_cand": tok.decode([best_cand]),
                "pred_vocab": tok.decode([am[j]]),
                "ok_cand": ok_c, "ok_vocab": ok_v,
            })
    n = len(prompts)
    return n_ok_cand / n, n_ok_vocab / n, details


def gen_samples(n, seed, n_boxes, n_swaps, min_hops, family, holders=None,
                template=None):
    rng = random.Random(seed)
    labels = task.NAME_POOL[:n_boxes] if holders == "names" else None
    cns = task.TEMPLATES[template].get("corrupt_n_swaps") if template else None
    return [task.make_sample(rng, n_boxes, n_swaps, min_hops, family, labels,
                             corrupt_n_swaps=cns)
            for _ in range(n)]


def mode_sweep(model, tok, args):
    configs = [
        (3, 2, 2),
        (4, 2, 2),
    ]
    if args.config:
        configs = [tuple(int(x) for x in args.config.split(","))]
    print(f"model={args.model}  n={args.n} per cell")
    print(f"{'template':<12} {'boxes':>5} {'swaps':>5} {'hops':>4} "
          f"{'cln_cand':>8} {'cln_vocab':>9} {'cor_cand':>8} {'cor_vocab':>9}")
    for template in args.templates:
        family = task.TEMPLATES[template]["family"]
        holders = task.TEMPLATES[template].get("holders")
        for n_boxes, n_swaps, min_hops in configs:
            samples = gen_samples(args.n, args.seed, n_boxes, n_swaps,
                                  min_hops, family, holders, template)
            cc, cv, _ = evaluate(model, tok, samples, template,
                                 args.batch_size, corrupt=False)
            kc, kv, det = evaluate(model, tok, samples, template,
                                   args.batch_size, corrupt=True)
            print(f"{template:<12} {n_boxes:>5} {n_swaps:>5} {min_hops:>4} "
                  f"{cc:>8.2f} {cv:>9.2f} {kc:>8.2f} {kv:>9.2f}")
            if args.dump:
                with open(args.dump, "a") as f:
                    for d in det:
                        d["template"] = template
                        d["config"] = [n_boxes, n_swaps, min_hops]
                        f.write(json.dumps(d) + "\n")


def mode_file(model, tok, args):
    records = [json.loads(l) for l in open(args.file)]
    if args.limit:
        records = records[:args.limit]
    template = records[0]["template"]
    # reconstruct structured samples from records
    samples = []
    for r in records:
        labels = r["labels"]
        init = [r["init"][labels[b]] for b in range(r["n_boxes"])]
        samples.append({
            "family": r["family"], "n_boxes": r["n_boxes"],
            "n_swaps": r["n_swaps"], "items": init, "init": init,
            "swaps": [[labels.index(a), labels.index(b)] for a, b in r["swaps"]],
            "corrupt_swaps": [[labels.index(a), labels.index(b)] for a, b in r["corrupt_swaps"]],
            "query_item": r["query_item"], "query_box": labels.index(r["query_box"]),
            "corrupt_n_swaps": r.get("corrupt_n_swaps", r["n_swaps"]),
            "labels": labels,
            "hops_clean": r["hops_clean"], "hops_corrupt": r["hops_corrupt"],
            # answers re-derived by simulation to guard against file corruption
            "clean_answer": None, "corrupt_answer": None,
        })
    # re-derive answers by simulation and check against file
    mismatches = 0
    for s, r in zip(samples, records):
        final = task.simulate(s["n_boxes"], s["init"], s["swaps"])
        if s["family"] == "itemloc":
            s["clean_answer"] = final.index(s["query_item"])
            s["corrupt_answer"] = s["query_box"]
        else:
            s["clean_answer"] = final[s["query_box"]]
            s["corrupt_answer"] = s["init"][s["query_box"]]
        if task.answer_str(s, template, "clean") != r["clean_answer"] or \
           task.answer_str(s, template, "corrupt") != r["corrupt_answer"]:
            mismatches += 1
    if mismatches:
        print(f"WARNING: {mismatches} records fail ground-truth re-derivation")

    cc, cv, _ = evaluate(model, tok, samples, template,
                         args.batch_size, corrupt=False)
    kc, kv, _ = evaluate(model, tok, samples, template,
                         args.batch_size, corrupt=True)
    print(f"file={args.file}  model={args.model}  n={len(samples)}")
    print(f"clean:   cand-acc={cc:.4f}  vocab-acc={cv:.4f}")
    print(f"corrupt: cand-acc={kc:.4f}  vocab-acc={kv:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sweep", "file"])
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--file")
    p.add_argument("--templates", nargs="+", default=list(task.TEMPLATES))
    p.add_argument("--config", help="e.g. '5,3,2' = boxes,swaps,min_hops")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dump", help="append corrupt-run details to this jsonl")
    args = p.parse_args()

    model, tok = load_model(args.model)
    for pool_name in ("ITEM_POOL", "NAME_POOL"):
        pool = getattr(task, pool_name)
        ok = task.filter_single_token(tok, pool)
        dropped = set(pool) - set(ok)
        if dropped:
            print(f"NOTE: {pool_name} dropped (not single-token for "
                  f"{args.model}): {sorted(dropped)}")
        pool[:] = ok

    if args.mode == "sweep":
        mode_sweep(model, tok, args)
    else:
        mode_file(model, tok, args)


if __name__ == "__main__":
    main()
