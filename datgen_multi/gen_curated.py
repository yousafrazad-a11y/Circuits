"""Generate the final curated dataset: sample a pool, keep only samples the
base model answers correctly on BOTH clean and corrupt (strict: both the
candidate-restricted and full-vocab argmax metrics), then write splits.

This guarantees the hard requirement (100% base accuracy on clean AND corrupt)
by construction. Filtering is standard dataset curation; it is documented in
RESEARCH.md and the kept fraction is reported.

Usage:
  python gen_curated.py --model meta-llama/Llama-3.2-1B \
      --template swapmv_ans_q_fs16_uq_l --n-boxes 3 --n-swaps 1 --min-hops 1 \
      --train 1024 --val 256 --test 256 --out datasets/
"""

import argparse
import json
import os
import random

import torch

import task
from eval_base import load_model, evaluate


def holders_labels(template, n_boxes):
    if task.TEMPLATES[template].get("holders") == "names":
        return task.NAME_POOL[:n_boxes]
    return None


@torch.no_grad()
def curate(model, tok, pool, template, batch_size):
    """Return the sub-pool correct on both clean and corrupt (cand AND vocab)."""
    _, _, det_c = evaluate(model, tok, pool, template, batch_size, corrupt=False)
    _, _, det_k = evaluate(model, tok, pool, template, batch_size, corrupt=True)
    keep = [s for s, dc, dk in zip(pool, det_c, det_k)
            if dc["ok_cand"] and dc["ok_vocab"] and dk["ok_cand"] and dk["ok_vocab"]]
    return keep


def gen_pool(n, seed, template, n_boxes, n_swaps, min_hops):
    rng = random.Random(seed)
    family = task.TEMPLATES[template]["family"]
    labels = holders_labels(template, n_boxes)
    cns = task.TEMPLATES[template].get("corrupt_n_swaps")
    return [task.make_sample(rng, n_boxes, n_swaps, min_hops, family, labels,
                             corrupt_n_swaps=cns)
            for _ in range(n)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--template", default="swapmv_ans_q_fs16_uq_l")
    p.add_argument("--n-boxes", type=int, default=3)
    p.add_argument("--n-swaps", type=int, default=1)
    p.add_argument("--min-hops", type=int, default=1)
    p.add_argument("--train", type=int, default=1024)
    p.add_argument("--val", type=int, default=256)
    p.add_argument("--test", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-factor", type=float, default=1.4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default="datasets")
    args = p.parse_args()

    model, tok = load_model(args.model)
    task.ITEM_POOL[:] = task.filter_single_token(tok, task.ITEM_POOL)
    task.NAME_POOL[:] = task.filter_single_token(tok, task.NAME_POOL)

    os.makedirs(args.out, exist_ok=True)
    base = f"multihop_{args.template}_{args.n_boxes}b{args.n_swaps}s{args.min_hops}h"

    id_offset = 0
    for split, n_want in [("train", args.train), ("val", args.val),
                          ("test", args.test)]:
        n_pool = int(n_want * args.pool_factor)
        kept = []
        round_ = 0
        while len(kept) < n_want:
            pool = gen_pool(n_pool, args.seed + 1000 * round_ + hash(split) % 997,
                            args.template, args.n_boxes, args.n_swaps,
                            args.min_hops)
            got = curate(model, tok, pool, args.template, args.batch_size)
            kept.extend(got)
            print(f"{split} round {round_}: kept {len(got)}/{len(pool)} "
                  f"(total {len(kept)}/{n_want})", flush=True)
            round_ += 1
            n_pool = int((n_want - len(kept)) * args.pool_factor) + 8
        kept = kept[:n_want]
        path = f"{args.out}/{base}_{split}.jsonl"
        with open(path, "w") as f:
            for i, s in enumerate(kept):
                rec = task.to_record(s, args.template, id_offset + i, split)
                f.write(json.dumps(rec) + "\n")
        id_offset += n_want
        print(f"wrote {len(kept)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
