"""Generate the final multi-hop tracking dataset (JSONL, repo conventions).

Usage:
  python gen_dataset.py --template swapmv_q --n-boxes 4 --n-swaps 2 --min-hops 2 \
      --train 1024 --val 256 --test 256 --seed 0 --out datasets/
"""

import argparse
import json
import os
import random

import task


def holders_labels(template, n_boxes):
    if task.TEMPLATES[template].get("holders") == "names":
        return task.NAME_POOL[:n_boxes]
    return None


def write_split(path, n, seed, template, n_boxes, n_swaps, min_hops, split,
                id_offset):
    rng = random.Random(seed)
    family = task.TEMPLATES[template]["family"]
    labels = holders_labels(template, n_boxes)
    n_written = 0
    with open(path, "w") as f:
        while n_written < n:
            s = task.make_sample(rng, n_boxes, n_swaps, min_hops, family, labels)
            rec = task.to_record(s, template, id_offset + n_written, split)
            f.write(json.dumps(rec) + "\n")
            n_written += 1
    print(f"wrote {n_written} -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--n-boxes", type=int, default=4)
    p.add_argument("--n-swaps", type=int, default=2)
    p.add_argument("--min-hops", type=int, default=2)
    p.add_argument("--train", type=int, default=1024)
    p.add_argument("--val", type=int, default=256)
    p.add_argument("--test", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="datasets")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = f"multihop_{args.template}_{args.n_boxes}b{args.n_swaps}s{args.min_hops}h"
    write_split(f"{args.out}/{base}_train.jsonl", args.train, args.seed,
                args.template, args.n_boxes, args.n_swaps, args.min_hops,
                "train", 0)
    write_split(f"{args.out}/{base}_val.jsonl", args.val, args.seed + 1,
                args.template, args.n_boxes, args.n_swaps, args.min_hops,
                "val", args.train)
    write_split(f"{args.out}/{base}_test.jsonl", args.test, args.seed + 2,
                args.template, args.n_boxes, args.n_swaps, args.min_hops,
                "test", args.train + args.val)


if __name__ == "__main__":
    main()
