"""Generate JSONL datasets for one or all template families.

Usage:
  python generate.py                 # all families, 50 samples each
  python generate.py --family t1_arithmetic_gate --n 50
"""

import argparse
import json
import os

from templates import FAMILIES, build_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, choices=list(FAMILIES) + [None])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    keys = [args.family] if args.family else list(FAMILIES)
    for key in keys:
        ds = build_dataset(key, n=args.n, seed=args.seed)
        # uniqueness sanity check
        assert len({d["clean_prompt"] for d in ds}) == len(ds), f"{key}: duplicate prompts"
        path = os.path.join(OUT_DIR, f"{key}.jsonl")
        with open(path, "w") as f:
            for d in ds:
                f.write(json.dumps(d) + "\n")
        print(f"wrote {path} ({len(ds)} samples)")
        print("  example clean:  ", ds[0]["clean_prompt"])
        print("  example corrupt:", ds[0]["corrupt_prompt"])


if __name__ == "__main__":
    main()
