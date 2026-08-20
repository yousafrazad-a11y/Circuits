"""Intersect per-model raw generations: keep pairs that ALL models pass.

Usage:
  ../venv/bin/python intersect_models.py OUT N raw1.jsonl raw2.jsonl [raw3.jsonl ...]

Every example in every raw file is re-verified with the strict exact-match
verifier. A pair is kept iff every model passed BOTH its clean and corrupt
sides. The first N such pairs (sorted by pair_id) are written to OUT, ONE
JSON LINE PER PAIR:

  {"pair_id": ..., "name": ..., "start_floor": ...,
   "clean":   {"moves": [...], "expected": ..., "prompt": ..., "answer": ...},
   "corrupt": {"moves": [...], "expected": ..., "prompt": ..., "answer": ...},
   "verified_models": [...]}

Strict passing means the answer text is identical across models, so a
single canonical "answer" is kept per side.
"""
import json
import sys

from gen_elevator_dataset import verify


def main():
    out_path, n_keep, raw_paths = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    models = []
    per_model = []
    for path in raw_paths:
        # model tag from filename, e.g. all3_8b-raw.jsonl -> 8b
        models.append(path.rsplit("/", 1)[-1].replace("-raw.jsonl", ""))
        rows = [json.loads(l) for l in open(path)]
        by_pair = {}
        for r in rows:
            r["correct"] = verify(r, r["answer"])
            by_pair.setdefault(r["pair_id"], {})[r["type"]] = r
        per_model.append(by_pair)

    common = set(per_model[0])
    for bp in per_model[1:]:
        common &= set(bp)

    kept, n_bad = [], 0
    for pid in sorted(common):
        if all(all(r["correct"] for r in bp[pid].values())
               for bp in per_model):
            kept.append(pid)
        else:
            n_bad += 1

    n_written = 0
    with open(out_path, "w") as f:
        for pid in kept[:n_keep]:
            sides = {}
            for typ in ("clean", "corrupt"):
                r = per_model[0][pid][typ]
                # sanity: strict pass => identical answers across models
                assert all(bp[pid][typ]["answer"] == r["answer"]
                           for bp in per_model[1:])
                sides[typ] = {"moves": r["moves"], "expected": r["expected"],
                              "prompt": r["prompt"], "answer": r["answer"]}
            f.write(json.dumps({
                "pair_id": pid,
                "name": per_model[0][pid]["clean"]["name"],
                "start_floor": per_model[0][pid]["clean"]["start_floor"],
                "clean": sides["clean"],
                "corrupt": sides["corrupt"],
                "verified_models": models,
            }) + "\n")
            n_written += 1

    print(f"models: {models}")
    print(f"pairs in all raws: {len(common)} | passed by ALL models: "
          f"{len(kept)} | failed by >= 1: {n_bad}")
    print(f"wrote {n_written} pairs -> {out_path}")
    if n_written < n_keep:
        print("WARNING: fewer than requested; need more candidates.")


if __name__ == "__main__":
    main()
