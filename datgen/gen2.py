#!/usr/bin/env python3
"""
gen_ioi_age_order_abc.py

Minimal controlled A/B/C dataset:

  C = IOI: identify the receiver
  A = C + retrieve the receiver's age
  B = C + find the next person after the receiver in the list order

A/B/C use the same fact table and transfer sentence. Only the final completion
changes. Corruption only replaces the receiver name in the transfer sentence.

Example:
  Alice age three.
  Bob age seven.
  Carol age six.
  Dave age one.
  Emma age five.
  Dave gave the book to Carol.

  C: The receiver is                            -> Carol
  A: The receiver's age is                      -> six
  B: The person after the receiver in the list is -> Dave

Generate:
  python -u gen_ioi_age_order_abc.py --mode gen \
    --model_name meta-llama/Llama-3.1-8B \
    --out_dir datasets/ioi_age_order \
    --n_train 1000 --n_val 250 --n_test 250

Base eval:
  python -u gen_ioi_age_order_abc.py --mode base_eval \
    --model_name meta-llama/Llama-3.1-8B \
    --data_dir datasets/ioi_age_order \
    --batch_size 4 --report_json base_eval_ioi_age_order.json
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter


NAMES = ["Alice", "Bob", "Carol", "Dave", "Emma"]

# Receivers are never Emma, so "next person in list" is always defined.
RECEIVER_NAMES = NAMES[:-1]

AGES = [
    "one", "two", "three", "four",
    "five", "six", "seven", "eight",
]

SPLITS = ("train", "val", "test")
DATASETS = ("A", "B", "C")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_seed(seed, split):
    data = f"{seed}|ioi-age-order|{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def unique_preserve(values):
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def build_next_of():
    return {
        NAMES[i]: NAMES[i + 1]
        for i in range(len(NAMES) - 1)
    }


def build_world(rng):
    ages = rng.sample(AGES, len(NAMES))
    age_of = dict(zip(NAMES, ages))
    next_of = build_next_of()

    clean_receiver, corrupt_receiver = rng.sample(RECEIVER_NAMES, 2)

    # Do not let the giver be either receiver or either B target. Otherwise the
    # model can get B right through the wrong "copy the giver" route.
    forbidden_givers = {
        clean_receiver,
        corrupt_receiver,
        next_of[clean_receiver],
        next_of[corrupt_receiver],
    }
    giver_candidates = [
        name for name in NAMES
        if name not in forbidden_givers
    ]

    # This should never trigger with five names, but keep a safe fallback.
    if not giver_candidates:
        giver_candidates = [
            name for name in NAMES
            if name not in (clean_receiver, corrupt_receiver)
        ]

    giver = rng.choice(giver_candidates)

    return {
        "age_of": age_of,
        "next_of": next_of,
        "giver": giver,
        "clean_receiver": clean_receiver,
        "corrupt_receiver": corrupt_receiver,
    }


def fact_text(world):
    return "\n".join(
        f"{name} age {world['age_of'][name]}."
        for name in NAMES
    )


def completion_text(ds):
    return {
        "C": "Question: Who received the book? Answer:",
        "A": "Question: How old is the person who received the book? Answer:",
        "B": "Question: Which person comes after the person who received the book in the list? Answer:",
    }[ds]


def answer_for(world, receiver, ds):
    if ds == "C":
        return " " + receiver

    if ds == "A":
        return " " + world["age_of"][receiver]

    if ds == "B":
        return " " + world["next_of"][receiver]

    raise ValueError(ds)


def naive_for(world, ds):
    """
    Wrong-role route: use the giver instead of the receiver.

    For B, the naive answer is the giver name. The more interesting intermediate
    failure for B is stored separately as core_answer = receiver.
    """
    if ds == "C":
        return " " + world["giver"]

    if ds == "A":
        return " " + world["age_of"][world["giver"]]

    if ds == "B":
        return " " + world["giver"]

    raise ValueError(ds)


def make_row(world, ds, split, idx):
    facts = fact_text(world)

    clean_sentence = (
        f"{world['giver']} gave the book to "
        f"{world['clean_receiver']}."
    )
    corrupt_sentence = (
        f"{world['giver']} gave the book to "
        f"{world['corrupt_receiver']}."
    )

    completion = completion_text(ds)

    clean_prompt = f"{facts}\n{clean_sentence}\n{completion}"
    corrupt_prompt = f"{facts}\n{corrupt_sentence}\n{completion}"

    clean_answer = answer_for(world, world["clean_receiver"], ds)
    corrupt_answer = answer_for(world, world["corrupt_receiver"], ds)

    naive_answer = naive_for(world, ds)
    corrupt_naive_answer = naive_answer  # giver does not change

    core_answer = " " + world["clean_receiver"]
    corrupt_core_answer = " " + world["corrupt_receiver"]

    assert clean_answer != corrupt_answer
    assert clean_prompt != corrupt_prompt

    return {
        "id": f"{ds}_{split}_{idx:05d}",
        "dataset": ds,
        "variant": "ioi_age_order",

        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "clean_answer": clean_answer,
        "corrupt_answer": corrupt_answer,

        "naive_answer": naive_answer,
        "corrupt_naive_answer": corrupt_naive_answer,
        "core_answer": core_answer,
        "corrupt_core_answer": corrupt_core_answer,

        "ld_candidates": unique_preserve([
            clean_answer,
            corrupt_answer,
            naive_answer,
            corrupt_naive_answer,
            core_answer,
            corrupt_core_answer,
        ]),

        "giver": world["giver"],
        "receiver": world["clean_receiver"],
        "corrupt_receiver": world["corrupt_receiver"],

        "receiver_age": world["age_of"][world["clean_receiver"]],
        "corrupt_receiver_age": world["age_of"][world["corrupt_receiver"]],

        "receiver_next": world["next_of"][world["clean_receiver"]],
        "corrupt_receiver_next": world["next_of"][world["corrupt_receiver"]],

        "ages": world["age_of"],
        "list_order": NAMES,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def load_tokenizer(args):
    if args.skip_verify:
        return None

    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"[verify skipped: {type(e).__name__}: {e}]")
        return None


def verify_rows(tok, rows_by_ds, split):
    if tok is None:
        return

    for ds, rows in rows_by_ds.items():
        for row in rows[:50]:
            for answer in row["ld_candidates"]:
                ids = tok.encode(answer, add_special_tokens=False)
                assert len(ids) == 1, (
                    f"{ds}/{split}: answer {answer!r} "
                    f"is not one token: {ids}"
                )

            clean_ids = tok.encode(
                row["clean_prompt"],
                add_special_tokens=True,
            )
            corrupt_ids = tok.encode(
                row["corrupt_prompt"],
                add_special_tokens=True,
            )

            assert len(clean_ids) == len(corrupt_ids), (
                f"{ds}/{split}: clean/corrupt token-length mismatch "
                f"{len(clean_ids)} vs {len(corrupt_ids)} in {row['id']}"
            )

    # A/B/C rows with the same index must describe the same world.
    if all(ds in rows_by_ds for ds in DATASETS):
        zipped = zip(*(rows_by_ds[ds] for ds in DATASETS))

        for i, rows in enumerate(zipped):
            for field in ("giver", "receiver", "corrupt_receiver"):
                values = [row[field] for row in rows]
                assert len(set(values)) == 1, (
                    f"{split}/{i}: A/B/C mismatch on {field}: {values}"
                )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def mode_gen(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tok = load_tokenizer(args)

    counts = {
        "train": args.n_train,
        "val": args.n_val,
        "test": args.n_test,
    }

    for split in SPLITS:
        n = counts[split]
        if n <= 0:
            continue

        rng = random.Random(stable_seed(args.seed, split))
        worlds = [build_world(rng) for _ in range(n)]

        rows_by_ds = {}

        for ds in DATASETS:
            rows = [
                make_row(world, ds, split, i)
                for i, world in enumerate(worlds)
            ]
            rows_by_ds[ds] = rows

            path = os.path.join(args.out_dir, f"{ds}_{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        verify_rows(tok, rows_by_ds, split)

    print(f"Wrote IOI age/order A/B/C dataset to {args.out_dir}/")

    for ds in DATASETS:
        path = os.path.join(args.out_dir, f"{ds}_val.jsonl")
        if not os.path.exists(path):
            path = os.path.join(args.out_dir, f"{ds}_train.jsonl")

        with open(path, encoding="utf-8") as f:
            row = json.loads(next(f))

        print(
            f"\n--- {ds} ---"
            f"\nCLEAN:   {row['clean_prompt']!r} -> {row['clean_answer']!r}"
            f"\nCORRUPT: {row['corrupt_prompt']!r} -> {row['corrupt_answer']!r}"
            f"\nCORE:    {row['core_answer']!r} | NAIVE: {row['naive_answer']!r}"
        )


# ---------------------------------------------------------------------------
# Base evaluation
# ---------------------------------------------------------------------------

def token_id_fn(tok):
    cache = {}

    def tid(text):
        if text not in cache:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"answer is not single-token: {text!r} -> {ids}"
                )
            cache[text] = ids[0]
        return cache[text]

    return tid


def classify_prediction(pred, ids, side):
    if side == "clean":
        target_key = "clean_answer"
        distractor_key = "corrupt_answer"
        naive_key = "naive_answer"
        core_key = "core_answer"
    else:
        target_key = "corrupt_answer"
        distractor_key = "clean_answer"
        naive_key = "corrupt_naive_answer"
        core_key = "corrupt_core_answer"

    if pred == ids[target_key]:
        return "correct"

    if pred == ids[distractor_key]:
        return "corrupt"

    if pred == ids[core_key]:
        return "core"

    if pred == ids[naive_key]:
        return "naive"

    return "other"


def eval_side(model, tok, rows, args, input_device, side):
    import torch

    prompt_field = (
        "clean_prompt" if side == "clean" else "corrupt_prompt"
    )
    target_field = (
        "clean_answer" if side == "clean" else "corrupt_answer"
    )
    distractor_field = (
        "corrupt_answer" if side == "clean" else "clean_answer"
    )

    tid = token_id_fn(tok)

    pair_ok = 0
    n = 0
    picks = Counter()
    others = Counter()

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]

            enc = tok(
                [row[prompt_field] for row in batch],
                padding=True,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(input_device)

            logits = model(
                **enc,
                use_cache=False,
            ).logits[:, -1, :].float()

            argmax = logits.argmax(-1)

            target_ids = torch.tensor(
                [tid(row[target_field]) for row in batch],
                device=logits.device,
            )
            distractor_ids = torch.tensor(
                [tid(row[distractor_field]) for row in batch],
                device=logits.device,
            )

            pair_ok += (
                logits.gather(1, target_ids[:, None])
                > logits.gather(1, distractor_ids[:, None])
            ).sum().item()

            for j, row in enumerate(batch):
                ids = {
                    key: tid(row[key])
                    for key in (
                        "clean_answer",
                        "corrupt_answer",
                        "naive_answer",
                        "corrupt_naive_answer",
                        "core_answer",
                        "corrupt_core_answer",
                    )
                }

                pred = argmax[j].item()
                pick = classify_prediction(pred, ids, side)
                picks[pick] += 1

                if pick == "other":
                    others[pred] += 1

            n += len(batch)

    return {
        "n": n,
        "pairwise": pair_ok / n,
        "picks": {
            key: picks.get(key, 0) / n
            for key in ("correct", "corrupt", "core", "naive", "other")
        },
        "other_top": [
            [tok.decode([token_id]), count]
            for token_id, count in others.most_common(10)
        ],
    }


def mode_base_eval(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
        "low_cpu_mem_usage": True,
    }

    if args.device_map == "auto":
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs,
        )
    except ImportError as e:
        if args.device_map == "auto" or "accelerate" not in str(e).lower():
            raise

        print("[accelerate missing; loading without low_cpu_mem_usage]")
        model_kwargs.pop("low_cpu_mem_usage", None)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs,
        )

    if args.device_map != "auto":
        model = model.to(device)

    model.eval()

    input_device = next(model.parameters()).device

    print(
        f"device: {device} | "
        f"input_device: {input_device} | "
        f"model: {args.model_name}"
    )

    summary = {}

    for ds in DATASETS:
        path = os.path.join(args.data_dir, f"{ds}_val.jsonl")

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}; run --mode gen first"
            )

        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        if args.max_rows:
            rows = rows[:args.max_rows]

        clean = eval_side(
            model, tok, rows, args, input_device, "clean"
        )
        corrupt = eval_side(
            model, tok, rows, args, input_device, "corrupt"
        )

        gate = (
            "PASS"
            if clean["pairwise"] >= args.gate_pairwise
            and corrupt["pairwise"] >= args.gate_pairwise
            and clean["picks"]["correct"] >= args.gate_correct
            and corrupt["picks"]["correct"] >= args.gate_correct
            else "FAIL"
        )

        clean_prof = " / ".join(
            f"{key} {value:.3f}"
            for key, value in clean["picks"].items()
        )
        corrupt_prof = " / ".join(
            f"{key} {value:.3f}"
            for key, value in corrupt["picks"].items()
        )

        clean_other = ", ".join(
            f"{text!r}x{count}"
            for text, count in clean["other_top"][:5]
        )
        corrupt_other = ", ".join(
            f"{text!r}x{count}"
            for text, count in corrupt["other_top"][:5]
        )

        print(
            f"{ds}: gate {gate} | n={clean['n']}\n"
            f"  clean:   pairwise {clean['pairwise']:.3f} | "
            f"{clean_prof} | other {clean_other}\n"
            f"  corrupt: pairwise {corrupt['pairwise']:.3f} | "
            f"{corrupt_prof} | other {corrupt_other}"
        )

        summary[ds] = {
            "gate": gate,
            "clean": clean,
            "corrupt": corrupt,
        }

    print(
        f"\nGATE: clean+corrupt pairwise >= {args.gate_pairwise:.2f} "
        f"AND clean+corrupt correct >= {args.gate_correct:.2f}."
    )

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": args.model_name,
                    "data_dir": args.data_dir,
                    "gate_pairwise": args.gate_pairwise,
                    "gate_correct": args.gate_correct,
                    "results": summary,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Wrote {args.report_json}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["gen", "base_eval"], default="gen")
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")

    ap.add_argument("--out_dir", default="datasets_ioi_age_order")
    ap.add_argument("--data_dir", default="datasets_ioi_age_order")

    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=250)
    ap.add_argument("--n_test", type=int, default=250)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_verify", action="store_true")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_rows", type=int, default=0)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device_map", choices=["none", "auto"], default="none")

    ap.add_argument("--gate_pairwise", type=float, default=0.90)
    ap.add_argument("--gate_correct", type=float, default=0.75)

    ap.add_argument("--report_json", default="")

    args = ap.parse_args()

    if args.mode == "gen":
        mode_gen(args)
    else:
        mode_base_eval(args)


if __name__ == "__main__":
    main()