"""Generate and verify the elevator floor-tracking dataset.

An "example" is a clean/corrupt PAIR: same name, same starting floor, same
move sequence — only the FIRST move is flipped in the corrupt version.

Prompting (tuned for Llama-3.1-8B-Instruct after failure analysis):
  - format rules live in the SYSTEM message (Llama 3.1 prioritizes it),
  - ONE-SHOT demo: a full example Q/A in the exact format, with the first
    move DOWN (the failure-prone case),
  - the answer format gives the starting floor a legitimate first line
    ("Start: 5th") — without it the model emits a phantom "up -> 5th"
    line whenever the first real move is down,
  - the user message is just the story, no format prose.

Answer format (verified exactly):
    Start: 5th
    down -> 4th
    up -> 5th
    ...
    FINAL ANSWER: 5th

Pipeline:
  1. Sample --train + --test candidate pairs. Every prompt is unique
     (name, start floor, move sequence). Both sequences of a pair are
     guaranteed to never go below the 1st floor at any step.
  2. Run the model (greedy) on every prompt, batched. Raw generations are
     saved to <out>-raw.jsonl so verification can be re-run without a GPU
     (--verify-only).
  3. Verify with a strict exact-match check: the answer's non-empty lines
     must be exactly the ground-truth lines (Start line, one step line per
     move, FINAL ANSWER line) — correct content AND exact format, no
     preamble, no numbering, no extra/phantom steps. A pair is kept only
     if BOTH clean and corrupt pass. The first --keep passing pairs go to
     <out>-train.jsonl / <out>-test.jsonl.

Usage:
  ../venv/bin/python gen_elevator_dataset.py                 # full run
  ../venv/bin/python gen_elevator_dataset.py --train 4 --test 2 --keep 1  # smoke
  ../venv/bin/python gen_elevator_dataset.py --verify-only   # re-check raw
"""
import argparse
import json
import random
from itertools import product

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FLOORS = (3, 4, 5, 6, 7)
MAX_NEW_TOKENS = 512

NAMES = [
    "Max", "Ava", "Liam", "Noah", "Emma", "Olivia", "Sophia", "Mason",
    "Ethan", "Mia", "Lucas", "Amelia", "Harper", "James", "Ben", "Ella",
    "Leo", "Zoe", "Lily", "Owen", "Nora", "Caleb", "Ruby", "Isaac",
    "Hazel", "Jonah", "Ivy", "Eli", "Clara", "Felix", "Maya", "Silas",
    "June", "Milo", "Ada", "Theo", "Esme", "Hugo", "Wren", "Jude",
    "Cora", "Ezra", "Nina", "Otto", "Iris", "Sam", "Grace", "Alex",
    "Rose", "Finn", "Lucy", "Jack", "Henry", "Alice", "Charlie", "Daniel",
    "Emily", "Jacob", "Michael", "Sarah", "David", "Laura", "Thomas",
    "Anna", "Peter", "Julia", "Mark", "Chloe", "Ryan", "Katie", "Adam",
    "Megan", "Dylan", "Lauren", "Aaron", "Paige", "Connor", "Sofia",
    "Nathan", "Bella", "Luke", "Hannah", "Oscar", "Stella", "Xavier",
    "Naomi", "Victor", "Willow", "Levi", "Aurora", "Miles", "Eden",
    "Blake", "Tessa", "Reid", "Fiona", "Colin", "Daphne", "Simon",
    "Vera", "Troy", "Bianca", "Marcus",
]

SYSTEM = ('You track elevator floors and reply in this EXACT format: first '
          'line "Start: <floor>", then one line per move "<up|down> -> '
          '<floor>", then a last line "FINAL ANSWER: <floor>". No other '
          'text, no numbering, no explanations.')


def ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def simulate(start, moves):
    """Return list of floors after each move, or None if it dips below 1st."""
    floors = []
    f = start
    for m in moves:
        f += 1 if m == "up" else -1
        if f < 1:
            return None
        floors.append(f)
    return floors


def flip_first(moves):
    return [("down" if moves[0] == "up" else "up")] + list(moves[1:])


def make_prompt(name, start, moves):
    seq = ", then ".join(f"one floor {m}" for m in moves)
    return (f"{name} is on the {ordinal(start)} floor in an elevator. "
            f"{name} goes {seq}. {name} is now on")


def answer_lines(start, moves):
    """The exact lines a perfect answer consists of, from ground truth."""
    floors = simulate(start, moves)
    return ([f"Start: {ordinal(start)}"]
            + [f"{mv} -> {ordinal(f)}" for mv, f in zip(moves, floors)]
            + [f"FINAL ANSWER: {ordinal(floors[-1])}"])


def build_demo(hops):
    """One-shot example Q/A in the exact format; first move DOWN on purpose."""
    name, start = "Jill", 6
    moves = (["down", "up", "down", "up", "up"] * 2)[:hops]
    return [{"role": "user", "content": make_prompt(name, start, moves)},
            {"role": "assistant", "content": "\n".join(answer_lines(start, moves))}]


def sample_candidates(n_pairs, seed, hops):
    """Enumerate every valid unique pair, shuffle, take the first n_pairs.

    A pair is {moves, flip_first(moves)}: both prompts must be unique, so
    each (name, start, unordered move-pair) is one candidate. Enumerating
    (instead of rejection sampling) guarantees termination and a clear
    error when n_pairs exceeds capacity.
    """
    space = []
    for name in NAMES:
        for start in FLOORS:
            for moves in product(("up", "down"), repeat=hops):
                corrupt = tuple(flip_first(moves))
                if corrupt < moves:  # count each unordered pair once
                    continue
                if simulate(start, moves) is None \
                        or simulate(start, corrupt) is None:
                    continue
                space.append({"name": name, "start": start,
                              "moves": list(moves)})
    if len(space) < n_pairs:
        raise SystemExit(f"only {len(space)} unique pairs possible "
                         f"({len(NAMES)} names x {len(FLOORS)} floors x "
                         f"{hops} hops), requested {n_pairs}")
    rng = random.Random(seed)
    rng.shuffle(space)
    return space[:n_pairs]


def expand(pairs, split):
    """Turn candidate pairs into individual prompt dicts (clean + corrupt).

    Which side is "clean" is randomized per pair (deterministically), so the
    clean stream is 50/50 up-first/down-first — the sampler's dedup keeps
    only one orientation, and without this every clean side would start
    with the same direction.
    """
    out = []
    for i, p in enumerate(pairs):
        if random.Random(f"orient-{split}-{i}").random() < 0.5:
            sides = (("clean", p["moves"]), ("corrupt", flip_first(p["moves"])))
        else:
            sides = (("corrupt", p["moves"]), ("clean", flip_first(p["moves"])))
        for typ, moves in sides:
            floors = simulate(p["start"], moves)
            out.append({
                "pair_id": f"{split}_{i:05d}",
                "type": typ,
                "name": p["name"],
                "start_floor": p["start"],
                "moves": moves,
                "expected": ordinal(floors[-1]),
                "prompt": make_prompt(p["name"], p["start"], moves),
            })
    return out


def verify(ex, answer):
    """Strict pass: correct content AND exact instructed format.

    The answer's non-empty lines must be EXACTLY the expected lines:
    "Start: <floor>", one "up -> 6th" line per move (correct direction and
    floor), and the "FINAL ANSWER: <floor>" line. Anything else fails:
    preamble, numbered lists, phantom/extra steps, "down -> 6th -> 5th"
    style, wrong floors, missing final line. No loose parsing -> no
    parser mistakes.
    """
    got = [l.strip() for l in answer.strip().splitlines() if l.strip()]
    return got == answer_lines(ex["start_floor"], ex["moves"])


def run_model(examples, model_name, batch_size, hops):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    prefix = ([{"role": "system", "content": SYSTEM}] + build_demo(hops))
    answers = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        texts = [tok.apply_chat_template(
            prefix + [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True) for ex in chunk]
        ids = tok(texts, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        for j, row in enumerate(out):
            answers.append(tok.decode(
                row[ids["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(examples))}"
              f"/{len(examples)}", flush=True)
    return answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--train", type=int, default=1000,
                    help="candidate pairs for the train split")
    ap.add_argument("--test", type=int, default=1000,
                    help="candidate pairs for the test split")
    ap.add_argument("--keep", type=int, default=500,
                    help="verified pairs to keep per split")
    ap.add_argument("--hops", type=int, default=5,
                    help="elevator moves per prompt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="elevator")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip generation; re-verify <out>-raw.jsonl")
    args = ap.parse_args()

    raw_path = f"{args.out}-raw.jsonl"

    if args.verify_only:
        raw = [json.loads(l) for l in open(raw_path)]
    else:
        # 1. sample unique candidate pairs (train/test share one pool: all unique)
        pairs = sample_candidates(args.train + args.test, args.seed, args.hops)
        train_pairs, test_pairs = pairs[:args.train], pairs[args.train:]
        examples = expand(train_pairs, "train") + expand(test_pairs, "test")
        assert len({e["prompt"] for e in examples}) == len(examples), \
            "prompts are not unique"
        print(f"{len(pairs)} candidate pairs -> {len(examples)} unique prompts")

        # 2. generate
        print(f"loading {args.model} ...", flush=True)
        answers = run_model(examples, args.model, args.batch_size, args.hops)
        raw = [dict(ex, answer=a) for ex, a in zip(examples, answers)]
        with open(raw_path, "w") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {raw_path}")

    # 3. verify each step + final answer; keep pairs where BOTH sides pass
    by_pair = {}
    for r in raw:
        r["correct"] = verify(r, r["answer"])
        by_pair.setdefault(r["pair_id"], {})[r["type"]] = r

    kept = {"train": [], "test": []}
    n_bad = 0
    for pid in sorted(by_pair):
        both = by_pair[pid]
        if len(both) == 2 and all(r["correct"] for r in both.values()):
            split = pid.rsplit("_", 1)[0]
            if len(kept[split]) < args.keep:
                kept[split].append(pid)
        else:
            n_bad += 1

    for split in ("train", "test"):
        path = f"{args.out}-{split}.jsonl"
        with open(path, "w") as f:
            for pid in kept[split]:
                for typ in ("clean", "corrupt"):
                    f.write(json.dumps(by_pair[pid][typ]) + "\n")
        print(f"{split}: kept {len(kept[split])} verified pairs -> {path}")
    print(f"rejected pairs (model wrong on >= 1 side): {n_bad}")
    if any(len(kept[s]) < args.keep for s in kept):
        print("WARNING: not enough verified pairs; raise --train/--test "
              "or inspect rejections.", flush=True)


if __name__ == "__main__":
    main()
