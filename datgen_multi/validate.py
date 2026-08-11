"""Structural + behavioral validation of the multi-hop pruning dataset.

Checks (see RESEARCH.md for rationale):
  1. minimality   — token/word diff between clean and corrupt prompts
  2. structural   — re-simulate from metadata: clean needs >=min_hops, corrupt
                    query item never moves (0 hops), answers re-derive exactly
  3. zero-swap    — strip ALL move sentences from the corrupt prompt: answer
                    unchanged by construction; base model must stay ~100%
  4. scramble     — replace corrupt swaps with random swaps that never touch
                    the query item: answer unchanged; accuracy must stay ~100%
  5. sensitivity  — replace CLEAN swaps with random swaps and re-simulate the
                    new ground truth: if the model truly tracks, accuracy vs
                    the NEW truth stays high (and flips when the truth flips)
  6. balance      — answer distribution over candidates

Usage:
  python validate.py --model meta-llama/Llama-3.2-1B \
      --file datasets/multihop_swapmv_q_4b2s2h_test.jsonl [--limit 200]
"""

import argparse
import json
import random
from collections import Counter

import task
from eval_base import load_model, evaluate


def records_to_samples(records):
    samples = []
    for r in records:
        labels = r["labels"]
        init = [r["init"][labels[b]] for b in range(r["n_boxes"])]
        s = {
            "family": r["family"], "n_boxes": r["n_boxes"],
            "n_swaps": r["n_swaps"], "items": init, "init": init,
            "swaps": [[labels.index(a), labels.index(b)] for a, b in r["swaps"]],
            "corrupt_swaps": [[labels.index(a), labels.index(b)]
                              for a, b in r["corrupt_swaps"]],
            "query_item": r["query_item"],
            "query_box": labels.index(r["query_box"]),
            "hops_clean": r["hops_clean"],
            "corrupt_n_swaps": r.get("corrupt_n_swaps", r["n_swaps"]),
            "labels": labels,
        }
        # re-derive answers by simulation
        final = task.simulate(s["n_boxes"], s["init"], s["swaps"])
        if s["family"] == "itemloc":
            s["clean_answer"] = final.index(s["query_item"])
            s["corrupt_answer"] = s["query_box"]
        else:
            s["clean_answer"] = final[s["query_box"]]
            s["corrupt_answer"] = s["init"][s["query_box"]]
        samples.append(s)
    return samples


def check_minimality(records, tok):
    import difflib
    diffs_tok, diffs_word = [], []
    for r in records:
        a = tok.encode(r["clean_prompt"], add_special_tokens=False)
        b = tok.encode(r["corrupt_prompt"], add_special_tokens=False)
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        diffs_tok.append(sum(max(i2 - i1, j2 - j1)
                             for tag, i1, i2, j1, j2 in sm.get_opcodes()
                             if tag != "equal"))
        wa, wb = r["clean_prompt"].split(), r["corrupt_prompt"].split()
        sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
        diffs_word.append(sum(max(i2 - i1, j2 - j1)
                              for tag, i1, i2, j1, j2 in sm.get_opcodes()
                              if tag != "equal"))
    diffs_tok.sort()
    diffs_word.sort()
    m = len(diffs_tok) // 2
    print(f"[1] minimality (edit distance): token diff median={diffs_tok[m]} "
          f"max={diffs_tok[-1]} | word diff median={diffs_word[m]} "
          f"max={diffs_word[-1]} (prompt ~{len(records[0]['clean_prompt'].split())} words)")


def check_structural(records, samples, template):
    bad = 0
    for r, s in zip(records, samples):
        q_home = s["query_box"]
        ok = (not any(q_home in sw for sw in s["corrupt_swaps"])
              and task.answer_str(s, template, "clean") == r["clean_answer"]
              and task.answer_str(s, template, "corrupt") == r["corrupt_answer"]
              and r["hops_clean"] >= 1
              and r["hops_corrupt"] == 0
              and r["clean_answer"] != r["corrupt_answer"])
        bad += not ok
    print(f"[2] structural: {len(records) - bad}/{len(records)} pass "
          f"(0-hop corrupt, >=1-hop clean, answers re-derive, clean!=corrupt)")


def check_no_leak(records):
    """The answer must not be derivable by last-mention copying. In the TARGET
    portion (after the few-shot prefix), the query item must appear exactly
    once (its init sentence) in both clean and corrupt; and the clean answer
    must differ from the query item's init box (so a last-mention cheater
    answers the corrupt answer and is wrong on clean)."""
    bad = 0
    for r in records:
        q = r["query_item"]
        leak = False
        for which in ("clean_prompt", "corrupt_prompt"):
            tgt = r[which].rsplit("\n\n", 1)[-1]
            # swap/move sentences must never name the query item
            for sent in tgt.split(". "):
                if ("swap" in sent or "moves" in sent) and q in sent:
                    leak = True
        ok = (not leak) and r["clean_answer"] != r["corrupt_answer"]
        bad += not ok
    print(f"[2b] no-leak: {len(records) - bad}/{len(records)} pass "
          f"(query item never named in swap sentences; clean ans != corrupt)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--file", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--n-probe", type=int, default=4, help="scramble variants")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    records = [json.loads(l) for l in open(args.file)][:args.limit]
    template = records[0]["template"]
    samples = records_to_samples(records)

    model, tok = load_model(args.model)
    task.ITEM_POOL[:] = task.filter_single_token(tok, task.ITEM_POOL)
    task.NAME_POOL[:] = task.filter_single_token(tok, task.NAME_POOL)

    check_minimality(records, tok)
    check_structural(records, samples, template)
    check_no_leak(records)

    # baseline accuracy on the file as-is
    cc, _, _ = evaluate(model, tok, samples, template, args.batch_size, False)
    kc, _, _ = evaluate(model, tok, samples, template, args.batch_size, True)
    print(f"[0] baseline: clean={cc:.4f} corrupt={kc:.4f} (cand-acc, n={len(samples)})")

    # [3] zero-swap probe: corrupt with no move sentences at all
    zero = [dict(s, corrupt_swaps=[]) for s in samples]
    acc, _, _ = evaluate(model, tok, zero, template, args.batch_size, True)
    print(f"[3] zero-swap probe on corrupt: acc={acc:.4f} "
          f"(must be ~1.0: swaps causally irrelevant to corrupt answer)")

    # [4] scramble probe on corrupt: random replacement swaps, query never moves
    rng = random.Random(99)
    accs = []
    for _ in range(args.n_probe):
        scr = []
        for s in samples:
            sw = task._random_swaps(rng, s["n_boxes"],
                                    s.get("corrupt_n_swaps", s["n_swaps"]),
                                    avoid_box=s["query_box"])
            scr.append(dict(s, corrupt_swaps=sw))
        a, _, _ = evaluate(model, tok, scr, template, args.batch_size, True)
        accs.append(a)
    print(f"[4] scramble probe on corrupt: accs={['%.3f' % a for a in accs]} "
          f"(NB: at 3 boxes the non-query pair is unique, so at corrupt_n_swaps=1 "
          f"this probe has no degrees of freedom and is vacuous)")

    # [4b] mutation probe on corrupt: replace the swap sentence(s) in the
    # target portion with content that CANNOT carry the answer. If accuracy
    # survives, the corrupt answer ignores swap content/meaning entirely.
    mutations = {
        "words-shuffled": lambda sent: " ".join(random.Random(5).sample(
            sent.split(), len(sent.split()))) + ".",
        "nonexistent-boxes": lambda sent: "Box D and box E are swapped.",
        "unrelated": lambda sent: "The sky is blue and the grass is green.",
    }
    import re
    for name, fn in mutations.items():
        mut_samples = []
        for s, r in zip(samples, records):
            pre, tgt = r["corrupt_prompt"].rsplit("\n\n", 1)
            tgt2 = re.sub(r"Box [A-Z] and box [A-Z] are swapped\.",
                          lambda m: fn(m.group(0)), tgt)
            mut_samples.append(dict(s, _override_prompt=pre + "\n\n" + tgt2))
        # evaluate with overridden prompts but corrupt gold answers
        acc = _eval_override(model, tok, mut_samples, template, args.batch_size)
        print(f"[4b] mutation probe on corrupt [{name}]: acc={acc:.3f} "
              f"(gold unchanged; ~1.0 = answer ignores swap content)")

    # [5] sensitivity on clean: random replacement swaps change the truth;
    # a truly tracking model stays accurate vs the NEW truth
    accs, flips = [], []
    for _ in range(args.n_probe):
        scr = []
        for s in samples:
            while True:
                sw = task._random_swaps(rng, s["n_boxes"], s["n_swaps"])
                final = task.simulate(s["n_boxes"], s["init"], sw)
                if s["family"] == "itemloc":
                    new_ans = final.index(s["query_item"])
                else:
                    new_ans = final[s["query_box"]]
                scr.append(dict(s, swaps=sw, clean_answer=new_ans))
                break
        a, _, det = evaluate(model, tok, scr, template, args.batch_size, False)
        accs.append(a)
        # flip rate: how often prediction changes when ground truth changes
        n_flip = n_ch = 0
        for s0, s1, d in zip(samples, scr, det):
            if s0["clean_answer"] != s1["clean_answer"]:
                n_ch += 1
                gold0 = task.answer_str(s0, template, "clean")
                if d["pred_cand"] != gold0.strip():
                    n_flip += 1
        flips.append(n_flip / max(n_ch, 1))
    print(f"[5] sensitivity on clean: acc-vs-new-truth={['%.3f' % a for a in accs]} "
          f"flip-rate={['%.3f' % f for f in flips]}")
    print("    (high acc vs new truth = model routes answer through swaps)")

    # [6] balance
    ca = Counter(r["clean_answer"] for r in records)
    ka = Counter(r["corrupt_answer"] for r in records)
    print(f"[6] balance: clean={dict(sorted(ca.items()))} "
          f"corrupt={dict(sorted(ka.items()))}")


def _eval_override(model, tok, samples, template, batch_size):
    """Evaluate samples whose '_override_prompt' replaces the rendered prompt;
    gold = corrupt answer."""
    prompts = [s["_override_prompt"] for s in samples]
    golds = [task.answer_str(s, template, "corrupt") for s in samples]
    import torch
    n_ok = 0
    for i in range(0, len(prompts), batch_size):
        bp = prompts[i:i + batch_size]
        enc = tok(bp, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        for j in range(len(bp)):
            s = samples[i + j]
            cands = [tok.encode(c, add_special_tokens=False)
                     for c in task.candidate_strs(s, template)]
            gold = tok.encode(golds[i + j], add_special_tokens=False)
            best = max(cands, key=lambda c: lg[j][c[0]].item())
            n_ok += best == gold
    return n_ok / len(prompts)


if __name__ == "__main__":
    main()
