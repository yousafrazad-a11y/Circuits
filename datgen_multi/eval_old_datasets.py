"""Evaluate the 7 pre-existing datasets in ../datasets/ against the same
requirements as our v3 multi-hop dataset, on the same model (Llama-3.2-1B).

Checks per dataset file:
  0. Base-model accuracy on clean and corrupt (span-logprob ranking between
     the record's clean_target and corrupted_target; also first-token acc).
  1. Minimality: difflib token-level similarity of clean/corrupt prompts.
  2. Structural 0-hop: corrupt answer is itemA's init container, and the
     chain never moves that container; clean_target != corrupted_target.
  2b. No-leak: no chain sentence contains the query item; answer not stated
      with the query item in the same sentence.
  3. Behavioral probes on corrupt (n=PROBE_N):
       zero-chain   : delete chain sentences -> answer should survive (~high)
       garble-chain : word-shuffle chain sentences -> answer should survive
  5. Clean chain-sensitivity: perturb chain so truth changes -> model must
     flip to the new truth.
  6. Answer balance over targets.
"""
import json, os, random, re, sys, difflib
from collections import Counter
import torch
from eval_base import load_model

DATA_DIR = "../datasets"
FILES = ["dataset.jsonl", "dataset2.jsonl", "dataset3.jsonl", "dataset4.jsonl",
         "dataset5.jsonl", "dataset6.jsonl", "dataset7.jsonl"]
MODEL = "meta-llama/Llama-3.2-1B"
BATCH = 16
PROBE_N = 150
rng = random.Random(0)

CHAIN_RE = re.compile(r"is moved to|is moved from|hands it to|is given to")


def load_records(path):
    return [json.loads(l) for l in open(path)]


def spans_score(model, tok, prompts, targets, batch=BATCH):
    """summed logprob of each target span -> (N, C) tensor."""
    C = len(targets)
    tids = [tok(t, add_special_tokens=False)["input_ids"] for t in targets]
    out = torch.zeros(len(prompts), C)
    for i in range(0, len(prompts), batch):
        bp = prompts[i:i + batch]
        enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        with torch.no_grad():
            logits = model(**enc).logits
        lp = torch.log_softmax(logits.float(), -1)
        for ci, ids in enumerate(tids):
            sc = torch.zeros(len(bp), device=model.device)
            for j, tid in enumerate(ids):
                pos = enc["input_ids"].shape[1] - len(ids) + j
                sc += lp[:, pos - 1, tid]
            out[i:i + len(bp), ci] = sc.cpu()
    return out


def first_token_acc(model, tok, prompts, targets, batch=BATCH):
    """unrestricted: does argmax next-token match gold's first token?"""
    gold0 = [tok(t, add_special_tokens=False)["input_ids"][0] for t in targets]
    hits, n = 0, 0
    for i in range(0, len(prompts), batch):
        bp = prompts[i:i + batch]
        enc = tok(bp, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
        with torch.no_grad():
            nl = model(**enc).logits[:, -1, :]
        pred = nl.argmax(-1).cpu()
        for k in range(len(bp)):
            hits += int(pred[k] == gold0[i + k]); n += 1
    return hits / max(n, 1)


def span_rank_accuracy(model, tok, recs, side):
    """side: 'clean' or 'corrupted'. Candidates = the record's two targets."""
    prompts = [r[f"{side}_prompt"] for r in recs]
    cands = [[r["clean_target"], r["corrupted_target"]] for r in recs]
    gold = [0 if side == "clean" else 1] * len(recs)
    correct, n = 0, 0
    for i in range(0, len(recs), BATCH):
        br = recs[i:i + BATCH]
        uniq = sorted({t for r in br for t in (r["clean_target"], r["corrupted_target"])})
        idx = {t: k for k, t in enumerate(uniq)}
        sc = spans_score(model, tok, [r[f"{side}_prompt"] for r in br], uniq)
        pred = sc.argmax(-1)
        for k, r in enumerate(br):
            correct += int(pred[k].item() == idx[r[f"{side}_target"]])
            n += 1
    return correct / max(n, 1)


def split_sentences(prompt):
    """-> (body_sentences, query_fragment). Prompts end with the query frag."""
    parts = prompt.split(". ")
    return parts[:-1], parts[-1]


def zero_chain(prompt):
    body, q = split_sentences(prompt)
    kept = [s for s in body if not CHAIN_RE.search(s)]
    return ". ".join(kept + [q])


def garble_chain(prompt):
    body, q = split_sentences(prompt)
    out = []
    for s in body:
        if CHAIN_RE.search(s):
            w = s.split(); rng.shuffle(w); out.append(" ".join(w))
        else:
            out.append(s)
    return ". ".join(out + [q])


def middle_container(prompt):
    """2-hop families: container named in the FIRST chain sentence = new truth
    when the two chain sentences are swapped."""
    body, _ = split_sentences(prompt)
    chain = [s for s in body if CHAIN_RE.search(s)]
    if len(chain) < 2:
        return None, None
    m = re.search(r"moved to the ([^\.]+)$", chain[0])
    if m:
        return m.group(1).strip(), "swap2"
    m = re.search(r"moved from the [^\.]+? to the ([^\.]+)$", chain[0])
    if m:
        return m.group(1).strip(), "swap2"
    return None, None


def swap_chain_2hop(prompt):
    body, q = split_sentences(prompt)
    chain = [s for s in body if CHAIN_RE.search(s)]
    if len(chain) != 2:
        return None
    out, ci = [], 0
    for s in body:
        if CHAIN_RE.search(s):
            out.append(chain[1 - ci]); ci += 1
        else:
            out.append(s)
    return ". ".join(out + [q])


def people_seq(prompt):
    """d1: people sequence from the handoff chain."""
    body, _ = split_sentences(prompt)
    ppl = []
    for s in body:
        m = re.search(r"is given to the ([^\.]+)$", s)
        if m:
            ppl.append(m.group(1).strip())
        m = re.search(r"hands it to the ([^\.]+)$", s)
        if m:
            ppl.append(m.group(1).strip())
    return ppl


def swap_last_two_people(prompt):
    """d1 sensitivity: rebuild the handoff chain with the last two people
    swapped -> chain stays valid, new truth = old second-to-last person."""
    body, q = split_sentences(prompt)
    ppl = people_seq(prompt)
    if len(ppl) < 2:
        return None, None
    m = re.search(r"^The ([^\.]+?) is given to the ", [s for s in body if CHAIN_RE.search(s)][0])
    if not m:
        return None, None
    cont = m.group(1)
    new_ppl = ppl[:-2] + [ppl[-1], ppl[-2]]
    chain = [f"The {cont} is given to the {new_ppl[0]}"]
    for a, b in zip(new_ppl, new_ppl[1:]):
        chain.append(f"The {a} hands it to the {b}")
    init = [s for s in body if not CHAIN_RE.search(s)]
    return ". ".join(init + chain + [q]), " " + ppl[-2]


def sensitivity_variant(rec):
    """-> (new_prompt, new_truth_str) or (None, None)."""
    p = rec["clean_prompt"]
    hops = rec.get("hops", 2)
    if hops and hops > 2:
        return swap_last_two_people(p)
    mid, kind = middle_container(p)
    if mid is None:
        return None, None
    newp = swap_chain_2hop(p)
    if newp is None or newp == p:
        return None, None
    # match the formatting of the original clean target (quoted or not)
    ct = rec["clean_target"]
    if ct.startswith(' "'):
        truth = f' "{mid}"'
    else:
        truth = " " + mid
    return newp, truth


def _holder_of(prompt, itemA, gold):
    """True if some init (non-chain) sentence mentions itemA together with the
    gold container word (i.e. gold is itemA's placement per the init)."""
    body, _ = split_sentences(prompt)
    g = gold.strip().strip('"')
    for s in body:
        if CHAIN_RE.search(s):
            continue
        if re.search(rf"\b{re.escape(itemA)}\b", s) and re.search(rf"\b{re.escape(g)}\b", s):
            return True
    return False


def structural_check(recs):
    """corrupt: gold = itemA's init container AND chain never mentions it (0-hop);
    clean: itemA's init container appears in the chain (genuine multi-hop)."""
    same_target = corrupt_moved = corrupt_not_init = clean_not_chained = 0
    for r in recs:
        itemA = r["entities"][0]
        if r["clean_target"] == r["corrupted_target"]:
            same_target += 1
        cb, _ = split_sentences(r["corrupted_prompt"])
        chain_txt = " ".join(s for s in cb if CHAIN_RE.search(s))
        cg = r["corrupted_target"].strip().strip('"')
        if not _holder_of(r["corrupted_prompt"], itemA, r["corrupted_target"]):
            corrupt_not_init += 1
        if re.search(rf"\b{re.escape(cg)}\b", chain_txt):
            corrupt_moved += 1
        clb, _ = split_sentences(r["clean_prompt"])
        cl_chain = " ".join(s for s in clb if CHAIN_RE.search(s))
        ok = False
        for s in clb:
            if CHAIN_RE.search(s) or not re.search(rf"\b{re.escape(itemA)}\b", s):
                continue
            for w in re.findall(r"the ([a-z_\.]+)", s):
                if w != itemA and re.search(rf"\b{re.escape(w)}\b", cl_chain):
                    ok = True
        if not ok:
            clean_not_chained += 1
    n = len(recs)
    return {"same_target": same_target / n,
            "corrupt_gold_not_in_init": corrupt_not_init / n,
            "corrupt_container_moved_by_chain": corrupt_moved / n,
            "clean_container_not_in_chain": clean_not_chained / n}


def no_leak_check(recs):
    """no sentence mentions the query item together with the gold answer."""
    bad = 0
    for r in recs:
        itemA = r["entities"][0]
        for side in ("clean", "corrupted"):
            gold = r[f"{side}_target"].strip().strip('"')
            body, _ = split_sentences(r[f"{side}_prompt"])
            for s in body:
                if not CHAIN_RE.search(s):
                    continue  # init stating the placement is the intended 0-hop path
                if re.search(rf"\b{re.escape(itemA)}\b", s) and re.search(rf"\b{re.escape(gold)}\b", s):
                    bad += 1
    return bad / (2 * len(recs))


def minimality(recs, tok):
    ratios = []
    for r in recs:
        a = tok(r["clean_prompt"], add_special_tokens=False)["input_ids"]
        b = tok(r["corrupted_prompt"], add_special_tokens=False)["input_ids"]
        ratios.append(difflib.SequenceMatcher(None, a, b).ratio())
    return sum(ratios) / len(ratios), min(ratios)


def probe_acc(model, tok, prompts, targets):
    """span-rank accuracy of a single gold target vs the record's other target."""
    correct, n = 0, 0
    for i in range(0, len(prompts), BATCH):
        bp = prompts[i:i + BATCH]
        bt = targets[i:i + BATCH]
        uniq = sorted(set(bt))
        idx = {t: k for k, t in enumerate(uniq)}
        sc = spans_score(model, tok, bp, uniq)
        pred = sc.argmax(-1)
        for k in range(len(bp)):
            correct += int(pred[k].item() == idx[bt[k]]); n += 1
    return correct / max(n, 1)


def main():
    print(f"loading {MODEL} ...", flush=True)
    model, tok = load_model(MODEL)
    print("model loaded", flush=True)
    report = {}
    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        recs = load_records(path)
        print(f"\n=== {fname} ({len(recs)} records) ===", flush=True)
        ent = Counter(tuple(r["entities"]) for r in recs)
        hops = Counter(r.get("hops") for r in recs)
        print(f"hops: {dict(hops)}  unique entity pairs: {len(ent)}", flush=True)

        clean_acc = span_rank_accuracy(model, tok, recs, "clean")
        print(f"clean span-rank acc:   {clean_acc:.3f}", flush=True)
        corr_acc = span_rank_accuracy(model, tok, recs, "corrupted")
        print(f"corrupt span-rank acc: {corr_acc:.3f}", flush=True)
        clean_ft = first_token_acc(model, tok, [r["clean_prompt"] for r in recs],
                                   [r["clean_target"] for r in recs])
        corr_ft = first_token_acc(model, tok, [r["corrupted_prompt"] for r in recs],
                                  [r["corrupted_target"] for r in recs])
        print(f"first-token acc: clean {clean_ft:.3f} / corrupt {corr_ft:.3f}", flush=True)

        mean_sim, min_sim = minimality(recs, tok)
        print(f"minimality: mean {mean_sim:.3f} min {min_sim:.3f}", flush=True)
        struct = structural_check(recs)
        print(f"structural: {struct}", flush=True)
        leak = no_leak_check(recs)
        print(f"no-leak violations: {leak:.4f}", flush=True)

        bal = Counter(r["clean_target"] for r in recs)
        balc = Counter(r["corrupted_target"] for r in recs)
        print(f"balance: {len(bal)} unique clean targets (top {bal.most_common(3)}), "
              f"{len(balc)} corrupt", flush=True)

        # probes on a sample
        sample = recs[:PROBE_N]
        cp = [r["corrupted_prompt"] for r in sample]
        ct = [r["corrupted_target"] for r in sample]
        zc = [zero_chain(p) for p in cp]
        zc_acc = probe_acc(model, tok, zc, ct)
        print(f"probe zero-chain (corrupt):   {zc_acc:.3f}", flush=True)
        gc = [garble_chain(p) for p in cp]
        gc_acc = probe_acc(model, tok, gc, ct)
        print(f"probe garble-chain (corrupt): {gc_acc:.3f}", flush=True)

        # clean sensitivity
        newp, newt, keep = [], [], []
        for r in sample:
            p2, t2 = sensitivity_variant(r)
            if p2 is not None:
                newp.append(p2); newt.append(t2); keep.append(r)
        if newp:
            sens_acc = probe_acc(model, tok, newp, newt)
            # flip rate: prediction on perturbed != prediction on original clean
            orig_pred_ok = None
            print(f"clean sensitivity: acc-vs-new-truth {sens_acc:.3f} (n={len(newp)})", flush=True)
        else:
            sens_acc = None
            print("clean sensitivity: could not build variants", flush=True)

        report[fname] = {
            "n": len(recs), "hops": dict(hops),
            "clean_span_acc": clean_acc, "corrupt_span_acc": corr_acc,
            "clean_first_tok": clean_ft, "corrupt_first_tok": corr_ft,
            "minimality_mean": mean_sim, "minimality_min": min_sim,
            "structural": struct, "no_leak_viol": leak,
            "unique_clean_targets": len(bal), "unique_corrupt_targets": len(balc),
            "probe_zero_chain_corrupt": zc_acc, "probe_garble_chain_corrupt": gc_acc,
            "clean_sensitivity_acc": sens_acc,
        }
        with open("old_datasets_report.json", "w") as f:
            json.dump(report, f, indent=2)
    print("\nsaved old_datasets_report.json", flush=True)


if __name__ == "__main__":
    main()
