"""Train circuit-pruning gates on one section-division file.

Dual-stream L0 pruning (models/llama_circuit.py) applied to the elevator
dataset. Each line of --data is a clean/corrupt pair targeting ONE token of
one section; everything before the target is context. Losses follow
circuit_pruning-argo/ioi_llama.py exactly:

  loss = kl_weight(1.5) * KL(full || pruned) summed over the answer span
       + task_weight(1.0) * margin task loss relu(4 - (logit_good - logit_bad))
         with good = clean target token, bad = corrupt target token
       + lambda_multiplier(1.0) * sparsity
         (LlamaPruningConfig lambdas: heads 0.8, mlp_hidden/output 1.0,
          attn_neurons 0.15, blocks 0.5, no full layers, no depth penalty)

The full model stays resident and its logits are computed per step
(identical math to ioi_llama's logit caching, without the GPU-memory cost).

Mask checkpoints (--out, and per-epoch with --save-every) store BOTH the raw
log_alpha values and the binarized 0/1 mask for every gate.

--init-masks PATH restarts from a checkpoint: gates that are OPEN in the
checkpoint continue from their saved log_alpha; gates that are CLOSED are
locked hard:
  - log_alpha is set to -1e6, so the gate output is exactly 0.0 for ANY
    noise draw (sigmoid((logit(u) - 1e6) / beta) == 0 in fp32) — the hard
    concrete's sampling randomness cannot reopen it, and
  - a gradient hook multiplies their grads by 0, so they cannot drift back.

Example:
  ../venv/bin/python train_circuit.py \
      --data datasets/divisions/train_*.jsonl \
      --val-data datasets/divisions/test_*.jsonl \
      --model meta-llama/Llama-3.2-3B-Instruct --epochs 50 \
      --out masks/full_circuit.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from utils import disable_dropout

ROOT = Path(__file__).resolve().parent
LOCK_VALUE = -1e6  # gate output is exactly 0 for any noise draw at this value

# Same context the dataset was generated under (see datgen_multi/gen_elevator_dataset.py)
SYSTEM = ('You track elevator floors and reply in this EXACT format: first '
          'line "Start: <floor>", then one line per move "<up|down> -> '
          '<floor>", then a last line "FINAL ANSWER: <floor>". No other '
          'text, no numbering, no explanations.')
DEMO_USER = ('Jill is on the 6th floor in an elevator. Jill goes one floor '
             'down, then one floor up, then one floor down, then one floor '
             'up, then one floor up. Jill is now on')
DEMO_ASSISTANT = ('Start: 6th\ndown -> 5th\nup -> 6th\ndown -> 5th\nup -> 6th\n'
                  'up -> 7th\nFINAL ANSWER: 7th')


def chat_text(tok, task_prompt):
    msgs = ([{"role": "system", "content": SYSTEM}]
            + [{"role": "user", "content": DEMO_USER},
               {"role": "assistant", "content": DEMO_ASSISTANT}]
            + [{"role": "user", "content": task_prompt}])
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


def load_examples(tok, paths, max_examples=0):
    """Encode every pair side; returns list of dicts with token ids."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out, skipped = [], 0
    for path in paths:
        for line in open(path):
            ex = json.loads(line)
            row = {"pair_id": ex["pair_id"], "section": ex["section"],
                   "part": ex["part"], "token_index": ex["token_index"],
                   "source": Path(path).name}
            ok = True
            for typ in ("clean", "corrupt"):
                s = ex[typ]
                chat_ids = tok(chat_text(tok, s["task_prompt"]),
                               add_special_tokens=True)["input_ids"]
                pre = tok(s["answer_prefix"], add_special_tokens=False)["input_ids"]
                full = tok(s["answer_prefix"] + s["target"],
                           add_special_tokens=False)["input_ids"]
                # target must be exactly one token and not merge across the cut
                if len(full) != len(pre) + 1 or full[:-1] != pre:
                    ok = False
                    break
                row[f"{typ}_ids"] = chat_ids + pre
                row[f"{typ}_target"] = full[-1]
            if ok:
                row["target_id"] = row["clean_target"]  # train/eval on clean stream
            if ok:
                out.append(row)
            else:
                skipped += 1
    if max_examples:
        out = out[:max_examples]
    if skipped:
        print(f"WARNING: skipped {skipped} examples (token boundary merge)")
    return out


def collate(batch, pad_id):
    maxlen = max(max(len(r["clean_ids"]), len(r["corrupt_ids"])) for r in batch)
    n = len(batch)
    clean = torch.full((n, maxlen), pad_id, dtype=torch.long)
    corrupt = torch.full((n, maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((n, maxlen), dtype=torch.long)
    pos = torch.zeros(n, dtype=torch.long)
    target = torch.zeros(n, dtype=torch.long)
    corrupt_target = torch.zeros(n, dtype=torch.long)
    for i, r in enumerate(batch):
        c, x = r["clean_ids"], r["corrupt_ids"]
        clean[i, :len(c)] = torch.tensor(c)
        corrupt[i, :len(x)] = torch.tensor(x)
        mask[i, :len(c)] = 1
        pos[i] = len(c) - 1  # logits here predict the target token
        target[i] = r["target_id"]
        corrupt_target[i] = r["corrupt_target"]  # distractor for margin loss
    return {"clean_ids": clean, "corrupt_ids": corrupt, "mask": mask,
            "pos": pos, "target": target,
            "corrupt_target": corrupt_target, "_rows": batch}


def gate_modules(model):
    from models.l0 import HardConcreteGate
    return {name + ".log_alpha": m
            for name, m in model.named_modules()
            if isinstance(m, HardConcreteGate)}


def save_masks(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()  # gates return deterministic hard 0/1
    with torch.no_grad():
        masks, log_alphas = {}, {}
        for pname, gate in gate_modules(model).items():
            masks[pname] = gate().detach().cpu()
            log_alphas[pname] = gate.log_alpha.detach().cpu()
    if was_training:
        model.train()
    torch.save({"masks": masks, "log_alpha": log_alphas, "meta": meta}, path)
    n_on = sum(int(m.sum()) for m in masks.values())
    n_all = sum(m.numel() for m in masks.values())
    print(f"saved {path} | gates open: {n_on}/{n_all} ({100*n_on/n_all:.1f}%)")


def load_and_lock_masks(model, path):
    """Start from a checkpoint; closed gates are locked open-proof (see header).
    Gates NOT present in the checkpoint (e.g. fine-grained neuron gates when
    resuming from a coarse heads+blocks mask) are left at their fresh init:
    open and trainable."""
    ckpt = torch.load(path, map_location="cpu")
    gates = gate_modules(model)
    n_locked = n_new = 0
    for pname, gate in gates.items():
        if pname not in ckpt["masks"]:
            n_new += 1
            continue
        m = ckpt["masks"][pname].to(gate.log_alpha.device).float()
        saved = ckpt["log_alpha"][pname].to(gate.log_alpha.device).float()
        with torch.no_grad():
            gate.log_alpha.copy_(
                torch.where(m > 0.5, saved,
                            torch.full_like(gate.log_alpha, LOCK_VALUE)))
        # closed positions get zero grad -> cannot drift open
        gate.log_alpha.register_hook(lambda g, m=m: g * m)
        n_locked += int((m <= 0.5).sum())
    print(f"loaded {path} | locked {n_locked} closed gates "
          f"(log_alpha={LOCK_VALUE}, grad=0, reopen-proof)"
          + (f" | {n_new} gate groups not in checkpoint (fresh, open, "
             f"trainable)" if n_new else ""))


def resume_masks(model, path):
    """Plain resume: restore every gate's log_alpha from the checkpoint and
    keep ALL gates trainable (no locking — closed gates can reopen)."""
    ckpt = torch.load(path, map_location="cpu")
    gates = gate_modules(model)
    for pname, gate in gates.items():
        saved = ckpt["log_alpha"][pname].to(gate.log_alpha.device).float()
        with torch.no_grad():
            gate.log_alpha.copy_(saved)
    n_closed = sum(int((ckpt["masks"][p] <= 0.5).sum()) for p in gates)
    print(f"resumed log_alpha from {path} | {n_closed} gates currently closed "
          f"(still trainable, no locking)")


def lock_orphaned_neuron_gates(model, ckpt):
    """Neuron gates sitting under components that are CLOSED in the checkpoint
    start locked closed too — the parent already swaps to corrupted, so they
    were invisible to the network from the start (they would only add noise
    and get harvested for free sparsity without any signal)."""
    masks = ckpt["masks"]
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    gates = gate_modules(model)
    n_locked = 0
    for i in range(len(model.model.layers)):
        pre = f"model.layers.{i}."
        hm = masks.get(pre + "attn.head_gates.log_alpha")
        nkey = pre + "attn.neuron_gates.log_alpha"
        if hm is not None and nkey in gates and nkey not in masks:
            g = gates[nkey]
            closed = (hm <= 0.5).repeat_interleave(head_dim).to(
                g.log_alpha.device)
            if closed.any():
                with torch.no_grad():
                    g.log_alpha[closed] = LOCK_VALUE
                keep = (~closed).float()
                g.log_alpha.register_hook(lambda gr, k=keep: gr * k)
                n_locked += int(closed.sum())
        bm = masks.get(pre + "mlp_block_gate.log_alpha")
        if bm is not None and float(bm.flatten()[0]) <= 0.5:
            for sub in ("hidden_gates", "output_gates"):
                key = pre + f"mlp.{sub}.log_alpha"
                if key in gates and key not in masks:
                    g = gates[key]
                    with torch.no_grad():
                        g.log_alpha.fill_(LOCK_VALUE)
                    g.log_alpha.register_hook(lambda gr: gr * 0)
                    n_locked += g.log_alpha.numel()
    if n_locked:
        print(f"locked {n_locked} neuron gates under already-closed parents "
              f"(invisible to the network)")


def evaluate(model, loader, device):
    """Dual-stream eval: closed gates swap in corrupted values, so accuracy
    actually reflects the pruned circuit. (Single-stream forward bypasses
    gates entirely — never use it for validation.)"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["clean_ids"].to(device),
                        corrupted_input_ids=batch["corrupt_ids"].to(device),
                        attention_mask=batch["mask"].to(device),
                        use_cache=False)
            idx = torch.arange(len(batch["pos"]), device=device)
            pred = out.logits[idx, batch["pos"].to(device)].argmax(-1)
            correct += (pred == batch["target"].to(device)).sum().item()
            total += len(batch["pos"])
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+",
                    help="one or more division jsonl files (mixed together)")
    ap.add_argument("--val-data", default=None, nargs="+")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--kl-weight", type=float, default=1.5)
    ap.add_argument("--task-weight", type=float, default=1.0)
    ap.add_argument("--lambda-multiplier", type=float, default=1.0)
    ap.add_argument("--sparsity-warmup", type=int, default=1000,
                    help="steps to ramp sparsity pressure in (default 1000; "
                         "lower this for small per-section datasets)")
    ap.add_argument("--heads-only", action="store_true",
                    help="prune ONLY attention head gates; all other gate "
                         "types (MLP hidden/output, attention neurons, "
                         "attention/MLP blocks) are disabled")
    ap.add_argument("--heads-mlp-blocks", action="store_true",
                    help="prune ONLY attention head gates + MLP block gates "
                         "(coarse-grained pass; finer grains can follow on "
                         "the surviving components)")
    ap.add_argument("--fine", action="store_true",
                    help="fine-grained mode: attention heads + neurons, MLP "
                         "hidden/output neurons + MLP blocks active; "
                         "attention blocks and full layers disabled")
    ap.add_argument("--init-masks", default=None,
                    help="checkpoint .pt to start from; closed gates get locked")
    ap.add_argument("--resume", default=None,
                    help="checkpoint .pt to resume from: restores log_alpha "
                         "for all gates, nothing locked")
    ap.add_argument("--out", default=None, help="mask output .pt (default: masks/<data-stem>.pt)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save masks every N epochs (0 = only at end)")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    out_path = args.out or f"masks/{Path(args.data[0]).stem}.pt"

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token

    print(f"loading {args.model} (prunable + frozen full) ...", flush=True)
    if args.heads_only:
        sel = dict(prune_mlp_hidden=False, prune_mlp_output=False,
                   prune_attention_neurons=False,
                   prune_attention_blocks=False, prune_mlp_blocks=False)
        print("heads-only mode: only attention head gates are active")
    elif args.heads_mlp_blocks:
        sel = dict(prune_mlp_hidden=False, prune_mlp_output=False,
                   prune_attention_neurons=False,
                   prune_attention_blocks=False)
        print("heads+mlp-blocks mode: attention head gates and MLP block "
              "gates are active")
    elif args.fine:
        sel = dict(prune_attention_blocks=False)
        print("fine mode: heads, attn neurons, MLP neurons + MLP blocks "
              "active (no attn blocks, no full layers)")
    else:
        sel = {}
    # equal sparsity pressure on every gate type (lambdas all 1.0),
    # no depth penalty, no full layers — rest follows ioi_llama.py
    pruning_config = PruningConfig(
        init_value=0.5,
        depth_penalty_scaling=0.0,
        lambda_attention_heads=1.0,
        lambda_mlp_hidden=1.0,
        lambda_mlp_output=1.0,
        lambda_attention_neurons=1.0,
        lambda_attention_blocks=1.0,
        lambda_mlp_blocks=1.0,
        prune_full_layers=False,
        **sel)
    pruning_config.sparsity_warmup_steps = args.sparsity_warmup
    # note: load on CPU then .to(device) — gates are created inside
    # from_pretrained_with_pruning and device_map would leave them stranded
    circuit_model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model, pruning_config, torch_dtype=torch.bfloat16).to(device)
    full_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    full_model.eval()
    for p in full_model.parameters():
        p.requires_grad = False

    disable_dropout(circuit_model)
    for name, p in circuit_model.named_parameters():
        is_gate = name.endswith(".log_alpha")
        p.requires_grad = is_gate
        if is_gate:
            p.data = p.data.float()

    if args.init_masks:
        load_and_lock_masks(circuit_model, args.init_masks)
        lock_orphaned_neuron_gates(
            circuit_model, torch.load(args.init_masks, map_location="cpu"))
    if args.resume:
        resume_masks(circuit_model, args.resume)

    examples = load_examples(tok, args.data, args.max_examples)
    from collections import Counter
    per_file = Counter(r["source"] for r in examples)
    print(f"{len(examples)} example pairs total from {len(args.data)} file(s):")
    for name, cnt in sorted(per_file.items()):
        print(f"  {name}: {cnt}")
    pad_id = tok.pad_token_id
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, pad_id))
    val_loader = None
    if args.val_data:
        val_examples = load_examples(tok, args.val_data)
        val_loader = DataLoader(val_examples, batch_size=args.batch_size,
                                shuffle=False,
                                collate_fn=lambda b: collate(b, pad_id))

    # ioi_llama-style: full model stays resident; KL is computed per step
    # over the whole answer span against on-the-fly full-model logits
    optimizer = torch.optim.AdamW(
        [p for p in circuit_model.parameters() if p.requires_grad],
        lr=args.lr)

    print(f"\ntraining: {args.epochs} epochs, lr={args.lr}, "
          f"kl={args.kl_weight} (span KL), task={args.task_weight} "
          f"(margin), lambda_mult={args.lambda_multiplier}")
    total_steps = 0
    circuit_model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        sums = {"kl": 0.0, "task": 0.0, "sparsity": 0.0, "acc": 0.0}
        nb = 0
        for batch in loader:
            optimizer.zero_grad()
            out = circuit_model(
                input_ids=batch["clean_ids"].to(device),
                corrupted_input_ids=batch["corrupt_ids"].to(device),
                attention_mask=batch["mask"].to(device),
                use_cache=False)
            with torch.no_grad():
                out_f = full_model(input_ids=batch["clean_ids"].to(device),
                                   attention_mask=batch["mask"].to(device),
                                   use_cache=False)

            # KL(full || pruned) at the target position (ioi_llama's span is
            # the answer-token positions; ours is the single target token) —
            # slice the position FIRST, then softmax, so the full-sequence
            # logits are never materialized in float
            idx = torch.arange(len(batch["pos"]), device=device)
            pos = batch["pos"].to(device)
            lp = F.log_softmax(out.logits[idx, pos].float(), dim=-1)
            lf = F.log_softmax(out_f.logits[idx, pos].float(), dim=-1)
            kl = F.kl_div(lp, lf, log_target=True,
                          reduction="none").sum(-1).mean()
            del lp, lf

            # margin task loss (ioi_llama): good = clean target token,
            # bad = corrupt target token at the target position
            pruned_logits = out.logits[idx, pos].float()
            tgt = batch["target"].to(device)
            bad = batch["corrupt_target"].to(device)
            logit_good = pruned_logits.gather(1, tgt[:, None]).squeeze(1)
            logit_bad = pruned_logits.gather(1, bad[:, None]).squeeze(1)
            task = F.relu(4.0 - (logit_good - logit_bad)).mean()

            sparsity = circuit_model.get_sparsity_loss(
                step=total_steps)["total_sparsity"]

            loss = (args.kl_weight * kl + args.task_weight * task
                    + args.lambda_multiplier * sparsity)
            loss.backward()
            optimizer.step()
            total_steps += 1
            nb += 1
            sums["kl"] += kl.item()
            sums["task"] += task.item()
            sums["sparsity"] += sparsity.item()
            sums["acc"] += (pruned_logits.argmax(-1) == tgt).float().mean().item()

        with torch.no_grad():
            gates = gate_modules(circuit_model)
            n_open = sum(int((g.log_alpha > 0).sum()) for g in gates.values())
            n_all = sum(g.log_alpha.numel() for g in gates.values())
        msg = (f"epoch {epoch + 1}/{args.epochs} "
               f"kl={sums['kl']/nb:.4f} task={sums['task']/nb:.4f} "
               f"sparsity={sums['sparsity']/nb:.4f} "
               f"train_acc={sums['acc']/nb:.4f} "
               f"gates_open={100*n_open/n_all:.1f}% ({time.time()-t0:.0f}s)")
        if val_loader and (epoch + 1) % args.eval_every == 0:
            acc = evaluate(circuit_model, val_loader, device)
            msg += f" | val acc {acc:.4f}"
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_masks(circuit_model,
                       Path(out_path).with_suffix(f".ep{epoch+1}.pt"),
                       {"args": vars(args), "epoch": epoch + 1})

    save_masks(circuit_model, out_path,
               {"args": vars(args), "epochs": args.epochs})


if __name__ == "__main__":
    main()
