#!/usr/bin/env python3
"""
train_abc.py — minimal A/B/C circuit training for intersection verification.

PHASE 1 — combined training (force the shared circuit)
    A+B+C mixed in one shuffled stream; HIGH sparsity lambda makes the
    cheapest solution ONE circuit for the shared algorithm + small wrappers.
    Output: combined checkpoint (continuous log_alpha) = parent superset.

PHASE 2 — per-dataset finetune (carve strict subsets)
    Binary restart from the parent (on -> +5, off -> -1e6 LOCKED via grad
    hook + re-pin). Train on ONE dataset: keeps what it needs, sheds the
    other formats' wrappers. Output: mask_A/B/C, strict subsets of parent.

Lambda sweep (find minimal circuits; outputs separated by --out_suffix):
    for lam in 0.10 0.15 0.20; do
      for ds in A B C; do
        python -u train_abc.py --mode finetune --datasets $ds \
          --lambda_finetune $lam --epochs_finetune 50 \
          --out_suffix _lam$lam 2>&1 | tee logs/p2_${ds}_${lam}.log &
      done; wait
    done
"""

import argparse, json, os, random, gc
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM

from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from models.l0 import HardConcreteGate

GATE_PATTERNS = ("_gates.", "_gate.", "layer_gates.", "log_alpha")
ALL_DATASETS = ["A", "B", "C"]


# ---------------------------------------------------------------- data ----
class JsonlDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.rows = [json.loads(l) for l in f]
    def __len__(self):  return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def make_collate(tokenizer):
    def collate(batch):
        tokenizer.padding_side = "left"
        clean = tokenizer([r["clean_prompt"] for r in batch], padding=True,
                          return_tensors="pt", add_special_tokens=True)
        corr  = tokenizer([r["corrupt_prompt"] for r in batch], padding=True,
                          return_tensors="pt", add_special_tokens=True)
        L = max(clean["input_ids"].size(1), corr["input_ids"].size(1))
        def lpad(t, val):
            pad = torch.full((t.size(0), L - t.size(1)), val, dtype=t.dtype)
            return torch.cat([pad, t], dim=1)
        tgt = [tokenizer.encode(r["clean_answer"],   add_special_tokens=False) for r in batch]
        dis = [tokenizer.encode(r["corrupt_answer"], add_special_tokens=False) for r in batch]
        assert all(len(x) == 1 for x in tgt + dis), "answers must be single tokens"
        return {
            "input_ids":           lpad(clean["input_ids"], tokenizer.pad_token_id),
            "attention_mask":      lpad(clean["attention_mask"], 0),
            "corrupted_input_ids": lpad(corr["input_ids"],  tokenizer.pad_token_id),
            "target_ids":          torch.tensor([x[0] for x in tgt]),
            "distractor_ids":      torch.tensor([x[0] for x in dis]),
        }
    return collate


# --------------------------------------------------------------- model ----
def build_config(args, lambda_heads, warmup):
    cfg = PruningConfig()                 # enables everything by default -> trim:
    cfg.prune_attention_heads   = True
    cfg.prune_attention_neurons = False
    cfg.prune_attention_blocks  = False
    cfg.prune_mlp_hidden        = False
    cfg.prune_mlp_output        = False
    cfg.prune_mlp_blocks        = args.prune_mlp_blocks
    cfg.prune_full_layers       = False
    cfg.lambda_attention_heads  = lambda_heads
    cfg.sparsity_warmup_steps   = warmup
    return cfg


def build_model(args, lambda_heads, warmup):
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model_name, pruning_config=build_config(args, lambda_heads, warmup),
        torch_dtype=torch.bfloat16).to(args.device)
    for name, p in model.named_parameters():       # freeze base weights
        p.requires_grad = any(g in name for g in GATE_PATTERNS)
        if p.requires_grad:
            p.data = p.data.float()                # gates in float32
    return model


def gate_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, HardConcreteGate) and hasattr(m, "log_alpha")]


def clamp_gates(model):
    with torch.no_grad():
        for _, m in gate_modules(model):
            if hasattr(m, "frozen_off"):
                m.log_alpha.data[~m.frozen_off].clamp_(-5.0, 5.0)
                m.log_alpha.data[m.frozen_off] = -1e6      # locked gates stay locked
            else:
                m.log_alpha.data.clamp_(-5.0, 5.0)


def prepare_finetune(model, ckpt_path, binary=True):
    """Load combined checkpoint; every OFF gate is locked at -1e6 (pin + grad hook).
    binary=True resets ON gates to +5 (pure binary start, no volatile near-zero gates)."""
    state = torch.load(ckpt_path, weights_only=True)
    n_on = n_frozen = 0
    with torch.no_grad():
        for n, m in gate_modules(model):
            la = state[n].to(m.log_alpha.device).float()
            on = la > 0
            m.log_alpha.data = torch.where(
                on, torch.full_like(la, 5.0), torch.full_like(la, -1e6)) if binary else la.clone()
            m.frozen_off = ~on
            m.log_alpha.register_hook(lambda g, f=m.frozen_off: g.masked_fill(f, 0))
            n_on += on.sum().item(); n_frozen += (~on).sum().item()
    print(f"  finetune start: {n_on} gates open (trainable), {n_frozen} locked off "
          f"<- {ckpt_path}")


def save_gates(model, path, binarize):
    state = {n: (m.log_alpha.data > 0 if binarize else m.log_alpha.data).cpu()
             for n, m in gate_modules(model)}
    torch.save(state, path)


def count_active(model):
    return sum((m.log_alpha.data > 0).sum().item() for _, m in gate_modules(model))


# ------------------------------------------------------------ training ----
def train(model, baseline, dl, args, epochs, tag):
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = 0
    last = {}
    for ep in range(epochs):
        tot = {"kl": 0.0, "hinge": 0.0, "sparse": 0.0}
        for batch in dl:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.no_grad():                                # fresh baseline, no cache bug
                bl = baseline(input_ids=batch["input_ids"],
                              attention_mask=batch["attention_mask"],
                              use_cache=False).logits[:, -1, :].float()
            out = model(input_ids=batch["input_ids"],
                        corrupted_input_ids=batch["corrupted_input_ids"],
                        attention_mask=batch["attention_mask"], use_cache=False)
            gl = out.logits[:, -1, :].float()                    # answer position

            kl = F.kl_div(F.log_softmax(gl, -1), F.log_softmax(bl, -1),
                          reduction="batchmean", log_target=True)
            good = gl.gather(1, batch["target_ids"].unsqueeze(1)).squeeze(1)
            bad  = gl.gather(1, batch["distractor_ids"].unsqueeze(1)).squeeze(1)
            hinge = F.relu(args.margin - (good - bad)).mean()
            sparse = model.get_sparsity_loss(step=step)["total_sparsity"]

            loss = args.kl_weight * kl + hinge + sparse
            loss.backward()
            opt.step()
            clamp_gates(model)
            step += 1
            tot["kl"] += kl.item(); tot["hinge"] += hinge.item(); tot["sparse"] += sparse.item()
        n = len(dl)
        last = {k: tot[k] / n for k in tot}
        print(f"[{tag}] epoch {ep+1}/{epochs} | KL {last['kl']:.4f} | "
              f"hinge {last['hinge']:.4f} | sparse {last['sparse']:.4f} | "
              f"active {count_active(model)}", flush=True)
    return last


# ------------------------------------------------------------ evaluate ----
@torch.no_grad()
def pairwise_acc(model, dl, args, gated):
    model.eval()
    c_ok = d_ok = n = 0
    for batch in dl:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        t, d = batch["target_ids"], batch["distractor_ids"]
        if gated:   # corrupt-side eval swaps the streams
            clean = model(input_ids=batch["input_ids"],
                          corrupted_input_ids=batch["corrupted_input_ids"],
                          attention_mask=batch["attention_mask"], use_cache=False).logits[:, -1, :].float()
            corr  = model(input_ids=batch["corrupted_input_ids"],
                          corrupted_input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"], use_cache=False).logits[:, -1, :].float()
        else:
            clean = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"], use_cache=False).logits[:, -1, :].float()
            corr  = model(input_ids=batch["corrupted_input_ids"],
                          attention_mask=batch["attention_mask"], use_cache=False).logits[:, -1, :].float()
        c_ok += (clean.gather(1, t.unsqueeze(1)) > clean.gather(1, d.unsqueeze(1))).sum().item()
        d_ok += (corr.gather(1, d.unsqueeze(1))  > corr.gather(1, t.unsqueeze(1))).sum().item()
        n += t.size(0)
    return c_ok / n, d_ok / n


def report(model, baseline, val_loaders, args, tag):
    print(f"\n=== {tag} === (active heads: {count_active(model)})")
    print(f"{'dataset':8s} {'base clean':>10s} {'base corr':>10s} {'circ clean':>10s} {'circ corr':>10s}")
    model.set_final_circuit_mode(True)
    res = {}
    for name, dl in val_loaders.items():
        bc, bd = pairwise_acc(baseline, dl, args, gated=False)
        cc, cd = pairwise_acc(model, dl, args, gated=True)
        res[name] = dict(base_clean=bc, base_corr=bd, circ_clean=cc, circ_corr=cd)
        print(f"{name:8s} {bc:10.3f} {bd:10.3f} {cc:10.3f} {cd:10.3f}")
    model.set_final_circuit_mode(False)
    return res


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--data_dir", default="datasets")
    ap.add_argument("--out_dir", default="masks_abc")
    ap.add_argument("--out_suffix", default="")      # e.g. _lam0.15 -> per-run outputs
    ap.add_argument("--ckpt_in", default=None)       # phase-2 start; default combined_checkpoint.pt
    ap.add_argument("--mode", choices=["all", "combined", "finetune"], default="all")
    ap.add_argument("--datasets", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs_combined", type=int, default=40)
    ap.add_argument("--epochs_finetune", type=int, default=30)
    ap.add_argument("--lambda_combined", type=float, default=0.10)   # high -> force common circuit
    ap.add_argument("--lambda_finetune", type=float, default=0.05)   # low  -> carve within parent
    ap.add_argument("--warmup_combined", type=int, default=1000)
    ap.add_argument("--warmup_finetune", type=int, default=200)
    ap.add_argument("--margin", type=float, default=4.0)
    ap.add_argument("--kl_weight", type=float, default=1.0)
    ap.add_argument("--prune_mlp_blocks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
    sfx = args.out_suffix
    print(f"device: {args.device} | suffix: '{sfx}'")
    torch.manual_seed(args.seed); random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = make_collate(tok)

    # train rows only for requested datasets; val for ALL (cross-eval in every report)
    train_rows = {d: JsonlDataset(os.path.join(args.data_dir, f"{d}_train.jsonl")).rows
                  for d in args.datasets}
    val_loaders = {d: DataLoader(JsonlDataset(os.path.join(args.data_dir, f"{d}_val.jsonl")),
                                 batch_size=args.batch_size, shuffle=False, collate_fn=collate)
                   for d in ALL_DATASETS if os.path.exists(os.path.join(args.data_dir, f"{d}_val.jsonl"))}

    baseline = LlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(args.device)
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    # ---------------- phase 1 ----------------
    if args.mode in ("all", "combined"):
        combined = [r for d in args.datasets for r in train_rows[d]]
        random.shuffle(combined)
        print(f"\n########## PHASE 1: combined ({len(combined)} rows) ##########")
        model = build_model(args, args.lambda_combined, args.warmup_combined)
        dl = DataLoader(combined, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        final = train(model, baseline, dl, args, args.epochs_combined, f"combined{sfx}")
        save_gates(model, os.path.join(args.out_dir, f"combined{sfx}_checkpoint.pt"), binarize=False)
        save_gates(model, os.path.join(args.out_dir, f"combined{sfx}_mask.pt"), binarize=True)
        res = report(model, baseline, val_loaders, args, f"combined{sfx} mask on val")
        with open(os.path.join(args.out_dir, f"combined{sfx}_summary.json"), "w") as f:
            json.dump(dict(dataset="combined", suffix=sfx, lambda_combined=args.lambda_combined,
                           epochs=args.epochs_combined, active=count_active(model),
                           train_kl=final["kl"], train_hinge=final["hinge"], val=res), f, indent=2)
        del model; gc.collect(); torch.cuda.empty_cache() if args.device == "cuda" else None

    # ---------------- phase 2 ----------------
    if args.mode in ("all", "finetune"):
        ckpt = args.ckpt_in or os.path.join(args.out_dir, "combined_checkpoint.pt")
        assert os.path.exists(ckpt), f"need {ckpt} (run --mode combined first)"
        for d in args.datasets:
            print(f"\n########## PHASE 2: finetune on {d} (lambda={args.lambda_finetune}) ##########")
            model = build_model(args, args.lambda_finetune, args.warmup_finetune)
            prepare_finetune(model, ckpt, binary=True)
            dl = DataLoader(train_rows[d], batch_size=args.batch_size, shuffle=True, collate_fn=collate)
            final = train(model, baseline, dl, args, args.epochs_finetune, f"{d}{sfx}")
            save_gates(model, os.path.join(args.out_dir, f"{d}{sfx}_checkpoint.pt"), binarize=False)
            save_gates(model, os.path.join(args.out_dir, f"{d}{sfx}_mask.pt"), binarize=True)
            res = report(model, baseline, val_loaders, args, f"{d}{sfx} mask on val")
            with open(os.path.join(args.out_dir, f"{d}{sfx}_summary.json"), "w") as f:
                json.dump(dict(dataset=d, suffix=sfx, lambda_finetune=args.lambda_finetune,
                               epochs=args.epochs_finetune, active=count_active(model),
                               train_kl=final["kl"], train_hinge=final["hinge"], val=res), f, indent=2)
            del model; gc.collect(); torch.cuda.empty_cache() if args.device == "cuda" else None

    print(f"\nDone. Gates in {args.out_dir}/  (suffix '{sfx}')")


if __name__ == "__main__":
    main()