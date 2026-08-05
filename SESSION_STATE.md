# SESSION STATE / HANDOFF

Last updated: 2026-07-29 (session on Jetstream2 A100 40GB)
Purpose: complete state snapshot so this work can be resumed from a fresh machine or fresh chat.
Give this file to the new chat/session as context.

---

## 1. Environment

- Machine: Jetstream2, A100 40GB GPU.
- Working dir / repo root: `/home/exouser/pruning`
- Python env: `/home/exouser/pruning/venv` (pinned deps in `requirements.txt`)
- Git: credential helper OFF on purpose — user runs `git push` themselves (browser auth).
- **Nothing running right now.** No screens, no background jobs. All experiments complete.

## 2. Project goal (big picture)

Circuit pruning / mechanistic interpretability: find the minimal circuit responsible for
logical sequence tracking (category word chains, e.g. "Sequence: peach kiwi mango peach
kiwi" -> "mango"). Datasets are category-based; the task is identical across categories,
only the vocabulary differs.

Key method: train binary masks (hard-concrete gates) per category, INTERSECT them to find
the shared category-agnostic core; anti-masks (NOT circuit) test necessity.

## 3. Datasets

- Generator: `induction_datasets/`. 10 categories: fruits, animals, colors, metals,
  vehicles (used for training) + clothing, furniture, instruments, professions, sports
  (heldout).
- Each category has `_1` (train), `_2` (test), `_3` (spare), 500 examples each.
- Copies in each experiment folder's `datasets/`; `categories.json` has the word lists.

## 4. Experiment folders and status

- `exp4_heads_mlp_main/` — **MAIN RESULTS, COMPLETE.** Heads + MLP-block gating,
  normal lambdas (combined λ=0.10, frozen λ=0.05, 600ep λ=0.05).
  Full write-up with every number: `exp4_heads_mlp_main/results/RESULTS_heads_mlp_normal_lambda.md` (sections 1-10).
- `exp2_heads_mlp_overpruned/` — same setup with 1.5x lambdas: OVER-PRUNED, failed.
  Write-up: `exp2_heads_mlp_overpruned/results/RESULTS_heads_mlp_run.md`.
- `exp2full_heads_mlp_neurons/` — older full-granularity run (heads+MLP+neurons).
- `exp3_dual_init_union/` — dual-init (Ns/Dn) union method (gates init open vs closed).
- `exp1_heads_only_original/` — oldest heads-only experiments; `intersection_report.md` at root.

## 5. MAIN FINDINGS (exp_4, all numbers in the results doc)

1. **Attention heads = the circuit.** One category's head gates alone (94/512 heads,
   all MLPs clean) do sequence tracking at 75% avg on all trained categories and 60%
   on 5 never-trained categories. The head circuit is category-general.
2. **Necessity of the shared core**: anti-intersections (corrupt only the 35-47 shared
   heads, keep 465+ heads + most MLPs clean) kill the task (<=6% gen) on ALL 10
   categories, trained and heldout.
3. **Intersection sufficiency**: head-only intersection of the 5 frozen masks keeps
   ~69% of individual-circuit performance; MLP-only intersection keeps ~17%.
4. **MLP gates carry NO category content** (cross-assembly leak test: zero vocabulary
   leakage when swapping fruits/colors MLP gate patterns).
5. **MLP blocks are fungible compute**: WHICH blocks are clean doesn't matter, HOW MANY
   does. Anti-MLP test (own heads + inverted MLP gates, only 3-5/16 clean): 0/10, but
   outputs stay on-category ("strawberry", "palladium") — heads set the category frame
   and rough position; clean MLP mass resolves the exact token.
6. **Two-phase (combined 300ep + frozen 300ep) intersection beats direct 600ep
   intersection ~3x** (0.165 vs 0.058 trained gen mean) — combined pre-training pushes
   more shared structure into the common core, as originally hypothesized.
7. **1.5x lambda (exp_2) over-prunes at heads+MLP granularity**; normal lambda (exp_4)
   trains cleanly. Combined mask generalizes to heldout categories (22-60%).
8. Paper framing: sparse, necessary, category-invariant ATTENTION backbone riding on
   dense, redundant MLP compute. Practical rule: prune heads, never gate MLP blocks
   (OFF gate = corrupted stream, so MLP removal is disproportionately destructive).

## 6. Code notes (do NOT regress)

- `train_mask.py`: `--dataset` (1+ names), `--epochs`, `--output_name`, `--mask`
  (phase-2 init: binary start, off-gates frozen off via pinning + grad hook + re-pinning
  at `pruning_manager.py:161-165`), `--lambda_sparsity` (default 0.05 heads, other levels
  scale proportionally), `--prune_mlp_blocks`.
- `evaluate_mask.py` / `eval_all.py`: auto-enable `prune_mlp_blocks` when mask file has
  `mlp_block_gate` keys; gates missing from a mask file are pinned OFF by `load_masks`.
- `intersect_masks.py`: `--mode intersect|difference`, `--components all|heads|mlp`
  (non-selected groups forced ON).
- Probe scripts in exp_4: `gen_probes.py`, `gen_decomposed.py`, `gen_swap.py`,
  `gen_anti_mlp.py`. `anti_mask.py` at repo root makes NOT-circuit masks.
- Model: meta-llama/Llama-3.2-1B. OFF gates receive the CORRUPTED stream (not bypass).

## 7. User preferences / operating rules

- Training in named `screen` sessions; evals as background bash tasks.
- Keep shell commands short; verify quickly (long foreground commands look stuck).
- Git mutations (commit/push) need explicit user request each time; user pushes.
- Machine may be shelved/unshelved; re-check `nvidia-smi` after restart.
