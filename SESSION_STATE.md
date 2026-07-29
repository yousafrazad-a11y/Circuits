# SESSION STATE / HANDOFF

Last updated: 2026-07-28 (session on Jetstream2 A100 40GB)
Purpose: complete state snapshot so this work can be resumed from a fresh machine or fresh chat.
Give this file to the new chat/session as context.

---

## 1. Environment

- Machine: Jetstream2, A100 40GB GPU.
- Working dir / repo root: `/home/exouser/pruning`
- Python env: `/home/exouser/pruning/venv` (pinned deps in `requirements.txt`)
- Git: credential helper OFF on purpose — user runs `git push` themselves (browser auth).
  Repo history was gc'd (1.82 GiB -> 43 MiB). Latest backup commit: `a152a5a`.

## 2. Project goal (big picture)

Circuit pruning / mechanistic interpretability: find the minimal circuit responsible for
logical sequence tracking (3-item chains, e.g. "apple is in the box, the box is on the
table, ... where is the apple?"). Datasets are category-based word chains.

Key hypothesis under test: a circuit found by **intersecting** per-category masks is the
category-agnostic "sequence tracking" core; the disjoint remainder is category-specific.
Compared two pipelines:
- Baseline: 5 masks trained 600 epochs individually per dataset -> intersect.
- Two-phase: 1 mask trained 300 epochs on all 5 datasets combined -> 5 frozen fine-tunes
  (300 ep each, gates OFF in phase-1 mask stay OFF) -> intersect.

## 3. Datasets

- Generator code: `induction_datasets/` (10 categories now: original 5 = fruits, animals,
  colors, metals, vehicles + 5 newer ones).
- Each category has `_1`, `_2`, `_3` versions, 500 examples each, no overlap between versions.
- Convention: **`_1` = training, `_2` = testing**, `_3` = spare.
- Copies live in `intersection_experiments_2/datasets/` (and `_full`, `_3` variants).
- `categories.json` regenerated for all versions.

## 4. Experiment folders

- `intersection_experiments_2/` — **CURRENT ACTIVE FOLDER**: attention heads + MLP blocks
  pruning (nothing else). This is where the retrain with fixed code is running now.
- `intersection_experiments_2_full/` — older full-granularity run (heads + MLP + neurons).
- `intersection_experiments_3/` — dual-init (Ns/Dn) union method experiment
  (gates-init-open vs gates-init-closed, union of both masks = complete+faithful circuit).
  Contains ns/dn/union masks + anti-masks and evals.
- `intersection_experiments/` — oldest heads-only experiments + report.
- Report file: `intersection_report.md` (root) — master tables of all findings.

## 5. Code fixes already applied (do NOT regress)

- `evaluate_mask.py` (in exp folders): auto-enables `prune_mlp_blocks` when the mask file
  contains `mlp_block_gate` keys. Old silent bug: MLP-block masks were evaluated heads-only.
- `load_masks`: any gate existing in the model but missing from a mask file is pinned OFF.
- `pruning_manager.py:161-165`: freeze re-pinning — when `--mask` is passed, gates that are
  OFF in the loaded mask are re-pinned OFF after every optimizer step (hard guarantee they
  cannot turn back on; protects against hard-concrete gate noise).
- `train_mask.py`: takes `--dataset` (name or list for combined), `--epochs`,
  `--output_name`, `--mask` (phase-2 init/freeze), `--lambda_sparsity` (default 0.05),
  `--prune_mlp_blocks` flag. Used for BOTH phases (same script).

## 6. RUNNING NOW (as of this snapshot)

Six screen sessions in `intersection_experiments_2` (masks/ and results/ were wiped first):
- `int_screen_1`: combined 5x`_1` datasets, 300 ep, lambda=0.15 (1.5x old combined lambda
  0.1), `--prune_mlp_blocks`, output `all_300ep_l015`. ETA ~4h from 21:44 UTC Jul 28.
- `int_screen_2..6`: fruits/animals/colors/metals/vehicles `_1`, 600 ep each,
  lambda=0.05 (default; user approved), `--prune_mlp_blocks`, output `{category}_600`.
  ETA ~1h from 21:44 UTC Jul 28.
- Logs: `intersection_experiments_2/masks/nohup_*.log`. Check: `screen -ls`, `nvidia-smi`.

## 7. NEXT STEPS (in order)

1. When combined 300ep finishes, launch 5x frozen phase-2 in screens int_screen_1..5:
   ```
   python train_mask.py --dataset {cat}_1 --epochs 300 \
     --output_name l01_frozen_{cat}_300ep \
     --mask intersection_experiments_2/masks/all_300ep_l015_mask.pt \
     --lambda_sparsity 0.075 --prune_mlp_blocks
   ```
   (lambda 0.075 = half of combined 0.15, as before.)
2. Take 5-way intersection of frozen masks, and intersection of the five 600ep masks.
3. Also anti-masks (whole network minus circuit) and disjoint masks
   (`intersect_masks.py` has options for component selection, anti, disjoint).
4. Evaluate everything on all 10 `_2` datasets with the FIXED evaluate_mask.py:
   accuracy (generative + probability) and KL divergence.
5. Update `intersection_report.md` tables; remove numbers whose masks were deleted.

## 8. Key user preferences / operating rules

- Training runs in named `screen` sessions; evals as background bash tasks.
- Keep shell commands short; verify quickly (long-running foreground commands look stuck).
- Git mutations (commit/push) need explicit user request each time; push is user-side.
- Machine may be shelved/unshelved (NVML mismatch fix) — re-check `nvidia-smi` after restart.
- After any server restart, screen sessions die — relaunch from this file.

## 9. Earlier findings (summary)

- Intersection of individually-trained masks recovers most sequence-tracking ability;
  anti-intersection masks keep talking about the prompt topic but fail tracking — supports
  "intersection = logical core, remainder = category detail" hypothesis.
- Two-phase (combined + frozen finetune) did NOT clearly beat direct individual intersection.
- Dual-init (Ns/Dn) union method tested in exp_3 at heads-only level.
- Natural-language probe set (10 prompts, sequence tracking in natural English) exists for
  qualitative generation checks of masks and anti-masks.
