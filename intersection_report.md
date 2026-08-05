# Circuit Pruning & Mask Intersection Experiments — Full Report

**Model:** Llama-3.2-1B (16 layers × 32 attention heads = 512 heads; hidden 2048; MLP intermediate 8192)
**Tasks:** 5 semantic-category induction datasets (fruits, animals, colors, metals, vehicles). Each category has 3 fully disjoint versions (`_1`, `_2`, `_3`), 500 samples each; no prompt appears in more than one version.
Prompt format: `Sequence: A B C A B` → predict `C`, with a corrupted version of the prompt used during pruning (clean/corrupted forward-pass trick).
**Random baseline:** 1/16 categories = 0.0625.

**Evaluation protocol:** all numbers in this report are **held-out** — every mask is evaluated on the `_1` version of each category, which no mask was ever trained or pruned on. (Masks were trained on the original category datasets; `_1`/`_2`/`_3` are freshly generated, word-disjoint splits of the same task.)

## Evaluation metrics

- **prob-acc**: single forward pass; argmax of last-position logits restricted to the 16 category tokens; compared to target.
- **gen-acc**: 2-token greedy generation from the full vocabulary; first word compared to target.
- **KL**: KL divergence between pruned-circuit logits and frozen baseline-model logits (mean per token).

## Method overview

Pruning uses L0 hard-concrete gates trained with loss = KL(circuit ‖ baseline) + λ·(sparsity). Only gate parameters train.

**Two-phase pipeline (proposed method):**
1. **Phase 1 (combined):** train one mask on the mixture of all 5 datasets (2500 samples), 300 epochs.
2. **Phase 2 (finetune):** start each category run from the *binary* combined mask with **off-gates frozen** (log_alpha pinned at −1e6, gradients zeroed via hooks, re-pinned each epoch — guarantees finetuned masks are strict subsets of the combined mask), 300 epochs per category.
3. **Intersection:** logical AND of the 5 finetuned masks.

**Baseline (scratch):** 5 masks trained independently for 600 epochs, then ANDed.

**Sparsity control:** `--lambda_sparsity` sets the attention-head λ and scales every other granularity level's λ by the same factor (default: heads 0.05 = factor 5; "λ×2" = 0.10 = factor 10).

---

# Part 1 — Attention-head-only pruning

Only `head_gates` active (512 gates). MLPs always dense.

## 1.1 Mask structure

| mask set | sizes (heads) | union | intersection |
|---|---|---|---|
| 600ep scratch | 76–164 | 226 | **42** |
| 300+300 finetune, unfrozen (λ=0.05 parent) | — | — | **49** |
| 300+300 finetune, frozen (λ=0.05 parent) | — | — | **50** |
| 300+300 finetune, frozen (λ=0.10 parent, 88h) | 63–86 | 88 | **59** |

(Per-category masks of the two λ=0.05-parent sets and the λ=0.05 parent itself were deleted; only their intersection masks survive, so only the ∩ sizes are reported for those rows. All conclusions below that depended on the deleted masks have been removed.)

- With freezing + the lean λ=0.10 parent, every finetuned mask is a strict subset of its parent (union = 88 = parent), pairwise Jaccard 0.71–0.94, and the intersection keeps **67%** of the parent. A leaner parent leaves less internal redundancy for the 5 runs to diverge on.

## 1.2 Parent (combined) mask — circuit performance

(The λ=0.05 / 136-head parent was deleted and could not be re-evaluated on held-out data; removed.)

| parent mask | metric | fruits | animals | colors | metals | vehicles |
|---|---|---|---|---|---|---|
| λ=0.10, 88 heads | prob | 0.824 | 0.944 | 0.814 | 0.840 | 0.892 |
| | gen | 0.788 | 0.942 | 0.814 | 0.832 | 0.876 |
| | KL | 0.365 | 0.068 | 0.354 | 0.257 | 0.096 |

## 1.3 Individual 600-epoch masks (own category)

| mask | prob | gen | KL | heads |
|---|---|---|---|---|
| fruits_600 | 0.962 | 0.946 | 0.176 | 154 |
| animals_600 | 0.978 | 0.974 | 0.045 | 76 |
| colors_600 | 0.972 | 0.966 | 0.199 | 164 |
| metals_600 | 0.918 | 0.902 | 0.200 | 127 |
| vehicles_600 | 0.980 | 0.976 | 0.053 | 95 |

## 1.4 Intersection results (the key table)

| intersection | metric | fruits | animals | colors | metals | vehicles |
|---|---|---|---|---|---|---|
| **600ep scratch (42h)** | prob | 0.292 | 0.532 | 0.264 | 0.336 | 0.448 |
| | gen | 0.190 | 0.468 | 0.226 | 0.256 | 0.338 |
| | KL | 0.867 | 0.169 | 0.768 | 0.568 | 0.201 |
| **300+300 unfrozen (49h)** | prob | 0.354 | 0.760 | 0.316 | 0.346 | 0.548 |
| | gen | 0.188 | 0.728 | 0.274 | 0.296 | 0.378 |
| | KL | 0.863 | 0.139 | 0.772 | 0.552 | 0.196 |
| **300+300 frozen (50h)** | prob | 0.256 | 0.530 | 0.294 | 0.374 | 0.450 |
| | gen | 0.178 | 0.480 | 0.268 | 0.298 | 0.308 |
| | KL | 0.858 | 0.167 | 0.767 | 0.549 | 0.199 |
| **300+300 frozen, λ=0.10 parent (59h)** | prob | **0.570** | **0.890** | **0.520** | **0.640** | **0.870** |
| | gen | **0.488** | **0.878** | **0.512** | **0.610** | **0.838** |
| | KL | **0.696** | **0.088** | **0.620** | **0.473** | **0.121** |

**Base model (unpruned, held-out `_1`):** prob 0.998/1.000/1.000/0.996/0.994; gen 0.998/0.998/1.000/0.990/0.978.

## 1.5 Findings (head level)

1. **The two-phase method beats independent training + intersection decisively.** Same-budget comparison: 59-head two-phase ∩ (0.52–0.89) vs 42-head scratch ∩ (0.26–0.53) — +0.26 to +0.42 accuracy on every category, better KL everywhere.
2. **Freezing alone didn't help intersection quality** (50h frozen ≈ 49h unfrozen) — it only guarantees the clean nesting structure (strict subsets). Divergence happens *within* the parent mask, not outside it.
3. **The lean parent (λ=0.10, 88 heads) was the real fix.** Less internal redundancy → the 5 phase-2 runs converge to nearly the same subset (Jaccard 0.71–0.94) → fatter, functional intersection at 67% of the parent.
4. **Two-tier task structure discovered:** the 59-head core is nearly sufficient for animals (0.890) and vehicles (0.870); fruits/colors/metals need ~25 additional heads, and they are the *same* heads across those three categories (pairwise Jaccard 0.92–0.94).

---

# Part 2 — Full-depth (fine-granularity) pruning

Enabled gate levels: attention heads (512), attention neurons (32,768), MLP hidden (131,072), MLP output (32,768) — 197,120 gates total. Coarse levels (attention blocks, MLP blocks, full layers) were **disabled after they proved unstable**: with all 7 levels on, training was bistable — runs either collapsed to a fully-pruned dead model (phase-1 λ×2 collapsed by epoch ~10, final KL 1.89; fruits/colors 600ep runs collapsed with KL 3.39/2.81) or barely pruned at all. The scalar per-layer gates could delete whole sub-blocks, cascading gradient loss to everything beneath them. All full-depth results below use the 4 fine levels only.

## 2.1 Parent (combined) mask — all_300ep_l01 (300ep, λ×2)

**Pruning overview: 36,279 / 197,120 gates kept (18.4%)**

| gate type | active | total | kept |
|---|---|---|---|
| attention heads | 70 | 512 | 13.7% |
| attention neurons | 2,800 | 32,768 | 8.5% |
| MLP hidden | 21,563 | 131,072 | 16.5% |
| MLP output | 11,846 | 32,768 | 36.2% |

| metric | fruits | animals | colors | metals | vehicles |
|---|---|---|---|---|---|
| prob | 0.952 | 0.994 | 0.968 | 0.912 | 0.944 |
| gen | 0.766 | 0.990 | 0.846 | 0.830 | 0.944 |
| KL | 0.317 | 0.108 | 0.244 | 0.274 | 0.175 |

Note: prob-acc 0.91–0.99 — **much better than the heads-only parent** (0.81–0.94) with fewer heads (70 vs 88). Fine granularity preserves task-relevant computation far more efficiently.

## 2.2 Frozen phase-2 masks (own category) — all strict subsets at every gate level (verified)

| mask | prob | gen | KL | heads | MLP hidden kept |
|---|---|---|---|---|---|
| frozen_fruits | 0.962 | 0.776 | 0.294 | 70 | 18,612 |
| frozen_animals | 0.984 | 0.986 | 0.097 | 56 | 11,326 |
| frozen_colors | 0.952 | 0.832 | 0.224 | 68 | 17,875 |
| frozen_metals | 0.914 | 0.852 | 0.252 | 70 | 17,057 |
| frozen_vehicles | 0.948 | 0.934 | 0.157 | 54 | 13,086 |

Phase-2 masks are individually excellent — near-parent accuracy everywhere.

## 2.3 Individual 600-epoch masks (own category)

| mask | prob | gen | KL | heads | MLP hidden kept |
|---|---|---|---|---|---|
| fruits_600 | 0.992 | 0.800 | 0.165 | 87 | 28,434 |
| animals_600 | 0.988 | 0.986 | 0.068 | 51 | 13,635 |
| colors_600 | 0.972 | 0.850 | 0.135 | 83 | 25,776 |
| metals_600 | 0.968 | 0.902 | 0.152 | 72 | 26,402 |
| vehicles_600 | 0.974 | 0.972 | 0.099 | 60 | 17,607 |

The best individual masks of the entire project: prob-acc 0.968–0.992 on held-out data. Yet their mutual intersection is functionally dead (Variant tables below) — the strongest evidence that fine-grained masks solve the same task with non-identifiable internal structure.

## 2.4 Full-depth intersection experiments

Four intersection variants were tested, for both the frozen two-phase set and the 600ep scratch set. "Hollow head" = head gate on but only the ANDed subset of its 64 neuron dims active.

**Variant A — ∩ of ALL fine gates (heads + neurons + MLP hidden + MLP output):**

| set | heads | neurons | MLP hid | MLP out | fruits | animals | colors | metals | vehicles |
|---|---|---|---|---|---|---|---|---|---|
| frozen ∩ | 52 | 1,032 | 5,105 | 5,257 | 0.018 | 0.036 | 0.040 | 0.022 | 0.028 |
| (gen) | | | | | 0.006 | 0.022 | 0.036 | 0.016 | 0.020 |
| (KL) | | | | | 3.528 | 0.586 | 3.071 | 2.161 | 0.665 |
| 600 ∩ | 29 | 593 | 1,906 | 2,024 | 0.020 | 0.020 | 0.040 | 0.018 | 0.028 |
| (gen) | | | | | 0.008 | 0.020 | 0.034 | 0.010 | 0.016 |
| (KL) | | | | | 3.654 | 0.640 | 3.180 | 2.225 | 0.703 |

**Variant B — ∩ heads (WHOLE, all 64 dims inherited) + ∩ MLP gates:**

| set | metric | fruits | animals | colors | metals | vehicles |
|---|---|---|---|---|---|---|
| frozen ∩ (52 whole heads) | prob | 0.026 | 0.056 | 0.036 | 0.030 | 0.034 |
| | gen | 0.008 | 0.054 | 0.036 | 0.016 | 0.032 |
| | KL | 3.473 | 0.527 | 3.005 | 2.126 | 0.639 |
| 600 ∩ (29 whole heads) | prob | 0.028 | 0.018 | 0.032 | 0.020 | 0.026 |
| | gen | 0.008 | 0.016 | 0.026 | 0.012 | 0.014 |
| | KL | 3.626 | 0.620 | 3.156 | 2.216 | 0.687 |

**Variant C — ∩ attention (heads + neurons, hollow) + MLP 100% dense:**

| set | metric | fruits | animals | colors | metals | vehicles |
|---|---|---|---|---|---|---|
| frozen ∩ (52 hollow heads) | prob | 0.078 | 0.164 | 0.080 | 0.106 | 0.130 |
| | gen | 0.014 | 0.056 | 0.054 | 0.038 | 0.024 |
| | KL | 1.034 | 0.378 | 0.881 | 0.714 | 0.391 |
| 600 ∩ (29 hollow heads) | prob | 0.042 | 0.046 | 0.046 | 0.052 | 0.068 |
| | gen | 0.010 | 0.004 | 0.024 | 0.014 | 0.010 |
| | KL | 1.104 | 0.484 | 0.975 | 0.758 | 0.471 |

**Reference (from Part 1):** ∩ whole heads + MLP always dense = heads-only 59-head ∩ → prob 0.520–0.890, gen 0.488–0.878.

## 2.5 Findings (full-depth)

1. **Intersection collapses below head granularity, even with perfect nesting.** All frozen masks were verified strict subsets of the parent at every gate level, phase-2 masks were individually excellent (0.91–0.98), yet every fine-grained AND variant scored ≤0.16 — near or below the 0.0625 random baseline.
2. **The MLP gate AND is the biggest destroyer.** Whole heads + ∩MLP is dead (≤0.06); hollow heads + dense MLP recovers somewhat (0.08–0.16, KL 3–8× better). Category content lives in MLP dims and does not align across independently trained masks — only 3.9% of MLP-hidden gates and 16% of MLP-output gates are shared by all 5.
3. **Neuron-level AND also kills heads.** 52 heads shared by all 5 masks (74% of parent — attention *structure* aligns well) become useless when hollowed to 3.1% of their dims. Compare: same 52–59 whole heads + dense MLP → 0.52–0.89.
4. **Mechanism vs content decomposition:** attention heads implement the shared, category-agnostic induction mechanism (74% agreement across masks); MLPs store category-specific vocabulary content (near-zero dim agreement). Intersection is valid only for the mechanism layer.
5. **Coarse gates (block/layer level) are untrainable in this setup** — bistable collapse/prune-nothing dynamics; disabled for all final runs.
6. **The two-phase + freeze pipeline is still the right way to train comparable masks** (strict nesting, individually excellent masks, best parent accuracy of all experiments) — but the shared core must be extracted at whole-head granularity, or the MLP must be combined by union rather than intersection.

---

## 2.6 Pairwise intersection analysis (all 10 category pairs)

Motivation: if pairwise intersections are very similar to each other, the intersection operation is extracting a stable shared core, and the 5-way collapse means it "left out things needed for each dataset individually" rather than failing outright. All 10 pairs intersected (AND over all 4 fine gate types). sim_X = mean Jaccard of that pair's ∩ against the other 9 pairwise ∩s. (Mask-structure analysis — independent of evaluation split.)

**300-combined + 300-finetune (frozen):**

| pair | heads | sim_h | neurons | sim_n | mlp_hidden | sim_mh | mlp_output | sim_mo |
|---|---|---|---|---|---|---|---|---|
| fruits–animals | 56 | 0.89 | 1648 | 0.65 | 9774 | 0.53 | 7383 | 0.68 |
| fruits–colors | 68 | 0.82 | 2067 | 0.60 | 15574 | 0.52 | 10566 | 0.68 |
| fruits–metals | 70 | 0.82 | 2060 | 0.61 | 14830 | 0.53 | 10266 | 0.69 |
| fruits–vehicles | 54 | 0.90 | 1613 | 0.64 | 11206 | 0.54 | 8353 | 0.69 |
| animals–colors | 54 | 0.89 | 1591 | 0.65 | 9439 | 0.53 | 7267 | 0.68 |
| animals–metals | 56 | 0.89 | 1656 | 0.65 | 9208 | 0.53 | 7118 | 0.68 |
| animals–vehicles | 53 | 0.89 | 1581 | 0.65 | 8015 | 0.51 | 6256 | 0.65 |
| colors–metals | 68 | 0.82 | 2008 | 0.61 | 14368 | 0.53 | 10134 | 0.69 |
| colors–vehicles | 53 | 0.89 | 1545 | 0.64 | 10774 | 0.54 | 8209 | 0.69 |
| metals–vehicles | 54 | 0.90 | 1583 | 0.65 | 10468 | 0.54 | 8040 | 0.69 |

**600-epoch individual (scratch):**

| pair | heads | sim_h | neurons | sim_n | mlp_hidden | sim_mh | mlp_output | sim_mo |
|---|---|---|---|---|---|---|---|---|
| fruits–animals | 36 | 0.65 | 1150 | 0.47 | 5918 | 0.31 | 4722 | 0.41 |
| fruits–colors | 63 | 0.48 | 1719 | 0.38 | 11554 | 0.27 | 8693 | 0.39 |
| fruits–metals | 54 | 0.57 | 1578 | 0.43 | 11077 | 0.28 | 8716 | 0.39 |
| fruits–vehicles | 39 | 0.60 | 1188 | 0.44 | 7021 | 0.30 | 5818 | 0.41 |
| animals–colors | 37 | 0.67 | 1199 | 0.49 | 5599 | 0.31 | 4345 | 0.41 |
| animals–metals | 42 | 0.67 | 1401 | 0.48 | 5550 | 0.31 | 4565 | 0.41 |
| animals–vehicles | 45 | 0.62 | 1554 | 0.45 | 5032 | 0.30 | 4007 | 0.38 |
| colors–metals | 56 | 0.58 | 1642 | 0.44 | 10632 | 0.28 | 8292 | 0.40 |
| colors–vehicles | 36 | 0.66 | 1132 | 0.48 | 6410 | 0.31 | 5352 | 0.41 |
| metals–vehicles | 45 | 0.62 | 1389 | 0.46 | 6775 | 0.30 | 5636 | 0.41 |

**Overall similarity among the 10 pairwise ∩s (Jaccard over all 45 pair-combinations):**

| gate type | frozen mean (min–max) | 600 scratch mean (min–max) |
|---|---|---|
| head_gates | **0.871** (0.754–1.000) | 0.614 (0.367–0.872) |
| neuron_gates | 0.635 (0.467–0.783) | 0.454 (0.261–0.672) |
| hidden_gates | 0.529 (0.336–0.765) | 0.298 (0.161–0.458) |
| output_gates | 0.683 (0.502–0.903) | 0.401 (0.223–0.611) |

**5-way ∩ vs union of pairwise ∩s (fraction retained):**

| gate type | frozen 5-way / union (retained) | 600 5-way / union (retained) |
|---|---|---|
| head_gates | 52 / 70 (**0.74**) | 29 / 95 (0.31) |
| neuron_gates | 1032 / 2760 (0.37) | 593 / 3652 (0.16) |
| hidden_gates | 5105 / 21102 (0.24) | 1906 / 28761 (0.07) |
| output_gates | 5257 / 11828 (0.44) | 2024 / 16462 (0.12) |

**Findings (pairwise analysis):**

1. **Hypothesis confirmed at head level for two-phase masks:** pairwise ∩s are 87% identical to each other (never below 0.75) and barely larger than the 5-way (53–70 vs 52 heads; 74% retained). Intersection extracts a stable, pair-independent shared head core.
2. **MLP sharing is pair-specific, not universal:** pairwise MLP ∩s are large (8–15.5k dims) but only ~0.51–0.54 similar to each other; the 5-way keeps just 24% of what any pair shares. Each pair shares *different* MLP content — the 5-way collapse is the absence of a universal MLP core, not an intersection artifact.
3. **600 scratch shows the same pattern degraded:** pairwise ∩s inconsistent even at head level (0.37–0.87), 5-way retains only 31% of heads and 7% of MLP-hidden — without the combined-mask anchor, each pair finds its own solution.

# Summary — where intersection helps and where it breaks

| setting | intersection works? | evidence |
|---|---|---|
| Whole attention heads + dense MLP | **YES** | 59h two-phase ∩: 0.52–0.89 prob-acc (vs 0.26–0.53 scratch) |
| Two-phase (combined → finetune) vs independent | **helps a lot** | +0.26–0.42 accuracy at same budget |
| Frozen off-gates in phase 2 | helps structure, neutral on accuracy | strict subsets; ∩ size unchanged |
| Lean parent (λ×2) | **helps a lot** | ∩ fraction 67% of parent; Jaccard 0.71–0.94 |
| Sub-head (neuron-dim) AND | **breaks** | hollows shared heads → ≤0.16 acc |
| MLP-dim AND | **breaks hardest** | category content doesn't align → ≤0.06 acc |
| Coarse (block/layer) gates | **breaks training** | bistable collapse |
| Pairwise (2-way) ∩ of two-phase masks | **works at head level** | pairwise ∩s 87% identical; stable 52-head core; MLP sharing is pair-specific (no universal MLP core) |

## Next steps (not yet run)

- **∩ whole heads + ∪ MLP** — shared attention mechanism + union of category memories; predicted best composite under the mechanism/content hypothesis.
- Per-category **delta analysis** on the nested frozen masks (which heads/dims each category keeps beyond the shared core).
- Save/report via PDF (this file is self-contained for conversion).

*Data sources: `exp2_heads_mlp_overpruned/results/*.csv`, `exp2full_heads_mlp_neurons/results/*.csv` (all evaluated on held-out `_1` datasets). Mask files in respective `masks/` dirs.*
