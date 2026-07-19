# Intersection Mask Comparison — intersection_experiments_2

Model: Llama-3.2-1B. Attention-head gates only (512 total = 16 layers x 32 heads).
Metric: circuit probability accuracy (argmax over the 16 category tokens) and KL divergence vs baseline.
Base model accuracies (unpruned): prob-acc / gen-acc.

Masks compared:
- `scratch_intersection_mask.pt` (42 heads): AND of five independently trained 600-epoch masks (`*_600_mask.pt`)
- `finetuned_intersection_mask.pt` (49 heads): AND of five masks finetuned (300ep each) from the old λ=0.05 combined mask (136 heads), unfrozen phase 2
- `frozen_intersection_mask.pt` (50 heads): same as above but with frozen off-gates in phase 2
- `l01_intersection_mask.pt` (59 heads): AND of five masks finetuned (300ep each, frozen off-gates) from the λ=0.10 combined mask (88 heads, `all_300ep_l01_mask.pt`)
- Parent (λ=0.10 combined, 88 heads) shown as the ceiling for the l01 intersection

## Circuit probability accuracy

| dataset | base (prob/gen) | ∩ 600-scratch (42h) | ∩ 300-finetuned (49h) | ∩ 300-frozen (50h) | ∩ λ=0.10 frozen (59h) | parent 88h |
|---|---|---|---|---|---|---|
| fruits   | 0.998 / 0.996 | 0.278 | 0.348 | 0.268 | **0.606** | 0.840 |
| animals  | 1.000 / 0.994 | 0.560 | 0.750 | 0.550 | **0.902** | 0.960 |
| colors   | 1.000 / 1.000 | 0.302 | 0.324 | 0.318 | **0.566** | 0.846 |
| metals   | 0.984 / 0.978 | 0.354 | 0.386 | 0.384 | **0.650** | 0.864 |
| vehicles | 0.994 / 0.990 | 0.428 | 0.520 | 0.426 | **0.848** | 0.874 |

## KL divergence vs baseline (lower is better)

| dataset | ∩ 600-scratch | ∩ 300-finetuned | ∩ 300-frozen | ∩ λ=0.10 frozen |
|---|---|---|---|---|
| fruits   | 0.884 | 0.877 | 0.878 | **0.711** |
| animals  | 0.167 | 0.139 | 0.168 | **0.087** |
| colors   | 0.730 | 0.733 | 0.732 | **0.586** |
| metals   | 0.537 | 0.518 | 0.512 | **0.438** |
| vehicles | 0.207 | 0.202 | 0.203 | **0.123** |

## Notes

- λ=0.10 pipeline beats all other intersections on every dataset, both metrics, with the largest shared core (59 heads = 67% of its 88-head parent; union of the five l01 masks = 88, zero waste).
- Two-tier structure: the 59-head core is nearly sufficient for animals (0.902) and vehicles (0.848); fruits/colors/metals additionally need a shared set of ~25 heads (their pairwise Jaccard is 0.92-0.94, their masks kept 83-86 of 88).
- Freezing off-gates in phase 2 makes every finetuned mask a strict subset of the combined mask (`pruning_manager.load_masks_for_finetuning`).
- Source CSVs: `scratch_intersection_eval.csv`, `finetuned_intersection_eval.csv`, `frozen_intersection_eval.csv`, `l01_intersection_eval.csv`, `all_300ep_l01_allcats_eval.csv` in this directory.
