# Heads + MLP-Block Run — Full Evaluation Results

Date: 2026-07-29. Folder: `exp2_heads_mlp_overpruned/`.
Granularity: attention heads + whole MLP blocks (nothing else).
Training: `_1` datasets; testing: `_2` datasets (500 examples each).
Trained categories: fruits, animals, colors, metals, vehicles.
Heldout categories: clothing, furniture, instruments, professions, sports.

Pipelines compared:
- **600**: each category trained from scratch, 600 epochs, λ=0.05.
- **Frozen (300+300)**: one mask trained 300 epochs on all 5 combined (λ=0.15),
  then fine-tuned 300 epochs per category with off-gates frozen off (λ=0.075).

OFF gates receive the *corrupted* activation stream (not a bypass), so every
component removed from a mask is actively replaced by noise.

## 1. Mask anatomy

| Mask | Heads on (/512) | MLP blocks on (/16) |
|---|---|---|
| all_300ep_l015 (combined) | 85 | 13 |
| fruits_600 | 147 | 14 |
| animals_600 | 76 | 8 |
| colors_600 | 147 | 15 |
| metals_600 | 115 | 15 |
| vehicles_600 | 80 | 12 |
| frozen_fruits | 83 | 13 |
| frozen_animals | 28 | 2 |
| frozen_colors | 82 | 13 |
| frozen_metals | 82 | 13 |
| frozen_vehicles | 56 | 9 |
| **intersection_600** | **37** | **7** |
| **intersection_frozen** | **25** | **2** |
| anti_intersection_600 | 475 | 9 |
| anti_intersection_frozen | 487 | 14 |

## 2. Individual masks on their own test set (generative acc / KL)

| Category | 600 (gen, KL) | Frozen (gen, KL) |
|---|---|---|
| fruits | 0.852, 0.69 | 0.354, 1.01 |
| animals | 0.722, 0.22 | 0.024, 0.53 |
| colors | 0.946, 0.25 | 0.478, 0.83 |
| metals | 0.874, 0.20 | 0.474, 0.55 |
| vehicles | 0.912, 0.17 | 0.202, 0.38 |

600 masks are all healthy (72–95%). Frozen masks are all degraded (2–48%),
including ones with near-normal head/MLP counts — phase-2 fine-tuning itself
damaged the circuits this run (λ=0.075 with heads+MLP blocks prunes too hard
from an already-pruned start).

## 3. Intersections and anti-intersections on all 10 test sets

Circuit probability accuracy / generative accuracy / KL (base acc ≈ 0.91–1.00 everywhere):

### intersection_frozen (25 heads, 2 MLP blocks)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.026 | 0.006 | 3.37 |
| animals_2 | trained | 0.034 | 0.024 | 0.54 |
| colors_2 | trained | 0.066 | 0.050 | 2.84 |
| metals_2 | trained | 0.060 | 0.038 | 1.83 |
| vehicles_2 | trained | 0.036 | 0.022 | 0.63 |
| clothing_2 | heldout | 0.022 | 0.016 | 0.43 |
| furniture_2 | heldout | 0.008 | 0.006 | 1.47 |
| instruments_2 | heldout | 0.068 | 0.006 | 5.61 |
| professions_2 | heldout | 0.026 | 0.000 | 1.90 |
| sports_2 | heldout | 0.060 | 0.028 | 3.21 |

### intersection_600 (37 heads, 7 MLP blocks)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.036 | 0.010 | 3.20 |
| animals_2 | trained | 0.054 | 0.040 | 0.47 |
| colors_2 | trained | 0.072 | 0.062 | 2.63 |
| metals_2 | trained | 0.056 | 0.038 | 1.68 |
| vehicles_2 | trained | 0.042 | 0.016 | 0.57 |
| clothing_2 | heldout | 0.036 | 0.018 | 0.41 |
| furniture_2 | heldout | 0.036 | 0.012 | 1.38 |
| instruments_2 | heldout | 0.058 | 0.006 | 5.24 |
| professions_2 | heldout | 0.038 | 0.002 | 1.78 |
| sports_2 | heldout | 0.052 | 0.018 | 3.01 |

### anti_intersection_frozen (487 heads, 14 MLP blocks)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.044 | 0.006 | 3.06 |
| animals_2 | trained | 0.046 | 0.030 | 0.55 |
| colors_2 | trained | 0.070 | 0.054 | 2.62 |
| metals_2 | trained | 0.078 | 0.050 | 1.62 |
| vehicles_2 | trained | 0.052 | 0.018 | 0.59 |
| clothing_2 | heldout | 0.032 | 0.016 | 0.41 |
| furniture_2 | heldout | 0.050 | 0.016 | 1.28 |
| instruments_2 | heldout | 0.046 | 0.000 | 4.91 |
| professions_2 | heldout | 0.054 | 0.004 | 1.73 |
| sports_2 | heldout | 0.042 | 0.016 | 2.88 |

### anti_intersection_600 (475 heads, 9 MLP blocks)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.064 | 0.006 | 3.15 |
| animals_2 | trained | 0.040 | 0.026 | 0.55 |
| colors_2 | trained | 0.066 | 0.056 | 2.69 |
| metals_2 | trained | 0.086 | 0.036 | 1.68 |
| vehicles_2 | trained | 0.050 | 0.018 | 0.62 |
| clothing_2 | heldout | 0.046 | 0.012 | 0.43 |
| furniture_2 | heldout | 0.028 | 0.008 | 1.36 |
| instruments_2 | heldout | 0.060 | 0.002 | 5.12 |
| professions_2 | heldout | 0.050 | 0.000 | 1.77 |
| sports_2 | heldout | 0.056 | 0.010 | 2.96 |

### all_300ep_l015 combined phase-1 mask (85 heads, 13 MLP blocks)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.524 | 0.378 | 1.01 |
| animals_2 | trained | 0.746 | 0.714 | 0.13 |
| colors_2 | trained | 0.514 | 0.490 | 0.83 |
| metals_2 | trained | 0.544 | 0.508 | 0.55 |
| vehicles_2 | trained | 0.644 | 0.528 | 0.18 |
| clothing_2 | heldout | 0.488 | 0.302 | 0.18 |
| furniture_2 | heldout | 0.294 | 0.218 | 0.55 |
| instruments_2 | heldout | 0.384 | 0.228 | 1.61 |
| professions_2 | heldout | 0.456 | 0.350 | 0.62 |
| sports_2 | heldout | 0.432 | 0.294 | 1.17 |

## 4. Findings

1. **Necessity holds**: ablating only the shared core (anti-intersection, with
   ~95% of heads and most MLP blocks still on) destroys the task everywhere
   (gen ≤ 5.6%). The 25–37 shared heads + few shared MLP blocks are necessary.
2. **Sufficiency fails at MLP blocks**: the intersections themselves score
   0–6% everywhere. Each category's mask keeps 8–15/16 MLP blocks but mostly
   *different* ones, so the 5-way AND keeps only 7/16 (600) or 2/16 (frozen).
   With OFF = corrupted stream, most layers lose their MLP and the circuit
   cannot run. Category vocabulary/knowledge lives in MLPs — exactly what an
   intersection removes.
3. **Contrast with heads-only runs**: in the earlier heads-only experiments
   intersections retained most accuracy, because MLPs were never gated and
   always passed clean. Gating MLP blocks turns intersection into a
   sufficiency killer.
4. **Both training stages over-pruned at heads+MLP granularity**: the combined
   phase-1 mask (λ=0.15) already reached only 38–71% gen on trained categories
   (600 individuals: 72–95%), and frozen phase-2 (λ=0.075) degraded further to
   2–48%. frozen_animals collapsed (28 heads, 2 MLP blocks). The 1.5× λ increase
   that was reasonable at heads-only granularity is too aggressive once MLP
   blocks (16 coarse, high-impact gates) are also being pruned.
5. No trained-vs-heldout gap for any intersection/anti mask — all fail
   equally on both splits, consistent with a broken circuit rather than a
   category-general one.
