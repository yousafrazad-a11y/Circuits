# Heads + MLP-Block Run, NORMAL lambdas — Full Evaluation Results

Date: 2026-07-29. Folder: `intersection_experiments_4/`.
Same setup as `intersection_experiments_2/` but with normal λ values instead of 1.5×.
Granularity: attention heads + whole MLP blocks. Train: `_1`; test: `_2` (500 ex each).
Trained: fruits, animals, colors, metals, vehicles.
Heldout: clothing, furniture, instruments, professions, sports.

Pipelines:
- **600**: per-category from scratch, 600 ep, λ=0.05.
- **Frozen (300+300)**: combined 300 ep on all 5 (λ=0.10), then per-category
  300 ep with off-gates frozen off (λ=0.05).

OFF gates receive the *corrupted* stream (not a bypass).

## 1. Mask anatomy

| Mask | Heads (/512) | MLP blocks (/16) |
|---|---|---|
| all_300ep_l01 (combined) | 95 | 13 |
| fruits_600 | 145 | 14 |
| animals_600 | 76 | 8 |
| colors_600 | 142 | 15 |
| metals_600 | 119 | 14 |
| vehicles_600 | 102 | 12 |
| frozen_fruits | 94 | 13 |
| frozen_animals | 52 | 8 |
| frozen_colors | 92 | 13 |
| frozen_metals | 94 | 13 |
| frozen_vehicles | 62 | 11 |
| **intersection_frozen** | **47** | **8** |
| **intersection_600** | **35** | **8** |
| anti_intersection_frozen | 465 | 8 |
| anti_intersection_600 | 477 | 8 |

## 2. Individual masks on own test set (gen acc / KL)

| Category | 600 | Frozen |
|---|---|---|
| fruits | 0.876, 0.58 | 0.526, 0.96 |
| animals | 0.754, 0.21 | 0.540, 0.29 |
| colors | 0.846, 0.45 | 0.612, 0.80 |
| metals | 0.780, 0.47 | 0.620, 0.53 |
| vehicles | 0.888, 0.16 | 0.324, 0.25 |

## 3. Combined phase-1 mask (95 heads, 13 MLP) — prob / gen / KL

| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.632 | 0.578 | 0.96 |
| animals_2 | trained | 0.858 | 0.802 | 0.12 |
| colors_2 | trained | 0.642 | 0.600 | 0.80 |
| metals_2 | trained | 0.630 | 0.620 | 0.53 |
| vehicles_2 | trained | 0.652 | 0.586 | 0.18 |
| clothing_2 | heldout | 0.634 | 0.602 | 0.16 |
| furniture_2 | heldout | 0.402 | 0.380 | 0.58 |
| instruments_2 | heldout | 0.474 | 0.224 | 1.63 |
| professions_2 | heldout | 0.530 | 0.346 | 0.65 |
| sports_2 | heldout | 0.500 | 0.420 | 1.18 |

## 4. Intersections — prob / gen / KL (base acc ≈ 0.91–1.00)

### intersection_frozen (47 heads, 8 MLP)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.138 | 0.068 | 3.09 |
| animals_2 | trained | 0.410 | 0.382 | 0.33 |
| colors_2 | trained | 0.148 | 0.126 | 2.47 |
| metals_2 | trained | 0.168 | 0.146 | 1.56 |
| vehicles_2 | trained | 0.132 | 0.102 | 0.49 |
| clothing_2 | heldout | 0.110 | 0.060 | 0.37 |
| furniture_2 | heldout | 0.048 | 0.022 | 1.30 |
| instruments_2 | heldout | 0.096 | 0.000 | 5.14 |
| professions_2 | heldout | 0.158 | 0.096 | 1.61 |
| sports_2 | heldout | 0.128 | 0.074 | 2.82 |

### intersection_600 (35 heads, 8 MLP)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.074 | 0.030 | 3.11 |
| animals_2 | trained | 0.118 | 0.102 | 0.42 |
| colors_2 | trained | 0.096 | 0.064 | 2.50 |
| metals_2 | trained | 0.108 | 0.084 | 1.60 |
| vehicles_2 | trained | 0.042 | 0.016 | 0.56 |
| clothing_2 | heldout | 0.064 | 0.026 | 0.39 |
| furniture_2 | heldout | 0.034 | 0.016 | 1.33 |
| instruments_2 | heldout | 0.076 | 0.004 | 5.12 |
| professions_2 | heldout | 0.052 | 0.012 | 1.70 |
| sports_2 | heldout | 0.074 | 0.030 | 2.88 |

## 5. Anti-intersections — prob / gen / KL

### anti_intersection_frozen (465 heads, 8 MLP)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.056 | 0.008 | 3.19 |
| animals_2 | trained | 0.050 | 0.034 | 0.56 |
| colors_2 | trained | 0.070 | 0.064 | 2.68 |
| metals_2 | trained | 0.068 | 0.046 | 1.74 |
| vehicles_2 | trained | 0.040 | 0.018 | 0.64 |
| clothing_2 | heldout | 0.044 | 0.022 | 0.43 |
| furniture_2 | heldout | 0.046 | 0.010 | 1.40 |
| instruments_2 | heldout | 0.056 | 0.002 | 5.36 |
| professions_2 | heldout | 0.050 | 0.002 | 1.82 |
| sports_2 | heldout | 0.046 | 0.016 | 3.06 |

### anti_intersection_600 (477 heads, 8 MLP)
| Dataset | Split | Prob | Gen | KL |
|---|---|---|---|---|
| fruits_2 | trained | 0.048 | 0.010 | 3.21 |
| animals_2 | trained | 0.048 | 0.038 | 0.56 |
| colors_2 | trained | 0.070 | 0.062 | 2.69 |
| metals_2 | trained | 0.066 | 0.040 | 1.74 |
| vehicles_2 | trained | 0.046 | 0.016 | 0.63 |
| clothing_2 | heldout | 0.040 | 0.018 | 0.42 |
| furniture_2 | heldout | 0.038 | 0.010 | 1.40 |
| instruments_2 | heldout | 0.056 | 0.002 | 5.35 |
| professions_2 | heldout | 0.052 | 0.002 | 1.82 |
| sports_2 | heldout | 0.044 | 0.014 | 3.07 |

## 6. Findings (vs exp_2's 1.5× λ run)

1. **Normal λ fixes the training collapse**: combined phase-1 now reaches
   58–80% gen on trained categories (exp_2: 38–71%) and even 22–60% on
   *heldout* categories it never saw — the combined mask does capture a
   partly category-general circuit. Frozen individuals: 32–62% (exp_2: 2–48%),
   no per-dataset collapse.
2. **Necessity confirmed again, stronger**: both anti-intersections (~465+ heads,
   8 MLP blocks on) fail at ≤6.4% gen everywhere. The shared 35–47 heads +
   8 MLP blocks are necessary for every category, trained and heldout alike.
3. **Sufficiency still fails**: intersections reach only ~2–38% gen. Even with
   8/16 MLP blocks kept, half the MLP layers run on corrupted input, which the
   circuit cannot tolerate. The MLP-gating problem from exp_2 persists at any λ.
4. **Two-phase beats direct for intersection quality**: the frozen ∩
   (47 heads) consistently outperforms the 600 ∩ (35 heads) — e.g. trained-split
   gen means 0.165 vs 0.058, and animals 0.382 vs 0.102. The combined-then-frozen
   pipeline does push more of the truly shared structure into the intersection,
   supporting the original hypothesis — but the absolute level is still far
   from a working circuit.
5. Consistent picture across both exp_2 and exp_4: at heads+MLP-block
   granularity, the shared core is *necessary* but never *sufficient*;
   heads-only granularity (older experiments) was sufficient because MLPs
   always passed clean.
