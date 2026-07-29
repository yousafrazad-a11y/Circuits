# Pairwise Intersection Similarity — full-depth masks

All 10 category pairs intersected (AND over heads, attention neurons, MLP hidden, MLP output gates). For each pair: gate counts of the pairwise ∩, and its mean similarity (Jaccard) to the other 9 pairwise ∩s.

## 300-combined + 300-finetune (frozen)

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

sim_X = mean Jaccard of this pair's intersection against the other 9 pairwise intersections.

**Overall similarity among the 10 pairwise intersections (Jaccard over all 45 pair-combinations):**

| gate type | mean J | min | max |
|---|---|---|---|
| head_gates | 0.871 | 0.754 | 1.000 |
| neuron_gates | 0.635 | 0.467 | 0.783 |
| hidden_gates | 0.529 | 0.336 | 0.765 |
| output_gates | 0.683 | 0.502 | 0.903 |

**5-way ∩ vs union of pairwise ∩s:**

| gate type | 5-way ∩ | union of 2-way ∩s | retained |
|---|---|---|---|
| head_gates | 52 | 70 | 0.74 |
| neuron_gates | 1032 | 2760 | 0.37 |
| hidden_gates | 5105 | 21102 | 0.24 |
| output_gates | 5257 | 11828 | 0.44 |

## 600-epoch individual (scratch)

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

sim_X = mean Jaccard of this pair's intersection against the other 9 pairwise intersections.

**Overall similarity among the 10 pairwise intersections (Jaccard over all 45 pair-combinations):**

| gate type | mean J | min | max |
|---|---|---|---|
| head_gates | 0.614 | 0.367 | 0.872 |
| neuron_gates | 0.454 | 0.261 | 0.672 |
| hidden_gates | 0.298 | 0.161 | 0.458 |
| output_gates | 0.401 | 0.223 | 0.611 |

**5-way ∩ vs union of pairwise ∩s:**

| gate type | 5-way ∩ | union of 2-way ∩s | retained |
|---|---|---|---|
| head_gates | 29 | 95 | 0.31 |
| neuron_gates | 593 | 3652 | 0.16 |
| hidden_gates | 1906 | 28761 | 0.07 |
| output_gates | 2024 | 16462 | 0.12 |
