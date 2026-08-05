# Intersection Circuit Similarity Analysis

This document analyzes the similarity between the 10 pairwise intersection circuits.

## Pairwise Similarity Matrix

| Circuit 1 | Circuit 2 | Size 1 | Size 2 | Intersection | Union | Jaccard | Overlap 1 | Overlap 2 |
|---|---|---|---|---|---|---|---|---|
| animals_vs_colors | animals_vs_fruits | 54 | 60 | 41 | 73 | 0.5616 | 0.7593 | 0.6833 |
| animals_vs_colors | animals_vs_metals | 54 | 56 | 43 | 67 | 0.6418 | 0.7963 | 0.7679 |
| animals_vs_colors | animals_vs_vehicles | 54 | 55 | 41 | 68 | 0.6029 | 0.7593 | 0.7455 |
| animals_vs_colors | colors_vs_fruits | 54 | 69 | 45 | 78 | 0.5769 | 0.8333 | 0.6522 |
| animals_vs_colors | colors_vs_metals | 54 | 63 | 43 | 74 | 0.5811 | 0.7963 | 0.6825 |
| animals_vs_colors | colors_vs_vehicles | 54 | 55 | 44 | 65 | 0.6769 | 0.8148 | 0.8000 |
| animals_vs_colors | fruits_vs_metals | 54 | 63 | 41 | 76 | 0.5395 | 0.7593 | 0.6508 |
| animals_vs_colors | fruits_vs_vehicles | 54 | 64 | 42 | 76 | 0.5526 | 0.7778 | 0.6562 |
| animals_vs_colors | metals_vs_vehicles | 54 | 66 | 45 | 75 | 0.6000 | 0.8333 | 0.6818 |
| animals_vs_fruits | animals_vs_metals | 60 | 56 | 44 | 72 | 0.6111 | 0.7333 | 0.7857 |
| animals_vs_fruits | animals_vs_vehicles | 60 | 55 | 46 | 69 | 0.6667 | 0.7667 | 0.8364 |
| animals_vs_fruits | colors_vs_fruits | 60 | 69 | 44 | 85 | 0.5176 | 0.7333 | 0.6377 |
| animals_vs_fruits | colors_vs_metals | 60 | 63 | 40 | 83 | 0.4819 | 0.6667 | 0.6349 |
| animals_vs_fruits | colors_vs_vehicles | 60 | 55 | 39 | 76 | 0.5132 | 0.6500 | 0.7091 |
| animals_vs_fruits | fruits_vs_metals | 60 | 63 | 44 | 79 | 0.5570 | 0.7333 | 0.6984 |
| animals_vs_fruits | fruits_vs_vehicles | 60 | 64 | 43 | 81 | 0.5309 | 0.7167 | 0.6719 |
| animals_vs_fruits | metals_vs_vehicles | 60 | 66 | 46 | 80 | 0.5750 | 0.7667 | 0.6970 |
| animals_vs_metals | animals_vs_vehicles | 56 | 55 | 44 | 67 | 0.6567 | 0.7857 | 0.8000 |
| animals_vs_metals | colors_vs_fruits | 56 | 69 | 43 | 82 | 0.5244 | 0.7679 | 0.6232 |
| animals_vs_metals | colors_vs_metals | 56 | 63 | 46 | 73 | 0.6301 | 0.8214 | 0.7302 |
| animals_vs_metals | colors_vs_vehicles | 56 | 55 | 42 | 69 | 0.6087 | 0.7500 | 0.7636 |
| animals_vs_metals | fruits_vs_metals | 56 | 63 | 46 | 73 | 0.6301 | 0.8214 | 0.7302 |
| animals_vs_metals | fruits_vs_vehicles | 56 | 64 | 45 | 75 | 0.6000 | 0.8036 | 0.7031 |
| animals_vs_metals | metals_vs_vehicles | 56 | 66 | 48 | 74 | 0.6486 | 0.8571 | 0.7273 |
| animals_vs_vehicles | colors_vs_fruits | 55 | 69 | 42 | 82 | 0.5122 | 0.7636 | 0.6087 |
| animals_vs_vehicles | colors_vs_metals | 55 | 63 | 42 | 76 | 0.5526 | 0.7636 | 0.6667 |
| animals_vs_vehicles | colors_vs_vehicles | 55 | 55 | 43 | 67 | 0.6418 | 0.7818 | 0.7818 |
| animals_vs_vehicles | fruits_vs_metals | 55 | 63 | 40 | 78 | 0.5128 | 0.7273 | 0.6349 |
| animals_vs_vehicles | fruits_vs_vehicles | 55 | 64 | 45 | 74 | 0.6081 | 0.8182 | 0.7031 |
| animals_vs_vehicles | metals_vs_vehicles | 55 | 66 | 44 | 77 | 0.5714 | 0.8000 | 0.6667 |
| colors_vs_fruits | colors_vs_metals | 69 | 63 | 49 | 83 | 0.5904 | 0.7101 | 0.7778 |
| colors_vs_fruits | colors_vs_vehicles | 69 | 55 | 45 | 79 | 0.5696 | 0.6522 | 0.8182 |
| colors_vs_fruits | fruits_vs_metals | 69 | 63 | 49 | 83 | 0.5904 | 0.7101 | 0.7778 |
| colors_vs_fruits | fruits_vs_vehicles | 69 | 64 | 50 | 83 | 0.6024 | 0.7246 | 0.7812 |
| colors_vs_fruits | metals_vs_vehicles | 69 | 66 | 48 | 87 | 0.5517 | 0.6957 | 0.7273 |
| colors_vs_metals | colors_vs_vehicles | 63 | 55 | 45 | 73 | 0.6164 | 0.7143 | 0.8182 |
| colors_vs_metals | fruits_vs_metals | 63 | 63 | 50 | 76 | 0.6579 | 0.7937 | 0.7937 |
| colors_vs_metals | fruits_vs_vehicles | 63 | 64 | 45 | 82 | 0.5488 | 0.7143 | 0.7031 |
| colors_vs_metals | metals_vs_vehicles | 63 | 66 | 48 | 81 | 0.5926 | 0.7619 | 0.7273 |
| colors_vs_vehicles | fruits_vs_metals | 55 | 63 | 42 | 76 | 0.5526 | 0.7636 | 0.6667 |
| colors_vs_vehicles | fruits_vs_vehicles | 55 | 64 | 47 | 72 | 0.6528 | 0.8545 | 0.7344 |
| colors_vs_vehicles | metals_vs_vehicles | 55 | 66 | 42 | 79 | 0.5316 | 0.7636 | 0.6364 |
| fruits_vs_metals | fruits_vs_vehicles | 63 | 64 | 45 | 82 | 0.5488 | 0.7143 | 0.7031 |
| fruits_vs_metals | metals_vs_vehicles | 63 | 66 | 48 | 81 | 0.5926 | 0.7619 | 0.7273 |
| fruits_vs_vehicles | metals_vs_vehicles | 64 | 66 | 46 | 84 | 0.5476 | 0.7188 | 0.6970 |

**Average Jaccard Similarity:** 0.5829

## Global Universal Circuit

The geometric intersection of ALL 10 circuits yields a core universal architecture.
**Total Core Heads:** 27

### Exact Head Indices `(Layer, Head)`
```json
[
  [
    1,
    22
  ],
  [
    2,
    26
  ],
  [
    2,
    27
  ],
  [
    2,
    29
  ],
  [
    3,
    11
  ],
  [
    4,
    12
  ],
  [
    4,
    16
  ],
  [
    5,
    8
  ],
  [
    5,
    10
  ],
  [
    6,
    0
  ],
  [
    6,
    6
  ],
  [
    7,
    23
  ],
  [
    8,
    19
  ],
  [
    8,
    26
  ],
  [
    9,
    8
  ],
  [
    9,
    16
  ],
  [
    9,
    18
  ],
  [
    9,
    27
  ],
  [
    10,
    1
  ],
  [
    10,
    20
  ],
  [
    10,
    21
  ],
  [
    10,
    23
  ],
  [
    12,
    13
  ],
  [
    14,
    15
  ],
  [
    14,
    28
  ],
  [
    15,
    1
  ],
  [
    15,
    14
  ]
]
```
