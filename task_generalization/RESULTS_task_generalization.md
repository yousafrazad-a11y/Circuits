# Task Generalization: MLP-block corruption on IOI and GT (Llama-3.2-1B)

Date: 2026-07-29. Question: does the exp_4 finding — "corrupting MLP blocks
(all heads on) degrades performance, and MLP gates carry no task-specific
content" — replicate on standard circuit tasks (IOI, GT)?

Method: `mlp_ablation_ioi.py` / `mlp_ablation_gt.py`. Eval sets filtered to
examples the full model answers correctly. All attention gates open; only
`mlp_block_gate` varies (OFF = that layer's MLP output replaced by the
corrupted-stream activation, using the task's own corrupted prompts).

## IOI (N=400; 898/900 passed base-correctness filter)

| Condition | Accuracy | Logit diff (IO−S) |
|---|---|---|
| Full model / gates-open | 1.000 | 5.89 |
| Fruits MLP pattern (layers 1,6,7 off) | 0.985 | 3.94 |
| Random 1 layer (3 seeds) | 0.996 | 5.36 |
| Random 3 layers (3 seeds) | 0.73 / 0.99 / 0.93 (mean 0.884) | 3.09 |
| Random 8 layers (3 seeds) | 0.12 / 0.39 / 0.00 (mean 0.168) | −3.32 |

## GT (N=400; base model only 53.8% unfiltered — weak signal, suggestive only)

| Condition | Accuracy | Prob diff |
|---|---|---|
| Full model / gates-open | 1.000 | 0.039 |
| Fruits MLP pattern | 0.890 | 0.035 |
| Random 1 layer | 0.950 | 0.038 |
| Random 3 layers | 0.894 | 0.034 |
| Random 8 layers | 0.790 | 0.027 |

## Findings

1. **The effect transfers**: MLP-block corruption degrades IOI and GT
   monotonically in the number of corrupted layers. MLP mass matters on
   standard tasks too.
2. **The fruits MLP set is not special**: on both tasks it is
   indistinguishable from a benign random 3-layer draw (IOI: 0.985 inside
   0.73–0.99; GT: 0.890 vs random-3 mean 0.894). Confirms "MLP gates =
   fungible compute, no task/category content" on independent tasks.
3. **Tasks differ in MLP-fragility**: our sequence-tracking task lost ~40
   points from 3 corrupted layers (0.755→0.456); IOI loses only ~12
   (1.0→0.88). IOI is attention-head-centric and tolerates MLP damage better.
4. **Layer position matters**: damage is strongly layer-dependent (IOI layers
   ~9–13 most sensitive; {12,10,9} alone → 0.73).

Note: GT on Llama required mapping two-digit tokens without the leading space
(" 42" splits into ['Ġ','42'] with the Llama tokenizer; bare "42" is one token).
