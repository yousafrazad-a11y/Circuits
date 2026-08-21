# Joint Automatic Position Discovery and Hierarchical Node Pruning

## 1. Scope

This document specifies the implemented automatic-position method closely enough to recreate its architecture, objective, data flow, training run, checkpoint semantics, evaluation, and structural reporting. It describes the code and the completed 2,000-epoch run, not a hypothetical redesign.

The method jointly learns:

1. a token router that assigns every valid token to one of `K=9` reusable learned positions; and
2. a complete hierarchical node-pruning mask for each learned position.

The learned positions are categorical circuit identities. They are not required to be contiguous spans, absolute token indices, token types, or the nine human IOI sections.

## 2. Base model and frozen components

The model is Hugging Face `gpt2` (GPT-2 small):

- 12 transformer blocks;
- hidden width `d_model=768`;
- 12 attention heads per block;
- head width `d_head=64`;
- MLP hidden width `d_mlp=3072`;
- vocabulary size 50,257;
- maximum experiment sequence length 64.

All pretrained GPT-2 parameters are frozen. Dropout is set to zero in both the frozen teacher/feature model and the prunable circuit. Only router parameters and hard-concrete `log_alpha` gate parameters receive gradients.

## 3. Data and preprocessing

### 3.1 Exact files and split behavior

The run uses `comparison_experiments/data`:

- train: first 500 rows of `discovery.csv`;
- validation: first 100 rows of `discovery.csv`;
- test: first 500 rows of `evaluation.csv`.

Consequently, the current validation set is a subset of the training set. This is suitable for internal checkpoint monitoring but is not a publication-quality train/validation separation. The test set is a separate file and is used only after checkpoint selection/training.

All requested splits are restricted to ABBA IOI examples. Before constructing the final datasets, the frozen base GPT-2 must assign the correct IO token a logit at least as large as the distractor logit. The completed run retained all requested `500/100/500` rows.

### 3.2 Tokenization

The tokenizer is `GPT2TokenizerFast` for `gpt2`. The padding token is set to the EOS token. Canonical rows receive the GPT-2 BOS token before the stored prompt tokens. Prompts are right-padded to length 64, and clean/corrupted sequences must have equal aligned lengths. Correct and distractor answers must each tokenize to one token.

The answer-prediction location for example `b` is:

```text
t_answer(b) = T_Start(b) - 1
```

All behavioral losses and metrics use the full 50,257-dimensional logits at this location.

### 3.3 Human sections used only for analysis

The dataset retains the PEAP-compatible labels:

```text
prefix, IO, and, S1, S1+1, action1, S2, action2, to
```

These labels are not router inputs and do not determine the learned assignments. They are used to produce the section-to-learned-position diagnostic matrix.

## 4. Frozen router evidence and cache

For token `t` in an example of valid length `L`, normalized relative position is:

```text
p_t = t / max(L - 1, 1)
```

Padding positions are multiplied by the attention mask. The configured frozen residual is `hidden_states[2]`, a 768-dimensional tensor. In Hugging Face indexing, `hidden_states[0]` is the embedding residual, so index 2 is the residual after two transformer blocks (blocks with zero-based indices 0 and 1).

The router input is:

```text
x_t = concat(hidden_states[2]_t, p_t) in R^769
```

The frozen tensor is detached; no gradient enters GPT-2.

Before optimization, one frozen GPT-2 pass per cache batch produces both:

- the `[64,769]` router feature tensor for each example; and
- the `[50257]` teacher-logit vector at `T_Start-1`.

Values are stored on the active device in dictionaries keyed by dataset `source_index`. This preserves identity under shuffled training. After all `500+100+500` examples are cached, the frozen GPT-2 object is deleted and the CUDA allocator cache is cleared. No frozen-model forward pass occurs during an epoch, validation, or final testing.

## 5. Token router

The same router is applied independently to every token in every example:

```text
LayerNorm(769)
Linear(769, 64)
GELU
Linear(64, 9)
```

Let its logits be `r_t in R^9`. The router has exactly 51,403 trainable parameters:

```text
LayerNorm:          2 * 769       = 1,538
Linear 769 -> 64:  769 * 64 + 64 = 49,280
Linear 64 -> 9:     64 * 9 + 9   =   585
Total:                              51,403
```

### 5.1 Training assignment

Training uses PyTorch straight-through hard Gumbel-Softmax:

```text
z_t = gumbel_softmax(r_t, tau, hard=True)
```

The forward value is exactly one-hot (`z_t in {0,1}^9`, `sum_k z_tk=1`). The backward value uses the soft Gumbel-Softmax derivative.

Temperature is exponentially annealed by epoch:

```text
f_e   = e / (E - 1),       e = 0,...,E-1
tau_e = 2.0 * exp(log(0.25 / 2.0) * f_e)
```

For the completed run, `E=2000`.

### 5.2 Evaluation assignment

Evaluation computes:

```text
q_t = softmax(r_t / 0.25)
z_t = one_hot(argmax_k r_tk)
```

Thus evaluation routing is deterministic and exactly one-hot. `q_t` is retained only for entropy, maximum-probability, and top-two-margin diagnostics.

## 6. Nine learned mask experts

Each learned position owns one full row of gates across all transformer blocks. For one position and one GPT-2 block, the gates are:

| Gate family | Shape per position per block | Count |
|---|---:|---:|
| Attention block | scalar | 1 |
| MLP block | scalar | 1 |
| Attention heads | `[12]` | 12 |
| Attention neurons | `[12,64]` | 768 |
| MLP hidden neurons | `[3072]` | 3,072 |
| MLP output dimensions | `[768]` | 768 |
| Total |  | 4,622 |

Across 12 blocks there are `4,622*12=55,464` gates per learned position. Across nine positions there are `499,176` gate parameters. Including the router, the method trains `550,579` scalar parameters.

Full-layer and embedding gates are disabled.

## 7. Hard-concrete gates

Every gate has one trainable `log_alpha`. Constants are:

```text
beta  = 2/3
gamma = -0.1
zeta  = 1.1
log_alpha initialization ~ Uniform(2.5, 3.5)
```

During training, for `u ~ Uniform(1e-8, 1-1e-8)`:

```text
noise  = log(u) - log(1-u)
s      = sigmoid((noise + log_alpha) / beta)
s_bar  = s * (zeta - gamma) + gamma
g_soft = clamp(s_bar, 0, 1)
g_hard = 1[g_soft > 0.5]
g      = stop_gradient(g_hard - g_soft) + g_soft
```

The forward pass is binary while gradients follow `g_soft`. In evaluation, noise is removed and the implementation uses:

```text
g_eval = 1[clamp(sigmoid(log_alpha)*(zeta-gamma)+gamma,0,1) > 0.5]
```

The expected probability of being open, used in the sparsity loss, is:

```text
P_open = sigmoid(log_alpha - beta * log(-gamma/zeta))
```

## 8. Routing a token to a mask

For any gate family, let `G in R^(9 x ...)` be its nine expert rows. The effective gate tensor for token `t` is:

```text
g_t = sum_(k=0)^8 z_tk * G_k
```

This operation is implemented by a tensor contraction between `[batch,token,9]` routing weights and the expert dimension of each gate. Because `z_t` is one-hot in the forward pass, each token uses exactly one complete mask row. The same nine rows are shared across all examples.

## 9. Prunable clean/corrupted computation

The prunable GPT-2 runs aligned clean and corrupted streams. At every gated unit:

```text
y = g * y_clean + (1-g) * y_corrupted
```

Therefore, a closed node is replaced by its activation from the corrupted prompt rather than by zero.

Within attention, clean and corrupted activations are separated before the output projection, reshaped to `[batch,token,12,64]`, and gated first by head and then neuron gates. The projected attention output is subsequently gated by the attention-block gate.

Within the MLP, post-activation hidden neurons are gated before `c_proj`; projected output dimensions are gated afterward; the complete output is subsequently gated by the MLP-block gate. Residual additions remain present.

Base GPT-2 weights are frozen, but gradients can reach the router and gates through all these mixtures.

## 10. Hierarchy semantics for circuit size

Training contains separately parameterized gates at every enabled level. Structural evaluation hardens them and enforces:

1. a closed layer closes both blocks and all descendants (layer gates are disabled in this run);
2. a closed attention block closes every head and attention neuron;
3. a closed head closes its 64 attention neurons;
4. a head with no surviving neurons is closed;
5. an attention block with no surviving heads is closed;
6. a closed MLP block closes its hidden and output neurons;
7. an MLP block is closed if either its hidden or output side is empty.

Reported hierarchy-effective counts use these rules. Granularities are nested and must not be summed.

## 11. Exact training objective

For circuit answer logits `c`, frozen full-model answer logits `f`, correct token `y+`, and distractor `y-`:

### 11.1 Faithfulness KL

```text
L_KL = D_KL(softmax(f) || softmax(c))
```

This direction follows PyTorch `kl_div(log_softmax(c), log_softmax(f), log_target=True)`.

### 11.2 IOI margin task loss

```text
m      = c[y+] - c[y-]
L_task = mean(ReLU(4 - m))
```

### 11.3 Router confidence

For valid-token probabilities `q_t=softmax(r_t/tau)`:

```text
L_conf = mean_t H(q_t) = mean_t[-sum_k q_tk log(q_tk)]
```

Its epoch coefficient scale is zero for the first 20% of training, then increases linearly:

```text
c_conf(f) = 0                              if f <= 0.2
            (f - 0.2)/(1 - 0.2)           otherwise
```

### 11.4 Load balance

Let `q_bar_k` be mean soft usage across valid tokens in a batch and `K=9`:

```text
L_balance = K * sum_k q_bar_k^2 - 1
```

It is zero at uniform usage and increases with concentration. It does not force equal hard counts.

### 11.5 Sparsity

For block index `l in {0,...,11}`, the depth multiplier is:

```text
d_l = 1 + 0.1 * (11-l)/11
```

Sparsity warmup by optimizer step `s` is:

```text
w_s = min(1, s/1000)
```

The expected-density coefficients are:

| Family | Lambda |
|---|---:|
| Attention block | 0.5 |
| MLP block | 0.5 |
| Attention heads | 0.8 |
| Attention neurons | 0.15 |
| MLP hidden | 1.0 |
| MLP output | 1.0 |

For each of the 72 `(12 blocks x 6 families)` gate modules, the implementation computes the mean `P_open` over all nine positions and components, multiplies it by `lambda * w_s * d_l`, sums the 72 terms, and divides by 72. This result is `L_sparsity`.

### 11.6 Total loss for the completed run

```text
L = 1.5*L_KL
  + L_task
  + 4.0*L_sparsity
  + 0.01*c_conf(f)*L_conf
  + 0.01*L_balance
```

## 12. Optimization and schedules

The optimizer is PyTorch `AdamW` with two parameter groups:

- router learning rate: `0.002`;
- hard-concrete gate learning rate: `0.06`.

Unspecified AdamW values are library defaults (`betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`). There is no learning-rate scheduler and no gradient clipping.

Run settings:

| Setting | Value |
|---|---:|
| Epochs | 2,000 |
| Training examples | 500 |
| Batch size | 64 |
| Batches per epoch | 8 |
| Total optimizer steps | 16,000 |
| Validation interval | 5 epochs |
| Seed | 42 |
| Data-loader workers | 2 |
| Router mode | hard Gumbel-Softmax |
| Temperature | exponential 2.0 -> 0.25 |
| Confidence warmup fraction | 0.2 |
| Minimum accuracy for sparsest-checkpoint selection | 0.95 |

Python, CPU torch, and CUDA random seeds are set to 42. Training loader shuffling is enabled. Validation and test loaders are not shuffled.

## 13. Checkpoints and selection

Every five epochs, the script evaluates deterministic hard routing and soft routing on validation and saves:

- epoch and optimizer-step count;
- router state;
- all gate parameters;
- optimizer state;
- command-line arguments;
- hard/soft validation metrics;
- raw gate density.

Two aliases are maintained:

- `best_kl.pt`: lowest hard-routing validation KL seen so far;
- `best.pt`: smallest raw hard-gate density among checkpoints with validation accuracy at least 0.95.

After epoch 2000, the script first evaluates the literal final model on test. It then loads `best.pt` and separately evaluates that checkpoint. For this run, `best.pt` is epoch 1995; all results explicitly called “epoch 2000” use `epoch_2000.pt` and `final_router_*_test`.

## 14. Metrics

At the answer position:

- accuracy: circuit vocabulary argmax equals the correct IO token;
- pairwise accuracy: `c[y+] - c[y-] >= 0`;
- logit difference: mean `c[y+] - c[y-]`;
- full-model logit difference: the same quantity for frozen GPT-2;
- soft faithfulness: circuit logit difference divided by full-model logit difference;
- KL divergence: mean `D_KL(full || circuit)` over examples;
- exact match: circuit vocabulary argmax equals full-model vocabulary argmax.

Router diagnostics are computed only over valid tokens:

- entropy of soft probabilities;
- maximum probability;
- top-one minus top-two probability margin;
- hard expert counts and usage;
- human-section by learned-position token-count matrix.

## 15. Structural metrics

`automatic_positions_project/evaluate_structure.py` loads a checkpoint, rebuilds the nine-position circuit, loads gate parameters, hardens gates, applies the hierarchy rules, and reports:

- effective position-specific blocks, heads, and neurons;
- physical union across positions;
- induced high-level directed edges using the adapted `circuit_pruning-argo` convention;
- an extractable-parameter proxy.

The physical union counts a component as active if any learned position uses it. The parameter metric is a structural extraction proxy; the evaluated PyTorch model remains physically dense.

## 16. Exact reproduction commands

From repository root:

```bash
comparison_experiments/node_eval_venv/bin/python -u -m automatic_positions_project.train_joint_positions \
  --router-input early_residual_plus_position \
  --output-dir automatic_positions_project/results/joint_9pos_2000ep_pressure4x_lr2x_bs64_cached_seed42 \
  --data-path /home/exouser/pruning/some_random_test/comparison_experiments/data \
  --model gpt2 \
  --num-positions 9 \
  --gate-learning-rate 0.06 \
  --global-sparsity-multiplier 4.0 \
  --confidence-weight 0.01 \
  --selection-min-accuracy 0.95 \
  --confidence-warmup-fraction 0.2 \
  --epochs 2000 \
  --batch-size 64 \
  --learning-rate 0.002 \
  --hidden-size 64 \
  --residual-layer 2 \
  --temperature-start 2.0 \
  --temperature-end 0.25 \
  --validation-interval 5 \
  --seed 42 \
  --num-workers 2 \
  --train-routing hard_gumbel \
  --load-balance-weight 0.01
```

Generate hierarchy-aware structure for the literal final checkpoint:

```bash
comparison_experiments/node_eval_venv/bin/python -m automatic_positions_project.evaluate_structure \
  automatic_positions_project/results/joint_9pos_2000ep_pressure4x_lr2x_bs64_cached_seed42/epoch_2000.pt \
  --output automatic_positions_project/results/joint_9pos_2000ep_pressure4x_lr2x_bs64_cached_seed42/structure.metrics.json
```

## 17. Completed-run result anchor

Literal epoch 2000, 500-example held-out test:

| Metric | Value |
|---|---:|
| KL | 0.1106354103 |
| Accuracy | 0.988 |
| Pairwise accuracy | 1.000 |
| Logit difference | 5.8669071045 |
| Full-model logit difference | 4.2677292480 |
| Soft faithfulness | 1.3747139904 |
| Raw gates pruned | 95.0708768050% |
| Hierarchy-effective attention neurons pruned | 95.0749903549% |
| Hierarchy-effective combined MLP neurons pruned | 95.1550443673% |
| High-level edges pruned | 98.6210604984% |

## 18. Environment and source identity

Observed environment:

```text
Python       3.12.3
PyTorch      2.13.0+cu130
CUDA build   13.0
Transformers 5.15.1
Tokenizers   0.22.2
```

SHA-256 source hashes used for this report:

```text
8dbf16cbdee6dc9066acaee495f094036ac2a1168704139463aaf9ac5fd54795  automatic_positions_project/train_joint_positions.py
ace3b73fb8484ca05e623b67dc0f160c5a4ed635725e98e04365a1a55b6e1003  automatic_positions_project/src/router.py
776d060afdebddb009100f61da5007c65d554ec8f830993ebd31c8dfa2871d91  automatic_positions_project/src/routed_circuit.py
ce2904c912a6dd3c276090d1263d68f64c9176878546609b969ca28f92acea0a  comparison_experiments/position_aware_node_pruning/models/l0.py
39ceb4d000c5a2346f3ae8f0d72dee5e72bfeab197da381b20b722f2549622d9  comparison_experiments/position_aware_node_pruning/models/gpt2_circuit.py
ecd9e624f51c55c3cc68598fe8aa2624a08b087710a7570d981ee9d800f29043  comparison_experiments/position_aware_node_pruning/dataset/ioi.py
56cba4b39143a9317d32822c0cd84f4fb823722d332f1ae31a4aabce7eefe112  comparison_experiments/common_metrics.py
```

Dataset and final-checkpoint hashes:

```text
22694614963c9fdc208bc86356326b4e30d7925e9e0ffd3df40190771cb9e1f0  comparison_experiments/data/discovery.csv
32c9d68402c3986d6686b58d6e601bffa40ce7713c5475d28ba86a2075219bee  comparison_experiments/data/evaluation.csv
855aae14bcb85b429e06f57201e37495a82408e2873295b13b8d80f79354da21  automatic_positions_project/results/joint_9pos_2000ep_pressure4x_lr2x_bs64_cached_seed42/epoch_2000.pt
```

## 19. Known limitations to address before publication

1. Validation overlaps training under the current CSV slicing behavior. Create a disjoint validation split before publication.
2. Results currently use one seed. Repeat across several seeds and report uncertainty.
3. Position labels are permutation-invariant. Cross-run comparisons should align positions by assignment or circuit similarity rather than numeric ID.
4. The cache is held on the active device and trades memory for speed. A CPU or reduced-precision cache should be tested for larger datasets.
5. The model remains dense at runtime. Structural percentages describe a circuit/extraction proxy, not measured sparse-kernel inference speed.
6. Automatic positions were tested only on IOI here. Demonstrating task and dataset generality requires additional behaviors and variable-length distributions.
