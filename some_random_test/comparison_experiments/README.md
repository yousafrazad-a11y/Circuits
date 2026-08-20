# Controlled IOI comparison

This workspace compares three circuit-discovery methods without modifying the
original folders:

- `position_aware_node_pruning`: learned hard-concrete node gates indexed by
  PEAP's human-defined IOI spans.
- `non_position_node_pruning`: the same trainer, losses, and data with one
  global gate row.
- `position_aware_eap`: PEAP, copied from the authors' repository, with common
  evaluation metrics added.

## Shared data

`data/discovery.csv` contains 500 GPT-2-correct ABBA examples and
`data/evaluation.csv` contains a disjoint 500. They were generated with seed 42
from PEAP's 15 templates, names, places, objects, and three-unrelated-name ABC
counterfactual intervention. `data/peap/` contains lossless PEAP CSV adapters.

The canonical files are BOS-free. Both execution adapters prepend GPT-2 BOS: the
node dataset adds the BOS token to `prefix`, while PEAP's CSV adds one to every
start index and to `length` for TransformerLens. The operative spans are:

`prefix IO and S1 S1+1 action1 S2 action2 to`

PEAP's CLI also receives `length`; it is a full-prompt aggregation marker, not
a tenth span. Node pruning therefore learns nine gate rows.

Regenerate the data with:

```bash
python -m comparison_experiments.prepare_ioi_data
```

## Node-pruning runs

From the repository root, with the project ML environment active:

```bash
python -m comparison_experiments.position_aware_node_pruning.ioi
python -m comparison_experiments.non_position_node_pruning.ioi
```

Both use all 500 discovery rows, a deterministic 100-row discovery view for
periodic monitoring, and all 500 evaluation rows for final metrics. Their JSONL
metrics contain top-token `accuracy`, `pairwise_accuracy`, `logit_diff`,
`kl_div`, exact agreement with GPT-2, and soft faithfulness.

Both 500-epoch entry points save resumable continuous-log-alpha checkpoints at
every validation cycle (every 10 epochs) under their own `outputs/checkpoints/`
directory. Select an epoch using validation metrics only, then report its
behavior once on the held-out 500-row evaluation set. Do not select using final
evaluation metrics.

## PEAP discovery and evaluation

Use the dependency versions in `position_aware_eap/environment.yml`, then:

```bash
PYTHONPATH=comparison_experiments/position_aware_eap/src \
python comparison_experiments/position_aware_eap/src/pos_aware_edge_attribution_patching.py \
  -at counterfactual -e ioi -m gpt2 \
  -cl comparison_experiments/data/peap/IOI_ABBA_data_clean.csv \
  -co comparison_experiments/data/peap/IOI_ABBA_data_counter_abc.csv \
  -sp prefix IO and S1 S1+1 action1 S2 action2 to length \
  -ds 500 -s 42 -p comparison_experiments/results/peap_scores.pkl
```

Evaluate discovered circuits (example top-k values shown):

```bash
PYTHONPATH=comparison_experiments/position_aware_eap/src \
python comparison_experiments/position_aware_eap/src/eval.py \
  -at counterfactual -e ioi -m gpt2 \
  -cl comparison_experiments/data/peap/IOI_ABBA_data_clean.csv \
  -co comparison_experiments/data/peap/IOI_ABBA_data_counter_abc.csv \
  -spn prefix IO and S1 S1+1 action1 S2 action2 to length \
  -n 500 -as 1 -tk 100 200 300 -s 42 -st abs -gpos \
  -p comparison_experiments/results/peap_scores.pkl \
  -sp comparison_experiments/results/peap_eval.pkl
```

`-as 1` makes faithfulness use each row's paired ABC counterfactual, matching
the intervention used by node pruning and avoiding PEAP's separate 24,000-row
mean-ablation pool. PEAP writes a sibling `peap_eval.metrics.json` containing,
for every top-k circuit: top-token accuracy, exact agreement with full GPT-2,
mean correct-minus-wrong logit difference, soft faithfulness, KL divergence to
full GPT-2, and mean concrete edge count.

PEAP has no epochs or optimizer checkpoints. One attribution run produces a
fixed score table, and `top_k` is its pruning/selection path. Evaluation now
saves each candidate graph separately as
`peap_eval.topk_<k>.circuit.pkl`, so a selected circuit is reproducible without
confusing it with an intermediate training state.

For matched comparisons, choose candidates by a predeclared behavioral target
(for example, highest compression/lowest edge count subject to an accuracy or
KL budget). Node slots and PEAP edges are different structural units, so matching
their raw percentages would not be meaningful.

## Interpretation

All methods now share prompts, counterfactuals, target/distractor tokens,
discovery/evaluation IDs, and semantic spans. They do not share circuit units:
node pruning reports active node slots/components, while PEAP reports edges.
Compare behavioral metrics directly, but report node and edge sizes in separate
columns rather than treating them as the same sparsity measure.

## Unified held-out evaluator

All three saved circuit types can be tested through one CLI. It uses the same
ordered 500-row held-out set and one shared implementation of accuracy,
pairwise accuracy, logit difference, full-model logit difference, soft
faithfulness, exact match, and `D_KL(full || circuit)`. Only the forward adapter
differs: node masks require the Hugging Face dual stream, while PEAP requires
TransformerLens graph hooks.

```bash
python -m comparison_experiments.evaluate_circuits --method global-node
python -m comparison_experiments.evaluate_circuits --method position-node

comparison_experiments/position_aware_eap/.venv/bin/python \
  -m comparison_experiments.evaluate_circuits --method peap
```

Use `--limit 10` for a smoke test and `--circuit PATH` to evaluate a selected
epoch checkpoint or another PEAP top-k graph. JSON results are written under
`comparison_experiments/results/`.

Each result also includes a `structure` object. Node methods report effective
closed/active percentages for blocks, heads, attention neurons, MLP hidden
neurons, and MLP output dimensions after propagating every parent closure. The
position-aware method additionally reports every section and the physical union
across sections. Its parameter percentage is an extraction proxy over base
GPT-2 parameters and excludes the learned gates themselves.

PEAP's native unit is an edge rather than a neuron. Its primary percentage is
the mean selected concrete token-level edges divided by the mean full
computation-graph edges on the exact evaluation prompts. Abstract graph counts
are also broken down by component and crossing-position relation. Do not compare
the PEAP edge percentage numerically as though it were node-neuron sparsity.

To add structural fields to an already completed PEAP evaluation without
repeating circuit inference:

```bash
comparison_experiments/position_aware_eap/.venv/bin/python \
  -m comparison_experiments.evaluate_circuits --method peap \
  --structure-only \
  --output comparison_experiments/results/unified_peap.metrics.json
```
