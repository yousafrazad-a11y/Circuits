# Experiment Audit and Future Fixes

## Status

The current results are suitable for exploratory comparison between methods and
for deciding whether the project is promising. The reported outputs were
actually produced by the saved circuits on the stated examples; no metric is
fabricated, rescaled differently between methods, or evaluated at a different
answer position.

They should not yet be described as a locked confirmatory benchmark. Some
choices were made after inspecting development/evaluation performance. This can
make the selected operating points look better than a completely unseen run,
but it does not invalidate the circuits or their measured behavior on these
examples.

## Confirmed Fair and Correct

- Normal node pruning, position-aware node pruning, and PEAP use exactly the
  same 500 discovery prompts and paired counterfactuals.
- They use exactly the same 500 evaluation prompts and paired
  counterfactuals.
- The discovery and evaluation prompt sets have zero overlap.
- All 1,000 selected prompts are solved by the base GPT-2 model.
- Node pruning uses Hugging Face GPT-2 and PEAP uses TransformerLens GPT-2 with
  compatible non-folded/non-centered settings. Their base logit-difference
  results agree to numerical precision.
- Final shared metrics are computed at the final answer-prediction position.
- KL is `D_KL(full || circuit)` over the complete vocabulary, summed per
  example and averaged over examples for every backend.
- Pairwise accuracy is `logit(correct) >= logit(wrong)`.
- Top-1 accuracy means the circuit's vocabulary-wide argmax is the correct IOI
  token.
- Exact match means the circuit and full model have the same vocabulary-wide
  argmax.
- Logit difference and soft faithfulness use the same definitions for every
  backend.
- The PEAP attribution and circuit-ablation algorithms are unchanged from the
  repository. Additions only save recovery files, circuit artifacts, and
  shared metrics.
- PEAP uses the paper-compatible GPT-2 IOI setup: 500 discovery examples, 500
  evaluation examples, the human nine-span ABBA schema, paired
  counterfactuals, absolute-score connected circuit construction, positional
  crossing edges, and query patching.
- Normal node pruning retains circuit_pruning-argo's dual-stream hard-concrete
  method and objective. Position-aware node pruning applies the same gates and
  losses with an additional logical-section dimension.

## Exploratory Selection Caveats

### Node validation overlap

The node loader currently uses the first 500 rows of `discovery.csv` for
training and the first 100 rows of the same file for validation. Validation is
therefore a subset of training.

Effect: validation curves and validation-based checkpoint choices can be
optimistic. This does not put final evaluation prompts into gradient training,
because `evaluation.csv` is disjoint.

### Repeated inspection of evaluation results

PEAP top-k choices and later node pruning pressures were informed by results on
the present evaluation collection. The final 500-example measurements are
therefore exploratory/development results rather than a single-shot estimate on
an untouched test set.

Effect: the numerical measurements on these examples are real, but selecting
the best checkpoint/top-k after seeing them introduces selection optimism.

### Continuation semantics

Node continuations load learned gate logits but start a new AdamW optimizer and
reset the sparsity-warmup step counter. They are new optimization phases from
the preceding masks, not exact optimizer-state resumes.

This is a valid pruning procedure, but it must be described as staged training.

### Structural edge percentage

PEAP's native size is a concrete directed-edge count. The shared 109,722-edge
denominator is an argo-derived high-level abstract comparison proxy. It applies
the same dense-survivor rule to head/MLP nodes, collapses Q/K/V for PEAP
compatibility, and adds causal logical-position attention transport.

This proxy is useful for method comparison but is not physical parameter
compression and is not a native metric reported by either original method.

## Interpretation of Current Results

The current results support statements such as:

> In exploratory experiments on a shared IOI discovery/evaluation collection,
> the saved circuits achieved the following measured performance and sparsity.

They should not yet support statements such as:

> These are unbiased final estimates on a test set that played no role in any
> model, checkpoint, hyperparameter, or circuit-size decision.

## Required Fixes for a Confirmatory Run

1. Generate a separate calibration set and a new locked test set.
2. Ensure train, calibration, and test prompts are all disjoint.
3. Select node checkpoints and PEAP top-k only on calibration.
4. Freeze all hyperparameters and selection rules before opening the test set.
5. Evaluate each selected circuit once on the locked test set with
   `comparison_experiments/common_metrics.py`.
6. Save dataset hashes, exact commands, software versions, seeds, checkpoint
   hashes, and the selection rule.
7. Label native structural sizes separately from the argo-derived abstract edge
   proxy.
8. If exact continuation is desired, restore optimizer/RNG/total-step state;
   otherwise explicitly call it staged continuation.

## Bottom Line

There is no identified formula or implementation error that artificially gives
PEAP, normal node pruning, or position-aware node pruning better behavioral
metrics than another method. PEAP is technically usable for the present
exploratory comparison. The remaining issues concern independence of model
selection and the interpretation of derived circuit-size proxies, and should be
fixed before a final confirmatory experiment.
