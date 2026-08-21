# Automatic Position Discovery for Circuit Pruning

This project is intentionally isolated from `comparison_experiments`.  Its
first goal is not full end-to-end automation.  It asks a smaller question:

> Can a small learned router assign each token to the useful position-specific
> pruning mask without being told the hand-written IOI section label?

## Why start with a router probe?

The existing position-aware circuit already provides nine trained masks.  They
act like nine specialist subnetworks.  We freeze those masks and GPT-2, remove
the hand-written `section_ids`, and train only a small router that selects (or
softly mixes) a specialist for every real token.

This separates two problems that would otherwise obscure each other:

1. discovering meaningful token groups;
2. discovering what each group's circuit should contain.

If the router cannot recover useful assignments when good specialist masks are
already available, jointly learning assignments and masks is unlikely to help.
If it succeeds, the next experiment can safely unfreeze the masks.

## First experiment

Use the strongest usable nine-section checkpoint as frozen experts.  For each
token, the router emits nine scores and forms a weighted mixture of the nine
expert masks. Padding tokens are ignored. Variable prompt length therefore does
not require equal-length examples.

Train the router with the same full-model KL and IOI task loss used by the node
experiments. During the diagnostic experiment, retain the known section labels
only for evaluation; never feed them to the router.

Run four input ablations:

| Router input | Question answered |
|---|---|
| Normalized token index only | Is ordinary relative position sufficient? |
| Token embedding only | Are words/token identities sufficient? |
| Frozen early residual state only | Does early context reveal the role? |
| Residual state + normalized index | Does content plus location work best? |

The early residual should initially come from `resid_pre` at layer 1 or 2. It
contains token identity plus a small amount of context, but is still cheap and
does not let the router inspect the answer produced by later layers.

## Success criteria

The pilot counts as positive evidence if an unlabeled router:

- retains at least 95% pairwise IOI accuracy;
- stays close to the frozen oracle circuit's KL;
- uses stable assignments across random seeds;
- produces a small number of consistently used groups;
- generalizes to held-out names and prompt lengths;
- aligns with known semantic sections better than position-only baselines.

Section agreement is diagnostic, not the objective. A router may discover a
different partition that performs better than the human sections.

## Progression after a successful probe

1. Freeze expert masks; learn only the router.
2. Unfreeze router and masks with a low mask learning rate.
3. Add a penalty for unused or redundant experts.
4. Allow adjacent experts to merge.
5. Only then learn the number of groups or boundaries end to end.

## Directory layout

- `experiments/router_probe/`: configurations and launch scripts for the pilot.
- `src/`: reusable router and evaluation code.
- `results/`: isolated metrics and checkpoints; no comparison results are
  written here.


## Hard-Gumbel probe

The primary second-stage probe uses a one-hot expert in every training forward
pass with a straight-through Gumbel-softmax gradient. Validation reports both
deterministic hard routing and the corresponding soft relaxation. Checkpoints
are selected by deterministic hard-routing validation KL. The run also records
router gradient norms, pre-hardening entropy and confidence, expert usage, and
the complete known-section-to-selected-expert matrix.

This frozen-expert probe optimizes assignment to masks that were already learned
by position-aware pruning; it does not yet learn new masks. Its purpose is to
test whether useful discrete routing is learnable before jointly optimizing the
router and pruning gates. The optional load-balance loss is an ablation and is
disabled by default because true semantic sections are not uniformly frequent.
