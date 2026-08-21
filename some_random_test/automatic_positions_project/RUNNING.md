# Running the router probe

Do not launch this while the current pruning continuations are using the GPU.

Run one ablation explicitly:

```bash
python -m automatic_positions_project.train_router_probe \
  --expert-checkpoint PATH/TO/ioi_abba_gates.pt \
  --router-input early_residual_plus_position \
  --output-dir automatic_positions_project/results/early_residual_plus_position
```

Or run all four sequentially:

```bash
automatic_positions_project/run_all_ablations.sh PATH/TO/ioi_abba_gates.pt
```

The checkpoint may be either a periodic training checkpoint or the finalized
`ioi_abba_gates.pt`. The base model and expert gates are frozen. Each validation
cycle writes a resumable epoch checkpoint and updates `best.pt` when KL improves.
The hand-written section circuit is evaluated as an oracle before router
training, using the identical test rows and shared metric accumulator.

Start with `early_residual_plus_position`; it is the strongest expected probe.
Then run `normalized_position` as the essential baseline. The remaining two
ablations tell us whether content or context provides the useful signal.

Hard-Gumbel primary probe:

    python -m automatic_positions_project.train_router_probe \
      --expert-checkpoint PATH/TO/ioi_abba_gates.pt \
      --router-input early_residual_plus_position \
      --train-routing hard_gumbel \
      --load-balance-weight 0.0 \
      --output-dir automatic_positions_project/results/hard_gumbel_seed42

Run a separate collapse-control ablation with a small nonzero
--load-balance-weight; do not mix its conclusions with the unregularized run.
