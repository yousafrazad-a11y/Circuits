# Non-position-aware node-pruning baseline

This is the controlled comparison run for `position_aware_node_pruning`.

It uses the same generated ABBA IOI examples, clean/corrupted alignment,
GPT-2 model, hard-concrete gates, pruning lambdas, optimizer, 1,000-step
sparsity warmup, validation schedule, hardware-aware batch sizing, random seed,
and 500 epochs. The only experimental change is that every token receives
section ID zero, so each component has one global gate rather than eight
logical-section gates.

Run from the repository root with:

```bash
python -m non_position_node_pruning.ioi
```

Runtime logs, JSONL metrics, and the final gate-only checkpoint are written to
`non_position_node_pruning/outputs/`.

The fresh 1,000-epoch comparison run is started with:

```bash
python -m non_position_node_pruning.train_1000
```

It validates every 10 epochs and saves a resumable continuous-log-alpha
checkpoint for every validation cycle under
`outputs/continuous_1000_run_01/checkpoints/`. A selected checkpoint can later
initialize the position-aware model by repeating its one global gate row across
all eight logical sections; those rows then specialize independently.
