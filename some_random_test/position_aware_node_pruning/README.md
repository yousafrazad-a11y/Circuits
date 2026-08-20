# Position-aware node pruning for IOI

This folder is an isolated implementation of the original clean/corrupted
hard-concrete node-pruning method. Existing project files are not imported or
modified.

## Position representation

Every prompt is divided into eight ordered logical sections:

1. prefix
2. first name
3. connector between names
4. second name
5. shared event/context
6. repeated subject
7. final action/object
8. final prediction relation (normally `to`)

The dataset stores an eight-element `section_lengths` vector. Its values may
differ between examples. `section_ids` expands those lengths along the model's
sequence dimension, so every token in a logical section uses the same gate row.

Clean/corrupted pairs are accepted only when every corresponding section has the
same GPT-2 token count. Synthetic counterfactuals select replacement names from
the same contextual token-length bucket and are validated after complete prompt
tokenization.

ABBA and BABA should be trained separately because their first and second name
positions have different semantic roles. `TEMPLATE_ORDER` in `ioi.py` selects
the circuit.

## Model

All original pruning granularities are position-aware:

- embedding
- complete layers
- attention and MLP blocks
- attention heads
- attention head dimensions
- MLP hidden neurons
- MLP output dimensions

The implementation gathers section gates using a `[batch, sequence]` index
tensor. It does not loop over tokens or run the model separately for each
section.

Run from the repository root with:

```bash
python -m position_aware_node_pruning.ioi
```

Set `IOI_DATASET_PATH` to load a saved Hugging Face `DatasetDict`. If it is not
set or does not exist, the code generates deterministic, section-aligned IOI
examples from the 15 templates.

The finalization report distinguishes position-specific gate slots from the
union of physical components active in any section. Its compression result is a
structural extraction proxy; the current model remains dense and computes both
clean and corrupted streams.
