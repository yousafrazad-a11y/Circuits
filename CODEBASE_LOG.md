# Pruning Codebase Log and Architecture Overview

## 1. Core Pruning Engine (`venn_circuit_discovery_v2/`)
This is the current, mathematically corrected iteration of the dual-task Venn circuit discovery algorithm. It finds the absolute minimal "Universal Core" circuit inside LLaMA by overlapping structural requirements across different datasets.

- **`models.py`**: Intercepts every Attention Head, MLP Neuron, and Decoder Layer in the HuggingFace LLaMA model. It runs 4 simultaneous streams (Clean A, Clean B, Corrupted A, Corrupted B) in a single forward pass, patching the network using Venn logic without breaking flash attention.
- **`gates.py`**: Contains the `HardConcreteGate` and `VennConcreteGate` logic. Uses the Straight-Through Estimator to allow differentiable binary pruning. Crucially, uses `.sum()` in `expected_l0` to mathematically guarantee that massive parameter groups (like 131,072 MLP neurons) still receive uniform, massive pruning pressure, crushing out random artifacts.
- **`trainer.py`**: The training loop. Replaced `AdamW` with pure `Adam` to prevent weight decay from passively destroying gates into 50/50 corrupted garbage states. Tracks exact accuracy and balances the dual-task losses.
- **`scheduler.py`**: A PID controller that dynamically adjusts the sparsity penalties ($\lambda$) based on how close the model's KL divergence is to the target.
- **`loss.py`**: Contains the Margin Loss (enforcing correctness), KL Divergence Loss (enforcing exact probability distributions against the dense unpruned model), and a Homoscedastic Uncertainty wrapper to autoscale multi-task losses.
- **`api.py`**: Provides the top-level API (`VennCircuitDiscoverer`) used by external batch scripts.

## 2. Research & Experiments (`data_research/`)
Contains all the high-level batch runners and analytical reports.

- **`batch_venn_discovery_ultra.py`**: The ultimate high-sparsity discovery script. Trains 10 pairs (all combinations of 5 category chain datasets) for 120 epochs per pair (10+ hours). Uses a relaxed `target_kl = 3.0` and `gate_lr = 0.01` so the network can rip out 80% of its inner logic without the PID controller violently shutting off. It dumps the full binary tensors and calculates the final global 10-way intersection.
- **`evaluate_dataset.py`, `25_test_dataset4_all_models.py`, etc.**: Analytical scripts from previous research phases evaluating dataset performance across model sizes (Llama, Qwen, etc.).
- **Markdown Reports (`29_semantic_gravity_evidence_report.md`, etc.)**: Logs and conclusions from various dataset failure investigations and prompt engineering proofs.

## 3. Datasets (`induction_datasets/`)
Contains all the procedural evaluation data.

- **`category_chains/`**: Small, pure JSONL datasets containing extremely simple 1-word object chains (e.g., `fruits.jsonl`, `animals.jsonl`, `metals.jsonl`). Designed to completely isolate the raw tracking logic inside the transformer without being distracted by complex syntax or grammar.
- **`generate_datasets.py`**: Uses an LLM locally (fitting in 40GB VRAM) to synthetically generate these clean and corrupted pure-category sets.

## 4. Older Versions & Proofs
These folders contain older logic or structural proofs that verified specific behaviors of the model before `v2` was implemented.

- **`proves/`**: Includes tests like `corrupted stream must not contain the chain` ensuring that our corrupted reference streams don't accidentally leak the correct answer to the model.
- **`mini_version/`**: Earlier, minimal reproducible test scripts for prompt engineering.
- **`venn_circuit_discovery/` (v1)**: The legacy v1 engine before accuracy tracking and stability fixes.
- **`circuit_pruning-argo/`**: The original reference codebase that inspired the L0-regularized Venn logic.

## Summary of Fixes (2026-07-10)
- Changed `.mean()` to `.sum()` in `gates.py` to fix gradient crushing on large inner circuits (MLPs).
- Removed Weight Decay (`AdamW` -> `Adam`) in `trainer.py` to prevent 50/50 gate decay.
- Relaxed `target_kl` to `3.0` in `batch_venn_discovery_ultra.py` to allow the network mathematical room to achieve 80% sparsity targets.
- Upped epochs to 120 and lowered gate LR to `0.01` for stability.
