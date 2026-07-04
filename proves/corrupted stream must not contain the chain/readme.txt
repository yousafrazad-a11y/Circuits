========================================================================
SEMANTIC GRAVITY WELL PROOF: "CORRUPTED STREAM MUST NOT CONTAIN THE CHAIN"
========================================================================

This folder contains the complete, reproducible evidence that language models process chained movement logic (e.g. A -> B -> C) as a structural attention trap ("Semantic Gravity Well") rather than through strict logical entity binding.

We proved this using Dataset 2 (Over-Eager Completion). The evidence is unified into a single master report containing three robust proofs across three different models (1B, 8B, and 32B).

## THE THREE PROOFS
1. **The Irrelevant Chain Ablation:** If an irrelevant movement chain (e.g. a random 'tray' moving across the room) is inserted into a prompt, the model's accuracy on the target object's location drops significantly, proving the chain acts as a blind attention sink.
2. **Distractor Probability Spikes:** The log-probability mass of the unmoved target ('B') versus the final destination of the moving distractor container ('D'). In a logical world, D should be near zero. Instead, it regularly captures a massive chunk of probability mass.
3. **Multi-Hop Probability Ratios:** The probability ratio between the containers inside the movement chain (Chain_0 : Chain_1 : Chain_2) remains mathematically identical regardless of whether the chain logically holds the target item or the distractor item.

## FILE STRUCTURE
1. `01_generate_unified_evidence.py`
   - Evaluates the 1B, 8B, and 32B models. Captures exact, raw log-probabilities of all structural nodes across Baseline, Clean, Corrupted, and Irrelevant Chain prompt variations.
2. `02_compile_master_report.py`
   - Synthesizes the raw JSON outputs into a single, beautifully formatted markdown/HTML paper showing all three proofs for each model.
3. `03_master_unified_report.md` / `.html`
   - The final, conclusive evidence document.
4. `dump/`
   - Contains the raw JSON evaluation outputs from vLLM.

## HOW TO REPRODUCE
All scripts are self-contained and use local, relative paths for saving and loading. 
1. Install `vllm`.
2. Run `python 01_generate_unified_evidence.py`
3. Run `python 02_compile_master_report.py`

*Note: Ensure your HuggingFace environment token is set to access the gated Llama 3 models if re-running.*
