import os

artifacts_dir = "/home/exouser/.gemini/antigravity-ide/brain/3b0d16b9-6980-4111-996d-ae6d2efb8502"

intro = """# Comprehensive Report: Breaking Logical Causal Tracking vs. Recency Heuristics in LLMs

## 1. Executive Summary
This report summarizes an extensive series of experiments conducted to isolate true logical reasoning (causal tracking) from superficial statistical heuristics (recency bias) across different scales of LLMs (Llama-3.2-1B, Llama-3.1-8B, and Qwen-2.5-32B). 

Our primary goal was to construct a robust dataset for circuit pruning (like E-MAC) where the model uses *pure logic* rather than semantic shortcuts.

### Key Discoveries:
1. **The Semantic "Living" Trap**: Standard transitive relationship prompts (e.g., "The ball is given to the boy... The ball is held by the") trigger strong semantic associations that completely override logical tracking.
2. **The Recency Heuristic**: Smaller models (1B, 8B) possess zero causal tracking circuitry for these zero-shot tasks. They rely entirely on a "Recency Heuristic," blindly guessing the noun closest to the question prompt.
3. **The 32B Threshold**: Only the Qwen2.5-32B model demonstrated true logical tracking, but *only* when the prompt was stripped of living entities and formulated with neutral, non-living verbs (e.g., "is placed in the", "is moved to the").
4. **Verb Sensitivity**: Even within the 32B model, the exact choice of internal and final verbs drastically altered whether the model relied on logic or defaulted back to heuristics.

---

## 2. Experimental Progression & Methodology

### Phase 1: The Initial Baseline (Mixed Categories)
Initially, we tested models on long, 6-hop chains involving "living" actors passing objects. We discovered that when we "corrupted" the chain (by introducing distractor statements that messed with the linear order of events), model accuracy plummeted. However, it wasn't because the logic was too complex; it was because the models were simply outputting the last noun they saw. 

### Phase 2: Category Isolation (Living vs. Non-Living)
We hypothesized that verbs like "given to" and "held by" trigger deep semantic biases. To test this, we isolated the prompts into two strict categories:
- **Living**: Actors passing items (Verbs: "given to", "held by")
- **Non-Living**: Objects moved between containers (Verbs: "put in", "located in")

**Result:** Qwen-32B achieved 100% accuracy on the Non-Living chain but failed the Living chain. The semantic association of "held by" was too strong, proving that Non-Living datasets are required for pure logic tracking.

### Phase 3: The "Shifted Distractor" Proof (Recency Bias)
To definitively prove the Recency Heuristic in smaller models, we tested 2-hop and 3-hop chains. We created a "Shifted" variant where the distractor statement was moved to the very end of the prompt, right before the question. 
**Result:** Smaller models suddenly achieved high accuracy on the Corrupted stream if the target was artificially moved to the end, confirming they possess zero causal logic for this task and merely regurgitate the most recent noun.

### Phase 4 & 5: Verb Optimization for 32B
Having isolated Qwen-32B + Non-Living categories as the only viable environment for true logic, we exhaustively tested 50, and then 288, verb combinations to find the most mathematically robust prompts.
We found that the combination of `is placed in the` (Initial Placement) + `is moved to the` (Movement) + `is in the` (Question) yielded near 90% accuracy across both clean and corrupted streams, making it the perfect foundation for our pruning dataset.

---

## 3. Detailed Experimental Data & Appendices

The following sections contain the raw, unedited data, tables, and accuracies from every stage of our investigation.

"""

files_to_append = [
    ("Appendix A: Pure Category Experiments (6-Hop)", "pure_category_experiments_summary.md"),
    ("Appendix B: Hop Scaling & Shifted Distractor Tests (2-Hop & 3-Hop)", "2_hop_experiments_summary.md"),
    ("Appendix C: Initial Verb Sensitivity (50 Combinations)", "internal_verbs_summary.md"),
    ("Appendix D: Extended Verb Optimization (288 Combinations)", "extended_verbs_sorted.md")
]

out_path = os.path.join(artifacts_dir, "final_logic_heuristics_report.md")

with open(out_path, "w") as f_out:
    f_out.write(intro)
    
    for title, filename in files_to_append:
        f_out.write(f"\n\n# {title}\n\n")
        file_path = os.path.join(artifacts_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r") as f_in:
                f_out.write(f_in.read())
        else:
            f_out.write(f"*(File {filename} not found)*\n")

print(f"Successfully compiled report to {out_path}")
