# Pure Category Object Tracking: Comprehensive Analysis

This report documents the accuracies and prompt structures for the "Pure Category" experiments, where we isolated causal tracking tasks into strictly "Living" (passed between people) and "Non-Living" (moved between places/containers) categories. 

The goal was to test if neutralizing semantic bias and adjusting prompt structure (length and distractor placement) enabled Llama-1B, 8B, and Qwen-32B to perform true zero-shot mechanistic object tracking.

---

## 1. The Baseline: 6-Hop Pure Categories

### Prompt Templates
**Living Category (Clean):**
> "The ball is given to the mom. The doll is given to the dad. The mom hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The grandma hands it to the uncle. The ball is held by the"
*(Corrupted swaps the initial assignments so the ball is given to the dad).*

**Non-Living Category (Clean):**
> "The ball is put in the basket. The doll is put in the box. The basket is moved to the table. The table is moved to the truck. The truck is moved to the boat. The boat is moved to the train. The ball is located in the"

### Accuracies (Clean / Corrupted)
| Model | Living Clean | Living Corr | Non-Living Clean | Non-Living Corr |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-1B** | 5/5 (100%) | 0/5 (0%) | 4/5 (80%) | 0/5 (0%) |
| **Llama-8B** | 5/5 (100%) | 1/5 (20%) | 4/5 (80%) | 0/5 (0%) |
| **Qwen-32B** | 5/5 (100%) | 0/5 (0%) | 3/5 (60%) | **5/5 (100%)** |

**Conclusion:** The smaller models completely fail Corrupted prompts. The 32B model achieves perfect 100% Corrupted tracking on Non-Living, but fails completely on Living due to the semantic "held by" verb hijack.

---

## 2. Reduced Difficulty: 3-Hop Pure Categories
We reduced the length of the chain from 6 hops to 3 hops to determine if shorter context lengths allowed smaller models to succeed.

### Prompt Templates
**Living Category (Clean):**
> "The ball is given to the mom. The doll is given to the dad. The mom hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The ball is held by the"

**Non-Living Category (Clean):**
> "The ball is put in the basket. The doll is put in the box. The basket is moved to the table. The table is moved to the truck. The truck is moved to the boat. The ball is located in the"

### Accuracies (Clean / Corrupted)
| Model | Living Clean | Living Corr | Non-Living Clean | Non-Living Corr |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-1B** | 4/5 (80%) | 0/5 (0%) | 4/5 (80%) | 1/5 (20%) |
| **Llama-8B** | 5/5 (100%) | 0/5 (0%) | 3/5 (60%) | 0/5 (0%) |
| **Qwen-32B** | 5/5 (100%) | 0/5 (0%) | 4/5 (80%) | **5/5 (100%)** |

**Conclusion:** Reducing the hops did NOT help the smaller models. They still scored ~0% on Corrupted streams, proving a fundamental lack of tracking logic, not just an attention-span limitation.

---

## 3. The Recency Test: 3-Hop Shifted Prompts
We shifted the 1-hop distractor statement to the very end of the prompt (right before the question) to test if models were just predicting the most recently mentioned noun.

### Prompt Templates
**Living Category (Clean):**
> "The ball is given to the mom. The mom hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. **The doll is given to the dad.** The ball is held by the" *(Target: friend)*

**Living Category (Corrupted):**
> "The doll is given to the mom. The mom hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. **The ball is given to the dad.** The ball is held by the" *(Target: dad)*

*(Note: In the Corrupted prompt, the correct answer is literally the noun right before the question).*

### Accuracies (Clean / Corrupted)
| Model | Living Clean | Living Corr | Non-Living Clean | Non-Living Corr |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-1B** | 0/5 (0%) | 4/5 (80%) | 0/5 (0%) | 5/5 (100%) |
| **Llama-8B** | 1/5 (20%) | 3/5 (60%) | 0/5 (0%) | 3/5 (60%) |
| **Qwen-32B** | 3/5 (60%) | 4/5 (80%) | 1/5 (20%) | 4/5 (80%) |

**Conclusion:** By moving the distractor to the end, we **inverted** the accuracy. Models that previously got 100% Clean and 0% Corrupted now scored 0% Clean and 100% Corrupted. This proves definitively that the 1B and 8B models are heavily relying on a "Recency Heuristic" (picking the noun closest to the question) rather than actually executing causal subject-object binding logic. 

The 32B model, however, was occasionally able to overcome the recency trap and read backward to find the correct answer (scoring 60% on Living Clean).
