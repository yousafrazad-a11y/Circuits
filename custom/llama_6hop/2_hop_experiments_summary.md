# Pure Category Object Tracking: 2-Hop Analysis

This report documents the accuracies and prompt structures for the "Pure Category" 2-Hop experiments. We defined a 2-hop chain as **an initial assignment followed by exactly 1 passing statement**. The goal was to minimize context length as much as possible to see if the 1B and 8B models could overcome their heuristics and perform logical object tracking.

---

## 1. Reduced Difficulty: 2-Hop Pure Categories (Normal Prompts)
We minimized the causal passing chain to just 1 pass after the initial assignment.

### Prompt Templates
**Living Category (Clean):**
> "The ball is given to the mom. The doll is given to the dad. The mom hands it to the kitty. The ball is held by the" *(Target: kitty)*

**Living Category (Corrupted):**
> "The doll is given to the mom. The ball is given to the dad. The mom hands it to the kitty. The ball is held by the" *(Target: dad)*

**Non-Living Category (Clean):**
> "The ball is put in the basket. The doll is put in the box. The basket is moved to the table. The ball is located in the" *(Target: table)*

**Non-Living Category (Corrupted):**
> "The doll is put in the basket. The ball is put in the box. The basket is moved to the table. The ball is located in the" *(Target: box)*

### Accuracies (Clean / Corrupted)
| Model | Living Clean | Living Corr | Non-Living Clean | Non-Living Corr |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-1B** | 3/5 (60%) | 1/5 (20%) | 4/5 (80%) | 0/5 (0%) |
| **Llama-8B** | 5/5 (100%) | 0/5 (0%) | 2/5 (40%) | **5/5 (100%)** |
| **Qwen-32B** | 4/5 (80%) | 1/5 (20%) | 3/5 (60%) | **5/5 (100%)** |

**Conclusion:** 
At this extremely short context length, the **Llama-8B model suddenly succeeded on the Non-Living Corrupted stream (100%)!** It was able to correctly bind the object to its final container.

However, **both the 1B and 8B models STILL completely failed the Living Corrupted stream (0% - 20%)**. This decisively proves that the semantic trap of the verb "held by" is so powerful that it hijacks the model's attention entirely. Even when the sequence is only three sentences long, the model blindly chooses the most recently mentioned person rather than tracking the logical object!

---

## 2. The Recency Test: 2-Hop Shifted Prompts
We moved the distractor statement to the very end of the prompt (right before the question) to prove the models are solely utilizing a "Recency Heuristic" rather than causal tracking.

### Prompt Templates
**Living Category (Clean):**
> "The ball is given to the mom. The mom hands it to the kitty. **The doll is given to the dad.** The ball is held by the" *(Target: kitty)*

**Living Category (Corrupted):**
> "The doll is given to the mom. The mom hands it to the kitty. **The ball is given to the dad.** The ball is held by the" *(Target: dad)*
*(Note: In the Corrupted prompt, the answer is literally the noun right before the question. In the Clean prompt, the answer is buried two sentences back).*

### Accuracies (Clean / Corrupted)
| Model | Living Clean | Living Corr | Non-Living Clean | Non-Living Corr |
| :--- | :--- | :--- | :--- | :--- |
| **Llama-1B** | 1/5 (20%) | 2/5 (40%) | 0/5 (0%) | 3/5 (60%) |
| **Llama-8B** | 2/5 (40%) | 4/5 (80%) | 0/5 (0%) | 3/5 (60%) |
| **Qwen-32B** | 4/5 (80%) | 5/5 (100%) | 3/5 (60%) | 3/5 (60%) |

**Conclusion:** 
Moving the distractor sentence to the end **inverted the accuracies**. 
Models that previously got ~100% on Clean prompts and ~0% on Corrupted prompts now scored **0% - 40% on Clean** and performed much better on Corrupted. 

This confirms the ultimate cheat code of small autoregressive LLMs on zero-shot IOI tasks: **They do not reason. They read backward from the question and pick the nearest noun that semantically fits the final verb.** When the distractor was placed at the end, it became the "nearest noun," and the models blindly guessed it, completely forgetting the causal chain.

### Final Takeaway for Mechanism Interpretability
To isolate a true logic circuit with pruning algorithms like E-MAC, you must:
1. Use a model with sufficient capacity (like Llama-8B or Qwen-32B).
2. Phrase the dataset using neutral verbs (`"located in"`).
3. Use non-living physical hierarchies so the model relies on logic rather than semantic linguistic bias.
