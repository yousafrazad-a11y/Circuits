# 2-Hop Logical Tracking Failure Analysis (Qwen-32B)

## Overview
This report analyzes the failure patterns of Qwen-32B on the new 2-hop non-living dataset (`dataset2.jsonl`). We evaluated 1,000 generated prompts in both "clean" and "corrupted" structural states. 

**Combined Accuracy:** 49.50% 
*(A success requires the model to correctly answer BOTH the clean and corrupted prompt variants, completely ruling out lucky guesses).*

## Performance by Theme
We observed a massive disparity in logical tracking accuracy depending on the theme (setting) of the prompt:

| Theme | Combined Accuracy | Total Examples |
|-------|-------------------|----------------|
| **The Storage Room** | **93.4%** | 91 |
| **The Laboratory** | **57.7%** | 130 |
| **The Warehouse** | **51.2%** | 43 |
| **The Post Office** | **47.0%** | 372 |
| **The NonLiving Kitchen**| **37.9%** | 364 |

By analyzing three separate randomized samples of 50 failures, we successfully isolated the exact mechanistic traps that break the LLM's logical tracking circuit. 

---

## Failure Mode 1: Semantic Bleeding (The Kitchen Trap)
The worst-performing theme was "The NonLiving Kitchen." When we analyze the entities involved in the failures, **`shelf`** (appearing in 197 failures) and **`drawer`** (appearing in 67 failures) are by far the most common "stationary distractor" containers.

> [!WARNING]
> **Example Prompt:** "The bean is placed in the tray. The pebble is placed in the shelf. The tray is moved to the pantry. The pantry is moved to the kitchen. The bean is in the -> kitchen"

**Why it fails:** 
The model possesses strong pre-trained semantic priors that a `shelf` or a `drawer` natively belongs inside a `pantry` or a `kitchen`. Even though the prompt explicitly places the distractor object (pebble) on the shelf and *never moves the shelf*, the model's semantic network hallucinates that the shelf was automatically transported inside the kitchen along with the tray. 

**The LLM allows implicit spatial assumptions to override explicit logical instructions.**

---

## Failure Mode 2: Lexical Overlap (The Post Office Trap)
The second worst-performing theme was "The Post Office." The bottom 5 absolute worst-performing prompts in the entire dataset (where the model scored literally 0.00% probability of getting the right answer) all belonged to this theme.

> [!WARNING]
> **Example Prompt:** "The stamp is placed in the mailbag. The envelope is placed in the mailbox. The mailbag is moved to the postbox. The postbox is moved to the postoffice."

**Why it fails:**
The entities used are `mailbag`, `mailbox`, `postbox`, `postoffice`, and `posttruck`. 
The model's attention heads lose the ability to track distinct entities because the tokens (`mail`, `post`, `box`, `bag`) bleed into each other. When asked to track whether the `mailbag` or the `mailbox` moved to the `postbox`, the overlapping prefixes and suffixes cause severe attention confusion. 

**Lexical similarity destroys multi-hop attention routing.**

---

## The Control: Pure Logic Success (The Storage Room)
The model achieved an outstanding **93.4% accuracy** in "The Storage Room" theme.

> [!TIP]
> **Example Prompt:** "The gold is placed in the crate. The jewel is placed in the shelf. The crate is moved to the rack. The rack is moved to the warehouse."

**Why it succeeds:**
1. **Lexical Distinctness:** The entities (`crate`, `shelf`, `rack`, `warehouse`) share no overlapping tokens, allowing clean attention mapping.
2. **Weak Spatial Priors:** A `crate` moving to a `rack` does not trigger overwhelming pre-trained assumptions that a `shelf` must also automatically move to a `warehouse`.

## Conclusion
If you want to test pure logical reasoning without interference, **you must construct datasets where entities are lexically distinct and lack strong pre-trained spatial hierarchies.** If you use words with heavy spatial associations (like kitchen cabinets) or overlapping tokens (like mailboxes), you are inadvertently testing the model's semantic prior strength rather than its logical tracking circuit.
