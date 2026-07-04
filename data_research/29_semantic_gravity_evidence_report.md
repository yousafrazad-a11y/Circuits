# Semantic Gravity Ratio Analysis
## Objective
To prove that the structural multi-hop chain (`Container 1 -> Container 2 -> Container 3`) acts as an attention trap ("Semantic Gravity Well") regardless of the logical variable binding. We hypothesize that the probability ratio between the containers in the movement chain remains mathematically similar in both the Clean state (where the target item is in the chain) and the Corrupted state (where the distractor item is in the chain).

## Methodology
- **Clean Prompt:** `The distractor is placed in Unmoved_0. The target is placed in Chain_0. Chain_0 -> Chain_1 -> Chain_2.`
- **Corrupted Prompt:** `The distractor is placed in Chain_0. The target is placed in Unmoved_0. Chain_0 -> Chain_1 -> Chain_2.`

We extract the probabilities of `Chain_0`, `Chain_1`, and `Chain_2` for N=500 queries and calculate the ratio `1.0 : (Chain_1/Chain_0) : (Chain_2/Chain_0)`.

## Model: 1B
**Clean Accuracy:** 62.6% | **Corrupted Accuracy:** 32.6%

### Clean Probabilities (Target is in the Chain)
| Node | Probability | Role |
|---|---|---|
| `Unmoved_0` | 0.1587 | Distractor Initial |
| `Chain_0` | 0.2082 | Target Initial |
| `Chain_1` | 0.1122 | Target Intermediate |
| `Chain_2` | 0.2785 | Target Final (Correct) |
**Chain Ratio (`Chain_0`:`Chain_1`:`Chain_2`):** `1.0 : 0.5 : 1.3`

### Corrupted Probabilities (Distractor is in the Chain)
| Node | Probability | Role |
|---|---|---|
| `Unmoved_0` | 0.1743 | Target Initial (Correct) |
| `Chain_0` | 0.1424 | Distractor Initial |
| `Chain_1` | 0.0975 | Distractor Intermediate |
| `Chain_2` | 0.2476 | Distractor Final |
**Chain Ratio (`Chain_0`:`Chain_1`:`Chain_2`):** `1.0 : 0.7 : 1.7`

> **Observation:** Note how the probability ratio of the containers within the multi-hop chain remains structurally similar across both prompts, even though the logical contents of the chain are entirely swapped!

---

## Model: 8B
**Clean Accuracy:** 82.2% | **Corrupted Accuracy:** 45.8%

### Clean Probabilities (Target is in the Chain)
| Node | Probability | Role |
|---|---|---|
| `Unmoved_0` | 0.1079 | Distractor Initial |
| `Chain_0` | 0.1703 | Target Initial |
| `Chain_1` | 0.1214 | Target Intermediate |
| `Chain_2` | 0.4593 | Target Final (Correct) |
**Chain Ratio (`Chain_0`:`Chain_1`:`Chain_2`):** `1.0 : 0.7 : 2.7`

### Corrupted Probabilities (Distractor is in the Chain)
| Node | Probability | Role |
|---|---|---|
| `Unmoved_0` | 0.2227 | Target Initial (Correct) |
| `Chain_0` | 0.1295 | Distractor Initial |
| `Chain_1` | 0.1040 | Distractor Intermediate |
| `Chain_2` | 0.4004 | Distractor Final |
**Chain Ratio (`Chain_0`:`Chain_1`:`Chain_2`):** `1.0 : 0.8 : 3.1`

> **Observation:** Note how the probability ratio of the containers within the multi-hop chain remains structurally similar across both prompts, even though the logical contents of the chain are entirely swapped!

---

## Model: 32B
**Clean Accuracy:** 96.8% | **Corrupted Accuracy:** 97.2%

### Clean Probabilities (Target is in the Chain)
| Node | Probability | Role |
|---|---|---|
| `Unmoved_0` | 0.0282 | Distractor Initial |
| `Chain_0` | 0.0732 | Target Initial |
| `Chain_1` | 0.0921 | Target Intermediate |
| `Chain_2` | 0.7690 | Target Final (Correct) |
**Chain Ratio (`Chain_0`:`Chain_1`:`Chain_2`):** `1.0 : 1.3 : 10.5`

### Corrupted Probabilities (Distractor is in the Chain)
| Node | Probability | Role |
|---|---|---|
| `Unmoved_0` | 0.6829 | Target Initial (Correct) |
| `Chain_0` | 0.0158 | Distractor Initial |
| `Chain_1` | 0.0318 | Distractor Intermediate |
| `Chain_2` | 0.1004 | Distractor Final |
**Chain Ratio (`Chain_0`:`Chain_1`:`Chain_2`):** `1.0 : 2.0 : 6.4`

> **Observation:** Note how the probability ratio of the containers within the multi-hop chain remains structurally similar across both prompts, even though the logical contents of the chain are entirely swapped!

---

## Example Walkthrough
**Generic Clean Prompt:**
`The bean is placed in the bucket. The pebble is placed in the shelf. The bucket is moved to the pantry. The pantry is moved to the kitchen. The bean is in the`
- **`Chain_0` (Target start):** `bucket`
- **`Chain_1` (Target inter):** `pantry`
- **`Chain_2` (Target final):** `kitchen`
- **`Unmoved_0` (Distractor):** `shelf`

**Generic Corrupted Prompt:**
`The pebble is placed in the bucket. The bean is placed in the shelf. The bucket is moved to the pantry. The pantry is moved to the kitchen. The bean is in the`
- **`Unmoved_0` (Target):** `shelf`
- **`Chain_0` (Distractor start):** `bucket`
- **`Chain_1` (Distractor inter):** `pantry`
- **`Chain_2` (Distractor final):** `kitchen`
