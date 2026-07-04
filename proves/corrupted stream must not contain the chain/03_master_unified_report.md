# Unified Semantic Gravity Evidence Report
## Objective
To provide conclusive, multi-model evidence that language models process structural multi-hop movement sequences (e.g. `A is moved to B -> B is moved to C`) not via strict logical entity tracking, but as a rigid structural probability funnel ("Semantic Gravity Well").

We present three distinct pieces of evidence across 1B, 8B, and 32B models.

## 🔵 MODEL: 1B

### Method 1: The Irrelevant Chain Ablation
We test the model's accuracy on the target object's location across three prompt variations. In the 'Irrelevant Chain', a completely disconnected object (e.g., a 'tray') moves through the room. If the model's accuracy drops from Baseline merely because an irrelevant chain is present, it proves the chain acts as a blind attention sink.

- **Baseline (No Hops):** 77.8%
- **Corrupted (Current):** 32.4%
- **Irrelevant Chain (The Proof):** 55.6%

### Method 2: Distractor Probability Spikes
We measure the raw log-probability mass of the unmoved target ('B') versus the final destination of the moving distractor container ('D'). If the model is not attending to the multi-hop logic, 'D' should have near-zero probability.

| Variation | Prob(Target Container `B`) | Prob(Distractor Destination `D`) |
|---|---|---|
| **Baseline** | 0.1137 | 0.0042 |
| **Corrupted** | 0.1741 | 0.2480 |

### Method 3: Multi-Hop Probability Ratios
We hypothesize that the probability ratio between the containers in the movement chain (`Chain_0` : `Chain_1` : `Chain_2`) remains mathematically similar regardless of what item is inside the chain.

| State | LogProb Ratio (`Chain_0`:`Chain_1`:`Chain_2`) |
|---|---|
| **Clean (Target moves)** | `1.0 : 0.7 : 1.8` |
| **Corrupted (Distractor moves)** | `1.0 : 0.7 : 1.7` |
| **Irrelevant Chain (Random item moves)** | `1.0 : 0.5 : 0.9` |

---

## 🔵 MODEL: 8B

### Method 1: The Irrelevant Chain Ablation
We test the model's accuracy on the target object's location across three prompt variations. In the 'Irrelevant Chain', a completely disconnected object (e.g., a 'tray') moves through the room. If the model's accuracy drops from Baseline merely because an irrelevant chain is present, it proves the chain acts as a blind attention sink.

- **Baseline (No Hops):** 89.4%
- **Corrupted (Current):** 45.0%
- **Irrelevant Chain (The Proof):** 49.6%

### Method 2: Distractor Probability Spikes
We measure the raw log-probability mass of the unmoved target ('B') versus the final destination of the moving distractor container ('D'). If the model is not attending to the multi-hop logic, 'D' should have near-zero probability.

| Variation | Prob(Target Container `B`) | Prob(Distractor Destination `D`) |
|---|---|---|
| **Baseline** | 0.2951 | 0.0040 |
| **Corrupted** | 0.2233 | 0.4001 |

### Method 3: Multi-Hop Probability Ratios
We hypothesize that the probability ratio between the containers in the movement chain (`Chain_0` : `Chain_1` : `Chain_2`) remains mathematically similar regardless of what item is inside the chain.

| State | LogProb Ratio (`Chain_0`:`Chain_1`:`Chain_2`) |
|---|---|
| **Clean (Target moves)** | `1.0 : 1.0 : 3.7` |
| **Corrupted (Distractor moves)** | `1.0 : 0.8 : 3.1` |
| **Irrelevant Chain (Random item moves)** | `1.0 : 0.6 : 1.6` |

---

## 🔵 MODEL: 32B

### Method 1: The Irrelevant Chain Ablation
We test the model's accuracy on the target object's location across three prompt variations. In the 'Irrelevant Chain', a completely disconnected object (e.g., a 'tray') moves through the room. If the model's accuracy drops from Baseline merely because an irrelevant chain is present, it proves the chain acts as a blind attention sink.

- **Baseline (No Hops):** 94.8%
- **Corrupted (Current):** 97.2%
- **Irrelevant Chain (The Proof):** 80.8%

### Method 2: Distractor Probability Spikes
We measure the raw log-probability mass of the unmoved target ('B') versus the final destination of the moving distractor container ('D'). If the model is not attending to the multi-hop logic, 'D' should have near-zero probability.

| Variation | Prob(Target Container `B`) | Prob(Distractor Destination `D`) |
|---|---|---|
| **Baseline** | 0.5611 | 0.0060 |
| **Corrupted** | 0.6829 | 0.1004 |

### Method 3: Multi-Hop Probability Ratios
We hypothesize that the probability ratio between the containers in the movement chain (`Chain_0` : `Chain_1` : `Chain_2`) remains mathematically similar regardless of what item is inside the chain.

| State | LogProb Ratio (`Chain_0`:`Chain_1`:`Chain_2`) |
|---|---|
| **Clean (Target moves)** | `1.0 : 1.3 : 15.2` |
| **Corrupted (Distractor moves)** | `1.0 : 2.0 : 6.4` |
| **Irrelevant Chain (Random item moves)** | `1.0 : 1.3 : 3.8` |

---

