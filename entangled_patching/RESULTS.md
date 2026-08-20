# Contextually-Entangled Dataset Families — Validation Results

Synthetic dataset families for demonstrating the failure of Activation Patching /
Causal Tracing on contextually entangled tasks. Each family: 50 unique samples,
`P_clean` vs `P_corrupt` differ ONLY in the routing/control token; token-length
aligned (verified per sample: equal length, single contiguous diff span).

## Files
- `templates.py` — 10 family definitions (control pairs tokenizer-verified equal-length)
- `generate.py` — writes `datasets/<family>.jsonl`
- `evaluate.py` — alignment check + greedy-gen eval + iteration reports → `logs/`
- `console_iter*.txt` — console logs of every iteration

## Final metrics (iteration 14, N=50 per family per side, candidate-restricted scoring)

### Llama-3.1-8B-Instruct — 3 families pass ≥90/90
| family | clean | corrupt | tricked | verdict |
|---|---|---|---|---|
| t2_kv_lookup | 100 | 100 | 0 | PASS |
| t4_boolean_override | 100 | 100 | 0 | PASS |
| t9_list_index | 90 | 98 | 2 | PASS |
| t1_arithmetic_gate | 100 | 0 | **100** | Corrupt Inertia |
| t3_spatial_containment | 100 | 8 | 92 | Corrupt Inertia |
| t5_relation_block | 96 | 0 | 94 | Corrupt Inertia |
| t6_concat_directive | 68 | 16 | 60 | both fail |
| t7_category_suppression | 100 | 2 | 98 | Corrupt Inertia |
| t8_unit_conversion | 100 | 0 | **100** | Corrupt Inertia |
| t10_state_tracking | 100 | 16 | 84 | Corrupt Inertia |

### Llama-3.2-3B-Instruct — 3 families pass ≥90/90
| family | clean | corrupt | tricked | verdict |
|---|---|---|---|---|
| t3_spatial_containment | 100 | 100 | 0 | PASS |
| t4_boolean_override | 100 | 100 | 0 | PASS |
| t10_state_tracking | 98 | 94 | 6 | PASS |
| t1_arithmetic_gate | 100 | 4 | 96 | Corrupt Inertia |
| t2_kv_lookup | 100 | 0 | 98 | Corrupt Inertia |
| t5_relation_block | 92 | 0 | 88 | Corrupt Inertia |
| t6_concat_directive | 70 | 10 | 58 | both fail |
| t7_category_suppression | 100 | 16 | 84 | Corrupt Inertia |
| t8_unit_conversion | 100 | 0 | **100** | Corrupt Inertia |
| t9_list_index | 22 | 88 | 10 | Clean Collapse |

## Key findings
1. **Corrupt Inertia (Failure Mode B) is robust**: on 8B, six families show
   84–100% tricked rates — the model outputs the computed task result despite
   an explicit override token. The task context is processed and wins over the
   routing directive: precisely the entanglement that makes P_clean/P_corrupt
   activation differences in task layers a false-positive trap for patching.
2. **Inertia is scale- and task-dependent**: T2/T9 obey the override on 8B but
   collapse on 3B (T2: 98% tricked); T3/T10 show inertia on 8B but are obeyed
   on 3B. T4 passes on BOTH models.
3. **Fallback-token semantics matter**: BYPASS (negative valence) was
   systematically confused with FAIL (model even hallucinated "Override is YES"
   on low scores); replacing with neutral AUTO gave 100/100.
4. Validated families for patching experiments: **T2, T4, T9 (8B)** and
   **T3, T4, T10 (3B)**; **T4 is validated on both scales**.

## Documented deviations from the original spec (all tokenizer-motivated, Llama-3 BPE)
- T2: `INACTIVE`(2 tok) → `CLOSED`(1) to match `ACTIVE`(1)
- T6: `COMBINE`(3) → `MERGE`(2) to match `HALT`(2)
- T7: `MUTE`(2) → `HOLD`(1) to match `REPORT`(1); fallback `N/A` → `NULL`
- T4: fallback `BYPASS` → `AUTO` (semantic confusion with FAIL, finding #3)
- All prompts: appended "Follow only the rule that matches the stated control
  value. Output only the final answer." (identical on both sides)
- T4: added "Check the Override value before looking at the score." (fixes 3B
  override-skipping; keeps 8B at 100/100)
- T4 FAIL samples: score within 10 below threshold (large-gap FAILs trigger
  spurious fallback association in small models)

## Reproduce
```bash
cd entangled_patching
../venv/bin/python generate.py                 # write datasets/*.jsonl
../venv/bin/python evaluate.py --model meta-llama/Llama-3.1-8B-Instruct --iter 14
../venv/bin/python evaluate.py --model meta-llama/Llama-3.2-3B-Instruct --iter 14
```
