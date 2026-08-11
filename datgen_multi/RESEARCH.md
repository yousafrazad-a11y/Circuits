# Multi-hop tracking circuit-pruning dataset — research log

Goal: a clean/corrupt dataset for circuit pruning (task loss + KL divergence guided,
corrupt-activation replacement) that isolates the **multi-hop tracking** circuit
(objects swapping places across boxes). Target base models: `meta-llama/Llama-3.2-1B`
(hard requirement) and `gpt2` (stretch). Base model must be at **100% accuracy on both
clean and corrupt** correct answers.

## Key requirements for a pruning dataset (and why)

1. **Clean/corrupt come in pairs from the same template/distribution.**
   Pruning scores components by "replace clean activation with corrupt activation, does
   the output (task loss / KL vs clean output) change?". If the two prompts come from
   different distributions, the circuit fills up with distribution-detection junk.

2. **Token-level diff must be minimal.** Every differing token is a potential
   confound. We measure and report the diff size (see validate.py).

3. **The corruption must diverge EARLY in the prompt, not only at the final query
   token.** In a causal LM, activations at all positions before the first differing
   token are *identical* between clean and corrupt. If clean and corrupt differ only
   in the final query (e.g. only the queried box number changes), then every
   activation upstream is the same in both runs, so corrupt-activation-replacement
   pruning can only ever find the last-token "which box was queried" retrieval
   components — NOT the swap-tracking chain. This is the flaw in the naive
   "just query a never-swapped position in corrupt" idea. The corrupt prompt must
   change the *swap statements* so the tracking computation itself produces
   different intermediate values.

4. **Corrupt must NOT require the target mechanism (multi-hop tracking) to answer.**
   Only clean should need it, otherwise pruning keeps nothing specific to tracking.
   We achieve this by making the corrupt query target an entity that is *never
   swapped*, so its answer is stated verbatim in the initial assignment (0 hops).

5. **We cannot prove a negative about the model's internals — but we can get
   structural + behavioral proof:**
   - *Structural:* in corrupt, the query entity appears in NO swap statement, so the
     answer is logically independent of every swap. Any correct computation of the
     answer needs 0 hops. (Verified by simulation in validate.py.)
   - *Behavioral probe 1 (zero-swap):* delete all swap sentences from the corrupt
     prompt — the answer is unchanged by construction; the base model must still be
     ~100%. Then swaps are causally irrelevant to the corrupt answer.
   - *Behavioral probe 2 (scramble):* replace corrupt swaps with random other swaps
     (still not touching the query entity) — answer unchanged, accuracy must stay
     ~100%. For clean prompts the same kind of scramble *changes* the ground-truth
     answer and the model's prediction must track that change (sensitivity check).
   If these hold, the corrupt computation demonstrably does not route the answer
   through swap content, while the clean one does.

6. **Base model at ceiling on BOTH clean and corrupt (100%).** If the base model
   fails on clean, pruning finds "the circuit for guessing", not the circuit for
   tracking. If it fails on corrupt, the KL reference distribution is garbage.
   100% on a large sample is a hard gate before the dataset is accepted.

7. **Corrupt answer = the "no-tracking" answer.** Corrupt's correct answer is
   exactly what a model that does *not* track swaps would answer on the clean
   prompt (the entity's initial location/content). This makes it a hard distractor:
   pruning against it specifically preserves components that distinguish
   "tracked" from "initial" answers.

8. **Single-token answers with consistent leading space**, matching this repo's
   convention (`clean_answer`, `corrupt_answer`, `ld_candidates`). Evaluated by
   candidate-restricted logit comparison at the final position, plus full-vocab
   argmax as a stricter metric.

9. **No statistical shortcuts.** Answers balanced over all boxes/items; correct
   answer not predictable from position, frequency, or recency. (Checked in
   validate.py.)

10. **Small-model wording.** GPT-2 / Llama-1B are base models: they complete
    statements far better than they answer questions, and they are easily confused
    by complex syntax. So: short declarative sentences, one fact per sentence,
    repetitive identical phrasing, query posed as a *statement completion*
    ("Now the key is in box") not a question ("Where is the key?"). All wording
    decisions below are made empirically by sweeping variants on the GPU, not by
    intuition.

## Task design (chosen)

- N boxes (default 5), each holding one distinct item, stated up front.
- K swap statements (default 3).
- **Clean:** query an item that was swapped >= 2 times (genuine multi-hop; its
  final location can only be found by chaining the swaps). Answer = box number.
- **Corrupt:** same initial assignment, same query item, but the swap statements
  are changed minimally so the query item NEVER swaps. Answer = its initial box,
  stated verbatim at the start. Diffs land mid-prompt (requirement 3).
- A second framing (query box contents: "Box 3 has the ___") is also implemented
  and swept; final choice is whichever hits 100%/100% on base models.

## Findings log (appended as we go)

- 2026-08-09: repo conventions read (`abc_circuits/datasets/*.jsonl`): JSONL with
  `clean_prompt`, `corrupt_prompt`, `clean_answer`, `corrupt_answer`,
  `ld_candidates`; answers carry a leading space and are single tokens; eval is
  top-1 argmax at the final position. We keep this schema + metadata.
- 2026-08-09: user confirmed existing `custom/llama_6hop` datasets/reports are
  failed/unproven — not to be trusted or reused. All wording decisions here are
  re-derived empirically.
- 2026-08-09 (sweep r1, Llama-3.2-1B, n=100/cell): all box-label/statement
  variants bad (clean 10-30%, corrupt 25-78%). Failure diagnosis on corrupt
  (0-hop!) shows the model predicts box " A" positionally — it is not even
  binding query item -> box. Also: box-number answers (" 1") are TWO tokens in
  the Llama-3 tokenizer; switched to letter labels A-E (single token in both
  Llama-3 and GPT-2 BPEs). All 20 items + 16 person names verified single-token.
- 2026-08-09 (sweep r2): question-form query ("Where is the X now? The X is in
  box") fixes 0-hop retrieval: corrupt jumps to 0.90-0.94 (itemloc_q, 5 boxes).
  Person-name holders do NOT help. Preamble "Track where each item is." helps a
  little. CLEAN (2-hop) still 1-30% everywhere.
- 2026-08-09 (sweep r3): swap-verb variants (swap/trade/switch/box-centric) no
  clean gain; clean-only few-shot (fs2/fs3) lifts clean to ~0.33 at 3 boxes but
  DAMAGES corrupt (0.34-0.67) — shots that always demonstrate tracking bias the
  model away from the never-moved answer. Lesson: if few-shot is used, shots
  must be mixed (tracked + never-moved examples).
- 2026-08-09 (diag r3.5): even ONE hop is only 30% — errors are mostly "other
  box", not the item's initial box. The model cannot infer "A and B swapped =>
  item moved" at all in this format. The bottleneck is state updating, not
  chaining depth.
- 2026-08-09 (sweep r4, explicit moves): rendering each swap as two "The X
  moves to box B." sentences makes corrupt 0.95-0.99 (excellent) but clean
  stays 0.15-0.35 even though the answer is then literally the LAST stated
  location of the query item. The model fails at preferring the latest
  state — the core "tracking" behavior we want the circuit for.
- 2026-08-09 (sweeps r5-r8 + diag3/diag4, Llama-3.2-1B): CAPABILITY CEILING
  FOUND. Sequence markers, "at the end" cues, "started/was" tense contrast,
  numbered steps, ledger/induction formats, pronoun narratives, arrow
  notation, instruction prefixes — none help. Even with ZERO distractors and
  the item's moves stated explicitly and contiguously
  ("The cup is in box A. The cup moves to box B. The cup moves to box C.
  Where is the cup now?") the model answers "A" (the FIRST-stated location)
  every time. Llama-3.2-1B base has an overriding first-mention binding;
  later state updates are ignored. 1-hop zero-shot caps at ~0.4-0.5,
  2-hop at ~0.25 (chance). This is a capability limit, not a wording issue.
- 2026-08-09 (sweeps r9-r10, few-shot): mixed shots (tracked + never-moved
  demos) with uniform items + uniform query item + "Answer:" marker recover
  corrupt to 1.00 and lift clean 2-hop to 0.62 at (3 boxes, 2 swaps) —
  learnable in-context but far from ceiling. Corrupt collapses if shots
  demonstrate tracking only (r3) or if "Answer:" is used zero-shot (r10).
- 2026-08-09: USER DECISION — relax difficulty: clean must be at least 1-hop
  vs corrupt 0-hop. Sweeping the easiest configs: (3,1,1), (3,2,1), (4,2,1).
- 2026-08-09 (sweeps r11-r12): 1-hop + heavy mixed few-shot breaks through
  (v1 design, later deprecated for answer leakage — see below).
- 2026-08-09: **USER CAUGHT A FATAL FLAW IN v1**: the item-named move
  sentences ("The book moves to box C") state the answer directly — the model
  can cheat by copying the LAST occurrence of the query item, no tracking
  needed. v1 dataset moved to `datasets/deprecated_v1_item_named/`.
  FIX (v2): swaps name only BOXES ("Box A and box C are swapped."), items are
  never named after the init. The answer then appears NOWHERE in the prompt
  and must be inferred (book in A; A<->C swapped => book in C). A
  last-mention cheater outputs the init box, which equals the CORRUPT answer
  — so cheating is wrong on clean by construction, and the corrupt answer is
  exactly the cheat answer (ideal hard distractor for KL-guided pruning).
- 2026-08-09: **USER CAUGHT v2's STRUCTURAL DEGENERACY**: with 3 boxes,
  excluding the query box leaves ONE unordered pair, so every corrupt prompt
  was forced into "(X,Y) swapped, (X,Y) swapped" — the same pair twice, which
  nets to identity. Risks: (a) "swaps cancel out" is a perfect clean/corrupt
  tell — pruning could keep no-op-detectors instead of tracking components;
  (b) near-zero corrupt diversity. 4-box rescue attempts failed (fs32 clean
  0.67, shot-seed variants 0.43-0.44 — capacity limit is 3 boxes). Shot-seed
  robustness of the main (3,2,2) design confirmed: seeds 12345/7/777 give
  clean 1.00/1.00/1.00, corrupt 0.96-0.99.
  FIX (v3): corrupt gets exactly ONE swap between the two non-query boxes
  (`corrupt_n_swaps=1`). Effect: corrupt no longer cancels (the two non-query
  items genuinely exchange places), and since clean swaps are (Q,M),(M,F)
  while the corrupt pair is (M,F), **corrupt = clean minus the first swap
  sentence** — a clean, interpretable minimal diff. Remaining tell: corrupt
  prompts have 1 swap sentence vs clean's 2 (unavoidable at 3 boxes; a
  surface confound, and benign for activation-replacement pruning because
  swap-count does not affect either answer). v2 files moved to
  `datasets/deprecated_v2_canceling_corrupt/`.
  v3 sweep (n=200): clean 0.99/1.00, corrupt 1.00/1.00 -> curation filters
  the rare clean miss.
- 2026-08-09 (sweep r13-r14): box-named swaps + the full few-shot package
  (16 mixed shots, uniform items + uniform query, "Answer:" marker) TEACHES
  THE MODEL SWAP SEMANTICS IN-CONTEXT: zero-shot clean = 0.07 (it cannot
  infer swaps), 16-shot clean = 1.00 at (3 boxes, 2 swaps, query item in both
  swaps = GENUINE 2-HOP). This also overturns the earlier "1B can't do
  2-hop" conclusion — the earlier failures used item-named moves, where the
  model has no incentive to learn swap semantics.
  v2 LOCKED DESIGN: template swapbox_ans_q_fs16_uq, config (3,2,2):
    Llama-1B n=200: clean 1.00/1.00, corrupt 0.99/1.00 (curation filters the
    rare corrupt miss). Frontier mapped: 4 boxes fails (0.46-0.51), 3 hops
    fails (0.42-0.52), "contents of" phrasing kills corrupt (0.21) — the
    exact phrasing "Box A and box B are swapped." matters.
    v1->v2 improvement: 2 hops instead of 1, and no answer leakage.
  --- v1 DESIGN (DEPRECATED, kept for the record) ---
    template: swapmv_ans_q_fs16_uq_l, config (3 boxes, 1 swap, 1 hop)
    Llama-3.2-1B, n=200: clean 1.00/1.00, corrupt 1.00/1.00 (cand/vocab).
  Ingredients that each mattered (ablations observed across rounds):
    * question-form query + explicit "Answer:" marker (zero-shot corrupt
      retrieval 0.23 without marker structure, 1.00 with the full package)
    * 16 in-context shots, MIXED: alternating tracked (1-hop) and never-moved
      (0-hop) solved examples — clean-only shots destroy corrupt accuracy
    * shots use the SAME items and SAME query item as the target (uniform)
      — without this, clean collapses to ~0.5
    * query item's move sentence rendered LAST within the swap
    * swap stated as two explicit "moves to" sentences, not "swap places"
  Adding a distractor swap (3,2,1) drops clean to 0.56-0.77 even with 24
  shots — the model tracks only a single swap. Kept as possible future
  "hard" tier; the main dataset is (3,1,1).
  Dataset curation: pool -> keep only samples correct on BOTH clean and
  corrupt (strict: cand-restricted AND full-vocab argmax) -> splits. This
  guarantees the 100%/100% requirement by construction; keep-rate reported
  in gen_curated.log.
  Prompt shape (target part): "The cup is in box A. The ball is in box B.
  The key is in box C. The ball moves to box A. The cup moves to box B.
  Where is the cup now? Answer:" -> " B"  (1 hop)
  Corrupt: same init; the single swap is between the two NON-query items, so
  the query item never moves and the answer is its initial box (0 hops).
  Diff is mid-prompt (the two move sentences), satisfying requirement 3.

TODO evidence to collect:
- [ ] template sweep results (Llama-3.2-1B, gpt2)
- [ ] final variant: clean/corrupt accuracy at N=full dataset (must be 100/100)
- [ ] minimality: token diff stats
- [ ] zero-swap + scramble probe results
- [ ] answer balance histogram

## Final validation results — v3 (2026-08-09), THE dataset to use

Dataset: `datasets/multihop_swapbox_ans_q_fs16_uq_c1_3b2s2h_{train,val,test}.jsonl`
(1024/256/256). Genuine 2-hop box-swap tracking; answers never stated;
corrupt = clean minus the first swap sentence (single non-canceling swap
between the two non-query boxes).

- Base accuracy from disk (Llama-3.2-1B, test split, n=256):
  **clean = 1.0000, corrupt = 1.0000**. Curation keep-rates: 1417/1433,
  356/358, 356/358 (98.9-99.4%).
- Diversity: corrupt swap pair uniform over the 3 possible pairs (82-89 each);
  clean paths balanced over all 6 box-paths; answers near-uniform
  (clean {' A': 84, ' B': 82, ' C': 90}, corrupt {' A': 89, ' B': 85, ' C': 82}).
- No-leak: all splits pass — swap sentences never name the query item.
- Structural: all splits pass (clean = 2 hops, corrupt = 0 hops).
- Minimality (edit distance): median 1 sentence diff (corrupt drops exactly
  one swap sentence of clean); few-shot prefix byte-identical on 256/256.
- PROOF corrupt does not track (updated after a probe bug was caught):
  * structural: query item appears in NO swap statement; answer = init box
    verbatim; any correct computation needs 0 hops.
  * clean-side contrast: sensitivity flip-rate = 1.000 x4 — clean predictions
    are 100% swap-content-dependent. (acc-vs-new-truth 0.27-0.35 on
    unconstrained random swap configs vs ~99% on the dataset's constrained
    distribution — curation selects the solvable slice; 100% on the dataset
    is by construction.)
  * mutation probes on the corrupt swap sentence (gold cannot change):
    - words shuffled into gibberish: acc 0.973 — survives destruction of the
      swap's meaning.
    - replaced by an unrelated sentence: acc 0.965.
    - replaced by "Box D and box E are swapped." (boxes outside this world):
      acc 0.297 — the model notices out-of-world box symbols and starts
      guessing. Domain-shift effect, not tracking.
  * scramble probe: VACUOUS at 3 boxes (only one pair avoids the query box;
    scrambling cannot change anything) — kept for larger-box future datasets.
  * zero-swap probe: 0.43 — off-format artifact (every shot contains one
    swap); not evidence of tracking.
- GPT-2 (124M), informative: clean 0.06, corrupt 0.996 — GPT-2 does the
  0-hop retrieval fine but cannot learn 2-hop swap inference from the prefix.
  Requirement met on Llama-3.2-1B only.

## Final validation results — v2 (DEPRECATED: canceling-corrupt degeneracy)

Dataset: `datasets/multihop_swapbox_ans_q_fs16_uq_3b2s2h_{train,val,test}.jsonl`
(1024/256/256; repo-convention fields + full metadata). Genuine 2-hop
box-swap tracking; answers never stated; query item never named after init.

- Base accuracy from disk (Llama-3.2-1B, test split, n=256):
  **clean = 1.0000, corrupt = 1.0000**. Curation keep-rates:
  train 1414/1433 (98.7%), val 352/358, test 358/358.
- No-leak: 1024/1024, 256/256, 256/256 — no swap sentence ever names the
  query item; clean answer != corrupt answer. A last-mention cheater outputs
  the corrupt answer and is therefore wrong on 100% of clean samples; the
  model scores 100% on clean => it is NOT cheating.
- Structural: all splits pass (clean = exactly 2 hops, corrupt = 0 hops,
  answers re-derive by simulation).
- Minimality: median 1 token, max 2 tokens diff (one box letter in one swap
  sentence); few-shot prefix byte-identical on 256/256. Diff is mid-prompt.
- Corrupt scramble probe: acc 1.000 x4 — corrupt answer invariant to swap
  content at constant format.
- Clean sensitivity probe: flip-rate 1.000 x4 (prediction always follows the
  swaps), acc-vs-new-truth 0.44-0.52. READ THIS HONESTLY: the model's 2-hop
  inference is ~98.7% reliable ON THE DATASET'S CONSTRAINED DISTRIBUTION
  (query item always in both swaps, exactly 2 hops) but only ~50% on
  unconstrained random swap configurations (where hops vary 0-2 and chains
  break the pattern). Curation selects the solvable slice; 100% on the
  dataset is by construction. The flip-rate proves predictions are fully
  swap-dependent — genuine routing, not memorization.
- Zero-swap probe: 0.27 — format-shift artifact (all shots contain 2 swaps;
  a move-less target is off-distribution). Not evidence of tracking; the
  valid invariance test is the scramble probe (passes).
- Balance: clean {' A': 77, ' B': 95, ' C': 84}, corrupt {' A': 86, ' B': 75,
  ' C': 95} — near-uniform.
- GPT-2 (124M), informative: clean 0.07, corrupt 0.86 — fails clean (GPT-2
  cannot learn swap semantics from the prefix). Requirement met on
  Llama-3.2-1B only.

## Final validation results — v1 (DEPRECATED: answer leak, 1-hop only)

(v1 files live in `datasets/deprecated_v1_item_named/` — do not use. Original
path: `datasets/multihop_swapmv_ans_q_fs16_uq_l_3b1s1h_*.jsonl`.)

(1024/256/256 samples; repo-convention fields: clean_prompt, corrupt_prompt,
clean_answer, corrupt_answer, ld_candidates, all_candidates + full metadata).

- Base accuracy from disk (Llama-3.2-1B, test split, n=256):
  **clean = 1.0000, corrupt = 1.0000** (candidate-restricted; full-vocab argmax
  also 1.00 during curation — keep-rate was 1433/1433, 358/358, 358/358, i.e.
  no samples needed filtering).
- Structural: 1024/1024, 256/256, 256/256 pass (clean = exactly 1 hop, corrupt
  = 0 hops with query item absent from all swaps, answers re-derive by
  simulation, clean answer != corrupt answer).
- Minimality: clean/corrupt differ by median 2 tokens, max 4 (the two move
  sentences); few-shot prefix verified byte-identical between clean and
  corrupt on all 256 test samples. Diff is mid-prompt, so corrupt activations
  diverge before the query — the tracking computation itself differs.
- Clean sensitivity probe (the important one): replacing clean swaps with
  random swaps changes the ground truth; the model's prediction follows the
  NEW truth with acc 1.000 and flip-rate 1.000 (4 probe rounds). The clean
  answer provably routes through the swap content — genuine tracking, not a
  shortcut.
- Corrupt scramble probe: replacing corrupt swaps with random swaps that never
  touch the query item (answer invariant by construction): acc stays 1.000
  (4 rounds). The corrupt answer provably does NOT route through swap content.
- Corrupt zero-swap probe (strip move sentences entirely): acc drops to 0.68 —
  this is a FORMAT-SHIFT artifact, not evidence of tracking: every few-shot
  example contains move sentences, so a move-less target is off-distribution.
  The scramble probe (content-invariance at constant format) is the valid
  test, and it passes at 1.000.
- Balance: clean answers {' A': 92, ' B': 87, ' C': 77}, corrupt
  {' A': 83, ' B': 84, ' C': 89} — near-uniform over boxes.
- GPT-2 (124M), informative only: clean 0.17, corrupt 0.68 — fails; GPT-2
  lacks even the 0-hop retrieval at this prompt length (16 shots, ~800
  tokens). The 100%/100% requirement is met on the bare-minimum model
  (Llama-3.2-1B) only.

## Remaining caveats (honest list)

- The dataset is 1-hop (per user decision: ">=1 hop vs no hop"). A 2-hop
  variant is NOT possible at ceiling on base Llama-3.2-1B (cap ~0.62 even with
  few-shot). A distractor-swap variant (3,2,1) caps at 0.77. Both are dead
  ends at 100%.
- The 16-shot prefix is part of every prompt (~800 tokens). Clean and corrupt
  share it byte-identically, so it does not confound the diff, but pruning
  compute is heavier and the circuit includes in-context-learning machinery
  use. If a zero-shot variant is ever needed, the best zero-shot template is
  swapmv_end_q (corrupt 1.00, but clean only ~0.3-0.5: unusable).
- We cannot prove the model does not internally simulate swaps on corrupt
  prompts; we proved the corrupt ANSWER is invariant to swap content
  (scramble probe) and absent from every swap statement (structural). That is
  the strongest obtainable guarantee.

## Files

- `task.py` — task construction, exact simulation, 40+ prompt templates,
  few-shot rendering. All ground truth is simulated, never parsed from text.
- `eval_base.py` — base-model eval (`sweep` / `file` modes).
- `gen_dataset.py` — raw (unfiltered) dataset writer.
- `gen_curated.py` — curated writer: keeps only samples the base model gets
  right on both sides (this run: 100% keep-rate, no filtering needed).
- `validate.py` — minimality / structural / sensitivity / scramble /
  zero-swap / balance checks.
- `sweep*.log`, `diag*.log`, `gen_curated.log`, `final_checks.log` — raw
  experiment logs backing every claim above.

