#!/usr/bin/env python3
"""
2-Hop State-Machine Dataset Generator (Non-Living Logic)  (single 40GB A100, one model at a time)
==============================================================================

Generates exactly N (default 1,000) unique, interpretability-grade 6-hop
"chain-of-custody" reasoning examples on ONE 40GB A100 (e.g. a Jetstream2
instance). Inference uses the high-throughput vLLM engine, and only ONE model
is resident in VRAM at any moment. No SLURM, no vLLM HTTP server -- everything
runs from a single `python generate_dataset_local.py` command.

Because an 8B generator and a 32B judge cannot coexist in 40GB, the pipeline
runs in alternating PHASES, and each phase runs as its OWN SUBPROCESS so the GPU
is guaranteed clean between models (when the child process exits, CUDA frees
100% of its VRAM -- far more reliable than trying to tear vLLM down in-process):

    while accepted < target:                         # parent orchestrator
        PHASE 1 (GENERATE, child)  spin up vLLM with Llama-3.1-8B-Instruct,
                                   batch-generate (remaining_needed + buffer)
                                   fresh, unique, structurally-valid 7-element
                                   arrays into a draft file, then EXIT (free VRAM).
        PHASE 2 (JUDGE, child)     spin up vLLM with Qwen2.5-32B-Instruct-AWQ --
                                   the strong semantic judge -- score every pending
                                   draft against the FULL interpretability checklist;
                                   write accepted rows to the final file and rejected
                                   ones (with reason) to a separate file; then EXIT.

    e.g. round 1 generates 1,100 -> judge keeps 600; round 2 generates 500
    (400 still needed + 100 buffer) -> judge keeps 400 -> done at 1,000.

vLLM uses paged-attention batching, so generation is dramatically faster than a
plain `transformers` .generate() loop. High GPU memory use is EXPECTED and good:
vLLM pre-reserves a KV-cache pool (`--gpu-memory-utilization`, default 0.90) to
maximize throughput; since only one model is loaded at a time that is free real
estate to use.

Element schema, dedupe, simplicity gate, theme rotation + randomized in-theme
sub-scenario injection, temperature cycling, clean/corrupted f-string
templating, and the resumable / rejected-cache logic are all UNCHANGED.

A NOTE ON THE JUDGE PRECISION
-----------------------------
The A100 is Ampere (sm80) with no native FP8 compute, so the judge defaults to
the official AWQ (int4) build `Qwen/Qwen2.5-32B-Instruct-AWQ` (~19GB weights),
which vLLM runs fast on Ampere via Marlin kernels and which keeps the 32B
model's reasoning quality. Override with `--judge-model` / `--judge-quantization`
(e.g. an FP8 build on a Hopper card).

Clean vs. corrupted (dual-chain minimal pair for activation patching)
--------------------------------------------------------------------
Each example has TWO objects and TWO containers: a PORTABLE mobile container
that is forwarded down a 6-role chain, and a STATIONARY fixed container that
stays put. Only the first two "is placed in" sentences differ, which flips the
answer:
    clean     : target_item in the mobile container -> rides 6 hops -> " role6"
    corrupted : target_item in the fixed container  -> never moves  -> " fixed_container"
So the clean prompt forces the model to track all six hops, while the corrupted
prompt traps the target item in stationary storage. The two TARGETS differ.
Vocabulary is kept toddler-simple (ball, cup, bag, dog, boy) and the frame ends
"... is held by the", which reads correctly for both a living role and a
container. Verbs: "is put in the" / "is given to the" / "hands it to the".

Why the requirements are written the way they are
--------------------------------------------------
This data later interprets a SEPARATE model (activation patching), so every
element must denote ONE stable, unambiguous concept that the studied model
will represent and route as we intend: no polysemy (one token == one meaning),
and no two elements it could collapse into the same internal representation
(near-synonyms, shared head nouns, hypernym/hyponym pairs).

Also implemented: 16-theme prompt rotation with randomized in-theme sub-scenario
injection, temperature cycling (0.1 -> +0.1 per 5 consecutive empty batches ->
cap 0.7, reset on a productive batch), robust JSON parsing, tqdm progress, and
resumable JSONL output.

--------------------------------------------------------------------------
RUNNING ON A JETSTREAM2 INSTANCE
--------------------------------------------------------------------------
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # (first run downloads the weights; set HF_HOME to a big volume if needed)
    python generate_dataset_local.py --target 1000

This single command is the parent orchestrator; it internally launches the
`generate` and `judge` phases as subprocesses (you can also run those phases
directly: `python generate_dataset_local.py generate --n 1100` etc.).

Outputs:
    draft_chains.jsonl                  pending raw drafts awaiting the judge
    6_hop_state_machine.jsonl           accepted examples
    6_hop_state_machine.rejected.jsonl  judge-rejected examples (+ reason)

The accepted and rejected signature sets are reloaded on restart, so re-runs
never regenerate or re-judge a chain you've already produced; any drafts left
in draft_chains.jsonl are judged first before more are generated.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from typing import List, Optional

from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. EXPANDED THEME MATRIX (THE PROMPT ROTATION)
# ---------------------------------------------------------------------------
# Each example is a 10-element array following the dual-chain corruption schema:
#   [0] target_item     - the object we ask about
#   [1] distractor_item - a second, different object
#   [2] mobile_container - PORTABLE container that is forwarded down the 6 roles
#   [3] fixed_container  - STATIONARY storage that stays put (the 1-hop "trap")
#   [4..9] role1..role6  - six DIFFERENT agents who hand the mobile container on
# In CLEAN, target_item starts in the mobile container -> ends at role6.
# In CORRUPTED, the two placements are swapped -> target_item sits in the fixed
# container and never moves -> answer is the fixed container.
THEMES: dict[str, List[str]] = {
    "The Warehouse": [
        "bolt", "tool", "box", "bin",
        "crate", "pallet", "truck", "train", "ship", "plane",
    ],
    "The Post Office": [
        "card", "note", "bag", "sack",
        "cart", "van", "truck", "plane", "train", "depot",
    ],
    "The Laboratory": [
        "slide", "tube", "rack", "case",
        "cooler", "fridge", "vault", "truck", "van", "ship",
    ],
    "The NonLiving Kitchen": [
        "bean", "nut", "cup", "jar",
        "bowl", "pan", "fridge", "cart", "van", "truck",
    ],
    "The Storage Room": [
        "coin", "gem", "safe", "vault",
        "chest", "truck", "train", "plane", "ship", "bag",
    ],
}

# ---------------------------------------------------------------------------
# RANDOMIZED SEED INJECTION (diversity without nonsense)
# ---------------------------------------------------------------------------
# For every attempt we pick a theme AND a random in-theme sub-scenario below,
# then ask the model to build a fresh, *logically realistic* chain around it.
# Because each variation stays inside the theme's real-world domain, diversity
# explodes (theme x variation x sampling) while every sequence remains valid --
# unlike forcing unrelated anchor words, which would manufacture nonsense.
# Each variation is a concrete TARGET-ITEM idea inside the theme's domain. The
# generator builds a fresh 10-element array around it (inventing the distractor
# item, both containers, and the six roles), which keeps every example realistic
# while exploding diversity (theme x variation x sampling).
THEME_VARIATIONS: dict[str, List[str]] = {
    "The Warehouse": ["bolt", "tool", "gear", "part", "nail", "wire"],
    "The Post Office": ["card", "note", "stamp", "bill", "form", "mail"],
    "The Laboratory": ["slide", "tube", "vial", "swab", "pill", "drug"],
    "The NonLiving Kitchen": ["bean", "nut", "seed", "plum", "pea", "egg"],
    "The Storage Room": ["coin", "gem", "ring", "gold", "cash", "bond"],
}

# ---------------------------------------------------------------------------
# Shared, interpretability-focused requirement list (used by BOTH prompts so
# the generator aims for what the verifier enforces -> higher accept rate).
# ---------------------------------------------------------------------------
INTERP_REQUIREMENTS = (
    "The data is a list of 6 elements in this exact order:\n"
    "  [1] target_item     - the physical object the question asks about\n"
    "  [2] distractor_item - a second, different physical object\n"
    "  [3] mobile_container - a container that holds the object and can be moved\n"
    "  [4] fixed_container  - a container that stays put (the trap)\n"
    "  [5..6] two outer_containers - two DIFFERENT NON-LIVING locations, vehicles, "
    "or larger containers, in a sensible nested order, that the mobile container is moved into.\n\n"
    "They are dropped into this fixed frame:\n"
    "  'The <item> is placed in the <container>. ... The <mobile> is moved to the "
    "<outer1>. The <outer1> is moved to the <outer2>. ... The <item> is in the "
    "<outer2 OR fixed_container>.'\n\n"
    "Approve when ALL of these hold:\n"
    "1. TODDLER-SIMPLE WORDS: Every element must be a fundamental, extremely common "
    "1-2 syllable noun (e.g. box, bag, cart, truck, train). REJECT technical jargon.\n"
    "2. DISTINCT: all 6 strings are different from each other.\n"
    "3. RIGHT KINDS: NO LIVING CREATURES ALLOWED. ALL items must be inanimate physical objects, "
    "and ALL containers must be non-living containers, locations, or vehicles.\n"
    "4. BOTH ITEMS FIT BOTH CONTAINERS: each item fits inside EITHER mobile or fixed.\n"
    "5. SENSIBLE CHAIN: the containers make a natural nesting/moving sequence (e.g., box -> truck -> ship).\n"
    "6. CLEAN FORMATTING: lowercase, normal spaces, no code-style tokens."
)

# ---- Generator prompt -----------------------------------------------------
GEN_SYSTEM_TEMPLATE = (
    "You are a strict data-generation engine for a mechanistic-interpretability "
    "dataset. You output ONLY a single JSON array of EXACTLY 6 strings — no "
    "prose, no keys, no markdown, no backticks.\n\n"
    "The 6 strings MUST be in this exact order:\n"
    "[target_item, distractor_item, mobile_container, fixed_container, "
    "outer1, outer2]\n\n"
    "These strings are dropped into this fixed sentence frame:\n"
    "  'The <target_item> is placed in the <mobile_container>. "
    "The <distractor_item> is placed in the <fixed_container>. "
    "The <mobile_container> is moved to the <outer1>. The <outer1> is moved to "
    "the <outer2>. "
    "The <target_item> is in the ...'\n"
    "So: BOTH items must fit inside BOTH inner containers, and the mobile_container "
    "is moved into larger and larger NON-LIVING locations/vehicles.\n\n"
    "USE TODDLER-SIMPLE WORDS ONLY. NO LIVING CREATURES ALLOWED.\n\n"
    "Follow EVERY requirement:\n"
    f"{INTERP_REQUIREMENTS}\n\n"
    "Theme to follow: {theme_name}.\n"
    "STYLE example only (invent a completely different scenario): {example}\n"
    "Output ONLY the raw JSON array of 6 strings."
)
GEN_USER_TEMPLATE = (
    "Generate ONE brand-new, original 6-element JSON array for the theme "
    '"{theme_name}". Output only the JSON array.'
)
# Used when a randomized in-theme target item is injected (the usual path).
GEN_USER_VARIATION_TEMPLATE = (
    'Generate ONE brand-new, original 6-element JSON array for the theme '
    '"{theme_name}".\n'
    "Use this as the target_item (element 1): {variation}. Then invent a "
    "different distractor_item, a portable mobile_container, a stationary "
    "fixed_container (both items must fit both containers), and six different "
    "realistic roles for this theme. Vary the wording from the style example. "
    "Output only the JSON array."
)

# ---- Verifier prompt ------------------------------------------------------
VERIFY_SYSTEM = (
    "You are a fair validator for a mechanistic-interpretability dataset. You are "
    "given a THEME and a 6-element array "
    "[target_item, distractor_item, mobile_container, fixed_container, "
    "outer1, outer2]. This data drives a clean/corrupted activation-patching pair, "
    "so each element should be a clean, distinct, everyday concept and the "
    "scenario should be realistic.\n\n"
    "Apply this checklist:\n"
    f"{INTERP_REQUIREMENTS}\n\n"
    "Be reasonable: APPROVE anything realistic, clearly distinct, and written in "
    "toddler-simple words; only reject on a concrete, specific violation (name "
    "which rule and why). The most common real reason to reject is rule 1 -- a "
    "word that is too complex/technical for a small child. Do NOT reject for "
    "mere stylistic taste.\n\n"
    "Respond with ONLY a JSON object on a single line, no markdown, no extra "
    'text. Use exactly: {"valid": true}  or  '
    '{"valid": false, "reason": "<which rule failed and why>"}.'
)
VERIFY_USER_TEMPLATE = (
    "Theme: {theme_name}\n"
    "Array (JSON, 6 strings): {chain}\n"
    "Validate against the checklist and reply with the JSON verdict."
)

# ---------------------------------------------------------------------------
# 3. STRICT TEMPLATING (pure Python f-strings; the LLM never writes prose).
# Dual-chain minimal pair. CLEAN puts the target item in the mobile container
# (it travels all 6 hops -> ends held by role6). CORRUPTED swaps the two
# placements so the target item sits in the fixed container (it never moves ->
# stays held by the fixed container). Only the first two sentences differ; the
# targets DIFFER (role6 vs fixed container). The "is held by the" ending is
# grammatically flawless for BOTH a living role and a static container.
# ---------------------------------------------------------------------------
_CHAIN_TEMPLATE = (
    " The {mobile} is moved to the {r1}. The {r1} is moved to the {r2}. "
    "The {target} is in the"
)
CLEAN_TEMPLATE = (
    "The {target} is placed in the {mobile}. "
    "The {distractor} is placed in the {fixed}." + _CHAIN_TEMPLATE
)
CORRUPTED_TEMPLATE = (
    "The {distractor} is placed in the {mobile}. "
    "The {target} is placed in the {fixed}." + _CHAIN_TEMPLATE
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no network)
# ---------------------------------------------------------------------------
def build_gen_messages(theme_name: str, example: List[str],
                       variation: str = "") -> List[dict]:
    """Generator messages. When a `variation` (random in-theme sub-scenario) is
    given, it is injected to force diverse-but-realistic chains."""
    if variation:
        user = GEN_USER_VARIATION_TEMPLATE.format(theme_name=theme_name,
                                                  variation=variation)
    else:
        user = GEN_USER_TEMPLATE.format(theme_name=theme_name)
    return [
        {"role": "system", "content": GEN_SYSTEM_TEMPLATE.format(
            theme_name=theme_name, example=json.dumps(example, ensure_ascii=False))},
        {"role": "user", "content": user},
    ]


def build_verify_messages(theme_name: str, arr: List[str]) -> List[dict]:
    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": VERIFY_USER_TEMPLATE.format(
            theme_name=theme_name, chain=json.dumps(arr, ensure_ascii=False))},
    ]


def strip_reasoning(raw: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit inline."""
    if not raw:
        return ""
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def extract_json_array(raw: str) -> Optional[List[str]]:
    """Extract/parse the LAST JSON array from `raw` (final answer wins)."""
    if not raw:
        return None
    cleaned = strip_reasoning(raw).replace("```json", "").replace("```", "").strip()
    matches = re.findall(r"\[.*?\]", cleaned, flags=re.DOTALL)
    candidates = matches[::-1] if matches else [cleaned]
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def extract_json_object(raw: str) -> Optional[dict]:
    """Extract/parse the LAST JSON object from `raw` (verifier verdict)."""
    if not raw:
        return None
    cleaned = strip_reasoning(raw).replace("```json", "").replace("```", "").strip()
    matches = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    candidates = matches[::-1] if matches else [cleaned]
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def is_structurally_valid(arr: object) -> bool:
    """Local structural gate (run before the judge) for the 6-element schema:
        - exactly 6 non-empty strings,
        - SIMPLE WORDING: each element is 1-2 whitespace-separated words,
        - all 6 elements are mutually distinct (case-insensitive).
    Semantic checks (portable vs stationary container, both items fit, realistic
    role chain) are left to the judge.
    """
    if not isinstance(arr, list) or len(arr) != 6:
        return False
    if not all(isinstance(s, str) and s.strip() for s in arr):
        return False
    if not all(1 <= len(s.split()) <= 2 for s in arr):   # 1-2 words each
        return False
    lowered = [s.strip().lower() for s in arr]
    if len(set(lowered)) != 6:                          # all 6 distinct
        return False
    return True


def normalize_entity(s: str) -> str:
    """Clean a single string so it reads as plain English, not a code token.

    Fixes the model's occasional snake_case / camelCase output:
      'encrypted_USB_drive' -> 'encrypted USB drive',
      'Station_Chief'        -> 'Station Chief'.
    """
    s = s.replace("_", " ")                       # snake_case -> spaces
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)    # camelCase  -> spaces
    s = re.sub(r"\s+", " ", s).strip()            # collapse whitespace
    return s


def normalize_array(arr: List[str]) -> List[str]:
    """Apply `normalize_entity` to every string element of a parsed array."""
    return [normalize_entity(s) if isinstance(s, str) else s for s in arr]


def signature_of(arr: List[str]) -> str:
    return "|".join(s.strip().lower() for s in arr)


def format_record(theme_name: str, arr: List[str]) -> dict:
    """Assemble the final JSONL record with pure f-string templating.

    Dual-chain minimal pair:
      clean     -> target item rides the mobile container 2 hops to outer2.
      corrupted -> target item is placed in the fixed container and stays.
    The two targets DIFFER (outer2 vs fixed container). Note the leading space on
    the targets (they continue "...currently in the").
    """
    target, distractor, mobile, fixed, r1, r2 = arr
    fields = dict(target=target, distractor=distractor, mobile=mobile, fixed=fixed,
                  r1=r1, r2=r2)
    return {
        "entities": list(arr),
        "clean_prompt": CLEAN_TEMPLATE.format(**fields),
        "clean_target": f" {r2}",
        "corrupted_prompt": CORRUPTED_TEMPLATE.format(**fields),
        "corrupted_target": f" {fixed}",
        "hops": 2,
        "theme": theme_name,
        "verified": True,
    }


def load_existing_signatures(path: str) -> set[str]:
    """Resume support: rebuild `seen_signatures` from an existing output file."""
    seen: set[str] = set()
    if not os.path.exists(path):
        return seen
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(rec, dict) and isinstance(rec.get("entities"), list):
                seen.add(signature_of(rec["entities"]))
    return seen


# ---------------------------------------------------------------------------
# vLLM phase helpers. Each phase runs in its OWN process (see the orchestrator),
# so simply building an `LLM` here claims the GPU and exiting the process frees
# 100% of it -- no fragile in-process teardown needed.
# ---------------------------------------------------------------------------
def _build_llm(model: str, *, dtype: str, gpu_mem_util: float, max_model_len: int,
               quantization: Optional[str], enforce_eager: bool, seed: int):
    from vllm import LLM
    kwargs = dict(
        model=model,
        dtype=dtype,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        seed=seed,
    )
    if quantization and quantization.lower() not in ("", "none", "auto"):
        kwargs["quantization"] = quantization
    return LLM(**kwargs)


def _chat_texts(llm, conversations, sampling_params, use_tqdm: bool = False) -> List[str]:
    """Run vLLM batched chat and return the assistant text for each conversation."""
    outputs = llm.chat(conversations, sampling_params, use_tqdm=use_tqdm)
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


# ---------------------------------------------------------------------------
# PHASE 1 -- GENERATE (runs as a subprocess): batch-produce fresh, unique,
# structurally-valid drafts with the fast 8B model until `--n` are written.
# ---------------------------------------------------------------------------
def cmd_generate(args) -> int:
    from vllm import SamplingParams

    seen = load_existing_signatures(args.output)
    rejected = load_existing_signatures(args.rejected_output)
    draft_sigs = load_existing_signatures(args.draft_output)
    avoid = seen | rejected
    theme_names = list(THEMES.keys())

    print(f"[generate] loading {args.gen_model} on vLLM "
          f"(util={args.gpu_memory_utilization}, max_len={args.max_model_len})...")
    llm = _build_llm(args.gen_model, dtype=args.dtype,
                     gpu_mem_util=args.gpu_memory_utilization,
                     max_model_len=args.max_model_len,
                     quantization=args.gen_quantization,
                     enforce_eager=args.enforce_eager, seed=args.seed)

    produced = 0
    temperature = 0.1
    stalls = 0
    batches = 0
    pbar = tqdm(total=args.n, desc="Phase 1 generate")
    with open(args.draft_output, "a", encoding="utf-8") as draft_fh:
        while produced < args.n and batches < args.max_gen_batches:
            batches += 1
            convos, themes = [], []
            for _ in range(args.gen_batch_size):
                theme = random.choice(theme_names)
                variation = random.choice(THEME_VARIATIONS.get(theme, [""]))
                convos.append(build_gen_messages(theme, THEMES[theme], variation))
                themes.append(theme)
            sp = SamplingParams(temperature=temperature, top_p=0.95,
                                max_tokens=args.gen_max_tokens)
            raws = _chat_texts(llm, convos, sp)

            new_this_batch = 0
            for theme, raw in zip(themes, raws):
                arr = extract_json_array(raw)
                if isinstance(arr, list):
                    arr = normalize_array(arr)
                if arr is None or not is_structurally_valid(arr):
                    continue
                sig = signature_of(arr)
                if sig in avoid or sig in draft_sigs:
                    continue
                draft_sigs.add(sig)
                draft_fh.write(json.dumps({"entities": list(arr), "theme": theme},
                                          ensure_ascii=False) + "\n")
                draft_fh.flush()
                produced += 1
                new_this_batch += 1
                pbar.update(1)
                if produced >= args.n:
                    break
            if new_this_batch == 0:
                stalls += 1
                if stalls % 5 == 0 and temperature < 0.7:
                    temperature = round(min(0.7, temperature + 0.1), 1)
                    tqdm.write(f"[temp] {stalls} empty batches -> temperature "
                               f"raised to {temperature}")
            else:
                stalls = 0
                temperature = 0.1
    pbar.close()
    print(f"[generate] wrote {produced} new drafts -> {args.draft_output}")
    if produced < args.n:
        print(f"[generate] WARNING: only {produced}/{args.n} (hit max-gen-batches).")
    return 0


# ---------------------------------------------------------------------------
# PHASE 2 -- JUDGE (runs as a subprocess): score every pending draft with the
# strong 32B model; write accepted / rejected; then clear the draft file.
# ---------------------------------------------------------------------------
def _read_pending(draft_path: str, seen: set, rejected: set):
    pending = []
    if not os.path.exists(draft_path):
        return pending
    with open(draft_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(rec, dict) and isinstance(rec.get("entities"), list):
                sig = signature_of(rec["entities"])
                if sig in seen or sig in rejected:
                    continue
                pending.append((rec.get("theme", "?"), rec["entities"]))
    return pending


def cmd_judge(args) -> int:
    seen = load_existing_signatures(args.output)
    rejected = load_existing_signatures(args.rejected_output)
    pending = _read_pending(args.draft_output, seen, rejected)
    print(f"[judge] {len(pending)} pending drafts; have {len(seen)}/{args.target}")

    accepted_now = 0
    with open(args.output, "a", encoding="utf-8") as out_fh, \
            open(args.rejected_output, "a", encoding="utf-8") as rej_fh:
        if args.no_verify:
            for theme, arr in pending:
                if len(seen) >= args.target:
                    break
                sig = signature_of(arr)
                seen.add(sig)
                out_fh.write(json.dumps(format_record(theme, arr),
                                        ensure_ascii=False) + "\n")
                out_fh.flush()
                accepted_now += 1
        elif pending:
            from vllm import SamplingParams
            print(f"[judge] loading {args.judge_model} on vLLM "
                  f"(util={args.gpu_memory_utilization})...")
            llm = _build_llm(args.judge_model, dtype=args.dtype,
                             gpu_mem_util=args.gpu_memory_utilization,
                             max_model_len=args.max_model_len,
                             quantization=args.judge_quantization,
                             enforce_eager=args.enforce_eager, seed=args.seed)
            sp = SamplingParams(temperature=args.verify_temperature, top_p=1.0,
                                max_tokens=args.verify_max_tokens)
            convos = [build_verify_messages(t, a) for (t, a) in pending]
            raws = _chat_texts(llm, convos, sp, use_tqdm=True)
            for (theme, arr), raw in zip(pending, raws):
                sig = signature_of(arr)
                obj = extract_json_object(raw)
                valid = bool(obj and obj.get("valid") is True)
                if valid:
                    if len(seen) < args.target:
                        seen.add(sig)
                        out_fh.write(json.dumps(format_record(theme, arr),
                                                ensure_ascii=False) + "\n")
                        out_fh.flush()
                        accepted_now += 1
                else:
                    reason = (str(obj.get("reason", "no reason given")) if obj
                              else "unparseable judge verdict")
                    rejected.add(sig)
                    rej_fh.write(json.dumps({"entities": list(arr), "theme": theme,
                                             "reason": reason,
                                             "rejected_by": args.judge_model},
                                            ensure_ascii=False) + "\n")
                    rej_fh.flush()

    open(args.draft_output, "w", encoding="utf-8").close()   # clear judged drafts
    print(f"[judge] accepted {accepted_now}; total now {len(seen)}/{args.target}")
    return 0


# ---------------------------------------------------------------------------
# Orchestrator (the default command): alternate generate / judge subprocesses,
# one model in VRAM at a time, until `--target` accepted rows exist.
# ---------------------------------------------------------------------------
def _phase_argv(subcmd: str, args, extra: List[str]) -> List[str]:
    """Build the argv for a child phase, forwarding all shared options."""
    argv = [sys.executable, os.path.abspath(__file__), subcmd,
            "--output", args.output,
            "--rejected-output", args.rejected_output,
            "--draft-output", args.draft_output,
            "--gen-model", args.gen_model,
            "--judge-model", args.judge_model,
            "--dtype", args.dtype,
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--gen-max-tokens", str(args.gen_max_tokens),
            "--verify-max-tokens", str(args.verify_max_tokens),
            "--verify-temperature", str(args.verify_temperature),
            "--gen-batch-size", str(args.gen_batch_size),
            "--max-gen-batches", str(args.max_gen_batches),
            "--target", str(args.target),
            "--seed", str(args.seed)]
    if args.gen_quantization:
        argv += ["--gen-quantization", args.gen_quantization]
    if args.judge_quantization:
        argv += ["--judge-quantization", args.judge_quantization]
    if args.enforce_eager:
        argv += ["--enforce-eager"]
    return argv + extra


def cmd_orchestrate(args) -> int:
    seen = load_existing_signatures(args.output)
    rejected = load_existing_signatures(args.rejected_output)
    if seen:
        print(f"[resume] {len(seen)} accepted examples already in {args.output}.")
    if rejected:
        print(f"[resume] {len(rejected)} previously-rejected chains "
              f"(won't be regenerated or re-judged).")
    if len(seen) >= args.target:
        print(f"[done] Target {args.target} already met. Nothing to do.")
        return 0

    print(f"[init] generator={args.gen_model}\n"
          f"[init] judge={'(disabled)' if args.no_verify else args.judge_model}\n"
          f"[init] target={args.target}, buffer={args.buffer}; each phase runs as "
          f"an isolated subprocess (one model in VRAM at a time)")

    no_progress = 0
    for rnd in range(1, args.max_rounds + 1):
        seen = load_existing_signatures(args.output)
        if len(seen) >= args.target:
            break
        rejected = load_existing_signatures(args.rejected_output)
        draft_sigs = load_existing_signatures(args.draft_output)
        pending_eff = len(draft_sigs - seen - rejected)
        remaining = args.target - len(seen)
        to_generate = max(0, remaining + args.buffer - pending_eff)
        print(f"\n===== Round {rnd}: have {len(seen)}/{args.target}, "
              f"{pending_eff} drafts pending, generating {to_generate} more =====")

        # ----- PHASE 1: GENERATE (isolated subprocess) -------------------
        if to_generate > 0:
            subprocess.run(_phase_argv("generate", args, ["--n", str(to_generate)]),
                           check=True)

        # ----- PHASE 2: JUDGE (isolated subprocess) ----------------------
        judge_extra = ["--no-verify"] if args.no_verify else []
        subprocess.run(_phase_argv("judge", args, judge_extra), check=True)

        seen_after = load_existing_signatures(args.output)
        gained = len(seen_after) - len(seen)
        print(f"[round {rnd}] gained {gained}; total {len(seen_after)}/{args.target}")
        no_progress = no_progress + 1 if gained == 0 else 0
        if no_progress >= 3:
            print("[stop] 3 rounds with zero new accepts; aborting to avoid a spin.")
            break

    final = len(load_existing_signatures(args.output))
    rej = len(load_existing_signatures(args.rejected_output))
    if final >= args.target:
        print(f"\n[done] {final} unique verified examples -> {args.output}")
    else:
        print(f"\n[stop] Produced {final}/{args.target} -> {args.output}")
    print(f"[info] {rej} rejected chains -> {args.rejected_output}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing (shared options live on a parent parser used by every cmd).
# ---------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output", type=str, default="2_hop_state_machine_logic.jsonl")
    p.add_argument("--rejected-output", type=str,
                   default="2_hop_state_machine_logic.rejected.jsonl")
    p.add_argument("--draft-output", type=str, default="draft_chains.jsonl")
    p.add_argument("--gen-model", type=str,
                   default=os.getenv("GEN_MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    p.add_argument("--judge-model", type=str,
                   default=os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"))
    p.add_argument("--gen-quantization", type=str, default=None,
                   help="vLLM quantization for the generator (default: none/bf16).")
    p.add_argument("--judge-quantization", type=str, default=None,
                   help="vLLM quantization for the judge (default: auto-detect, "
                        "e.g. AWQ from the model config).")
    p.add_argument("--dtype", type=str, default="auto",
                   help="vLLM dtype (auto/bfloat16/float16).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                   help="Fraction of the 40GB A100 vLLM may reserve (KV cache pool).")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gen-batch-size", type=int, default=256,
                   help="Prompts submitted to vLLM per generation batch.")
    p.add_argument("--gen-max-tokens", type=int, default=256)
    p.add_argument("--verify-max-tokens", type=int, default=256)
    p.add_argument("--verify-temperature", type=float, default=0.0)
    p.add_argument("--max-gen-batches", type=int, default=2000,
                   help="Safety cap on batches inside one generate phase.")
    p.add_argument("--target", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--enforce-eager", action="store_true",
                   help="Disable CUDA graphs (lower VRAM, slightly slower).")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the 32B judge and accept all valid drafts.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-hop interpretability dataset (non-living logic) on a single 40GB A100 via "
                    "vLLM; one model at a time using isolated phase subprocesses.")
    sub = parser.add_subparsers(dest="command")

    p_orch = sub.add_parser("orchestrate", help="(default) run the full loop.")
    _add_common(p_orch)
    p_orch.add_argument("--buffer", type=int, default=100,
                        help="Extra drafts per round to absorb judge rejections.")
    p_orch.add_argument("--max-rounds", type=int, default=25)

    p_gen = sub.add_parser("generate", help="Phase 1 only (used as a subprocess).")
    _add_common(p_gen)
    p_gen.add_argument("--n", type=int, required=True,
                       help="Number of new unique drafts to produce.")

    p_judge = sub.add_parser("judge", help="Phase 2 only (used as a subprocess).")
    _add_common(p_judge)

    # Default to `orchestrate` when no subcommand is given.
    argv = sys.argv[1:]
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["orchestrate"] + argv
    args = parser.parse_args(argv)

    random.seed(args.seed)
    if args.command == "generate":
        sys.exit(cmd_generate(args))
    elif args.command == "judge":
        sys.exit(cmd_judge(args))
    else:
        sys.exit(cmd_orchestrate(args))


if __name__ == "__main__":
    main()
