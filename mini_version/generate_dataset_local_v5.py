#!/usr/bin/env python3
"""
2-Hop State-Machine Dataset Generator V4 (Unified 32B Model)
==============================================================================

Uses Qwen2.5-32B-Instruct-GPTQ-Int8 for BOTH generating and judging.
Since both phases use the exact same model, they run in the SAME process
without any subprocess ping-pong, massively speeding up generation.

Fixes for Dataset 6:
1. Implements item-movement tracking instead of container-movement tracking.
2. Uses most successful Non-Living themes from Dataset 2, but applies v4 surface bans.
"""

import json
import argparse
import random
import os
import re
from typing import List, Set

# vLLM MUST be loaded inside the process after env vars are set
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["HF_TOKEN"] = "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"

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# PHYSICAL ONTOLOGY SEEDS
# ---------------------------------------------------------------------------
# We provide 6 elements:
# [target_item, distractor_item, mobile_container, fixed_container, outer1, outer2]
# We strictly enforce that containers are enclosures (fit for "in the")
# and that physical size logically allows mobile -> outer1 -> outer2.

THEMES: dict[str, List[str]] = {
    "The Laboratory": ["tube", "beaker", "briefcase", "cabinet", "safe", "laboratory"],
    "The NonLiving Kitchen": ["bean", "pebble", "bucket", "pot", "pantry", "kitchen"],
    "The Post Office": ["card", "letter", "box", "safe", "truck", "warehouse"],
    "The Storage Room": ["coin", "key", "drawer", "chest", "crate", "warehouse"],
    "The Warehouse": ["bolt", "screw", "crate", "locker", "bin", "warehouse"],
}

THEME_VARIATIONS: dict[str, List[str]] = {
    "The Laboratory": ["tube", "beaker", "flask", "vial", "slide", "lens"],
    "The NonLiving Kitchen": ["bean", "pebble", "seed", "nut", "grain", "salt"],
    "The Post Office": ["card", "letter", "stamp", "note", "bill", "mail"],
    "The Storage Room": ["coin", "key", "ring", "gem", "gold", "cash"],
    "The Warehouse": ["bolt", "screw", "nail", "nut", "gear", "part"],
}

# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------
# Extremely concise, direct prompts because small models perform better with less fluff.

GENERATOR_SYS_PROMPT = """You are an expert ontology designer. Your task is to output a JSON array of 6 strictly related, physically plausible everyday objects and locations that follow a strict size hierarchy.

The array MUST be exactly 6 strings in this exact order:
[target, distractor, mobile_container, fixed_container, outer_transport, final_destination]

RULES:
1. `target` and `distractor`: Two DIFFERENT small objects (e.g., "coin", "key").
2. `mobile_container`: An ENCLOSURE (e.g., "box", "chest", "case"). NEVER use a surface like "shelf" or "table". 
3. `fixed_container`: An ENCLOSURE (e.g., "safe", "vault", "locker"). NEVER use a surface. 
4. `outer_transport`: A larger container or location (e.g., "bin", "truck", "rack"). 
5. `final_destination`: A massive location or building (e.g., "warehouse", "yard", "depot"). 
6. SINGLE WORD NOUNS ONLY. Do not use multi-word entities (use "box" instead of "cardboard box").
7. VERBS ARE HARDCODED. The verbs in the prompt will be "is placed in the" and "is moved to the". Your generated nouns MUST make perfect grammatical sense with these verbs.
8. Output strictly a single JSON array and nothing else."""

JUDGE_SYS_PROMPT = """You are a strict data-quality logic judge. You will be given a JSON array of 6 elements:
[target, distractor, mobile_container, fixed_container, outer_transport, final_destination]

You must output a single JSON object:
{"valid": true_or_false, "reason": "brief reason"}

Rules for validation:
1. NO SURFACES. `mobile_container` and `fixed_container` MUST be enclosures (box, chest, safe). Reject "shelf", "table", "desk", "counter".
2. SINGLE WORD NOUNS. Reject any array that contains multi-word elements (e.g., reject "cardboard box", accept "box").
3. SIZE HIERARCHY. The objects must be able to be placed inside the containers, and the containers inside the outer locations.
4. If ANY rule is broken, return {"valid": false}. If perfectly logical, return {"valid": true}."""

def build_generator_prompt(theme_name: str, target_item: str) -> str:
    demo = THEMES[theme_name]
    msg = [
        {"role": "system", "content": GENERATOR_SYS_PROMPT},
        {"role": "user", "content": f"Generate a 6-element array for the theme '{theme_name}' starting with the item '{demo[0]}'."},
        {"role": "assistant", "content": json.dumps(demo)},
        {"role": "user", "content": f"Generate a completely new 6-element array for the theme '{theme_name}' starting with the item '{target_item}'. Make sure the containers are different from the example but fit the theme."}
    ]
    # We construct the chat manually because we are using base vLLM
    prompt = ""
    for m in msg:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n["
    return prompt

def build_judge_prompt(array_json_str: str) -> str:
    msg = [
        {"role": "system", "content": JUDGE_SYS_PROMPT},
        {"role": "user", "content": f"Evaluate this array: {array_json_str}"}
    ]
    prompt = ""
    for m in msg:
        prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n{"
    return prompt

def clean_array(text: str) -> List[str]:
    try:
        if not text.startswith("["):
            text = "[" + text
        end_idx = text.find("]")
        if end_idx != -1:
            text = text[:end_idx+1]
        arr = json.loads(text)
        if isinstance(arr, list) and len(arr) == 6 and all(isinstance(x, str) for x in arr):
            return [x.lower().strip() for x in arr]
    except:
        pass
    return []

def python_pre_validate(arr: List[str]) -> bool:
    if len(arr) != 6:
        return False
    # Hard ban on surfaces
    banned = ["shelf", "table", "desk", "counter", "floor", "rack", "roof", "ground"]
    for w in banned:
        if w in arr[2] or w in arr[3]: # containers
            return False
    return True

def generate_prompts(entities: List[str], theme_name: str) -> dict:
    target, distractor, mobile_c, fixed_c, outer1, outer2 = entities
    
    clean_p = (
        f"The {target} is placed in the {mobile_c}. "
        f"The {distractor} is placed in the {fixed_c}. "
        f"The item is moved from the {mobile_c} to the {outer1}. "
        f"The item is moved from the {outer1} to the {outer2}. "
        f"The {target} is in the"
    )
    corr_p = (
        f"The {distractor} is placed in the {mobile_c}. "
        f"The {target} is placed in the {fixed_c}. "
        f"The item is moved from the {mobile_c} to the {outer1}. "
        f"The item is moved from the {outer1} to the {outer2}. "
        f"The {target} is in the"
    )
    return {
        "entities": entities,
        "clean_prompt": clean_p,
        "clean_target": f" {outer2}",
        "corrupted_prompt": corr_p,
        "corrupted_target": f" {fixed_c}",
        "hops": 2,
        "theme": theme_name,
        "verified": True
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--output", type=str, default="/home/exouser/pruning/datasets/dataset6.jsonl")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading Qwen2.5-32B-Instruct-GPTQ-Int8...")
    llm = LLM(
        model="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        quantization="gptq",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        trust_remote_code=True
    )
    
    gen_params = SamplingParams(temperature=1.2, max_tokens=100, stop=["<|im_end|>"])
    judge_params = SamplingParams(temperature=0.0, max_tokens=50, stop=["<|im_end|>"])

    # Load existing to avoid dupes
    seen_entities = set()
    accepted_records = []
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            for line in f:
                rec = json.loads(line)
                accepted_records.append(rec)
                seen_entities.add(tuple(rec["entities"]))
                
    print(f"Loaded {len(accepted_records)} existing records.")
    
    themes = list(THEMES.keys())
    
    while len(accepted_records) < args.target:
        needed = args.target - len(accepted_records)
        to_gen = min(args.batch_size, needed * 2 + 10) # Overgenerate
        
        print(f"\n--- GENERATING {to_gen} DRAFTS ---")
        gen_prompts = []
        gen_themes = []
        for _ in range(to_gen):
            thm = random.choice(themes)
            tgt = random.choice(THEME_VARIATIONS[thm])
            gen_prompts.append(build_generator_prompt(thm, tgt))
            gen_themes.append(thm)
            
        outputs = llm.generate(gen_prompts, gen_params, use_tqdm=True)
        
        drafts = []
        for out, thm in zip(outputs, gen_themes):
            text = out.outputs[0].text
            arr = clean_array(text)
            if python_pre_validate(arr):
                if tuple(arr) not in seen_entities:
                    drafts.append((arr, thm))
                    seen_entities.add(tuple(arr))
                    
        print(f"Generated {len(drafts)} valid-looking drafts.")
        if not drafts:
            continue
            
        print(f"\n--- JUDGING {len(drafts)} DRAFTS ---")
        judge_prompts = [build_judge_prompt(json.dumps(d[0])) for d in drafts]
        j_outputs = llm.generate(judge_prompts, judge_params, use_tqdm=True)
        
        valid_count = 0
        for (arr, thm), out in zip(drafts, j_outputs):
            text = "{" + out.outputs[0].text
            try:
                j_res = json.loads(text)
                if j_res.get("valid") is True:
                    # ACCEPTED
                    record = generate_prompts(arr, thm)
                    accepted_records.append(record)
                    valid_count += 1
                    with open(args.output, "a") as f:
                        f.write(json.dumps(record) + "\n")
                    if len(accepted_records) >= args.target:
                        break
            except:
                pass
                
        print(f"Round complete: {valid_count} accepted. Total so far: {len(accepted_records)}/{args.target}")
        
    print("Done! Dataset complete.")

if __name__ == "__main__":
    main()
