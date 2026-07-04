import json
import re
import random
import os
import multiprocessing
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = [
    {"name": "meta-llama/Llama-3.2-1B", "id": "1B", "quant": None},
    {"name": "meta-llama/Llama-3.1-8B", "id": "8B", "quant": None},
    {"name": "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8", "id": "32B", "quant": "gptq"}
]

def evaluate_model(model_config, examples, output_file):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Please install vllm: pip install vllm")
        return
        
    print(f"\n[{model_config['id']}] Loading {model_config['name']}...")
    kwargs = {
        "model": model_config['name'],
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "dtype": "float16" if model_config['quant'] else "bfloat16",
        "max_logprobs": 50,
        "trust_remote_code": True
    }
    if model_config['quant']:
        kwargs['quantization'] = model_config['quant']
        
    llm = LLM(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'], trust_remote_code=True)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        logprobs=50
    )
    
    def get_prob(logprobs_dict, word):
        word_lower = word.lower().strip()
        word_space = " " + word_lower
        prob_sum = 0.0
        
        ids_nospace = tokenizer.encode(word_lower, add_special_tokens=False)
        ids_space = tokenizer.encode(word_space, add_special_tokens=False)
        exp_nospace = tokenizer.decode([ids_nospace[0]]).lower().strip() if ids_nospace else ""
        exp_space = tokenizer.decode([ids_space[0]]).lower() if ids_space else ""
        
        for token_id, logprob_obj in logprobs_dict.items():
            dt = tokenizer.decode([token_id]).lower()
            if dt == word_lower or dt == word_space or (exp_nospace and dt == exp_nospace) or (exp_space and dt == exp_space):
                prob_sum += np.exp(logprob_obj.logprob)
        return prob_sum

    print(f"[{model_config['id']}] Running Baseline Prompts...")
    base_prompts = [ex['baseline_prompt'] for ex in examples]
    base_outputs = llm.generate(base_prompts, sampling_params, use_tqdm=True)

    print(f"[{model_config['id']}] Running Clean Prompts...")
    clean_prompts = [ex['clean_prompt'] for ex in examples]
    clean_outputs = llm.generate(clean_prompts, sampling_params, use_tqdm=True)
    
    print(f"[{model_config['id']}] Running Corrupted Prompts...")
    corr_prompts = [ex['corrupted_prompt'] for ex in examples]
    corr_outputs = llm.generate(corr_prompts, sampling_params, use_tqdm=True)
    
    print(f"[{model_config['id']}] Running Irrelevant Chain Prompts...")
    irrel_prompts = [ex['irrel_prompt'] for ex in examples]
    irrel_outputs = llm.generate(irrel_prompts, sampling_params, use_tqdm=True)
    
    results = []
    for i, ex in enumerate(examples):
        base_out = base_outputs[i].outputs[0]
        c_out = clean_outputs[i].outputs[0]
        corr_out = corr_outputs[i].outputs[0]
        irrel_out = irrel_outputs[i].outputs[0]
        
        def extract_all_probs(logprobs):
            return {
                "chain_0": get_prob(logprobs, ex['chain_0']),
                "chain_1": get_prob(logprobs, ex['chain_1']),
                "chain_2": get_prob(logprobs, ex['chain_2']),
                "unmoved_0": get_prob(logprobs, ex['unmoved_0']),
            }
            
        results.append({
            "example_id": i,
            "containers": {
                "chain_0": ex['chain_0'],
                "chain_1": ex['chain_1'],
                "chain_2": ex['chain_2'],
                "unmoved_0": ex['unmoved_0']
            },
            "baseline": {
                "pred": base_out.text,
                "probs": extract_all_probs(base_out.logprobs[0])
            },
            "clean": {
                "pred": c_out.text,
                "probs": extract_all_probs(c_out.logprobs[0])
            },
            "corrupted": {
                "pred": corr_out.text,
                "probs": extract_all_probs(corr_out.logprobs[0])
            },
            "irrelevant": {
                "pred": irrel_out.text,
                "probs": extract_all_probs(irrel_out.logprobs[0])
            }
        })
        
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_config['id']}] Saved results to {output_file}")


def main():
    dataset_path = "/home/exouser/pruning/datasets/dataset2.jsonl"
    
    raw_examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            raw_examples.append(json.loads(line))
            
    random.seed(42)
    sample_examples = random.sample(raw_examples, 500)
    
    pattern = re.compile(r"The (.*?) is placed in the (.*?). The (.*?) is placed in the (.*?). The (.*?) is moved to the (.*?). The (.*?) is moved to the (.*?). The (.*?) is in the")
    irrelevant_nouns = ["tray", "bag", "bowl", "basket", "bucket", "cart", "wagon", "bin"]
    
    parsed_examples = []
    for ex in sample_examples:
        corrupted = ex['corrupted_prompt']
        match = pattern.search(corrupted)
        if not match:
            continue
            
        distractor_item = match.group(1)
        target_item = match.group(3)
        
        unmoved_0 = match.group(4).strip() # Target is placed here in corrupted. Doesn't move.
        chain_0 = match.group(2).strip()   # Distractor is placed here in corrupted. Moves.
        chain_1 = match.group(6).strip()   # Intermediate
        chain_2 = match.group(8).strip()   # Final
        
        # 1. Baseline
        baseline_prompt = f"The {distractor_item} is placed in the {chain_0}. The {target_item} is placed in the {unmoved_0}. The {target_item} is in the"
        
        # 2. Clean
        # Target in chain_0, moves to chain_1, chain_2. Distractor in unmoved_0.
        clean_prompt = f"The {distractor_item} is placed in the {unmoved_0}. The {target_item} is placed in the {chain_0}. The {chain_0} is moved to the {chain_1}. The {chain_1} is moved to the {chain_2}. The {target_item} is in the"
        
        # 3. Corrupted (Original)
        # Distractor in chain_0, moves. Target in unmoved_0.
        corrupted_prompt = corrupted
        
        # 4. Irrelevant
        # Target in unmoved_0, Distractor in chain_0. An irrelevant item moves.
        valid_nouns = [n for n in irrelevant_nouns if n != chain_0 and n != unmoved_0]
        irrel_noun = random.choice(valid_nouns)
        irrel_prompt = f"The {distractor_item} is placed in the {chain_0}. The {target_item} is placed in the {unmoved_0}. The {irrel_noun} is moved to the {chain_1}. The {chain_1} is moved to the {chain_2}. The {target_item} is in the"
        
        parsed_examples.append({
            "baseline_prompt": baseline_prompt,
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
            "irrel_prompt": irrel_prompt,
            "chain_0": chain_0,
            "chain_1": chain_1,
            "chain_2": chain_2,
            "unmoved_0": unmoved_0
        })
        
    print(f"Successfully constructed 4 states for {len(parsed_examples)} examples.")
    
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
        
    script_dir = Path(__file__).parent.resolve()
    results_dir = script_dir / "dump"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for model_config in MODELS:
        output_file = results_dir / f"{model_config['id']}_unified_raw.json"
        p = multiprocessing.Process(
            target=evaluate_model,
            args=(model_config, parsed_examples, str(output_file))
        )
        p.start()
        p.join()
        
        import subprocess
        try:
            smi_out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
            gpu_pids = [pid.strip() for pid in smi_out.decode().split('\\n') if pid.strip()]
            current_pid = str(os.getpid())
            for pid in gpu_pids:
                if pid != current_pid:
                    os.system(f"kill -9 {pid}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
