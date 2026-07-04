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
        
        # Tokenizer check fallback
        ids_nospace = tokenizer.encode(word_lower, add_special_tokens=False)
        ids_space = tokenizer.encode(word_space, add_special_tokens=False)
        exp_nospace = tokenizer.decode([ids_nospace[0]]).lower().strip() if ids_nospace else ""
        exp_space = tokenizer.decode([ids_space[0]]).lower() if ids_space else ""
        
        for token_id, logprob_obj in logprobs_dict.items():
            dt = tokenizer.decode([token_id]).lower()
            if dt == word_lower or dt == word_space or (exp_nospace and dt == exp_nospace) or (exp_space and dt == exp_space):
                prob_sum += np.exp(logprob_obj.logprob)
        return prob_sum

    print(f"[{model_config['id']}] Running Clean Prompts...")
    clean_prompts = [ex['clean_prompt'] for ex in examples]
    clean_outputs = llm.generate(clean_prompts, sampling_params, use_tqdm=True)
    
    print(f"[{model_config['id']}] Running Corrupted Prompts...")
    corr_prompts = [ex['corrupted_prompt'] for ex in examples]
    corr_outputs = llm.generate(corr_prompts, sampling_params, use_tqdm=True)
    
    results = []
    for i, ex in enumerate(examples):
        c_out = clean_outputs[i]
        c_logprobs = c_out.outputs[0].logprobs[0]
        c_pred = c_out.outputs[0].text
        
        corr_out = corr_outputs[i]
        corr_logprobs = corr_out.outputs[0].logprobs[0]
        corr_pred = corr_out.outputs[0].text
        
        c_probs = {
            "chain_0": get_prob(c_logprobs, ex['chain_0']),
            "chain_1": get_prob(c_logprobs, ex['chain_1']),
            "chain_2": get_prob(c_logprobs, ex['chain_2']),
            "unmoved_0": get_prob(c_logprobs, ex['unmoved_0']),
        }
        
        corr_probs = {
            "chain_0": get_prob(corr_logprobs, ex['chain_0']),
            "chain_1": get_prob(corr_logprobs, ex['chain_1']),
            "chain_2": get_prob(corr_logprobs, ex['chain_2']),
            "unmoved_0": get_prob(corr_logprobs, ex['unmoved_0']),
        }
        
        results.append({
            "example_id": i,
            "containers": {
                "chain_0": ex['chain_0'],
                "chain_1": ex['chain_1'],
                "chain_2": ex['chain_2'],
                "unmoved_0": ex['unmoved_0']
            },
            "clean_probs": c_probs,
            "corr_probs": corr_probs,
            "clean_pred": c_pred,
            "corr_pred": corr_pred
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
            
    # Sample 500 examples to make it fast but statistically significant
    random.seed(42)
    sample_examples = random.sample(raw_examples, 500)
    
    pattern = re.compile(r"The (.*?) is placed in the (.*?). The (.*?) is placed in the (.*?). The (.*?) is moved to the (.*?). The (.*?) is moved to the (.*?). The (.*?) is in the")
    
    parsed_examples = []
    for ex in sample_examples:
        corrupted = ex['corrupted_prompt']
        match = pattern.search(corrupted)
        if not match:
            continue
            
        unmoved_0 = match.group(4).strip() # In corrupted, target is placed here. Doesn't move.
        chain_0 = match.group(2).strip()   # Distractor is placed here. Moves to chain_1.
        chain_1 = match.group(6).strip()   # Intermediate
        chain_2 = match.group(8).strip()   # Final
        
        parsed_examples.append({
            "clean_prompt": ex['clean_prompt'],
            "corrupted_prompt": ex['corrupted_prompt'],
            "chain_0": chain_0,
            "chain_1": chain_1,
            "chain_2": chain_2,
            "unmoved_0": unmoved_0
        })
        
    print(f"Successfully parsed {len(parsed_examples)} examples.")
    
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
        
    results_dir = Path("/home/exouser/pruning/data_research/ratio_evidence")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for model_config in MODELS:
        output_file = results_dir / f"{model_config['id']}_results.json"
        p = multiprocessing.Process(
            target=evaluate_model,
            args=(model_config, parsed_examples, str(output_file))
        )
        p.start()
        p.join()
        
        # Cleanup VRAM aggressively
        import subprocess
        try:
            smi_out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
            gpu_pids = [pid.strip() for pid in smi_out.decode().split('\\n') if pid.strip()]
            current_pid = str(os.getpid())
            for pid in gpu_pids:
                if pid != current_pid:
                    os.system(f"kill -9 {pid}")
                    print(f"Force-killed zombie GPU process {pid}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
