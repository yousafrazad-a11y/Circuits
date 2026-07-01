import json
import math
import os
import argparse
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Crucial fix for environments without nvcc compiler installed
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_TOKEN"] = "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"

MODELS = {
    "llama_3_2_1b_base": {
        "name": "meta-llama/Llama-3.2-1B",
        "quant": None
    },
    "llama_3_1_8b_base": {
        "name": "meta-llama/Llama-3.1-8B",
        "quant": None
    }
}

def evaluate_model(model_key, model_config, input_file, output_file):
    # This runs in a completely separate process to guarantee clean VRAM on exit
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Please install vllm: pip install vllm")
        return

    print(f"\n[{model_key.upper()}] Loading dataset from {input_file}...")
    dataset = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    print(f"[{model_key.upper()}] Loaded {len(dataset)} examples.")
    
    print(f"[{model_key.upper()}] Loading model {model_config['name']} via vLLM...")
    
    kwargs = {
        "model": model_config['name'],
        "tensor_parallel_size": 1,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.80,
        "trust_remote_code": True,
        "enforce_eager": True,
        "max_logprobs": 50
    }
    if model_config['quant']:
        kwargs["quantization"] = model_config['quant']
        
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        logprobs=50 
    )
    
    clean_prompts = [ex["clean_prompt"] for ex in dataset]
    corr_prompts = [ex["corrupted_prompt"] for ex in dataset]
    all_prompts = clean_prompts + corr_prompts
    
    print(f"[{model_key.upper()}] Running inference on {len(all_prompts)} prompts...")
    outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)
    
    clean_outputs = outputs[:len(dataset)]
    corr_outputs = outputs[len(dataset):]
    
    results = []
    for i, item in enumerate(dataset):
        c_out = clean_outputs[i]
        c_top = c_out.outputs[0].logprobs[0]
        c_pred_top1 = c_out.outputs[0].text
        
        corr_out = corr_outputs[i]
        corr_top = corr_out.outputs[0].logprobs[0]
        corr_pred_top1 = corr_out.outputs[0].text
        
        import re
        
        def build_target_regex(target_str):
            target_str = target_str.strip().lower()
            words = re.split(r'[\s\-\_\.]+', target_str)
            words = [re.escape(w) for w in words if w]
            if not words:
                return ""
            
            last_word = words[-1]
            last_word_pattern = last_word + r'(?:s|es)?'
            
            if len(words) > 1:
                core_pattern = r'[\s\-\_\.]*'.join(words[:-1]) + r'[\s\-\_\.]*' + last_word_pattern
            else:
                core_pattern = last_word_pattern
                
            return r'(?:\b|\W|^)(?:a\s+|an\s+|the\s+)?' + core_pattern + r'(?:\b|\W|$)'
        
        c_target_full = item["clean_target"]
        corr_target_full = item["corrupted_target"]
        
        c_target = c_target_full.strip().lower()
        corr_target = corr_target_full.strip().lower()
        
        # Get expected first tokens for robust logprob matching
        c_ids = tokenizer.encode(c_target_full, add_special_tokens=False)
        c_expected_token = tokenizer.decode([c_ids[0]]).strip().lower() if c_ids else ""
        
        corr_ids = tokenizer.encode(corr_target_full, add_special_tokens=False)
        corr_expected_token = tokenizer.decode([corr_ids[0]]).strip().lower() if corr_ids else ""
        
        # Get expected first tokens for ALL entities to handle multi-token words (like "lockbox" -> "lock")
        entity_expected_tokens = {}
        for ent in item.get("entities", []):
            ent_with_space = " " + ent
            e_ids = tokenizer.encode(ent_with_space, add_special_tokens=False)
            if e_ids:
                e_tok = tokenizer.decode([e_ids[0]]).strip().lower()
                entity_expected_tokens[ent] = e_tok
        
        c_pred_clean = c_pred_top1.strip().lower()
        corr_pred_clean = corr_pred_top1.strip().lower()
        
        # Top 1 regex check: string must contain target at a word boundary/punctuation
        c_pattern = build_target_regex(c_target)
        c_pass_top1 = bool(re.search(c_pattern, c_pred_clean)) if c_pattern else False
        
        corr_pattern = build_target_regex(corr_target)
        corr_pass_top1 = bool(re.search(corr_pattern, corr_pred_clean)) if corr_pattern else False
        
        c_pass_top3 = c_pass_top1
        c_pass_top10 = c_pass_top1
        c_prob = 0.0
        c_entity_probs = {ent: 0.0 for ent in item.get("entities", [])}
        
        sorted_c_logprobs = sorted(c_top.values(), key=lambda x: x.logprob, reverse=True)
        for rank, lp_obj in enumerate(sorted_c_logprobs):
            t = lp_obj.decoded_token
            if not t: continue
            t_clean = t.strip().lower()
            if not t_clean: continue
            
            prob = math.exp(lp_obj.logprob)
            
            # Check for entities using both full string and tokenizer-expected first token
            for ent in c_entity_probs.keys():
                exp_tok = entity_expected_tokens.get(ent, "")
                if t_clean == ent.lower() or (t_clean.startswith(ent.lower()) and len(t_clean) <= len(ent) + 2) or (exp_tok and t_clean == exp_tok):
                    if prob > c_entity_probs[ent]:
                        c_entity_probs[ent] = prob
            
            if rank < 10:
                if t_clean == c_target or (c_expected_token and t_clean == c_expected_token):
                    if rank < 1: c_pass_top1 = True
                    if rank < 3: c_pass_top3 = True
                    if rank < 10: c_pass_top10 = True
                    if c_prob == 0.0: c_prob = prob
                
        corr_pass_top3 = corr_pass_top1
        corr_pass_top10 = corr_pass_top1
        corr_prob = 0.0
        corr_entity_probs = {ent: 0.0 for ent in item.get("entities", [])}
        
        sorted_corr_logprobs = sorted(corr_top.values(), key=lambda x: x.logprob, reverse=True)
        for rank, lp_obj in enumerate(sorted_corr_logprobs):
            t = lp_obj.decoded_token
            if not t: continue
            t_clean = t.strip().lower()
            if not t_clean: continue
            
            prob = math.exp(lp_obj.logprob)
            
            # Check for entities using both full string and tokenizer-expected first token
            for ent in corr_entity_probs.keys():
                exp_tok = entity_expected_tokens.get(ent, "")
                if t_clean == ent.lower() or (t_clean.startswith(ent.lower()) and len(t_clean) <= len(ent) + 2) or (exp_tok and t_clean == exp_tok):
                    if prob > corr_entity_probs[ent]:
                        corr_entity_probs[ent] = prob
            
            if rank < 10:
                if t_clean == corr_target or (corr_expected_token and t_clean == corr_expected_token):
                    if rank < 1: corr_pass_top1 = True
                    if rank < 3: corr_pass_top3 = True
                    if rank < 10: corr_pass_top10 = True
                    if corr_prob == 0.0: corr_prob = prob
                
        item["clean_pred_top1"] = c_pred_top1
        item["clean_pass_top1"] = c_pass_top1
        item["clean_pass_top3"] = c_pass_top3
        item["clean_pass_top10"] = c_pass_top10
        
        item["corrupted_pred_top1"] = corr_pred_top1
        item["corrupted_pass_top1"] = corr_pass_top1
        item["corrupted_pass_top3"] = corr_pass_top3
        item["corrupted_pass_top10"] = corr_pass_top10
        
        item["combined_pass"] = c_pass_top1 and corr_pass_top1
        item["combined_pass_top3"] = c_pass_top3 and corr_pass_top3
        item["combined_pass_top10"] = c_pass_top10 and corr_pass_top10
        
        for k in list(item.keys()):
            if "logprob" in k or k == "entities":
                pass
                
        item["clean_prob"] = c_prob
        item["corrupted_prob"] = corr_prob
        item["avg_prob"] = (c_prob + corr_prob) / 2.0
        
        item["clean_entity_probs"] = c_entity_probs
        item["corrupted_entity_probs"] = corr_entity_probs
        
        results.append(item)
        
    results.sort(key=lambda x: x["avg_prob"], reverse=True)
    
    print(f"[{model_key.upper()}] Saving sorted results to {output_file}...")
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    clean_acc_top1 = sum(1 for r in results if r["clean_pass_top1"]) / len(results)
    clean_acc_top3 = sum(1 for r in results if r["clean_pass_top3"]) / len(results)
    clean_acc_top10 = sum(1 for r in results if r["clean_pass_top10"]) / len(results)
    
    corr_acc_top1 = sum(1 for r in results if r["corrupted_pass_top1"]) / len(results)
    corr_acc_top3 = sum(1 for r in results if r["corrupted_pass_top3"]) / len(results)
    corr_acc_top10 = sum(1 for r in results if r["corrupted_pass_top10"]) / len(results)
    
    comb_acc_top1 = sum(1 for r in results if r["combined_pass"]) / len(results)
    comb_acc_top3 = sum(1 for r in results if r["combined_pass_top3"]) / len(results)
    comb_acc_top10 = sum(1 for r in results if r["combined_pass_top10"]) / len(results)
    
    summary = {
        "model": model_config['name'],
        "clean_top1": clean_acc_top1,
        "clean_top3": clean_acc_top3,
        "clean_top10": clean_acc_top10,
        "corrupted_top1": corr_acc_top1,
        "corrupted_top3": corr_acc_top3,
        "corrupted_top10": corr_acc_top10,
        "combined_top1": comb_acc_top1,
        "combined_top3": comb_acc_top3,
        "combined_top10": comb_acc_top10
    }
    
    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*50}")
    print(f"{model_key.upper()} FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Top-1  -> Clean: {clean_acc_top1*100:.2f}% | Corr: {corr_acc_top1*100:.2f}% | Comb: {comb_acc_top1*100:.2f}%")
    print(f"Top-3  -> Clean: {clean_acc_top3*100:.2f}% | Corr: {corr_acc_top3*100:.2f}% | Comb: {comb_acc_top3*100:.2f}%")
    print(f"Top-10 -> Clean: {clean_acc_top10*100:.2f}% | Corr: {corr_acc_top10*100:.2f}% | Comb: {comb_acc_top10*100:.2f}%")
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a dataset across all local models.")
    parser.add_argument("dataset_file", type=str, help="Absolute path to the dataset file (e.g. /path/to/dataset5.jsonl)")
    args = parser.parse_args()

    input_file = args.dataset_file
    if not os.path.exists(input_file):
        print(f"Error: Dataset file {input_file} not found.")
        return

    # Extract dataset name, e.g., 'dataset5' from 'dataset5.jsonl'
    dataset_name = Path(input_file).stem
    results_dir = Path(input_file).parent / f"{dataset_name}_results"
    
    # Create the results directory
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Force 'spawn' for multiprocessing to ensure clean GPU VRAM between models
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    for model_key, model_config in MODELS.items():
        output_file = str(results_dir / f"{model_key}_results.jsonl")
        print(f"\n{'#'*60}")
        print(f" STARTING EVALUATION ROUND: {model_key.upper()}")
        print(f" {'#'*60}")
        
        p = multiprocessing.Process(
            target=evaluate_model, 
            args=(model_key, model_config, input_file, output_file)
        )
        p.start()
        p.join()
        
        # Kill any zombie background EngineCore processes to guarantee clean VRAM for the next model
        import subprocess
        try:
            smi_out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
            gpu_pids = [pid.strip() for pid in smi_out.decode().split('\\n') if pid.strip()]
            current_pid = str(os.getpid())
            for pid in gpu_pids:
                if pid != current_pid:
                    os.system(f"kill -9 {pid}")
                    print(f"Force-killed zombie GPU process {pid}")
        except Exception as e:
            pass
        
        if p.exitcode != 0:
            print(f"ERROR: {model_key.upper()} phase failed with exit code {p.exitcode}. VRAM has been forcefully cleared.")
            
        # Clear huggingface cache for this model to save disk space
        model_repo_name = model_config["name"]
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_repo_name.replace('/', '--')}")
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"Successfully cleared cache for {model_repo_name} at {cache_dir}")
            
    # Combine summaries
    all_summaries = {}
    for model_key in MODELS.keys():
        summary_file = results_dir / f"{model_key}_results_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                all_summaries[model_key] = json.load(f)
                
    final_summary_path = results_dir / "final_accuracy_summary.json"
    with open(final_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=4)
        
    print(f"\nAll models successfully evaluated.")
    print(f"Final compiled summaries saved to: {final_summary_path}")

if __name__ == "__main__":
    main()
