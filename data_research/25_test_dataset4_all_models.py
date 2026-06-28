import json
import math
import os
import multiprocessing
from tqdm import tqdm

# Crucial fix for environments without nvcc compiler installed
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_TOKEN"] = "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"

MODELS = {
    "qwen_normal_32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        "quant": "gptq"
    },
    "qwen_coder_32b": {
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8",
        "quant": "gptq"
    },
    "qwen_coder_7b": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8",
        "quant": "gptq"
    },
    "deepseek_coder_33b": {
        "name": "TheBloke/deepseek-coder-33B-instruct-AWQ",
        "quant": "awq"
    },
    "deepseek_thinking_32b": {
        "name": "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8",
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
        "max_model_len": 4096, # Very safe for 8-bit 32B on 40GB A100
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True, # Skips CUDA graphs to save host memory overhead
        "trust_remote_code": True
    }
    if model_config['quant']:
        kwargs["quantization"] = model_config['quant']
        
    llm = LLM(**kwargs)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1, # 1 hop prediction
        logprobs=20 
    )
    
    clean_prompts = [item["clean_prompt"] for item in dataset]
    corr_prompts = [item["corrupted_prompt"] for item in dataset]
    
    print(f"[{model_key.upper()}] Running inference on Clean Prompts...")
    clean_outputs = llm.generate(clean_prompts, sampling_params)
    
    print(f"[{model_key.upper()}] Running inference on Corrupted Prompts...")
    corr_outputs = llm.generate(corr_prompts, sampling_params)
    
    print(f"[{model_key.upper()}] Processing results...")
    results = []
    
    for i in tqdm(range(len(dataset)), desc=f"Processing {model_key.upper()}"):
        item = dataset[i].copy()
        
        clean_target = item["clean_target"].strip().lower()
        corr_target = item["corrupted_target"].strip().lower()
        
        c_out = clean_outputs[i].outputs[0]
        c_logprobs_dict = c_out.logprobs[0] if c_out.logprobs else {}
        
        corr_out = corr_outputs[i].outputs[0]
        corr_logprobs_dict = corr_out.logprobs[0] if corr_out.logprobs else {}
        
        # Sort logprobs to find top 1, 3, 10
        sorted_c_logprobs = sorted(c_logprobs_dict.values(), key=lambda x: x.logprob, reverse=True)
        c_top_tokens = [x.decoded_token.strip().lower() if x.decoded_token else "" for x in sorted_c_logprobs[:10]]
        
        sorted_corr_logprobs = sorted(corr_logprobs_dict.values(), key=lambda x: x.logprob, reverse=True)
        corr_top_tokens = [x.decoded_token.strip().lower() if x.decoded_token else "" for x in sorted_corr_logprobs[:10]]
        
        item["clean_pass_top1"] = (clean_target in c_top_tokens[:1])
        item["clean_pass_top3"] = (clean_target in c_top_tokens[:3])
        item["clean_pass_top10"] = (clean_target in c_top_tokens[:10])
        
        item["corrupted_pass_top1"] = (corr_target in corr_top_tokens[:1])
        item["corrupted_pass_top3"] = (corr_target in corr_top_tokens[:3])
        item["corrupted_pass_top10"] = (corr_target in corr_top_tokens[:10])
        
        # Keep original fields for backward compatibility in other scripts
        item["clean_pass"] = item["clean_pass_top1"]
        item["corrupted_pass"] = item["corrupted_pass_top1"]
        item["combined_pass"] = item["clean_pass_top1"] and item["corrupted_pass_top1"]
        
        item["combined_pass_top3"] = item["clean_pass_top3"] and item["corrupted_pass_top3"]
        item["combined_pass_top10"] = item["clean_pass_top10"] and item["corrupted_pass_top10"]
        
        c_prob = 0.0
        for tok_id, lp_obj in c_logprobs_dict.items():
            if lp_obj.decoded_token and lp_obj.decoded_token.strip().lower() == clean_target:
                c_prob = math.exp(lp_obj.logprob)
                break
                
        corr_prob = 0.0
        for tok_id, lp_obj in corr_logprobs_dict.items():
            if lp_obj.decoded_token and lp_obj.decoded_token.strip().lower() == corr_target:
                corr_prob = math.exp(lp_obj.logprob)
                break
                
        item["clean_prob"] = c_prob
        item["corrupted_prob"] = corr_prob
        item["avg_prob"] = (c_prob + corr_prob) / 2.0
        
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
    # Force 'spawn' for multiprocessing to ensure clean GPU VRAM between models
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    
    input_file = "/home/exouser/pruning/datasets/dataset4.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    for model_key, model_config in MODELS.items():
        output_file = f"/home/exouser/pruning/datasets/dataset3_results_{model_key}.jsonl"
        print(f"\n{'#'*60}")
        print(f" STARTING EVALUATION ROUND: {model_key.upper()}")
        print(f" {'#'*60}")
        
        p = multiprocessing.Process(
            target=evaluate_model, 
            args=(model_key, model_config, input_file, output_file)
        )
        p.start()
        p.join()
        
        if p.exitcode != 0:
            print(f"ERROR: {model_key.upper()} phase failed with exit code {p.exitcode}. VRAM will be cleared before next model.")
            
    # Combine summaries
    all_summaries = {}
    for model_key in MODELS.keys():
        summary_file = f"/home/exouser/pruning/datasets/dataset3_results_{model_key}_summary.json"
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                all_summaries[model_key] = json.load(f)
                
    final_summary_path = "/home/exouser/pruning/datasets/dataset4_accuracy_summary.json"
    with open(final_summary_path, "w") as f:
        json.dump(all_summaries, f, indent=4)
        
    print(f"\nAll models successfully evaluated.")
    print(f"Final compiled summaries saved to: {final_summary_path}")

if __name__ == "__main__":
    main()
