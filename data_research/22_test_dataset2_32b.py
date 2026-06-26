import json
import math
import os
from tqdm import tqdm

# Crucial fix for environments without nvcc compiler installed
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["HF_TOKEN"] = "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"

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Please install vllm: pip install vllm")
    exit(1)

def main():
    input_file = "/home/exouser/pruning/datasets/dataset2.jsonl"
    output_file = "/home/exouser/pruning/datasets/dataset2_results.jsonl"
    model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"
    
    print(f"Loading dataset from {input_file}...")
    dataset = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    print(f"Loaded {len(dataset)} examples.")
    
    print(f"Loading model {model_name} via vLLM (with flashinfer bypassed)...")
    
    llm = LLM(
        model=model_name,
        quantization="awq",
        tensor_parallel_size=1,
        max_model_len=1024, # Smaller len to save KV cache memory
        gpu_memory_utilization=0.90,
        enforce_eager=True # Disables CUDAGraphs overhead for one-off scripts
    )
    
    # 20 is the max logprobs supported by vLLM without increasing server args
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20 
    )
    
    clean_prompts = [item["clean_prompt"] for item in dataset]
    corr_prompts = [item["corrupted_prompt"] for item in dataset]
    
    print("Running inference on Clean Prompts...")
    clean_outputs = llm.generate(clean_prompts, sampling_params)
    
    print("Running inference on Corrupted Prompts...")
    corr_outputs = llm.generate(corr_prompts, sampling_params)
    
    print("Processing results...")
    results = []
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i].copy()
        
        clean_target = item["clean_target"].strip().lower()
        corr_target = item["corrupted_target"].strip().lower()
        
        c_out = clean_outputs[i].outputs[0]
        c_logprobs_dict = c_out.logprobs[0] if c_out.logprobs else {}
        
        corr_out = corr_outputs[i].outputs[0]
        corr_logprobs_dict = corr_out.logprobs[0] if corr_out.logprobs else {}
        
        # Determine pass/fail based on top-1 prediction
        c_pred = c_out.text.strip().lower()
        corr_pred = corr_out.text.strip().lower()
        
        item["clean_pass"] = (c_pred == clean_target)
        item["corrupted_pass"] = (corr_pred == corr_target)
        
        # Calculate probability for clean target
        c_prob = 0.0
        for tok_id, lp_obj in c_logprobs_dict.items():
            if lp_obj.decoded_token and lp_obj.decoded_token.strip().lower() == clean_target:
                c_prob = math.exp(lp_obj.logprob)
                break
                
        # Calculate probability for corrupted target
        corr_prob = 0.0
        for tok_id, lp_obj in corr_logprobs_dict.items():
            if lp_obj.decoded_token and lp_obj.decoded_token.strip().lower() == corr_target:
                corr_prob = math.exp(lp_obj.logprob)
                break
                
        item["clean_prob"] = c_prob
        item["corrupted_prob"] = corr_prob
        item["avg_prob"] = (c_prob + corr_prob) / 2.0
        
        results.append(item)
        
    # Sort by avg_prob descending
    results.sort(key=lambda x: x["avg_prob"], reverse=True)
    
    print(f"Saving sorted results to {output_file}...")
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print("Done! Summary:")
    clean_acc = sum(1 for r in results if r["clean_pass"]) / len(results)
    corr_acc = sum(1 for r in results if r["corrupted_pass"]) / len(results)
    print(f"Clean Accuracy: {clean_acc*100:.2f}%")
    print(f"Corrupted Accuracy: {corr_acc*100:.2f}%")

if __name__ == "__main__":
    main()
