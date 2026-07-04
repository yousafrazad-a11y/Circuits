import json
import re
import random
import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import numpy as np

os.environ["HF_TOKEN"] = "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"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def generate_evidence():
    dataset_path = "/home/exouser/pruning/datasets/dataset2.jsonl"
    
    # We need some random nouns to serve as the "Irrelevant Noun"
    # We'll use typical container nouns
    irrelevant_nouns = ["tray", "bag", "bowl", "basket", "bucket", "cart", "wagon", "bin"]
    
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            
    # Sample 500 examples to make it fast but statistically significant
    random.seed(42)
    sample_examples = random.sample(examples, 500)
    
    pattern = re.compile(r"The (.*?) is placed in the (.*?). The (.*?) is placed in the (.*?). The (.*?) is moved to the (.*?). The (.*?) is moved to the (.*?). The (.*?) is in the")
    
    ablations = []
    
    for ex in sample_examples:
        corrupted = ex['corrupted_prompt']
        match = pattern.search(corrupted)
        if not match:
            continue
            
        distractor_item = match.group(1)
        distractor_container = match.group(2)
        target_item = match.group(3)
        target_container = match.group(4)
        intermediate = match.group(6)
        final_dest = match.group(8)
        
        # 1. Baseline (No Hops)
        baseline_prompt = f"The {distractor_item} is placed in the {distractor_container}. The {target_item} is placed in the {target_container}. The {target_item} is in the"
        
        # 2. Corrupted (Original)
        corrupted_prompt = corrupted
        
        # 3. Irrelevant Chain
        # Pick an irrelevant noun that is not the distractor or target container
        valid_nouns = [n for n in irrelevant_nouns if n != distractor_container and n != target_container]
        irrel_noun = random.choice(valid_nouns)
        
        irrel_prompt = f"The {distractor_item} is placed in the {distractor_container}. The {target_item} is placed in the {target_container}. The {irrel_noun} is moved to the {intermediate}. The {intermediate} is moved to the {final_dest}. The {target_item} is in the"
        
        ablations.append({
            "target": target_container.strip().lower(),
            "distractor_dest": final_dest.strip().lower(),
            "baseline_prompt": baseline_prompt,
            "corrupted_prompt": corrupted_prompt,
            "irrel_prompt": irrel_prompt
        })
        
    print(f"Successfully generated {len(ablations)} ablation examples.")
    
    # Now run Llama 3.1 8B on these
    model_id = "meta-llama/Llama-3.1-8B"
    print(f"Loading {model_id}...")
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="bfloat16",
        max_logprobs=50
    )
    
    # Pre-calculate expected tokens with leading space
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
        logprobs=50
    )
    
    def get_word_prob_from_logprobs(logprobs_dict, target_word):
        # target_word is e.g. "box"
        target_with_space = " " + target_word
        prob_sum = 0.0
        
        # We look for either "box" or " box"
        for token_id, logprob_obj in logprobs_dict.items():
            decoded_token = tokenizer.decode([token_id])
            if decoded_token.lower() == target_word.lower() or decoded_token.lower() == target_with_space.lower():
                prob_sum += np.exp(logprob_obj.logprob)
        return prob_sum
    
    def run_batch(prompts, ablations, variation_name):
        print(f"Running {variation_name}...")
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        results = []
        for out, ab in zip(outputs, ablations):
            pred_text = out.outputs[0].text
            logprobs = out.outputs[0].logprobs[0] # first token logprobs
            
            target_prob = get_word_prob_from_logprobs(logprobs, ab['target'])
            distractor_prob = get_word_prob_from_logprobs(logprobs, ab['distractor_dest'])
            
            # Check if prediction is correct
            first_word = re.split(r'[\s\.\,\;]+', pred_text.strip().lower())[0]
            is_correct = (first_word == ab['target'])
            
            results.append({
                "pred_text": pred_text,
                "first_word": first_word,
                "is_correct": is_correct,
                "target_prob": float(target_prob),
                "distractor_prob": float(distractor_prob)
            })
            
        return results

    baseline_results = run_batch([x['baseline_prompt'] for x in ablations], ablations, "Baseline")
    corrupted_results = run_batch([x['corrupted_prompt'] for x in ablations], ablations, "Corrupted")
    irrel_results = run_batch([x['irrel_prompt'] for x in ablations], ablations, "Irrelevant Chain")
    
    # Combine results
    final_output = []
    for i in range(len(ablations)):
        final_output.append({
            "ablation_info": ablations[i],
            "baseline": baseline_results[i],
            "corrupted": corrupted_results[i],
            "irrelevant": irrel_results[i]
        })
        
    with open("/home/exouser/pruning/data_research/ablation_evidence_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    print("Saved raw results.")

if __name__ == "__main__":
    generate_evidence()
