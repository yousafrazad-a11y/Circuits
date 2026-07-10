import os
import torch
from transformers import AutoTokenizer

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def get_prob(tokenizer, logprobs_dict, target_word):
    # Logprobs from vllm
    word_lower = target_word.lower().strip()
    word_space = " " + word_lower
    prob_sum = 0.0
    
    for token_id, logprob_obj in logprobs_dict.items():
        dt = tokenizer.decode([token_id]).lower()
        if dt == word_lower or dt == word_space or dt.strip() == word_lower:
            import numpy as np
            prob_sum += np.exp(logprob_obj.logprob)
    return prob_sum

def main():
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Please install vllm")
        return

    model_name = "meta-llama/Llama-3.2-1B"
    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        dtype="bfloat16",
        max_logprobs=50,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=2, logprobs=50)

    # Dataset 1: Name Binding
    d1_clean = "Mr. John Smith went to the store. The manager asked for his name, and he replied 'John"
    d1_corr_a = "Mr. John Miller went to the store. The manager asked for his name, and he replied 'John"
    d1_corr_b = "Mr. David Smith went to the store. The manager asked for his name, and he replied 'John"
    d1_target = "Smith"

    # Dataset 2: Nonsense Word Translation
    d2_clean = "In the fictional language of Zog, 'glib' translates to 'fast'. So the word 'glib' translates to '"
    d2_corr_a = "In the fictional language of Zog, 'glib' translates to 'slow'. So the word 'glib' translates to '"
    d2_corr_b = "In the fictional language of Zog, 'florp' translates to 'fast'. So the word 'glib' translates to '"
    d2_target = "fast"

    prompts = [d1_clean, d1_corr_a, d1_corr_b, d2_clean, d2_corr_a, d2_corr_b]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    print("\n--- DATASET 1: NAME BINDING (Target: 'Smith') ---")
    d1_res = outputs[:3]
    for name, out in zip(["Clean", "Corr A (Value: Miller)", "Corr B (Query Prefix: David)"], d1_res):
        text = out.outputs[0].text
        prob = get_prob(tokenizer, out.outputs[0].logprobs[0], d1_target)
        print(f"{name:30s} | Prob of '{d1_target}': {prob:.4f} | Top output: {text!r}")

    print("\n--- DATASET 2: ACRONYM BINDING (Target: 'WHO') ---")
    d2_res = outputs[3:]
    for name, out in zip(["Clean", "Corr A (Value: FBI)", "Corr B (Query Prefix: Federal...)"], d2_res):
        text = out.outputs[0].text
        prob = get_prob(tokenizer, out.outputs[0].logprobs[0], d2_target)
        print(f"{name:30s} | Prob of '{d2_target}': {prob:.4f} | Top output: {text!r}")

if __name__ == "__main__":
    main()
