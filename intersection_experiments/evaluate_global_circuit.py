import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

CATEGORIES = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "pear", "plum", "kiwi", "melon", "cherry", "lemon", "lime", "fig", "date", "papaya"],
    "animals": ["cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow", "pig", "sheep", "goat", "frog", "snake", "bird"],
    "colors": ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "gray", "brown", "cyan", "magenta", "teal", "navy", "maroon"],
    "metals": ["iron", "gold", "silver", "copper", "zinc", "lead", "tin", "nickel", "bronze", "brass", "steel", "aluminum", "platinum", "titanium", "chrome", "cobalt"],
    "vehicles": ["car", "bus", "truck", "train", "plane", "boat", "ship", "bike", "scooter", "van", "jeep", "taxi", "tram", "cart", "wagon", "jet"]
}

def load_data(cat_name):
    path = f"/home/exouser/pruning/induction_datasets/category_chains/{cat_name}.jsonl"
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_log_prob(model, tokenizer, prompt, candidate):
    """Calculate the log probability of candidate string given the prompt."""
    full_text = prompt + " " + candidate
    tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].cuda()
    
    # Safely get the number of tokens the candidate adds
    cand_tokens = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
    cand_len = len(cand_tokens)
    
    with torch.no_grad():
        outputs = model(tokens)
        logits = outputs.logits[0, :-1, :] # Shifted
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = tokens[0, 1:] # Shifted targets
        
        # We only care about the log probs of the LAST `cand_len` tokens
        start_idx = len(target_ids) - cand_len
        cand_log_probs = []
        for i in range(start_idx, len(target_ids)):
            cand_log_probs.append(log_probs[i, target_ids[i]].item())
            
    return sum(cand_log_probs)

def evaluate_and_save(model, tokenizer, cat_name, out_dir):
    data = load_data(cat_name)
    candidates = CATEGORIES[cat_name]
    
    prob_correct_total = 0
    gen_correct_total = 0
    total = len(data)
    
    print(f"Evaluating {cat_name.upper()} on 5-Way Global Circuit...")
    
    evaluated_samples = []
    
    for idx, item in enumerate(data):
        prompt = item["clean_prompt"]
        target = item["target"]
        
        # 1. Probability Accuracy
        candidate_probs = {}
        for cand in candidates:
            candidate_probs[cand] = calculate_log_prob(model, tokenizer, prompt, cand)
            
        best_cand = max(candidate_probs, key=candidate_probs.get)
        passed_acc1 = (best_cand == target)
        if passed_acc1:
            prob_correct_total += 1
            
        # 2. Generative Accuracy
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        gen_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Clean generated text
        cleaned_gen = re.sub(r'[^a-zA-Z\s]', '', gen_text).strip().lower()
        
        passed_acc2 = (target.lower() in cleaned_gen)
        if passed_acc2:
            gen_correct_total += 1
            
        # Append boolean flags to the original dictionary
        item["passed_acc1"] = passed_acc1
        item["passed_acc2"] = passed_acc2
        evaluated_samples.append(item)
            
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{total} samples...")
            
    # Save the updated dataset
    out_file = os.path.join(out_dir, f"{cat_name}_evaluated.jsonl")
    with open(out_file, "w") as f:
        for s in evaluated_samples:
            f.write(json.dumps(s) + "\n")
            
    prob_acc = prob_correct_total / total
    gen_acc = gen_correct_total / total
    return prob_acc, gen_acc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = PruningConfig(
        prune_attention_heads=True, lambda_attention_heads=0.8,
        prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
        prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
    )
    
    print("Initializing Model...")
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        "meta-llama/Llama-3.2-1B",
        pruning_config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # Load 5-Way Global Circuit
    circuit_path = "/home/exouser/pruning/intersection_experiments/results_5way/5WAY_GLOBAL_CIRCUIT.pt"
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Global circuit not found at {circuit_path}")
        
    print(f"\n--- Loading 5-Way Global Circuit ---")
    mask = torch.load(circuit_path, weights_only=True)
    
    model.set_final_circuit_mode(True)
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                if name in mask:
                    module.log_alpha.data = torch.where(
                        mask[name].to(device),
                        torch.tensor(5.0, device=device),
                        torch.tensor(-1e6, device=device)
                    )
                else:
                    module.log_alpha.data.fill_(-1e6)
                    
    # Prepare output directory
    out_dir = "/home/exouser/pruning/intersection_experiments/results_5way/evaluated_datasets"
    os.makedirs(out_dir, exist_ok=True)
    
    results = {}
    
    for cat_name in CATEGORIES.keys():
        prob_acc, gen_acc = evaluate_and_save(model, tokenizer, cat_name, out_dir)
        results[cat_name] = {"Probability_Acc": prob_acc, "Generative_Acc": gen_acc}
        print(f"{cat_name.upper()} -> Prob Acc: {prob_acc*100:.2f}% | Gen Acc: {gen_acc*100:.2f}%")
        
    # Save final global report
    report_path = os.path.join(out_dir, "global_circuit_evaluation_summary.txt")
    with open(report_path, "w") as f:
        f.write("FINAL 5-WAY GLOBAL CIRCUIT RESULTS SUMMARY\n")
        f.write("="*50 + "\n")
        for cat, metrics in results.items():
            line = f"{cat.upper():10} | Prob Acc: {metrics['Probability_Acc']*100:6.2f}% | Gen Acc: {metrics['Generative_Acc']*100:6.2f}%\n"
            f.write(line)
            
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for cat, metrics in results.items():
        print(f"{cat.upper():10} | Prob Acc: {metrics['Probability_Acc']*100:6.2f}% | Gen Acc: {metrics['Generative_Acc']*100:6.2f}%")
        
    print(f"\nAll updated datasets securely saved in: {out_dir}")

if __name__ == "__main__":
    main()
