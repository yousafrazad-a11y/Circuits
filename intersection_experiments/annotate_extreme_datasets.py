import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

CATEGORIES = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "pear", "plum", "kiwi", "melon", "cherry", "lemon", "lime", "fig", "date", "papaya"],
    "animals": ["cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow", "pig", "sheep", "goat", "frog", "snake", "bird"],
    "colors": ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "gray", "brown", "cyan", "magenta", "teal", "navy", "maroon"],
    "metals": ["iron", "gold", "silver", "copper", "zinc", "lead", "tin", "nickel", "bronze", "brass", "steel", "aluminum", "platinum", "titanium", "chrome", "cobalt"],
    "vehicles": ["car", "bus", "truck", "train", "plane", "boat", "ship", "bike", "scooter", "van", "jeep", "taxi", "tram", "cart", "wagon", "jet"]
}

def pad_to_equal_length(t1, t2, pad_token_id):
    max_len = max(t1.shape[1], t2.shape[1])
    if t1.shape[1] < max_len:
        t1 = torch.cat([torch.full((1, max_len - t1.shape[1]), pad_token_id, device='cuda'), t1], dim=1)
    if t2.shape[1] < max_len:
        t2 = torch.cat([torch.full((1, max_len - t2.shape[1]), pad_token_id, device='cuda'), t2], dim=1)
    return t1, t2

def calculate_log_prob(model, tokenizer, prompt, corr_prompt, candidate):
    full_text = prompt + " " + candidate
    corr_text = corr_prompt + " " + candidate
    
    tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].cuda()
    corr_tokens = tokenizer(corr_text, return_tensors="pt")["input_ids"].cuda()
    tokens, corr_tokens = pad_to_equal_length(tokens, corr_tokens, tokenizer.pad_token_id)
    
    cand_tokens = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
    cand_len = len(cand_tokens)
    
    with torch.no_grad():
        outputs = model(input_ids=tokens, corrupted_input_ids=corr_tokens)
        logits = outputs.logits[0, :-1, :] 
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = tokens[0, 1:]
        start_idx = len(target_ids) - cand_len
        cand_log_probs = []
        for i in range(start_idx, len(target_ids)):
            cand_log_probs.append(log_probs[i, target_ids[i]].item())
            
    return sum(cand_log_probs)

def generate_manual(model, tokenizer, prompt, corr_prompt):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    corr_tokens = tokenizer(corr_prompt, return_tensors="pt")["input_ids"].cuda()
    tokens, corr_tokens = pad_to_equal_length(tokens, corr_tokens, tokenizer.pad_token_id)
    
    gen_ids = []
    for _ in range(4):
        with torch.no_grad():
            outputs = model(input_ids=tokens, corrupted_input_ids=corr_tokens)
        next_tok = torch.argmax(outputs.logits[0, -1, :])
        gen_ids.append(next_tok.item())
        if next_tok.item() == tokenizer.eos_token_id:
            break
            
        tokens = torch.cat([tokens, next_tok.unsqueeze(0).unsqueeze(0)], dim=-1)
        corr_tokens = torch.cat([corr_tokens, torch.tensor([[tokenizer.eos_token_id]], device='cuda')], dim=-1)
        
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def apply_mask(model, mask, device):
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = PruningConfig(
        prune_attention_heads=True, lambda_attention_heads=0.8,
        prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
        prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
    )
    
    print("Initializing Prunable Model...")
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    
    DIR = "/home/exouser/pruning/intersection_experiments/results_5way_extreme"
    OUT_DIR = os.path.join(DIR, "annotated_datasets")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    intersect_mask = torch.load(f"{DIR}/5WAY_EXTREME_GLOBAL_CIRCUIT.pt", weights_only=True)
    joint_600_mask = torch.load(f"{DIR}/JOINT_600_EXTREME_CIRCUIT.pt", weights_only=True)
    
    for cat, candidates in CATEGORIES.items():
        print(f"\nProcessing {cat.upper()}...")
        
        path = f"/home/exouser/pruning/induction_datasets/category_chains/{cat}.jsonl"
        out_path = os.path.join(OUT_DIR, f"{cat}_annotated.jsonl")
        
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
            
        annotated_data = []
        
        for idx, item in enumerate(data):
            prompt = item["clean_prompt"]
            corr_prompt = item["corr_prompt"] if "corr_prompt" in item else item.get("corrupted_prompt", "")
            target = item["target"]
            
            # --- EVALUATE INTERSECT CIRCUIT ---
            apply_mask(model, intersect_mask, device)
            
            # Prob Acc
            probs_int = {cand: calculate_log_prob(model, tokenizer, prompt, corr_prompt, cand) for cand in candidates}
            best_cand_int = max(probs_int, key=probs_int.get)
            item["passed_acc1_intersect"] = bool(best_cand_int == target)
            
            # Gen Acc
            gen_text_int = generate_manual(model, tokenizer, prompt, corr_prompt)
            cleaned_gen_int = re.sub(r'[^a-zA-Z\s]', '', gen_text_int).strip().lower()
            item["passed_acc2_intersect"] = bool(target.lower() in cleaned_gen_int)
            
            # --- EVALUATE JOINT 600 CIRCUIT ---
            apply_mask(model, joint_600_mask, device)
            
            # Prob Acc
            probs_j = {cand: calculate_log_prob(model, tokenizer, prompt, corr_prompt, cand) for cand in candidates}
            best_cand_j = max(probs_j, key=probs_j.get)
            item["passed_acc1_joint"] = bool(best_cand_j == target)
            
            # Gen Acc
            gen_text_j = generate_manual(model, tokenizer, prompt, corr_prompt)
            cleaned_gen_j = re.sub(r'[^a-zA-Z\s]', '', gen_text_j).strip().lower()
            item["passed_acc2_joint"] = bool(target.lower() in cleaned_gen_j)
            
            annotated_data.append(item)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(data)}")
                
        # Save dataset
        with open(out_path, 'w') as f:
            for item in annotated_data:
                f.write(json.dumps(item) + "\n")
                
        print(f"Saved {out_path}")
        
    print("\nAll datasets annotated successfully!")

if __name__ == "__main__":
    main()
