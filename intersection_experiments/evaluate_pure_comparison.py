import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def pad_to_equal_length(t1, t2, pad_token_id):
    max_len = max(t1.shape[1], t2.shape[1])
    if t1.shape[1] < max_len:
        t1 = torch.cat([torch.full((1, max_len - t1.shape[1]), pad_token_id, device='cuda'), t1], dim=1)
    if t2.shape[1] < max_len:
        t2 = torch.cat([torch.full((1, max_len - t2.shape[1]), pad_token_id, device='cuda'), t2], dim=1)
    return t1, t2

def calculate_log_prob(model, tokenizer, prompt, corr_prompt, candidate, is_base=False):
    full_text = prompt + " " + candidate
    corr_text = corr_prompt + " " + candidate
    
    tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].cuda()
    corr_tokens = tokenizer(corr_text, return_tensors="pt")["input_ids"].cuda()
    tokens, corr_tokens = pad_to_equal_length(tokens, corr_tokens, tokenizer.pad_token_id)
    
    cand_tokens = tokenizer(" " + candidate, add_special_tokens=False)["input_ids"]
    cand_len = len(cand_tokens)
    
    with torch.no_grad():
        if is_base:
            outputs = model(tokens)
        else:
            outputs = model(input_ids=tokens, corrupted_input_ids=corr_tokens)
            
        logits = outputs.logits[0, :-1, :] 
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = tokens[0, 1:]
        start_idx = len(target_ids) - cand_len
        cand_log_probs = []
        for i in range(start_idx, len(target_ids)):
            cand_log_probs.append(log_probs[i, target_ids[i]].item())
            
    return sum(cand_log_probs)

def generate_manual(model, tokenizer, prompt, corr_prompt, is_base=False):
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
    corr_tokens = tokenizer(corr_prompt, return_tensors="pt")["input_ids"].cuda()
    tokens, corr_tokens = pad_to_equal_length(tokens, corr_tokens, tokenizer.pad_token_id)
    
    gen_ids = []
    for _ in range(4):
        with torch.no_grad():
            if is_base:
                outputs = model(tokens)
            else:
                outputs = model(input_ids=tokens, corrupted_input_ids=corr_tokens)
        next_tok = torch.argmax(outputs.logits[0, -1, :])
        gen_ids.append(next_tok.item())
        if next_tok.item() == tokenizer.eos_token_id:
            break
            
        tokens = torch.cat([tokens, next_tok.unsqueeze(0).unsqueeze(0)], dim=-1)
        corr_tokens = torch.cat([corr_tokens, torch.tensor([[tokenizer.eos_token_id]], device='cuda')], dim=-1)
        
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

def build_base_cache(base_model, tokenizer, cat_name):
    data = load_data(cat_name)
    candidates = CATEGORIES[cat_name]
    base_cache = []
    
    print(f"  -> Caching Base Model metrics for {cat_name.upper()}...")
    for item in data:
        prompt = item["clean_prompt"]
        corr_prompt = item["corr_prompt"] if "corr_prompt" in item else item.get("corrupted_prompt", "")
        target = item["target"]
        
        # 1. Base log probs
        base_cand_log_probs = []
        for cand in candidates:
            base_cand_log_probs.append(calculate_log_prob(base_model, tokenizer, prompt, corr_prompt, cand, is_base=True))
            
        lps = torch.tensor(base_cand_log_probs)
        probs = F.softmax(lps, dim=0)
        target_idx = candidates.index(target)
        base_target_prob = probs[target_idx].item()
        
        # 2. Golden logits
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            outputs = base_model(tokens)
            golden_logits = outputs.logits[0, -1, :].float().cpu()
            
        base_cache.append({
            "base_target_prob": base_target_prob,
            "golden_logits": golden_logits
        })
        
    return base_cache

def evaluate_condition(model, tokenizer, cat_name, base_cache):
    data = load_data(cat_name)
    candidates = CATEGORIES[cat_name]
    
    prob_correct = 0
    gen_correct = 0
    kl_total = 0.0
    prob_diff_total = 0.0
    total = len(data)
    
    for idx, item in enumerate(data):
        prompt = item["clean_prompt"]
        corr_prompt = item["corr_prompt"] if "corr_prompt" in item else item.get("corrupted_prompt", "")
        target = item["target"]
        
        # 1. Circuit Probabilities
        circuit_cand_log_probs = []
        for cand in candidates:
            circuit_cand_log_probs.append(calculate_log_prob(model, tokenizer, prompt, corr_prompt, cand, is_base=False))
            
        lps = torch.tensor(circuit_cand_log_probs)
        probs = F.softmax(lps, dim=0)
        target_idx = candidates.index(target)
        circuit_target_prob = probs[target_idx].item()
        
        best_idx = torch.argmax(probs).item()
        if candidates[best_idx] == target:
            prob_correct += 1
            
        base_target_prob = base_cache[idx]["base_target_prob"]
        prob_diff_total += (circuit_target_prob - base_target_prob)
        
        # 2. Gen Acc & KL
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        corr_tokens = tokenizer(corr_prompt, return_tensors="pt")["input_ids"].cuda()
        tokens, corr_tokens = pad_to_equal_length(tokens, corr_tokens, tokenizer.pad_token_id)
        
        with torch.no_grad():
            circuit_outputs = model(input_ids=tokens, corrupted_input_ids=corr_tokens)
            circuit_logits = circuit_outputs.logits[0, -1, :].float()
            
        golden_logits = base_cache[idx]["golden_logits"].cuda()
        kl_div = F.kl_div(
            F.log_softmax(circuit_logits.unsqueeze(0), dim=-1),
            F.log_softmax(golden_logits.unsqueeze(0), dim=-1),
            reduction='batchmean',
            log_target=True
        ).item()
        kl_total += kl_div
        
        gen_text = generate_manual(model, tokenizer, prompt, corr_prompt, is_base=False)
        cleaned_gen = re.sub(r'[^a-zA-Z\s]', '', gen_text).strip().lower()
        if target.lower() in cleaned_gen:
            gen_correct += 1
            
    return {
        "Prob_Acc": prob_correct / total,
        "Gen_Acc": gen_correct / total,
        "KL_Div": kl_total / total,
        "Prob_Diff": prob_diff_total / total
    }

def count_heads(mask):
    return sum(v.sum().item() for v in mask.values())

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
    
    print("Initializing Unpruned Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16).to(device)
    base_model.eval()
    
    print("Initializing Prunable Model...")
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    
    DIR = "/home/exouser/pruning/intersection_experiments/results_5way_pure"
    
    intersect_mask_300 = torch.load(f"{DIR}/INTERSECT_PURE_300.pt", weights_only=True)
    intersect_heads_300 = count_heads(intersect_mask_300)
    
    intersect_mask_600 = torch.load(f"{DIR}/INTERSECT_PURE_600.pt", weights_only=True)
    intersect_heads_600 = count_heads(intersect_mask_600)
    
    indiv_masks_300, indiv_heads_300 = {}, {}
    indiv_masks_600, indiv_heads_600 = {}, {}
    for cat in CATEGORIES.keys():
        m_300 = torch.load(f"{DIR}/{cat}_pure_300.pt", weights_only=True)
        indiv_masks_300[cat] = m_300
        indiv_heads_300[cat] = count_heads(m_300)
        
        m_600 = torch.load(f"{DIR}/{cat}_pure_600.pt", weights_only=True)
        indiv_masks_600[cat] = m_600
        indiv_heads_600[cat] = count_heads(m_600)
        
    print(f"--- CIRCUIT SIZES ---")
    print(f"Intersection 300 Circuit: {intersect_heads_300} heads")
    print(f"Intersection 600 Circuit: {intersect_heads_600} heads")
    
    results = {}
    
    for cat in CATEGORIES.keys():
        print(f"\n{'-'*50}\nEVALUATING {cat.upper()} (Indiv 300 Heads: {indiv_heads_300[cat]} | Indiv 600 Heads: {indiv_heads_600[cat]})\n{'-'*50}")
        base_cache = build_base_cache(base_model, tokenizer, cat)
        results[cat] = {}
        
        apply_mask(model, indiv_masks_300[cat], device)
        r = evaluate_condition(model, tokenizer, cat, base_cache)
        results[cat]["Indiv300"] = r
        print(f"  [Indiv 300]     Prob: {r['Prob_Acc']*100:.2f}% | Gen: {r['Gen_Acc']*100:.2f}% | KL: {r['KL_Div']:.4f} | ProbDiff: {r['Prob_Diff']*100:+.2f}%")
        
        apply_mask(model, indiv_masks_600[cat], device)
        r = evaluate_condition(model, tokenizer, cat, base_cache)
        results[cat]["Indiv600"] = r
        print(f"  [Indiv 600]     Prob: {r['Prob_Acc']*100:.2f}% | Gen: {r['Gen_Acc']*100:.2f}% | KL: {r['KL_Div']:.4f} | ProbDiff: {r['Prob_Diff']*100:+.2f}%")
        
        apply_mask(model, intersect_mask_300, device)
        r = evaluate_condition(model, tokenizer, cat, base_cache)
        results[cat]["Intersect300"] = r
        print(f"  [Intersect 300] Prob: {r['Prob_Acc']*100:.2f}% | Gen: {r['Gen_Acc']*100:.2f}% | KL: {r['KL_Div']:.4f} | ProbDiff: {r['Prob_Diff']*100:+.2f}%")
        
        apply_mask(model, intersect_mask_600, device)
        r = evaluate_condition(model, tokenizer, cat, base_cache)
        results[cat]["Intersect600"] = r
        print(f"  [Intersect 600] Prob: {r['Prob_Acc']*100:.2f}% | Gen: {r['Gen_Acc']*100:.2f}% | KL: {r['KL_Div']:.4f} | ProbDiff: {r['Prob_Diff']*100:+.2f}%")

    print("\n\n==========================================================================================================================================")
    print("FINAL PURE TRAINING (NO PRETRAIN) MULTI-METRIC ACCURACY TABLE")
    print("==========================================================================================================================================")
    
    h1 = "| Dataset | Indiv 300 Prob | Indiv 300 Gen | Indiv 300 KL | "
    h2 = "| Indiv 600 Prob | Indiv 600 Gen | Indiv 600 KL | "
    h3 = f"Intersect 300 Prob ({intersect_heads_300}h) | Intersect 300 Gen | Intersect 300 KL | "
    h4 = f"Intersect 600 Prob ({intersect_heads_600}h) | Intersect 600 Gen | Intersect 600 KL |"
    
    print(h1 + h2 + h3 + h4)
    print("| :--- " * 13 + "|")
    
    for cat in CATEGORIES.keys():
        r = results[cat]
        
        s1 = f"| **{cat.upper()}** | {r['Indiv300']['Prob_Acc']*100:.2f}% | {r['Indiv300']['Gen_Acc']*100:.2f}% | {r['Indiv300']['KL_Div']:.4f} | "
        s2 = f"{r['Indiv600']['Prob_Acc']*100:.2f}% | {r['Indiv600']['Gen_Acc']*100:.2f}% | {r['Indiv600']['KL_Div']:.4f} | "
        s3 = f"{r['Intersect300']['Prob_Acc']*100:.2f}% | {r['Intersect300']['Gen_Acc']*100:.2f}% | {r['Intersect300']['KL_Div']:.4f} | "
        s4 = f"{r['Intersect600']['Prob_Acc']*100:.2f}% | {r['Intersect600']['Gen_Acc']*100:.2f}% | {r['Intersect600']['KL_Div']:.4f} |"
        
        print(s1 + s2 + s3 + s4)

if __name__ == "__main__":
    main()
