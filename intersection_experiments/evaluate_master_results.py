import os
import sys
import torch
import json
import re
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from utils import _get_model_info

CATEGORIES = ["fruits", "animals", "colors", "metals", "vehicles"]
DIR = "/home/exouser/pruning/intersection_experiments/results_master"

class CategoryDataset(Dataset):
    def __init__(self, cat, tokenizer):
        self.data = []
        self.all_targets = set()
        with open(f"/home/exouser/pruning/induction_datasets/category_chains/{cat}.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                target_tok = tokenizer.encode(" " + item["target"], add_special_tokens=False)[-1]
                self.all_targets.add(target_tok)
                self.data.append({
                    "clean_prompt": item["clean_prompt"],
                    "corr_prompt": item["corr_prompt"] if "corr_prompt" in item else item.get("corrupted_prompt", ""),
                    "target_tok": target_tok,
                    "target_str": item["target"].lower()
                })
        self.tokenizer = tokenizer
        self.all_targets = list(self.all_targets)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch, tokenizer, all_targets):
    clean_input_ids = []
    corr_input_ids = []
    target_tokens, target_strs = [], []
    for item in batch:
        c_seq = torch.tensor(tokenizer.encode(item["clean_prompt"], add_special_tokens=True))
        corr_seq = torch.tensor(tokenizer.encode(item["corr_prompt"], add_special_tokens=True))
        clean_input_ids.append(c_seq)
        corr_input_ids.append(corr_seq)
        target_tokens.append(item["target_tok"])
        target_strs.append(item["target_str"])
        
    global_max_len = max(max(len(s) for s in clean_input_ids), max(len(s) for s in corr_input_ids))
    
    clean_padded, corr_padded, attn_mask = [], [], []
    for c_seq, corr_seq in zip(clean_input_ids, corr_input_ids):
        pad_len = global_max_len - len(c_seq)
        pad_len_corr = global_max_len - len(corr_seq)
        clean_padded.append(torch.cat([torch.full((pad_len,), tokenizer.pad_token_id), c_seq]))
        corr_padded.append(torch.cat([torch.full((pad_len_corr,), tokenizer.pad_token_id), corr_seq]))
        attn_mask.append(torch.cat([torch.zeros(pad_len), torch.ones(len(c_seq))]))
        
    return {
        "clean_input_ids": torch.stack(clean_padded).to("cuda"),
        "corr_input_ids": torch.stack(corr_padded).to("cuda"),
        "attention_mask": torch.stack(attn_mask).to("cuda"),
        "target_tokens": torch.tensor(target_tokens, device="cuda"),
        "target_strs": target_strs,
        "all_targets": torch.tensor(all_targets, device="cuda")
    }

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

def eval_prob_accuracy(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corr_input_ids"],
                attention_mask=batch["attention_mask"]
            )
            b_idx = torch.arange(batch["clean_input_ids"].size(0), device="cuda")
            all_target_logits = outputs.logits[:, -1, :][:, batch["all_targets"]]
            best_target_idx = torch.argmax(all_target_logits, dim=-1)
            predicted_tokens = batch["all_targets"][best_target_idx]
            correct += (predicted_tokens == batch["target_tokens"]).sum().item()
            total += batch["clean_input_ids"].size(0)
    return correct / total

def eval_gen_accuracy(model, tokenizer, dataloader):
    model.eval()
    correct = total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch_size = batch["clean_input_ids"].size(0)
            target_strs = batch["target_strs"]
            
            tokens = batch["clean_input_ids"]
            corr_tokens = batch["corr_input_ids"]
            attn_mask = batch["attention_mask"]
            
            gen_ids = [[] for _ in range(batch_size)]
            
            for _ in range(4):
                outputs = model(input_ids=tokens, corrupted_input_ids=corr_tokens, attention_mask=attn_mask)
                next_toks = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                
                for i in range(batch_size):
                    gen_ids[i].append(next_toks[i].item())
                    
                tokens = torch.cat([tokens, next_toks.unsqueeze(1)], dim=-1)
                corr_tokens = torch.cat([corr_tokens, torch.full((batch_size, 1), tokenizer.pad_token_id, device='cuda')], dim=-1)
                attn_mask = torch.cat([attn_mask, torch.ones((batch_size, 1), device='cuda')], dim=-1)
                
            for i in range(batch_size):
                text = tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
                first_word = re.sub(r'[^a-zA-Z]', '', text.split()[0] if text else "").lower()
                if target_strs[i] == first_word:
                    correct += 1
            total += batch_size
            
    return correct / total

def format_row(name, ds, prob, gen, heads, kl):
    return f"| {name:<35} | {ds:<15} | {prob*100:>7.2f}% | {gen*100:>7.2f}% | {heads:>6} | {kl:>7.2f} |"

def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = PruningConfig(prune_attention_heads=True, lambda_attention_heads=0.8)
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    
    datasets = {cat: CategoryDataset(cat, tokenizer) for cat in CATEGORIES}
    dataloaders = {cat: DataLoader(ds, batch_size=64, shuffle=False, collate_fn=lambda b, ds=ds: collate_fn(b, tokenizer, ds.all_targets)) for cat, ds in datasets.items()}
    
    print("\n" + "="*95, flush=True)
    print(f"| {'Mask Type':<35} | {'Eval Dataset':<15} | {'Prob Acc':>8} | {'Gen Acc':>8} | {'Heads':>6} | {'KL Div':>7} |", flush=True)
    print("-" * 95, flush=True)
    
    indiv_pure = {cat: torch.load(f"{DIR}/{cat}_pure_600.pt", weights_only=True) for cat in CATEGORIES}
    indiv_extreme = {cat: torch.load(f"{DIR}/{cat}_extreme_circuit.pt", weights_only=True) for cat in CATEGORIES}
    global_600 = torch.load(f"{DIR}/global_circuit_600.pt", weights_only=True)
    
    # 1. Pure Individual
    for cat in CATEGORIES:
        apply_mask(model, indiv_pure[cat], device)
        sparsity = model.get_sparsity_loss(step=1000)["total_sparsity"].item()
        heads = sum(v.sum().item() for v in indiv_pure[cat].values())
        prob = eval_prob_accuracy(model, dataloaders[cat])
        gen = eval_gen_accuracy(model, tokenizer, dataloaders[cat])
        print(format_row(f"Pure Circuit ({cat})", cat, prob, gen, heads, sparsity), flush=True)
        
    print("-" * 95, flush=True)
    
    # 2. Extreme Individual
    for cat in CATEGORIES:
        apply_mask(model, indiv_extreme[cat], device)
        sparsity = model.get_sparsity_loss(step=1000)["total_sparsity"].item()
        heads = sum(v.sum().item() for v in indiv_extreme[cat].values())
        prob = eval_prob_accuracy(model, dataloaders[cat])
        gen = eval_gen_accuracy(model, tokenizer, dataloaders[cat])
        print(format_row(f"Extreme Circuit ({cat})", cat, prob, gen, heads, sparsity), flush=True)
        
    print("-" * 95, flush=True)
    
    # 3. Global 600
    apply_mask(model, global_600, device)
    sparsity = model.get_sparsity_loss(step=1000)["total_sparsity"].item()
    heads = sum(v.sum().item() for v in global_600.values())
    
    total_prob_corr, total_gen_corr, total = 0, 0, 0
    for cat in CATEGORIES:
        prob = eval_prob_accuracy(model, dataloaders[cat])
        gen = eval_gen_accuracy(model, tokenizer, dataloaders[cat])
        print(format_row("Global Joint (600 epochs)", cat, prob, gen, heads, sparsity), flush=True)
        total_prob_corr += prob * len(dataloaders[cat].dataset)
        total_gen_corr += gen * len(dataloaders[cat].dataset)
        total += len(dataloaders[cat].dataset)
    print(format_row("Global Joint (600 epochs)", "ALL AVERAGED", total_prob_corr/total, total_gen_corr/total, heads, sparsity), flush=True)
        
    print("-" * 95, flush=True)
    
    # 4. Universal Core (5-way Intersection of Extreme)
    universal_mask = {k: indiv_extreme[CATEGORIES[0]][k].clone() for k in indiv_extreme[CATEGORIES[0]]}
    for cat in CATEGORIES[1:]:
        for k in universal_mask:
            universal_mask[k] &= indiv_extreme[cat][k]
            
    apply_mask(model, universal_mask, device)
    sparsity = model.get_sparsity_loss(step=1000)["total_sparsity"].item()
    heads = sum(v.sum().item() for v in universal_mask.values())
    
    total_prob_corr, total_gen_corr, total = 0, 0, 0
    for cat in CATEGORIES:
        prob = eval_prob_accuracy(model, dataloaders[cat])
        gen = eval_gen_accuracy(model, tokenizer, dataloaders[cat])
        print(format_row("Universal Core (5-way Int.)", cat, prob, gen, heads, sparsity), flush=True)
        total_prob_corr += prob * len(dataloaders[cat].dataset)
        total_gen_corr += gen * len(dataloaders[cat].dataset)
        total += len(dataloaders[cat].dataset)
    print(format_row("Universal Core (5-way Int.)", "ALL AVERAGED", total_prob_corr/total, total_gen_corr/total, heads, sparsity), flush=True)
        
    print("=" * 95, flush=True)

if __name__ == "__main__":
    main()
