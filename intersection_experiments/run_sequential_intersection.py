import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
import argparse

# Insert circuit_pruning-argo path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

# Import datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from induction_datasets.test_venn_induction import load_dataset

def get_pruning_config(level):
    """Return the exact PruningConfig based on the requested level."""
    if level == "blocks":
        return PruningConfig(
            prune_attention_blocks=True, lambda_attention_blocks=0.5,
            prune_mlp_blocks=True, lambda_mlp_blocks=0.5,
            prune_full_layers=True, lambda_full_layers=0.0,
            prune_attention_heads=False, prune_attention_neurons=False,
            prune_mlp_hidden=False, prune_mlp_output=False
        )
    elif level == "heads":
        return PruningConfig(
            prune_attention_heads=True, lambda_attention_heads=0.8,
            prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
            prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
        )
    elif level == "heads_mlp":
        return PruningConfig(
            prune_attention_heads=True, lambda_attention_heads=0.8,
            prune_mlp_hidden=False,
            prune_mlp_blocks=True, lambda_mlp_blocks=0.5, 
            prune_attention_blocks=False, prune_full_layers=False,
            prune_attention_neurons=False
        )
    elif level == "neurons":
        # The exact neuron-level config from circuit_pruning-argo
        return PruningConfig(
            prune_attention_heads=True, lambda_attention_heads=0.8,
            prune_mlp_hidden=True, lambda_mlp_hidden=1.0,
            prune_mlp_output=True, lambda_mlp_output=1.0,
            prune_attention_neurons=True, lambda_attention_neurons=0.15,
            prune_attention_blocks=True, lambda_attention_blocks=0.5,
            prune_mlp_blocks=True, lambda_mlp_blocks=0.5,
            prune_full_layers=False,
            prune_embedding=False
        )
    else:
        raise ValueError(f"Unknown pruning level: {level}")

def collate_fn(batch, tokenizer):
    clean_input_ids = []
    corr_input_ids = []
    target_tokens = []
    distractor_tokens = []
    answer_positions = []
    
    for item in batch:
        prompt = item["clean_prompt"]
        clean_input_ids.append(torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)))
        corr_input_ids.append(torch.tensor(tokenizer.encode(item["corr_prompt"], add_special_tokens=True)))
        target_tokens.append(tokenizer.encode(item["target"], add_special_tokens=False)[0])
        distractor_tokens.append(tokenizer.encode(item["distractor"], add_special_tokens=False)[0])
    
    max_len_clean = max(len(seq) for seq in clean_input_ids)
    max_len_corr = max(len(seq) for seq in corr_input_ids)
    max_len = max(max_len_clean, max_len_corr)
    
    clean_padded, corr_padded, attn_mask = [], [], []
    
    for c_seq, corr_seq in zip(clean_input_ids, corr_input_ids):
        pad_len_c = max_len - len(c_seq)
        pad_len_corr = max_len - len(corr_seq)
        clean_padded.append(torch.cat([c_seq, torch.full((pad_len_c,), tokenizer.pad_token_id)]))
        corr_padded.append(torch.cat([corr_seq, torch.full((pad_len_corr,), tokenizer.pad_token_id)]))
        attn_mask.append(torch.cat([torch.ones(len(c_seq)), torch.zeros(pad_len_c)]))
        answer_positions.append(len(c_seq) - 1)
        
    return {
        "clean_input_ids": torch.stack(clean_padded).long(),
        "corr_input_ids": torch.stack(corr_padded).long(),
        "attention_mask": torch.stack(attn_mask).long(),
        "target_tokens": torch.tensor(target_tokens).long(),
        "distractor_tokens": torch.tensor(distractor_tokens).long(),
        "answer_positions": torch.tensor(answer_positions).long(),
    }

def train_phase(model, dataloader, epochs, lr, device="cuda"):
    # Freeze all base model parameters, only unfreeze log_alpha (gates)
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "log_alpha" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float() # stable training
            trainable_params += param.numel()
            
    print(f"Trainable gate parameters: {trainable_params} / {total_params}")
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_kl = 0
        epoch_task = 0
        epoch_sparsity = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size_curr = batch["clean_input_ids"].size(0)
            
            # Golden logits
            model.eval()
            with torch.no_grad():
                golden_outputs = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                golden_logits = golden_outputs.logits.detach()
            
            # Forward pass
            model.train()
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corr_input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            
            # KL divergence calculation at answer position
            pos = batch["answer_positions"]
            batch_indices = torch.arange(batch_size_curr, device=device)
            
            circuit_logits = logits[batch_indices, pos].float()
            target_logits = golden_logits[batch_indices, pos].float()
            
            kl_loss = F.kl_div(
                F.log_softmax(circuit_logits, dim=-1),
                F.log_softmax(target_logits, dim=-1),
                reduction='batchmean',
                log_target=True
            )
            
            # Task loss calculation exactly as in ioi_llama.py
            logit_good = logits[batch_indices, pos, batch["target_tokens"]].float()
            logit_bad = logits[batch_indices, pos, batch["distractor_tokens"]].float()
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()
            
            # Sparsity loss (no adaptive lambda, purely from config)
            sparsity_loss = model.get_sparsity_loss(step=1000)["total_sparsity"]
            
            # Combined Loss (identical to argo)
            loss = kl_loss * 1.5 + sparsity_loss + task_loss
            loss.backward()
            optimizer.step()
            
            # Clamping
            with torch.no_grad():
                for name, module in model.named_modules():
                    if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                        module.log_alpha.clamp_(-5.0, 5.0)
                        
            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_task += task_loss.item()
            epoch_sparsity += sparsity_loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.3f} | KL: {epoch_kl/len(dataloader):.3f} | Task: {epoch_task/len(dataloader):.3f} | Sparsity: {epoch_sparsity/len(dataloader):.3f}")

def extract_mask(model):
    """Extract binary mask for all gates > 0.5 probability"""
    model.eval()
    mask = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                # Standard sigmoidal probability map as used in hard concrete gates
                s = torch.sigmoid(module.log_alpha)
                # Apply stretch exactly as hard concrete does before rounding
                s_stretched = s * 1.2 - 0.1
                mask[name] = (s_stretched > 0.5).bool().cpu()
    return mask

def calculate_metrics(mask_a, mask_b):
    metrics = {}
    total_a = 0
    total_b = 0
    total_intersect = 0
    total_union = 0
    
    for k in mask_a.keys():
        ma = mask_a[k]
        mb = mask_b[k]
        intersect = ma & mb
        union = ma | mb
        
        total_a += ma.sum().item()
        total_b += mb.sum().item()
        total_intersect += intersect.sum().item()
        total_union += union.sum().item()
        
    jaccard = total_intersect / total_union if total_union > 0 else 0
    overlap_a = total_intersect / total_a if total_a > 0 else 0
    overlap_b = total_intersect / total_b if total_b > 0 else 0
    
    metrics = {
        "Size_A": total_a,
        "Size_B": total_b,
        "Intersection": total_intersect,
        "Union": total_union,
        "Jaccard_Similarity": jaccard,
        "Overlap_A_Percent": overlap_a,
        "Overlap_B_Percent": overlap_b
    }
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--task_a", type=str, required=True)
    parser.add_argument("--task_b", type=str, required=True)
    parser.add_argument("--epochs_joint", type=int, default=50)
    parser.add_argument("--epochs_fine", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--level", type=str, choices=["blocks", "heads", "heads_mlp", "neurons"], default="neurons")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading datasets: {args.task_a} and {args.task_b}")
    ds_a = load_dataset(args.task_a)
    ds_b = load_dataset(args.task_b)
    joint_ds = ConcatDataset([ds_a, ds_b])
    
    dl_joint = DataLoader(joint_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    dl_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    config = get_pruning_config(args.level)
    print(f"Initializing Model at pruning level: {args.level}")
    
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model,
        pruning_config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    print(f"\n{'='*50}\nPHASE 1: JOINT TRAINING (A+B)\n{'='*50}")
    train_phase(model, dl_joint, args.epochs_joint, args.lr, device=device)
    
    # Save Joint State
    joint_state = {k: v.clone() for k, v in model.state_dict().items() if 'log_alpha' in k}
    
    print(f"\n{'='*50}\nPHASE 2A: FINE-TUNE ON {os.path.basename(args.task_a).upper()}\n{'='*50}")
    train_phase(model, dl_a, args.epochs_fine, args.lr, device=device)
    mask_a = extract_mask(model)
    
    print(f"\n{'='*50}\nPHASE 2B: FINE-TUNE ON {os.path.basename(args.task_b).upper()}\n{'='*50}")
    # Restore Joint State to branch
    model.load_state_dict(joint_state, strict=False)
    train_phase(model, dl_b, args.epochs_fine, args.lr, device=device)
    mask_b = extract_mask(model)
    
    print(f"\n{'='*50}\nPHASE 3: INTERSECTION & METRICS\n{'='*50}")
    metrics = calculate_metrics(mask_a, mask_b)
    print("\n--- RESULTS ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{os.path.basename(args.task_a).split('.')[0]}_vs_{os.path.basename(args.task_b).split('.')[0]}_{args.level}.pt")
    torch.save({
        "mask_a": mask_a,
        "mask_b": mask_b,
        "metrics": metrics
    }, out_path)
    print(f"\nSaved masks and metrics to {out_path}")

if __name__ == "__main__":
    main()
