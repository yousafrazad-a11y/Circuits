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
    if level == "heads":
        return PruningConfig(
            prune_attention_heads=True, lambda_attention_heads=0.8,
            prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
            prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
        )
    elif level == "neurons":
        return PruningConfig(
            prune_attention_heads=True, lambda_attention_heads=0.8,
            prune_mlp_hidden=True, lambda_mlp_hidden=1.0,
            prune_mlp_output=True, lambda_mlp_output=1.0,
            prune_attention_neurons=True, lambda_attention_neurons=0.15,
            prune_attention_blocks=True, lambda_attention_blocks=0.5,
            prune_mlp_blocks=True, lambda_mlp_blocks=0.5,
            prune_full_layers=False, prune_embedding=False
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
            
            # KL divergence calculation
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
            
            # Sparsity loss
            sparsity_loss = model.get_sparsity_loss(step=1000)["total_sparsity"]
            
            # Combined Loss
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
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.3f} | KL: {epoch_kl/len(dataloader):.3f} | Task: {epoch_task/len(dataloader):.3f} | Sparsity: {epoch_sparsity/len(dataloader):.3f}")

def extract_mask(model):
    """Extract binary mask for all gates > 0.5 probability"""
    model.eval()
    mask = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                s = torch.sigmoid(module.log_alpha)
                s_stretched = s * 1.2 - 0.1
                mask[name] = (s_stretched > 0.5).bool().cpu()
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--level", type=str, default="heads")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    data_dir = "/home/exouser/pruning/induction_datasets/category_chains"
    dataset_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    dataset_files.sort()
    
    print(f"Loading {len(dataset_files)} datasets...")
    datasets = {}
    for f in dataset_files:
        name = os.path.basename(f).split('.')[0]
        datasets[name] = load_dataset(f)
        print(f" - Loaded {name}: {len(datasets[name])} samples")
        
    joint_ds = ConcatDataset(list(datasets.values()))
    dl_joint = DataLoader(joint_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    config = get_pruning_config(args.level)
    print(f"\nInitializing Model at pruning level: {args.level}")
    
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model,
        pruning_config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    print(f"\n{'='*60}\nPHASE 1: JOINT TRAINING ON ALL 5 DATASETS\n{'='*60}")
    train_phase(model, dl_joint, args.epochs, args.lr, device=device)
    
    # Save Joint State
    joint_state = {k: v.clone() for k, v in model.state_dict().items() if 'log_alpha' in k}
    
    # Phase 2: Fine-Tuning branches
    final_masks = {}
    for name, ds in datasets.items():
        print(f"\n{'='*60}\nPHASE 2: FINE-TUNE ON {name.upper()}\n{'='*60}")
        # Restore Joint State
        model.load_state_dict(joint_state, strict=False)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        train_phase(model, dl, args.epochs, args.lr, device=device)
        final_masks[name] = extract_mask(model)
        
    print(f"\n{'='*60}\nPHASE 3: 5-WAY GLOBAL INTERSECTION\n{'='*60}")
    global_circuit = {}
    first_name = list(final_masks.keys())[0]
    for k in final_masks[first_name].keys():
        global_circuit[k] = final_masks[first_name][k].clone()
        
    for name in list(final_masks.keys())[1:]:
        for k in global_circuit.keys():
            global_circuit[k] = global_circuit[k] & final_masks[name][k]
            
    global_size = sum(global_circuit[k].sum().item() for k in global_circuit.keys())
    print(f"The 5-Way Universal Circuit contains {global_size} active components.")
    
    # Save results
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_5way")
    os.makedirs(out_dir, exist_ok=True)
    
    for name, mask in final_masks.items():
        torch.save(mask, os.path.join(out_dir, f"{name}_circuit.pt"))
        print(f"Saved {name} specific circuit to {out_dir}")
        
    torch.save(global_circuit, os.path.join(out_dir, "5WAY_GLOBAL_CIRCUIT.pt"))
    print(f"Saved global 5-way circuit to {out_dir}/5WAY_GLOBAL_CIRCUIT.pt")
    
    print("\nAll 5-way processing completed perfectly.")

if __name__ == "__main__":
    main()
