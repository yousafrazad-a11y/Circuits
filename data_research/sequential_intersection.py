import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer

# Fix import path for circuit_pruning-argo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

# Import existing datasets and losses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from induction_datasets.test_venn_induction import load_dataset
from venn_circuit_discovery_v2.loss import kl_divergence_loss, margin_loss

def collate_fn(batch, tokenizer):
    # Standard 2-stream collation for single dataset
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
    
    # Pad sequences
    max_len_clean = max(len(seq) for seq in clean_input_ids)
    max_len_corr = max(len(seq) for seq in corr_input_ids)
    max_len = max(max_len_clean, max_len_corr)
    
    clean_padded = []
    corr_padded = []
    attn_mask = []
    
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
        "target": torch.tensor(target_tokens).long(),
        "distractor": torch.tensor(distractor_tokens).long(),
        "answer_positions": torch.tensor(answer_positions).long(),
    }

def train_phase(model, dataloader, epochs, lr, margin, target_kl, lambda_mult=1.0, device="cuda"):
    # Freeze all base model parameters
    for name, param in model.named_parameters():
        if "log_alpha" not in name:
            param.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    optimizer = AdamW(trainable_params, lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get Golden logits using base model (no corruption)
            model.eval()
            with torch.no_grad():
                golden_outputs = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                golden_logits = golden_outputs.logits.detach()
            
            # Forward pass with dual streams
            model.train()
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corr_input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # PrunableLlamaForCausalLM returns (CausalLMOutput, corrupted_output) or similar?
            # Wait, looking at ioi_llama_hybrid_adaptive.py, they do `outputs.logits` directly!
            # Let's check: outputs = circuit_model(...)
            # outputs.logits... 
            # If PrunableLlamaForCausalLM subclasses LlamaForCausalLM, it returns a CausalLMOutputWithPast.
            # But the forward signature might be different. Let's assume outputs.logits exists.
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple) and hasattr(outputs[0], "logits"):
                logits = outputs[0].logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
            # KL Loss
            # Generate mask for answer positions to match venn behavior
            seq_len = logits.shape[1]
            pos_mask = torch.zeros((logits.shape[0], seq_len), device=device)
            pos_mask[torch.arange(logits.shape[0]), batch["answer_positions"]] = 1.0
            
            kl = kl_divergence_loss(logits, golden_logits, pos_mask)
            
            # Margin Loss
            mrg = margin_loss(
                logits, batch["target"], batch["distractor"], batch["answer_positions"], margin=margin
            )
            
            # Sparsity Loss
            # PrunableLlamaForCausalLM usually has a get_sparsity_loss method or we can extract it.
            if hasattr(model, "get_sparsity_loss"):
                sparsity = model.get_sparsity_loss(step=1000)["total_sparsity"] * lambda_mult
            elif hasattr(model.model, "get_sparsity_loss"):
                sparsity = model.model.get_sparsity_loss(step=1000)["total_sparsity"] * lambda_mult
            else:
                sparsity = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Simple combined loss
            loss = kl + mrg + sparsity
            
            loss.backward()
            optimizer.step()
            
            # Clamping to avoid death by epsilon
            with torch.no_grad():
                for name, module in model.named_modules():
                    if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                        module.log_alpha.clamp_(-5.0, 5.0)
                        
        print(f"Epoch {epoch+1}/{epochs} | KL: {kl.item():.4f} | Margin: {mrg.item():.4f} | Sparsity: {sparsity.item():.4f}")

def extract_mask(model):
    """Extract >0.5 binary mask for all gates."""
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--task_a", type=str, default="fruits")
    parser.add_argument("--task_b", type=str, default="animals")
    parser.add_argument("--epochs_phase1", type=int, default=10)
    parser.add_argument("--epochs_phase2", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda_mult", type=float, default=0.01)
    
    # Pruning toggles
    parser.add_argument("--heads", action="store_true")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--attention_neurons", action="store_true")
    parser.add_argument("--blocks", action="store_true")
    
    args = parser.parse_args()
    
    # If nothing selected, default to all
    if not any([args.heads, args.mlp, args.attention_neurons, args.blocks]):
        args.heads = True
        args.mlp = True
        args.attention_neurons = True
        args.blocks = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading datasets...")
    ds_a = load_dataset(f"induction_datasets/category_chains/{args.task_a}.jsonl")
    ds_b = load_dataset(f"induction_datasets/category_chains/{args.task_b}.jsonl")
    
    joint_ds = ConcatDataset([ds_a, ds_b])
    
    dl_joint = DataLoader(joint_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    dl_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    print("Initializing Model...")
    config = PruningConfig(
        prune_attention_heads=args.heads,
        prune_attention_neurons=args.attention_neurons,
        prune_mlp_hidden=args.mlp,
        prune_mlp_output=args.mlp,
        prune_attention_blocks=args.blocks,
        prune_mlp_blocks=args.blocks,
        prune_full_layers=args.blocks,
    )
    
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model,
        pruning_config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    print(f"\n--- PHASE 1: JOINT TRAINING (A+B) ---")
    train_phase(model, dl_joint, args.epochs_phase1, args.lr, margin=4.0, target_kl=0.5, lambda_mult=args.lambda_mult, device=device)
    
    print("\nSaving joint weights...")
    joint_state = {k: v.clone() for k, v in model.state_dict().items() if 'log_alpha' in k}
    
    print(f"\n--- PHASE 2A: FINE-TUNE ON {args.task_a.upper()} ---")
    train_phase(model, dl_a, args.epochs_phase2, args.lr, margin=4.0, target_kl=0.5, lambda_mult=args.lambda_mult, device=device)
    mask_a = extract_mask(model)
    
    print(f"\n--- PHASE 2B: FINE-TUNE ON {args.task_b.upper()} ---")
    # Restore joint weights
    model.load_state_dict(joint_state, strict=False)
    train_phase(model, dl_b, args.epochs_phase2, args.lr, margin=4.0, target_kl=0.5, lambda_mult=args.lambda_mult, device=device)
    mask_b = extract_mask(model)
    
    print(f"\n--- PHASE 3: INTERSECTION ---")
    intersection = {}
    total_overlap = 0
    total_gates = 0
    for k in mask_a.keys():
        intersection[k] = mask_a[k] & mask_b[k]
        total_overlap += intersection[k].sum().item()
        total_gates += intersection[k].numel()
        
    print(f"Global Intersection Density: {total_overlap} / {total_gates} ({(total_overlap/total_gates)*100:.4f}%)")
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequential_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.task_a}_vs_{args.task_b}.pt")
    torch.save(intersection, out_path)
    print(f"Saved intersection to {out_path}")

if __name__ == "__main__":
    main()
