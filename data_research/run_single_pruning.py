import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Fix import path for circuit_pruning-argo
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

# Import existing datasets and losses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from induction_datasets.test_venn_induction import load_dataset
from venn_circuit_discovery_v2.loss import kl_divergence_loss, margin_loss

def collate_fn(batch, tokenizer):
    clean_input_ids = []
    corr_input_ids = []
    target_tokens = []
    distractor_tokens = []
    answer_positions = []
    
    for item in batch:
        prompt = item["clean_prompt"] if "clean_prompt" in item else item["prompt"]
        corr_prompt = item["corr_prompt"] if "corr_prompt" in item else item["corrupted_prompt"]
        clean_input_ids.append(torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)))
        corr_input_ids.append(torch.tensor(tokenizer.encode(corr_prompt, add_special_tokens=True)))
        target_str = item["target"] if "target" in item else item["clean_target"]
        dist_str = item["distractor"] if "distractor" in item else item["corrupted_target"]
        target_tokens.append(tokenizer.encode(target_str, add_special_tokens=False)[0])
        distractor_tokens.append(tokenizer.encode(dist_str, add_special_tokens=False)[0])
    
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--task", type=str, default="induction_datasets/category_chains/fruits.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda_mult", type=float, default=0.01)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading dataset from {args.task}...")
    ds = load_dataset(args.task)
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    print("Initializing Default Prunable Model (All Masks Enabled)...")
    config = PruningConfig(
        prune_attention_heads=True,
        prune_attention_neurons=True,
        prune_mlp_hidden=True,
        prune_mlp_output=True,
        prune_attention_blocks=True,
        prune_mlp_blocks=True,
        prune_full_layers=True,
    )
    
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        args.model,
        pruning_config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # Freeze all base model parameters
    for name, param in model.named_parameters():
        if "log_alpha" not in name:
            param.requires_grad = False
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    optimizer = AdamW(trainable_params, lr=args.lr)
    
    print(f"\nStarting Normal Circuit Pruning on {args.task}...")
    
    for epoch in range(args.epochs):
        model.train()
        total_kl = 0
        total_margin = 0
        total_acc = 0
        total_golden_acc = 0
        total_sparsity = 0
        batches = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 1. Evaluate baseline Golden logic
            model.eval()
            with torch.no_grad():
                golden_outputs = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                golden_logits = golden_outputs.logits.detach()
                
            ans_pos = batch["answer_positions"]
            b_indices = torch.arange(golden_logits.shape[0])
            gt_logits = golden_logits[b_indices, ans_pos, batch["target"]]
            gd_logits = golden_logits[b_indices, ans_pos, batch["distractor"]]
            golden_acc = (gt_logits > gd_logits).float().mean()
            
            # 2. Forward pass with corrupted stream
            model.train()
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corr_input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            elif isinstance(outputs, tuple) and hasattr(outputs[0], "logits"):
                logits = outputs[0].logits
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
            # 3. KL Loss vs Golden
            seq_len = logits.shape[1]
            pos_mask = torch.zeros((logits.shape[0], seq_len), device=device)
            pos_mask[torch.arange(logits.shape[0]), batch["answer_positions"]] = 1.0
            
            kl = kl_divergence_loss(logits, golden_logits, pos_mask)
            
            # 4. Margin Loss (Task loss proxy)
            mrg = margin_loss(
                logits, batch["target"], batch["distractor"], batch["answer_positions"], margin=4.0
            )
            
            # Exact Top-1 Accuracy over batch (did it predict target over distractor?)
            ans_pos = batch["answer_positions"]
            b_indices = torch.arange(logits.shape[0])
            t_logits = logits[b_indices, ans_pos, batch["target"]]
            d_logits = logits[b_indices, ans_pos, batch["distractor"]]
            acc = (t_logits > d_logits).float().mean()
            
            # 5. Sparsity Loss
            sparsity_info = model.get_sparsity_loss(step=1000)
            sparsity = sparsity_info["total_sparsity"] * args.lambda_mult
            
            loss = kl + mrg + sparsity
            loss.backward()
            optimizer.step()
            
            # Gate clipping
            with torch.no_grad():
                for name, module in model.named_modules():
                    if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                        module.log_alpha.clamp_(-5.0, 5.0)
                        
            total_kl += kl.item()
            total_margin += mrg.item()
            total_acc += acc.item()
            total_golden_acc += golden_acc.item()
            total_sparsity += sparsity_info["total_sparsity"].item()
            batches += 1
            
        print(f"Epoch {epoch+1:02d} | KL vs Base: {total_kl/batches:.4f} | Margin: {total_margin/batches:.4f} | Golden Acc: {total_golden_acc/batches*100:.1f}% | Pruned Acc: {total_acc/batches*100:.1f}% | Mask Sparsity: {total_sparsity/batches:.4f}")

if __name__ == "__main__":
    main()
