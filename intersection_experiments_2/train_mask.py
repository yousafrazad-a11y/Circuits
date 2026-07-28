import os
import sys
import json
import torch
import argparse
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager

class CategoryDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class MemoryDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    tokenizer.padding_side = 'left'
    clean_texts = [item['clean_prompt'] for item in batch]
    corr_texts = [item['corr_prompt'] for item in batch]
    
    clean_encoded = tokenizer(clean_texts, padding=True, return_tensors='pt', add_special_tokens=True)
    corr_encoded = tokenizer(corr_texts, padding=True, return_tensors='pt', add_special_tokens=True)
    
    max_len = max(clean_encoded['input_ids'].size(1), corr_encoded['input_ids'].size(1))
    
    def pad_left(tensor, pad_val, target_len):
        pad_len = target_len - tensor.size(1)
        if pad_len > 0:
            pads = torch.full((tensor.size(0), pad_len), pad_val, dtype=tensor.dtype)
            return torch.cat([pads, tensor], dim=1)
        return tensor
        
    input_ids = pad_left(clean_encoded['input_ids'], tokenizer.pad_token_id, max_len)
    attention_mask = pad_left(clean_encoded['attention_mask'], 0, max_len)
    corrupted_input_ids = pad_left(corr_encoded['input_ids'], tokenizer.pad_token_id, max_len)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'corrupted_input_ids': corrupted_input_ids,
    }

def main():
    parser = argparse.ArgumentParser(description="Train a pruning mask on one or more datasets.")
    parser.add_argument("--dataset", type=str, nargs='+', required=True,
                        help="One or more dataset names (resolved to intersection_experiments_2/datasets/<name>.jsonl) "
                             "or direct .jsonl paths. Multiple datasets are mixed and shuffled (seed 42).")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("--output_name", type=str, required=True, help="Base name for the saved mask and checkpoint files.")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to a binary mask file to finetune from. Starts from the pure binary mask; gates that are off in the mask are frozen off for the whole run.")
    parser.add_argument("--lambda_sparsity", type=float, default=None, help="Global sparsity lambda: sets the attention-head lambda to this value and scales every other granularity level's lambda by the same factor (default: keep PruningConfig values, heads=0.05).")
    parser.add_argument("--prune_mlp_blocks", action="store_true", help="Also prune whole MLP blocks (one scalar gate per MLP per layer). Default: attention heads only.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset(s): names resolve to intersection_experiments_2/datasets/<name>.jsonl,
    # or pass direct .jsonl paths. Multiple datasets are mixed and shuffled (seed 42).
    mixed_data = []
    for name in args.dataset:
        path = name if name.endswith(".jsonl") else f"intersection_experiments_2/datasets/{name}.jsonl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")
        with open(path, 'r') as f:
            for line in f:
                mixed_data.append(json.loads(line))
    random.seed(42)
    random.shuffle(mixed_data)
    ds = MemoryDataset(mixed_data)
    
    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, manager.tokenizer))
    
    config = manager._get_default_config()
    if args.prune_mlp_blocks:
        config.prune_mlp_blocks = True
        print("MLP block pruning ENABLED (heads + whole MLP blocks).")
    manager.initialize_model(config)
    
    if args.lambda_sparsity is not None:
        manager.set_global_sparsity_lambda(args.lambda_sparsity)
    
    if args.mask:
        print(f"Finetuning from binary mask {args.mask} (off-gates frozen)...")
        manager.load_masks_for_finetuning(args.mask)
    
    print(f"\n--- TRAINING MASKS ON {', '.join(args.dataset).upper()} ({len(ds)} samples) ---")
    manager.train_masks(dl, epochs=args.epochs)
    
    # Save mask and checkpoint
    os.makedirs("intersection_experiments_2/masks", exist_ok=True)
    mask_path = f"intersection_experiments_2/masks/{args.output_name}_mask.pt"
    ckpt_path = f"intersection_experiments_2/masks/{args.output_name}_checkpoint.pt"
    
    manager.save_masks(mask_path)
    manager.save_checkpoint(ckpt_path)
    print(f"Done! Final mask saved to {mask_path}")
    print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
