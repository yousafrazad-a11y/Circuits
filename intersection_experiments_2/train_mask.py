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
    parser = argparse.ArgumentParser(description="Train a pruning mask on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., fruits, all).")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train.")
    parser.add_argument("--output_name", type=str, required=True, help="Base name for the saved mask and checkpoint files.")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to a binary mask file to finetune from. Starts from the pure binary mask; gates that are off in the mask are frozen off for the whole run.")
    parser.add_argument("--lambda_sparsity", type=float, default=None, help="Sparsity lambda for attention-head gates. Default: keep the PruningConfig value (0.05).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    if args.dataset == "all":
        dataset_names = ["fruits", "animals", "colors", "metals", "vehicles"]
        mixed_data = []
        for name in dataset_names:
            path = f"intersection_experiments_2/datasets/{name}.jsonl"
            with open(path, 'r') as f:
                for line in f:
                    mixed_data.append(json.loads(line))
        random.seed(42)
        random.shuffle(mixed_data)
        ds = MemoryDataset(mixed_data)
    else:
        dataset_path = f"intersection_experiments_2/datasets/{args.dataset}.jsonl"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        ds = CategoryDataset(dataset_path)
    
    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, manager.tokenizer))
    
    manager.initialize_model()
    
    if args.lambda_sparsity is not None:
        manager.model.pruning_config.lambda_attention_heads = args.lambda_sparsity
        print(f"Sparsity lambda (attention heads) overridden to {args.lambda_sparsity}")
    
    if args.mask:
        print(f"Finetuning from binary mask {args.mask} (off-gates frozen)...")
        manager.load_masks_for_finetuning(args.mask)
    
    print(f"\n--- TRAINING MASKS ON {args.dataset.upper()} ({len(ds)} samples) ---")
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
