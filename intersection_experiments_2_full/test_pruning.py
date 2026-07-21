import os
import sys
import json
import torch
import re
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

def collate_fn(batch, tokenizer):
    tokenizer.padding_side = 'left'
    clean_texts = [item['clean_prompt'] for item in batch]
    corr_texts = [item['corr_prompt'] for item in batch]
    targets = [item['target'] for item in batch]
    
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
        'targets': targets
    }

def evaluate_accuracies(model, dataloader, tokenizer, category_set, device="cuda"):
    model.eval()
    
    cat_tokens = {}
    for word in category_set:
        # We must compare the FIRST token of the word, because the model is predicting the next immediate token.
        tok_id = tokenizer.encode(" " + word, add_special_tokens=False)[0]
        cat_tokens[word] = tok_id
        
    tok_to_word = {v: k for k, v in cat_tokens.items()}
    all_cat_toks = list(cat_tokens.values())
    
    prob_correct = 0
    gen_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            corrupted_input_ids = batch['corrupted_input_ids'].to(device)
            targets = batch['targets']
            
            # Probability accuracy: forward pass
            if hasattr(model, 'set_final_circuit_mode'):
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    corrupted_input_ids=corrupted_input_ids,
                    use_cache=False
                )
            else:
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    use_cache=False
                )
            
            # Since padding is on the left, the last token is at index -1 for all batches
            last_logits = outputs.logits[:, -1, :]
            
            cat_logits = last_logits[:, all_cat_toks]
            best_idx = torch.argmax(cat_logits, dim=-1)
            
            for i in range(len(targets)):
                pred_tok = all_cat_toks[best_idx[i].item()]
                pred_word = tok_to_word[pred_tok]
                if pred_word.lower() == targets[i].lower():
                    prob_correct += 1
                    
            # Generation Accuracy
            gen_ids = []
            curr_input = input_ids
            curr_mask = attention_mask
            if hasattr(model, 'set_final_circuit_mode'):
                curr_corr = corrupted_input_ids
            else:
                curr_corr = None
                
            batch_gen_tokens = [[] for _ in range(input_ids.size(0))]
            for _ in range(2):
                if curr_corr is not None:
                    out = model(input_ids=curr_input, attention_mask=curr_mask, corrupted_input_ids=curr_corr, use_cache=False)
                else:
                    out = model(input_ids=curr_input, attention_mask=curr_mask, use_cache=False)
                
                next_toks = torch.argmax(out.logits[:, -1, :], dim=-1)
                for i in range(input_ids.size(0)):
                    batch_gen_tokens[i].append(next_toks[i].item())
                    
                curr_input = torch.cat([curr_input, next_toks.unsqueeze(-1)], dim=-1)
                curr_mask = torch.cat([curr_mask, torch.ones((input_ids.size(0), 1), device=device)], dim=-1)
                if curr_corr is not None:
                    curr_corr = torch.cat([curr_corr, torch.full((input_ids.size(0), 1), tokenizer.pad_token_id, device=device)], dim=-1)
            
            for i in range(len(targets)):
                text = tokenizer.decode(batch_gen_tokens[i], skip_special_tokens=True).strip()
                # strip first word intelligently
                first_word = re.sub(r'[^a-zA-Z]', '', text.split()[0] if text else "").lower()
                if first_word == targets[i].lower():
                    gen_correct += 1
                    
            total += len(targets)
            
    return prob_correct / total, gen_correct / total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("intersection_experiments_2_full/categories.json", "r") as f:
        categories = json.load(f)
    fruits_set = categories["fruits"]
    
    ds = CategoryDataset("intersection_experiments_2_full/datasets/fruits.jsonl")
    
    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=lambda b: collate_fn(b, manager.tokenizer))
    
    manager.initialize_model()
    
    print("\n--- BASELINE MODEL EVALUATION ---")
    base_prob, base_gen = evaluate_accuracies(manager.baseline_model, dl, manager.tokenizer, fruits_set, device)
    print(f"Base Probability Accuracy: {base_prob:.4f}")
    print(f"Base Generation Accuracy: {base_gen:.4f}")
    
    print("\n--- TRAINING MASKS ---")
    # Using 300 epochs to demonstrate significant pruning and accuracy drop
    manager.train_masks(dl, epochs=0)
    
    print("\n--- CIRCUIT MODEL EVALUATION ---")
    manager.use_model(enable_masks=True)
    circ_prob, circ_gen = evaluate_accuracies(manager.model, dl, manager.tokenizer, fruits_set, device)
    print(f"Circuit Probability Accuracy: {circ_prob:.4f}")
    print(f"Circuit Generation Accuracy: {circ_gen:.4f}")
    
    manager.evaluate_kl_divergence(dl)
    
    # Save test mask
    manager.save_masks("intersection_experiments_2_full/masks/test_fruits_mask.pt")

if __name__ == "__main__":
    main()
