import os
import sys
import torch
import subprocess
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig


OUT_DIR = "/home/exouser/pruning/intersection_experiments/results_5way_pure"

BATCH_SIZE = 4
LR = 0.1
EPOCHS = 300 # 300 + 300 = 600 total
LEVEL = "head"

def get_pruning_config(level):
    return PruningConfig(
        prune_attention_heads=True, lambda_attention_heads=0.8,
        prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
        prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
    )

def load_dataset(path):
    import json
    class DummyDS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]
        
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return DummyDS(data)

def collate_fn(batch, tokenizer):
    clean_texts = [b["clean_prompt"] + " " + b["target"] for b in batch]
    corr_texts = [(b["corr_prompt"] if "corr_prompt" in b else b.get("corrupted_prompt", "")) + " " + b["target"] for b in batch]
    
    clean_toks = tokenizer(clean_texts)
    corr_toks = tokenizer(corr_texts)
    
    clean_input_ids = [torch.tensor(t) for t in clean_toks["input_ids"]]
    corr_input_ids = [torch.tensor(t) for t in corr_toks["input_ids"]]
    
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
        
    return {
        "clean_input_ids": torch.stack(clean_padded).long(),
        "corr_input_ids": torch.stack(corr_padded).long(),
        "attention_mask": torch.stack(attn_mask).long()
    }

def train_phase(model, dataloader, epochs, lr, device="cuda"):
    model.train()
    optimizer = torch.optim.Adam(
        [p for n, p in model.named_parameters() if hasattr(p, 'is_gate') or 'log_alpha' in n],
        lr=lr
    )
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_kl = 0
        epoch_task = 0
        epoch_sparsity = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            clean_ids = batch["clean_input_ids"].to(device)
            corr_ids = batch["corr_input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=clean_ids,
                corrupted_input_ids=corr_ids,
                attention_mask=attn_mask
            )
            
            logits = outputs.logits
            
            target_ids = clean_ids[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            task_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), target_ids.view(-1))
            
            sparsity_dict = model.get_sparsity_loss()
            sparsity_loss = sum(sparsity_dict.values())
            
            loss = task_loss + sparsity_loss
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                for name, module in model.named_modules():
                    if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                        module.log_alpha.clamp_(-5.0, 5.0)
                        
            epoch_loss += loss.item()
            epoch_task += task_loss.item()
            epoch_sparsity += sparsity_loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.3f} | Task: {epoch_task/len(dataloader):.3f} | Sparsity: {epoch_sparsity/len(dataloader):.3f}")
            
    return model._orig_mod if hasattr(model, '_orig_mod') else model

def extract_mask(model):
    model.eval()
    mask = {}
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                s = torch.sigmoid(module.log_alpha)
                s_stretched = s * 1.2 - 0.1
                mask[name] = (s_stretched > 0.5).bool().cpu()
    return mask

def get_datasets():
    data_dir = "/home/exouser/pruning/induction_datasets/category_chains"
    dataset_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    dataset_files.sort()
    
    datasets = {}
    for f in dataset_files:
        name = os.path.basename(f).split('.')[0]
        datasets[name] = load_dataset(f)
    return datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="If set, run training for this dataset only")
    args = parser.parse_args()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = get_pruning_config(LEVEL)
    
    if args.dataset is not None:
        name = args.dataset
        print(f"\n{'='*60}\nTRAINING {name.upper()} FROM SCRATCH (600 Epochs)\n{'='*60}")
        datasets = get_datasets()
        ds = datasets[name]
        
        model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
            "meta-llama/Llama-3.2-1B",
            pruning_config=config,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        
        print(f"[{name.upper()}] Starting first 300 epochs...")
        model = train_phase(model, dl, EPOCHS, LR, device=device)
        mask = extract_mask(model)
        torch.save(mask, os.path.join(OUT_DIR, f"{name}_pure_300.pt"))
        print(f"[{name.upper()}] Saved 300-epoch circuit.")
        
        print(f"[{name.upper()}] Starting next 300 epochs...")
        model = train_phase(model, dl, EPOCHS, LR, device=device)
        mask = extract_mask(model)
        torch.save(mask, os.path.join(OUT_DIR, f"{name}_pure_600.pt"))
        print(f"[{name.upper()}] Saved 600-epoch circuit.")
        return
        
    print(f"\n{'='*60}\nORCHESTRATING PURE INDIVIDUAL TRAINING (300 & 600 EPOCHS)\n{'='*60}")
    
    datasets = get_datasets()
    procs = []
    
    # 1. Run all 5 individually in parallel
    for name in datasets.keys():
        cmd = [sys.executable, __file__, "--dataset", name]
        p = subprocess.Popen(cmd)
        procs.append(p)
        
    for p in procs:
        p.wait()
        
    print("\nAll independent trainings completed.")
    
    # 2. Intersect the 300-epoch circuits
    print("Intersecting 300-epoch circuits...")
    intersect_mask_300 = None
    for name in datasets.keys():
        m = torch.load(os.path.join(OUT_DIR, f"{name}_pure_300.pt"), weights_only=True)
        if intersect_mask_300 is None:
            intersect_mask_300 = {k: v.clone() for k, v in m.items()}
        else:
            for k in intersect_mask_300:
                intersect_mask_300[k] = intersect_mask_300[k] & m[k]
                
    torch.save(intersect_mask_300, os.path.join(OUT_DIR, "INTERSECT_PURE_300.pt"))
    print(f"Saved INTERSECT_PURE_300.pt (Heads: {sum(v.sum().item() for v in intersect_mask_300.values())})")
    
    # 3. Intersect the 600-epoch circuits
    print("Intersecting 600-epoch circuits...")
    intersect_mask_600 = None
    for name in datasets.keys():
        m = torch.load(os.path.join(OUT_DIR, f"{name}_pure_600.pt"), weights_only=True)
        if intersect_mask_600 is None:
            intersect_mask_600 = {k: v.clone() for k, v in m.items()}
        else:
            for k in intersect_mask_600:
                intersect_mask_600[k] = intersect_mask_600[k] & m[k]
                
    torch.save(intersect_mask_600, os.path.join(OUT_DIR, "INTERSECT_PURE_600.pt"))
    print(f"Saved INTERSECT_PURE_600.pt (Heads: {sum(v.sum().item() for v in intersect_mask_600.values())})")
    
    print("\nAll Pure Training completed successfully!")

if __name__ == "__main__":
    main()
