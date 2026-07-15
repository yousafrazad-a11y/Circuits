import os
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer
import argparse
import subprocess
import json

# Insert circuit_pruning-argo path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

# Import datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from induction_datasets.test_venn_induction import load_dataset

EPOCHS = 300
BATCH_SIZE = 4
LR = 0.05
LEVEL = "heads"
OUT_DIR = "/home/exouser/pruning/intersection_experiments/results_5way_extreme"

def get_pruning_config(level):
    if level == "heads":
        return PruningConfig(
            prune_attention_heads=True, lambda_attention_heads=0.8,
            prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
            prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
        )
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

def train_phase(model, dataloader, epochs, lr, device="cuda", compile_model=True):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "log_alpha" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float() 
            trainable_params += param.numel()
            
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_kl = 0
        epoch_task = 0
        epoch_sparsity = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size_curr = batch["clean_input_ids"].size(0)
            
            model.eval()
            with torch.no_grad():
                golden_outputs = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                golden_logits = golden_outputs.logits.detach()
            
            model.train()
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corr_input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            
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
            
            logit_good = logits[batch_indices, pos, batch["target_tokens"]].float()
            logit_bad = logits[batch_indices, pos, batch["distractor_tokens"]].float()
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()
            
            if hasattr(model, 'get_sparsity_loss'):
                sparsity_loss = model.get_sparsity_loss(step=1000)["total_sparsity"]
            else:
                sparsity_loss = model._orig_mod.get_sparsity_loss(step=1000)["total_sparsity"]
            
            loss = kl_loss * 1.5 + sparsity_loss + task_loss
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                for name, module in base_model.named_modules():
                    if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                        module.log_alpha.clamp_(-5.0, 5.0)
                        
            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_task += task_loss.item()
            epoch_sparsity += sparsity_loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.3f} | KL: {epoch_kl/len(dataloader):.3f} | Task: {epoch_task/len(dataloader):.3f} | Sparsity: {epoch_sparsity/len(dataloader):.3f}")
            
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
    parser.add_argument("--phase2_dataset", type=str, default=None, help="If set, run Phase 2 only for this dataset")
    parser.add_argument("--phase4_direct", action="store_true", help="If set, run direct 300 epochs on joint")
    args = parser.parse_args()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = get_pruning_config(LEVEL)
    
    # -------------------------------------------------------------------------
    # PHASE 2: INDEPENDENT SUB-PROCESS FINE-TUNING
    # -------------------------------------------------------------------------
    if args.phase2_dataset is not None:
        name = args.phase2_dataset
        print(f"\n{'='*60}\nPHASE 2: FINE-TUNE ON {name.upper()} (300 Epochs)\n{'='*60}")
        datasets = get_datasets()
        ds = datasets[name]
        
        model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
            "meta-llama/Llama-3.2-1B",
            pruning_config=config,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        joint_state = torch.load(os.path.join(OUT_DIR, "joint_checkpoint.pt"), weights_only=True)
        model.load_state_dict(joint_state, strict=False)
        
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        model = train_phase(model, dl, EPOCHS, LR, device=device)
        
        mask = extract_mask(model)
        torch.save(mask, os.path.join(OUT_DIR, f"{name}_extreme_circuit.pt"))
        print(f"[{name.upper()}] Finished Phase 2 and saved circuit.")
        return
        
    # -------------------------------------------------------------------------
    # PHASE 4: DIRECT 600-EPOCH METHOD ON JOINT
    # -------------------------------------------------------------------------
    if args.phase4_direct:
        print(f"\n{'='*60}\nPHASE 4: DIRECT METHOD ON JOINT DATASET (Remaining 300 Epochs)\n{'='*60}")
        datasets = get_datasets()
        joint_ds = ConcatDataset(list(datasets.values()))
        
        model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
            "meta-llama/Llama-3.2-1B",
            pruning_config=config,
            torch_dtype=torch.bfloat16,
        ).to(device)
        
        joint_state = torch.load(os.path.join(OUT_DIR, "joint_checkpoint.pt"), weights_only=True)
        model.load_state_dict(joint_state, strict=False)
        
        dl_joint = DataLoader(joint_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
        model = train_phase(model, dl_joint, EPOCHS, LR, device=device)
        
        mask = extract_mask(model)
        torch.save(mask, os.path.join(OUT_DIR, "JOINT_600_EXTREME_CIRCUIT.pt"))
        print(f"[DIRECT METHOD] Finished Phase 4 and saved JOINT_600_EXTREME_CIRCUIT.pt.")
        return

    # -------------------------------------------------------------------------
    # ORCHESTRATOR
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\nEXTREME 5-WAY RUN (300 EPOCHS)\n{'='*60}")
    datasets = get_datasets()
    
    print(f"\n{'='*60}\nPHASE 1: JOINT TRAINING ON ALL 5 DATASETS\n{'='*60}")
    joint_ds = ConcatDataset(list(datasets.values()))
    dl_joint = DataLoader(joint_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning(
        "meta-llama/Llama-3.2-1B",
        pruning_config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    model = train_phase(model, dl_joint, EPOCHS, LR, device=device)
    
    joint_state = {k: v.clone() for k, v in model.state_dict().items() if 'log_alpha' in k}
    torch.save(joint_state, os.path.join(OUT_DIR, "joint_checkpoint.pt"))
    print("Phase 1 completed. Checkpoint saved.")
    
    del model
    torch.cuda.empty_cache()
    
    print("\nSpawning 5 parallel Phase 2 workers (Intersection Method)...")
    processes = []
    log_files = []
    for name in datasets.keys():
        log_f = open(os.path.join(OUT_DIR, f"log_phase2_{name}.txt"), "w")
        cmd = [sys.executable, __file__, "--phase2_dataset", name]
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append(p)
        log_files.append(log_f)
        print(f"Launched {name} (PID: {p.pid})")
        
    print("Waiting for 5-Way Intersection workers to finish...")
    for p in processes:
        p.wait()
        
    for log_f in log_files:
        log_f.close()
        
    print("\nIntersection Method Complete! Collecting Extreme Masks...")
    final_masks = {}
    for name in datasets.keys():
        path = os.path.join(OUT_DIR, f"{name}_extreme_circuit.pt")
        final_masks[name] = torch.load(path, weights_only=True)
        
    print(f"\n{'='*60}\nPHASE 3: 5-WAY EXTREME GLOBAL INTERSECTION\n{'='*60}")
    global_circuit = {}
    first_name = list(final_masks.keys())[0]
    for k in final_masks[first_name].keys():
        global_circuit[k] = final_masks[first_name][k].clone()
        
    for name in list(final_masks.keys())[1:]:
        for k in global_circuit.keys():
            global_circuit[k] = global_circuit[k] & final_masks[name][k]
            
    global_size = sum(global_circuit[k].sum().item() for k in global_circuit.keys())
    print(f"The EXTREME 5-Way Universal Circuit contains {global_size} active components.")
    
    torch.save(global_circuit, os.path.join(OUT_DIR, "5WAY_EXTREME_GLOBAL_CIRCUIT.pt"))
    print(f"Saved EXTREME global 5-way circuit.")
    
    print("\nSpawning final Direct Method worker (Joint Dataset for +300 Epochs)...")
    log_f = open(os.path.join(OUT_DIR, "log_phase4_direct.txt"), "w")
    cmd = [sys.executable, __file__, "--phase4_direct"]
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    print(f"Launched Direct Method Phase 4 (PID: {p.pid})")
    
    print("Waiting for Direct Method to finish...")
    p.wait()
    log_f.close()
    
    print("\nAll Extreme 5-way processing completed perfectly.")

if __name__ == "__main__":
    main()
