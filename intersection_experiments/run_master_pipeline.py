import os
import sys
import json
import torch
import argparse
import subprocess
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from core_trainer import collate_fn, train_phase, extract_mask

class DummyDS(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataset(name):
    data_dir = "/home/exouser/pruning/induction_datasets/category_chains"
    data = []
    with open(f"{data_dir}/{name}.jsonl", 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return DummyDS(data)

def get_combined_dataset():
    data_dir = "/home/exouser/pruning/induction_datasets/category_chains"
    data = []
    for name in ["fruits", "animals", "colors", "metals", "vehicles"]:
        with open(f"{data_dir}/{name}.jsonl", 'r') as f:
            for line in f:
                data.append(json.loads(line))
    return DummyDS(data)

def get_pruning_config():
    return PruningConfig(
        prune_attention_heads=True, lambda_attention_heads=0.8,
        prune_attention_blocks=False, prune_mlp_blocks=False, prune_full_layers=False,
        prune_attention_neurons=False, prune_mlp_hidden=False, prune_mlp_output=False
    )

def worker_pure(dataset_name, out_dir):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = get_pruning_config()
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    
    ds = get_dataset(dataset_name)
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    print(f"[{dataset_name.upper()}] Starting Phase 1 (300 epochs)...")
    model = train_phase(model, dataloader, epochs=300, lr=0.05, device=device)
    torch.save(extract_mask(model), os.path.join(out_dir, f"{dataset_name}_pure_300.pt"))
    
    print(f"[{dataset_name.upper()}] Starting Phase 2 (300-600 epochs)...")
    model = train_phase(model, dataloader, epochs=300, lr=0.03, device=device)
    torch.save(extract_mask(model), os.path.join(out_dir, f"{dataset_name}_pure_600.pt"))

def worker_joint(out_dir):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = get_pruning_config()
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    
    ds = get_combined_dataset()
    dataloader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    print(f"[JOINT] Starting Phase 1 (300 epochs)...")
    model = train_phase(model, dataloader, epochs=300, lr=0.03, device=device)
    
    os.makedirs(out_dir, exist_ok=True)
    # Save the full model parameters for extreme fine-tuning
    torch.save(model.state_dict(), os.path.join(out_dir, "global_checkpoint_300.pth"))
    # Save the mask for evaluation
    torch.save(extract_mask(model), os.path.join(out_dir, "global_circuit_300.pt"))
    
    print(f"[JOINT] Starting Phase 2 (300-600 epochs)...")
    model = train_phase(model, dataloader, epochs=300, lr=0.03, device=device)
    torch.save(extract_mask(model), os.path.join(out_dir, "global_circuit_600.pt"))

def worker_extreme(dataset_name, out_dir):
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = get_pruning_config()
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    
    # Load the joint checkpoint
    checkpoint = torch.load(os.path.join(out_dir, "global_checkpoint_300.pth"), weights_only=True)
    model.load_state_dict(checkpoint)
    
    ds = get_dataset(dataset_name)
    dataloader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    
    print(f"[{dataset_name.upper()}] Starting EXTREME Fine-tuning (300 epochs)...")
    model = train_phase(model, dataloader, epochs=300, lr=0.03, device=device)
    torch.save(extract_mask(model), os.path.join(out_dir, f"{dataset_name}_extreme_circuit.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=str, choices=["pure", "joint", "extreme"])
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    
    OUT_DIR = "/home/exouser/pruning/intersection_experiments/results_master"
    
    if args.worker:
        os.makedirs(OUT_DIR, exist_ok=True)
        if args.worker == "pure":
            worker_pure(args.dataset, OUT_DIR)
        elif args.worker == "joint":
            worker_joint(OUT_DIR)
        elif args.worker == "extreme":
            worker_extreme(args.dataset, OUT_DIR)
        return

    print("============================================================")
    print("STARTING MASTER ORCHESTRATION PIPELINE")
    print("============================================================")
    
    datasets = ["fruits", "animals", "colors", "metals", "vehicles"]
    
    print("\n>>> PHASE 1: PURE INDEPENDENT TRAINING (Sequential) <<<")
    for ds in datasets:
        p = subprocess.Popen([sys.executable, __file__, "--worker", "pure", "--dataset", ds])
        p.wait()
        
    print("\n>>> PHASE 2: JOINT GLOBAL TRAINING (1 process) <<<")
    p = subprocess.Popen([sys.executable, __file__, "--worker", "joint"])
    p.wait()
    
    print("\n>>> PHASE 3: EXTREME FINE-TUNING (Sequential) <<<")
    for ds in datasets:
        p = subprocess.Popen([sys.executable, __file__, "--worker", "extreme", "--dataset", ds])
        p.wait()
        
    print("\n============================================================")
    print("MASTER ORCHESTRATION PIPELINE COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()
