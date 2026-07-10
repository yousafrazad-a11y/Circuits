import os
import json
import torch
import itertools
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venn_circuit_discovery_v2 import VennCircuitDiscoverer, VennBatch, VennHyperparameters

os.environ["HF_TOKEN"] = ""
MODEL_NAME = "meta-llama/Llama-3.2-1B"

def load_dataset(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def collate_fn(batch_a, batch_b, tokenizer):
    clean_a = [item["clean_prompt"] for item in batch_a]
    corr_a = [item["corr_prompt"] for item in batch_a]
    clean_b = [item["clean_prompt"] for item in batch_b]
    corr_b = [item["corr_prompt"] for item in batch_b]
    
    def pad_to_max(enc, max_len):
        if enc.input_ids.shape[1] < max_len:
            pad_shape = (enc.input_ids.shape[0], max_len - enc.input_ids.shape[1])
            pad_tensor = torch.full(pad_shape, tokenizer.pad_token_id, dtype=enc.input_ids.dtype)
            input_ids = torch.cat([pad_tensor, enc.input_ids], dim=1)
            mask_pad = torch.zeros(pad_shape, dtype=enc.attention_mask.dtype)
            attention_mask = torch.cat([mask_pad, enc.attention_mask], dim=1)
            return input_ids, attention_mask
        return enc.input_ids, enc.attention_mask

    enc_clean_a = tokenizer(clean_a, return_tensors="pt", padding=True)
    enc_corr_a = tokenizer(corr_a, return_tensors="pt", padding=True)
    max_a = max(enc_clean_a.input_ids.shape[1], enc_corr_a.input_ids.shape[1])
    clean_a_ids, clean_a_mask = pad_to_max(enc_clean_a, max_a)
    corr_a_ids, _ = pad_to_max(enc_corr_a, max_a)

    enc_clean_b = tokenizer(clean_b, return_tensors="pt", padding=True)
    enc_corr_b = tokenizer(corr_b, return_tensors="pt", padding=True)
    max_b = max(enc_clean_b.input_ids.shape[1], enc_corr_b.input_ids.shape[1])
    clean_b_ids, clean_b_mask = pad_to_max(enc_clean_b, max_b)
    corr_b_ids, _ = pad_to_max(enc_corr_b, max_b)

    def get_token_id(word):
        return tokenizer.encode(" " + word, add_special_tokens=False)[-1]

    target_a = torch.tensor([get_token_id(item["target"]) for item in batch_a])
    distractor_a = torch.tensor([get_token_id(item["distractor"]) for item in batch_a])
    target_b = torch.tensor([get_token_id(item["target"]) for item in batch_b])
    distractor_b = torch.tensor([get_token_id(item["distractor"]) for item in batch_b])
    
    return VennBatch(
        clean_a_input_ids=clean_a_ids,
        clean_b_input_ids=clean_b_ids,
        corr_a_input_ids=corr_a_ids,
        corr_b_input_ids=corr_b_ids,
        answer_positions_a=torch.tensor([clean_a_ids.shape[1] - 1] * len(batch_a)),
        answer_positions_b=torch.tensor([clean_b_ids.shape[1] - 1] * len(batch_b)),
        target_a=target_a,
        distractor_a=distractor_a,
        target_b=target_b,
        distractor_b=distractor_b,
        attention_mask_a=clean_a_mask,
        attention_mask_b=clean_b_mask
    )

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, ds_a, ds_b):
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.length = min(len(ds_a), len(ds_b))
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        return self.ds_a[idx], self.ds_b[idx]

def paired_collate(batch, tokenizer):
    batch_a = [item[0] for item in batch]
    batch_b = [item[1] for item in batch]
    return collate_fn(batch_a, batch_b, tokenizer)

def flatten_mask(circuit_dict, key="core"):
    """Flatten all core boolean masks from all layers into a single 1D tensor for easy comparison"""
    flat = []
    # sort keys to guarantee same order
    for name in sorted(circuit_dict.keys()):
        if "layer" in name:
            flat.append(circuit_dict[name][key].flatten())
    return torch.cat(flat)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    data_dir = Path("/home/exouser/pruning/induction_datasets/category_chains")
    categories = ["fruits", "animals", "colors", "metals", "vehicles"]
    
    datasets = {}
    for cat in categories:
        datasets[cat] = load_dataset(data_dir / f"{cat}.jsonl")
        
    pairs = list(itertools.combinations(categories, 2))
    
    hp = VennHyperparameters(
        gate_lr=0.15,  
        target_kl_a=3.0, 
        target_kl_b=3.0,
        init_lambda=5.0,
        max_lambda=200.0,
        pid_kp=2.0
    )
    
    results = {}
    saved_cores = {}
    
    for idx, (cat_a, cat_b) in enumerate(pairs):
        print(f"\n=======================================================")
        print(f"[{idx+1}/{len(pairs)}] Intersecting '{cat_a}' and '{cat_b}'")
        print(f"=======================================================\n")
        
        ds_a = datasets[cat_a]
        ds_b = datasets[cat_b]
        
        paired_ds = PairedDataset(ds_a, ds_b)
        dl = DataLoader(paired_ds, batch_size=8, collate_fn=lambda b: paired_collate(b, tokenizer), shuffle=True)

        discoverer = VennCircuitDiscoverer(model_name=MODEL_NAME, mode="intersection", hyperparameters=hp)
        
        # Train for 5 epochs to push sparsity down heavily
        discoverer.fit(dl, epochs=5)
        
        circuit = discoverer.extract_circuit()
        summary = discoverer.circuit_summary()
        
        # Convert dictionary to a single flat boolean mask representing the 'core' circuit
        core_mask = flatten_mask(circuit, key="core")
        saved_cores[f"{cat_a}_vs_{cat_b}"] = core_mask
        
        total_intersection = core_mask.sum().item()
        total_gates = core_mask.shape[0]
        density_pct = (total_intersection / total_gates) * 100
        
        print(f"\n=> Pair {cat_a} & {cat_b} Core Density: {density_pct:.2f}% ({total_intersection} / {total_gates} gates)")
        
        results[f"{cat_a}_vs_{cat_b}"] = {
            "core_density_pct": float(density_pct),
            "core_gates": int(total_intersection)
        }
        
        del discoverer
        torch.cuda.empty_cache()
        
    print("\n\n#######################################################")
    print("PAIRWISE CIRCUIT SIMILARITY (IoU BETWEEN PAIRS)")
    print("#######################################################")
    
    pair_names = list(saved_cores.keys())
    cross_results = []
    for i in range(len(pair_names)):
        for j in range(i+1, len(pair_names)):
            name1 = pair_names[i]
            name2 = pair_names[j]
            mask1 = saved_cores[name1]
            mask2 = saved_cores[name2]
            
            intersection = (mask1 & mask2).sum().item()
            union = (mask1 | mask2).sum().item()
            iou = intersection / union if union > 0 else 0
            
            cross_results.append({
                "pair1": name1,
                "pair2": name2,
                "iou": iou
            })
            
    # Calculate global intersection
    global_mask = saved_cores[pair_names[0]]
    for name in pair_names[1:]:
        global_mask = global_mask & saved_cores[name]
    global_active = global_mask.sum().item()
    
    # Save the huge results to json
    out_dict = {
        "individual_pair_results": results,
        "cross_pair_similarities": cross_results,
        "global_intersection_gates": int(global_active),
        "total_gates": int(saved_cores[pair_names[0]].shape[0])
    }
    
    out_file = Path("/home/exouser/pruning/data_research/deep_universal_circuit_results.json")
    with open(out_file, "w") as f:
        json.dump(out_dict, f, indent=4)
        
    print(f"\nGlobal Intersection (Active in ALL 10 pairs): {global_active} gates")
    print(f"Saved highly detailed results to {out_file}")

if __name__ == "__main__":
    main()
