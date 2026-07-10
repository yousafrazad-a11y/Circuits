import os
import json
import torch
import itertools
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venn_circuit_discovery_v2 import VennCircuitDiscoverer, VennBatch

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

    # Encode A
    enc_clean_a = tokenizer(clean_a, return_tensors="pt", padding=True)
    enc_corr_a = tokenizer(corr_a, return_tensors="pt", padding=True)
    max_a = max(enc_clean_a.input_ids.shape[1], enc_corr_a.input_ids.shape[1])
    clean_a_ids, clean_a_mask = pad_to_max(enc_clean_a, max_a)
    corr_a_ids, _ = pad_to_max(enc_corr_a, max_a)

    # Encode B
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

    # Since we left pad, the answer position is the last token (which is the index len - 1)
    # Actually, answer_position is just the length of the non-padded tokens - 1 ? No, left padding means 
    # the real tokens are shifted to the right, so the last token is always at index `seq_len - 1`.
    answer_positions_a = clean_a_mask.sum(dim=1) - 1 + (clean_a_ids.shape[1] - clean_a_mask.sum(dim=1))
    answer_positions_b = clean_b_mask.sum(dim=1) - 1 + (clean_b_ids.shape[1] - clean_b_mask.sum(dim=1))
    # Simplify:
    answer_positions_a = torch.tensor([clean_a_ids.shape[1] - 1] * len(batch_a))
    answer_positions_b = torch.tensor([clean_b_ids.shape[1] - 1] * len(batch_b))
    
    return VennBatch(
        clean_a_input_ids=clean_a_ids,
        clean_b_input_ids=clean_b_ids,
        corr_a_input_ids=corr_a_ids,
        corr_b_input_ids=corr_b_ids,
        answer_positions_a=answer_positions_a,
        answer_positions_b=answer_positions_b,
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
    
    results = {}
    
    for idx, (cat_a, cat_b) in enumerate(pairs):
        print(f"\n=======================================================")
        print(f"[{idx+1}/{len(pairs)}] Intersecting '{cat_a}' and '{cat_b}'")
        print(f"=======================================================\n")
        
        ds_a = datasets[cat_a]
        ds_b = datasets[cat_b]
        
        paired_ds = PairedDataset(ds_a, ds_b)
        dl = DataLoader(paired_ds, batch_size=8, collate_fn=lambda b: paired_collate(b, tokenizer), shuffle=True)

        discoverer = VennCircuitDiscoverer(model_name=MODEL_NAME, mode="intersection", target_kl=2.0)
        
        discoverer.fit(dl, epochs=2)
        circuit = discoverer.extract_circuit()
        
        # Calculate overlap
        total_intersection = 0
        total_union = 0
        
        for name, gate in circuit.items():
            if "layer" not in name: continue
            
            c_core = gate["core"]
            total_intersection += c_core.sum().item()
            total_union += (gate["mask_a"] | gate["mask_b"]).sum().item()
            
        iou = total_intersection / total_union if total_union > 0 else 0
        print(f"\n=> Overlap (IoU) for {cat_a} & {cat_b}: {iou:.4f} ({total_intersection} / {total_union})")
        
        results[f"{cat_a}_vs_{cat_b}"] = {
            "iou": float(iou),
            "intersection": int(total_intersection),
            "union": int(total_union)
        }
        
        del discoverer
        torch.cuda.empty_cache()
        
    print("\n\nFINAL RESULTS:")
    for pair, res in results.items():
        print(f"{pair}: IoU = {res['iou']:.4f}")
        
    out_file = Path("/home/exouser/pruning/data_research/universal_circuit_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()
