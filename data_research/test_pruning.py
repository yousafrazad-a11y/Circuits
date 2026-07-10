import os
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venn_circuit_discovery_v2 import VennCircuitDiscoverer, VennBatch

MODEL_NAME = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

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
        target_a=target_a, distractor_a=distractor_a,
        target_b=target_b, distractor_b=distractor_b,
        attention_mask_a=clean_a_mask,
        attention_mask_b=clean_b_mask
    )

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, ds_a, ds_b):
        self.ds_a = ds_a
        self.ds_b = ds_b
    def __len__(self): return min(len(self.ds_a), len(self.ds_b))
    def __getitem__(self, idx): return self.ds_a[idx], self.ds_b[idx]

data_dir = Path("/home/exouser/pruning/induction_datasets/category_chains")
ds_a = load_dataset(data_dir / "fruits.jsonl")
ds_b = load_dataset(data_dir / "animals.jsonl")
dl = DataLoader(PairedDataset(ds_a, ds_b), batch_size=16, collate_fn=lambda b: collate_fn(b, tokenizer))

from venn_circuit_discovery_v2 import VennHyperparameters
hp = VennHyperparameters(
    gate_lr=0.1,  
    target_kl_a=3.0, 
    target_kl_b=3.0,
    init_lambda=10.0,
    max_lambda=200.0,
    pid_kp=2.0
)

discoverer = VennCircuitDiscoverer(
    model_name=MODEL_NAME, 
    mode="intersection", 
    hyperparameters=hp
)
discoverer.fit(dl, epochs=4)
summary = discoverer.circuit_summary()
print("Sparsity Core:", summary["core"])
print("Total Expected Active:", summary["core"] + summary["a_only"] + summary["b_only"])

