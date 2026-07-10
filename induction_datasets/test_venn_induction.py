import os
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# The library relies on local imports and needs to be imported securely
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from venn_circuit_discovery import VennCircuitDiscoverer, VennBatch

MODEL_NAME = "meta-llama/Llama-3.2-1B"

def load_dataset(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def collate_fn(batch, tokenizer):
    clean_prompts = [item["clean_prompt"] for item in batch]
    corr_a_prompts = [item["corr_a_prompt"] for item in batch]
    corr_b_prompts = [item["corr_b_prompt"] for item in batch]
    
    clean_enc = tokenizer(clean_prompts, padding=True, return_tensors="pt")
    corr_a_enc = tokenizer(corr_a_prompts, padding=True, return_tensors="pt")
    corr_b_enc = tokenizer(corr_b_prompts, padding=True, return_tensors="pt")
    
    # Ensure all streams have the exact same sequence length for four-stream pass
    max_len = max(clean_enc.input_ids.shape[1], corr_a_enc.input_ids.shape[1], corr_b_enc.input_ids.shape[1])
    
    def pad_to_max(tensor, pad_token_id):
        if tensor.shape[1] < max_len:
            pad_shape = (tensor.shape[0], max_len - tensor.shape[1])
            pad_tensor = torch.full(pad_shape, pad_token_id, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([pad_tensor, tensor], dim=1)
        return tensor

    clean_ids = pad_to_max(clean_enc.input_ids, tokenizer.pad_token_id)
    corr_a_ids = pad_to_max(corr_a_enc.input_ids, tokenizer.pad_token_id)
    corr_b_ids = pad_to_max(corr_b_enc.input_ids, tokenizer.pad_token_id)
    clean_mask = pad_to_max(clean_enc.attention_mask, 0)
    
    # Target and Distractor tokens
    def get_token_id(word):
        return tokenizer.encode(" " + word, add_special_tokens=False)[-1]

    target_a = torch.tensor([get_token_id(item["target"]) for item in batch])
    distractor_a = torch.tensor([get_token_id(item["distractor_a"]) for item in batch])
    
    target_b = target_a.clone()
    distractor_b = distractor_a.clone() # Fallback for dataset 2's corr B

    answer_positions = clean_mask.sum(dim=1) - 1 # Last token index
    
    return VennBatch(
        clean_input_ids=clean_ids,
        corr_a_input_ids=corr_a_ids,
        corr_b_input_ids=corr_b_ids,
        answer_positions=answer_positions,
        target_a=target_a,
        distractor_a=distractor_a,
        target_b=target_b,
        distractor_b=distractor_b,
        attention_mask=clean_mask
    )

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    script_dir = Path(__file__).parent
    ds1 = load_dataset(script_dir / "dataset1_names.jsonl")
    ds2 = load_dataset(script_dir / "dataset2_nonsense.jsonl")

    dl1 = DataLoader(ds1, batch_size=8, collate_fn=lambda b: collate_fn(b, tokenizer), shuffle=True)
    dl2 = DataLoader(ds2, batch_size=8, collate_fn=lambda b: collate_fn(b, tokenizer), shuffle=True)

    print("Initializing discoverer...")
    discoverer = VennCircuitDiscoverer(
        model_name=MODEL_NAME,
        mode="intersection",
        target_kl=0.1
    )

    print("--- Running Venn Discovery on Dataset 1 (Name Binding) ---")
    discoverer.fit(dl1, epochs=2)
    circuit1 = discoverer.extract_circuit()

    # To be safe against state leakage (gates, scheduler PIDs), we re-initialize.
    # To save VRAM, delete the first discoverer and clear cache.
    del discoverer
    torch.cuda.empty_cache()

    print("\nRe-initializing discoverer for Dataset 2...")
    discoverer = VennCircuitDiscoverer(
        model_name=MODEL_NAME,
        mode="intersection",
        target_kl=0.1
    )

    print("--- Running Venn Discovery on Dataset 2 (Nonsense Word Translation) ---")
    discoverer.fit(dl2, epochs=2)
    circuit2 = discoverer.extract_circuit()

    print("\n--- Calculating Intersection-over-Union (IoU) of g_core ---")
    intersection = 0
    union = 0
    for name in circuit1:
        c1_core = circuit1[name]["core"]
        c2_core = circuit2[name]["core"]
        
        intersection += (c1_core & c2_core).sum().item()
        union += (c1_core | c2_core).sum().item()

    iou = intersection / union if union > 0 else 0
    print(f"Total Shared Core Gates (Intersection): {intersection}")
    print(f"Total Unique Core Gates (Union): {union}")
    print(f"IoU Score: {iou:.4f}")

if __name__ == "__main__":
    main()
