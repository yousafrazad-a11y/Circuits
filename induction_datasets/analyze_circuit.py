import os
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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
    
    max_len = max(
        len(tokenizer.encode(p)) for p in clean_prompts + corr_a_prompts + corr_b_prompts
    )
    
    clean_enc = tokenizer(clean_prompts, padding=True, return_tensors="pt")
    corr_a_enc = tokenizer(corr_a_prompts, padding=True, return_tensors="pt")
    corr_b_enc = tokenizer(corr_b_prompts, padding=True, return_tensors="pt")
    
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
    
    def get_token_id(word):
        return tokenizer.encode(" " + word, add_special_tokens=False)[-1]

    target_a = torch.tensor([get_token_id(item["target"]) for item in batch])
    distractor_a = torch.tensor([get_token_id(item["distractor_a"]) for item in batch])
    
    target_b = target_a.clone()
    distractor_b = distractor_a.clone()

    answer_positions = clean_mask.sum(dim=1) - 1
    
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
    discoverer = VennCircuitDiscoverer(model_name=MODEL_NAME, mode="intersection", target_kl=2.0)

    print("--- Running Venn Discovery on Dataset 1 ---")
    discoverer.fit(dl1, epochs=2)
    circuit1 = discoverer.extract_circuit()

    del discoverer
    torch.cuda.empty_cache()

    discoverer = VennCircuitDiscoverer(model_name=MODEL_NAME, mode="intersection", target_kl=2.0)
    print("--- Running Venn Discovery on Dataset 2 ---")
    discoverer.fit(dl2, epochs=2)
    circuit2 = discoverer.extract_circuit()

    print("\n--- Layer-by-Layer Intersection Analysis ---")
    
    # We will aggregate by layer index and module type (attn vs mlp)
    layer_stats = {}
    
    total_intersection = 0
    total_union = 0
    
    for name in circuit1:
        # e.g., layer.15.attention_heads
        parts = name.split(".")
        if parts[0] != "layer":
            continue
            
        layer_idx = int(parts[1])
        module_type = "attn" if "attention" in name else "mlp" if "mlp" in name else "other"
        
        c1_core = circuit1[name]["core"]
        c2_core = circuit2[name]["core"]
        
        intersection = (c1_core & c2_core)
        union = (c1_core | c2_core)
        
        total_intersection += intersection.sum().item()
        total_union += union.sum().item()
        
        shared_active = intersection.sum().item()
        total_gates = intersection.numel()
        
        if layer_idx not in layer_stats:
            layer_stats[layer_idx] = {"attn_active": 0, "attn_total": 0, "mlp_active": 0, "mlp_total": 0}
            
        if module_type == "attn":
            layer_stats[layer_idx]["attn_active"] += shared_active
            layer_stats[layer_idx]["attn_total"] += total_gates
        elif module_type == "mlp":
            layer_stats[layer_idx]["mlp_active"] += shared_active
            layer_stats[layer_idx]["mlp_total"] += total_gates

    iou = total_intersection / total_union if total_union > 0 else 0

    print(f"\nOverall IoU: {iou:.4f} ({total_intersection} / {total_union})")

    print("\nSummary of Shared 'Core' Gates:")
    print("Layer | Attn Active / Total (Density) | MLP Active / Total (Density)")
    print("-" * 70)
    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        attn_d = (stats['attn_active'] / stats['attn_total'] * 100) if stats['attn_total'] > 0 else 0
        mlp_d = (stats['mlp_active'] / stats['mlp_total'] * 100) if stats['mlp_total'] > 0 else 0
        
        attn_str = f"{stats['attn_active']:>5} / {stats['attn_total']:>5} ({attn_d:>5.1f}%)"
        mlp_str = f"{stats['mlp_active']:>5} / {stats['mlp_total']:>5} ({mlp_d:>5.1f}%)"
        
        print(f"{layer:>5} | {attn_str} | {mlp_str}")
        
    # Extra logging to write to file so we don't lose it
    with open("circuit_layer_summary.txt", "w") as f:
        f.write("Summary of Shared 'Core' Gates:\n")
        f.write("Layer | Attn Active / Total (Density) | MLP Active / Total (Density)\n")
        f.write("-" * 70 + "\n")
        for layer in sorted(layer_stats.keys()):
            stats = layer_stats[layer]
            attn_d = (stats['attn_active'] / stats['attn_total'] * 100) if stats['attn_total'] > 0 else 0
            mlp_d = (stats['mlp_active'] / stats['mlp_total'] * 100) if stats['mlp_total'] > 0 else 0
            attn_str = f"{stats['attn_active']:>5} / {stats['attn_total']:>5} ({attn_d:>5.1f}%)"
            mlp_str = f"{stats['mlp_active']:>5} / {stats['mlp_total']:>5} ({mlp_d:>5.1f}%)"
            f.write(f"{layer:>5} | {attn_str} | {mlp_str}\n")

if __name__ == "__main__":
    main()
