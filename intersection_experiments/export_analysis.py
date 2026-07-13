import os
import torch
import json
from itertools import combinations

RESULTS_DIR = "/home/exouser/pruning/intersection_experiments/results"
OUTPUT_DIR = "/home/exouser/pruning/intersection_experiments/final_analysis"

def load_intersection_masks():
    masks = {}
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".pt"):
            name = f.replace("_heads.pt", "")
            path = os.path.join(RESULTS_DIR, f)
            data = torch.load(path, weights_only=True)
            mask_a = data["mask_a"]
            mask_b = data["mask_b"]
            
            intersection_circuit = {}
            for k in mask_a.keys():
                intersection_circuit[k] = mask_a[k] & mask_b[k]
                
            masks[name] = intersection_circuit
    return masks

def compute_similarity(mask1, mask2):
    total_1 = 0
    total_2 = 0
    total_intersect = 0
    total_union = 0
    
    for k in mask1.keys():
        m1 = mask1[k]
        m2 = mask2[k]
        intersect = m1 & m2
        union = m1 | m2
        
        total_1 += m1.sum().item()
        total_2 += m2.sum().item()
        total_intersect += intersect.sum().item()
        total_union += union.sum().item()
        
    jaccard = total_intersect / total_union if total_union > 0 else 0
    overlap_1 = total_intersect / total_1 if total_1 > 0 else 0
    overlap_2 = total_intersect / total_2 if total_2 > 0 else 0
    
    return {
        "Size_1": total_1,
        "Size_2": total_2,
        "Intersection": total_intersect,
        "Union": total_union,
        "Jaccard": jaccard,
        "Overlap_1": overlap_1,
        "Overlap_2": overlap_2
    }

def extract_head_indices(circuit):
    """Convert boolean mask dict to a list of (layer, head_index) tuples."""
    active_heads = []
    for k, mask in circuit.items():
        # k looks like: model.layers.X.attn.head_gates
        if "layers" in k:
            layer_idx = int(k.split("layers.")[1].split(".")[0])
            indices = torch.nonzero(mask).squeeze()
            if indices.dim() == 0:
                indices = [indices.item()]
            else:
                indices = indices.tolist()
            
            for idx in indices:
                active_heads.append((layer_idx, idx))
    return sorted(active_heads)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    circuits_dir = os.path.join(OUTPUT_DIR, "saved_circuits")
    os.makedirs(circuits_dir, exist_ok=True)
    
    print("Loading raw pairwise results...")
    intersection_masks = load_intersection_masks()
    names = sorted(list(intersection_masks.keys()))
    
    # 1. Save all 10 pairwise intersection circuits
    print("Saving 10 pairwise intersection circuits...")
    for name, mask in intersection_masks.items():
        torch.save(mask, os.path.join(circuits_dir, f"{name}_circuit.pt"))
        
        # Save readable indices
        indices = extract_head_indices(mask)
        with open(os.path.join(circuits_dir, f"{name}_indices.json"), "w") as f:
            json.dump({"total_heads": len(indices), "heads": indices}, f, indent=2)

    # 2. Compute similarities and create tables
    print("Computing similarities...")
    pairs = list(combinations(names, 2))
    
    md_content = "# Intersection Circuit Similarity Analysis\n\n"
    md_content += "This document analyzes the similarity between the 10 pairwise intersection circuits.\n\n"
    md_content += "## Pairwise Similarity Matrix\n\n"
    md_content += "| Circuit 1 | Circuit 2 | Size 1 | Size 2 | Intersection | Union | Jaccard | Overlap 1 | Overlap 2 |\n"
    md_content += "|---|---|---|---|---|---|---|---|---|\n"
    
    all_jaccards = []
    for n1, n2 in pairs:
        sim = compute_similarity(intersection_masks[n1], intersection_masks[n2])
        all_jaccards.append(sim['Jaccard'])
        md_content += f"| {n1} | {n2} | {sim['Size_1']} | {sim['Size_2']} | {sim['Intersection']} | {sim['Union']} | {sim['Jaccard']:.4f} | {sim['Overlap_1']:.4f} | {sim['Overlap_2']:.4f} |\n"

    avg_jaccard = sum(all_jaccards) / len(all_jaccards)
    md_content += f"\n**Average Jaccard Similarity:** {avg_jaccard:.4f}\n\n"

    # 3. Compute GLOBAL UNIVERSAL CIRCUIT
    print("Computing Global Universal Circuit...")
    global_circuit = {}
    first_name = names[0]
    for k in intersection_masks[first_name].keys():
        global_circuit[k] = intersection_masks[first_name][k].clone()
        
    for name in names[1:]:
        for k in global_circuit.keys():
            global_circuit[k] = global_circuit[k] & intersection_masks[name][k]
            
    torch.save(global_circuit, os.path.join(circuits_dir, "GLOBAL_UNIVERSAL_CIRCUIT.pt"))
    
    global_indices = extract_head_indices(global_circuit)
    with open(os.path.join(circuits_dir, "GLOBAL_UNIVERSAL_CIRCUIT_indices.json"), "w") as f:
        json.dump({"total_heads": len(global_indices), "heads": global_indices}, f, indent=2)

    md_content += "## Global Universal Circuit\n\n"
    md_content += "The geometric intersection of ALL 10 circuits yields a core universal architecture.\n"
    md_content += f"**Total Core Heads:** {len(global_indices)}\n\n"
    
    md_content += "### Exact Head Indices `(Layer, Head)`\n"
    md_content += "```json\n"
    md_content += json.dumps(global_indices, indent=2)
    md_content += "\n```\n"

    with open(os.path.join(OUTPUT_DIR, "analysis_report.md"), "w") as f:
        f.write(md_content)
        
    print(f"All data successfully exported to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
