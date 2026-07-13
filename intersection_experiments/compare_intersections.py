import os
import torch
from itertools import combinations

RESULTS_DIR = "/home/exouser/pruning/intersection_experiments/results"

def load_intersection_masks():
    masks = {}
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".pt"):
            name = f.replace("_heads.pt", "")
            path = os.path.join(RESULTS_DIR, f)
            data = torch.load(path, weights_only=True)
            mask_a = data["mask_a"]
            mask_b = data["mask_b"]
            
            # Compute intersection circuit
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

def main():
    print("Loading intersection circuits...")
    intersection_masks = load_intersection_masks()
    names = sorted(list(intersection_masks.keys()))
    print(f"Loaded {len(names)} intersection circuits.\n")
    
    print("="*60)
    print("PAIRWISE SIMILARITY BETWEEN INTERSECTION CIRCUITS")
    print("="*60)
    
    pairs = list(combinations(names, 2))
    all_jaccards = []
    
    for n1, n2 in pairs:
        sim = compute_similarity(intersection_masks[n1], intersection_masks[n2])
        all_jaccards.append(sim['Jaccard'])
        print(f"[{n1}] vs [{n2}]:")
        print(f"  Sizes: {sim['Size_1']} and {sim['Size_2']}")
        print(f"  Intersection: {sim['Intersection']} / Union: {sim['Union']}")
        print(f"  Jaccard: {sim['Jaccard']:.4f} | Overlap 1: {sim['Overlap_1']:.4f} | Overlap 2: {sim['Overlap_2']:.4f}\n")
        
    print(f"Average Jaccard Similarity across all {len(pairs)} pairs: {sum(all_jaccards)/len(all_jaccards):.4f}\n")
    
    print("="*60)
    print("GLOBAL UNIVERSAL CIRCUIT (INTERSECTION OF ALL 10)")
    print("="*60)
    
    global_circuit = {}
    first_name = names[0]
    for k in intersection_masks[first_name].keys():
        global_circuit[k] = intersection_masks[first_name][k].clone()
        
    for name in names[1:]:
        for k in global_circuit.keys():
            global_circuit[k] = global_circuit[k] & intersection_masks[name][k]
            
    global_size = sum(global_circuit[k].sum().item() for k in global_circuit.keys())
    print(f"The Universal Circuit shared by ALL pairs contains {global_size} components.")
    
    # Analyze exactly which layers these heads belong to
    print("\nLayer Breakdown of Global Universal Circuit:")
    for k in sorted(global_circuit.keys()):
        active = global_circuit[k].sum().item()
        if active > 0:
            print(f"  {k}: {active} heads active")

if __name__ == "__main__":
    main()
