import torch
import argparse
import os

def get_active_heads(mask_state):
    total = 0
    active = 0
    for k, v in mask_state.items():
        if 'head_gates' in k:
            total += v.numel()
            active += v.sum().item()
    return active, total

def main():
    parser = argparse.ArgumentParser(description="Compute the logical intersection of multiple pruning masks.")
    parser.add_argument("--masks", nargs='+', required=True, help="List of paths to the mask files.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the resulting intersected mask.")
    args = parser.parse_args()

    if len(args.masks) < 2:
        print("Warning: Only one mask provided. The output will just be a copy of this mask.")
        
    intersected_state = None
    
    for i, mask_path in enumerate(args.masks):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
        mask = torch.load(mask_path, weights_only=True)
        active, total = get_active_heads(mask)
        print(f"Mask {i+1} ({os.path.basename(mask_path)}) has {active}/{total} active heads.")
        
        if intersected_state is None:
            intersected_state = mask
        else:
            for k in intersected_state.keys():
                if k in mask:
                    # Logical AND for boolean tensors
                    intersected_state[k] = intersected_state[k] & mask[k]
                    
    final_active, final_total = get_active_heads(intersected_state)
    print(f"\nFinal Intersected Mask has {final_active}/{final_total} active heads ({(final_active/final_total)*100:.1f}%).")
    
    # Save the output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    torch.save(intersected_state, args.output)
    print(f"Saved intersected mask to {args.output}")

if __name__ == "__main__":
    main()
