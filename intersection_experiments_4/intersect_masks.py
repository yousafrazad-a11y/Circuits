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
    parser = argparse.ArgumentParser(description="Compute the logical intersection of multiple pruning masks, "
                                                 "or each mask's disjoint part (mask minus the intersection).")
    parser.add_argument("--masks", nargs='+', required=True, help="List of paths to the mask files.")
    parser.add_argument("--output", type=str, required=True,
                        help="Intersect mode: path to save the intersected mask. "
                             "Difference mode: directory to save the per-mask disjoint masks.")
    parser.add_argument("--mode", choices=["intersect", "difference"], default="intersect",
                        help="intersect: logical AND of all masks (default). "
                             "difference: for each input mask, save mask_i AND NOT (intersection of all masks) "
                             "— the part of each circuit that is NOT shared.")
    parser.add_argument("--components", choices=["all", "heads", "mlp"], default="all",
                        help="Which gate group to intersect. Non-selected groups are forced fully ON. "
                             "heads: intersect attention head gates only, all MLP block gates ON. "
                             "mlp: intersect MLP block gates only, all attention head gates ON.")
    args = parser.parse_args()

    if len(args.masks) < 2:
        print("Warning: Only one mask provided. The output will just be a copy of this mask.")

    # Load all masks
    loaded = []  # list of (path, mask_state)
    intersected_state = None

    for i, mask_path in enumerate(args.masks):
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        mask = torch.load(mask_path, weights_only=True)
        active, total = get_active_heads(mask)
        print(f"Mask {i+1} ({os.path.basename(mask_path)}) has {active}/{total} active heads.")
        loaded.append((mask_path, mask))

        if intersected_state is None:
            intersected_state = {k: v.clone() for k, v in mask.items()}
        else:
            for k in intersected_state.keys():
                if k in mask:
                    # Logical AND for boolean tensors
                    intersected_state[k] = intersected_state[k] & mask[k]

    if args.components != "all":
        # Force non-selected gate groups fully ON
        for k in intersected_state.keys():
            if args.components == "heads" and 'mlp_block_gate' in k:
                intersected_state[k] = torch.ones_like(intersected_state[k])
            elif args.components == "mlp" and 'head_gates' in k:
                intersected_state[k] = torch.ones_like(intersected_state[k])

    if args.mode == "difference":
        # Per-mask disjoint part: mask_i AND NOT (intersection of all masks)
        os.makedirs(args.output, exist_ok=True)
        for mask_path, mask in loaded:
            name = os.path.splitext(os.path.basename(mask_path))[0]
            diff_state = {k: v.clone() for k, v in mask.items()}
            for k in diff_state.keys():
                if k in intersected_state:
                    diff_state[k] = mask[k] & ~intersected_state[k]
            active, total = get_active_heads(diff_state)
            print(f"\nDisjoint part of {name} has {active}/{total} active heads ({(active/total)*100:.1f}%).")
            out_path = os.path.join(args.output, f"{name}_minus_intersection.pt")
            torch.save(diff_state, out_path)
            print(f"Saved disjoint mask to {out_path}")
        return

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
