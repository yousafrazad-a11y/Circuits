import torch
import argparse
import os

GATE_TYPES = ["head_gates", "neuron_gates", "hidden_gates", "output_gates",
              "attention_block_gate", "mlp_block_gate", "layer_gates"]

# Group aliases for --components
COMPONENT_GROUPS = {
    "attention": ["head_gates", "neuron_gates"],
    "mlp": ["hidden_gates", "output_gates"],
}

def gate_type(key):
    for t in GATE_TYPES:
        if t in key:
            return t
    return None

def get_active_heads(mask_state):
    total = 0
    active = 0
    for k, v in mask_state.items():
        if 'head_gates' in k:
            total += v.numel()
            active += v.sum().item()
    return active, total

def handle_excluded_components(mask_state, components):
    """Excluded gate types are NOT intersected: neuron_gates inherit their head's
    state (all neurons on if the head is on), all other excluded types are set fully on."""
    for k in mask_state.keys():
        t = gate_type(k)
        if t is None or t in components:
            continue
        if t == "neuron_gates":
            head_key = k.replace("neuron_gates", "head_gates")
            if "head_gates" in components and head_key in mask_state:
                # Inherit head state: all neurons on inside surviving heads
                head_dim = mask_state[k].numel() // mask_state[head_key].numel()
                mask_state[k] = mask_state[head_key].repeat_interleave(head_dim)
            else:
                mask_state[k] = torch.ones_like(mask_state[k])
        else:
            mask_state[k] = torch.ones_like(mask_state[k])
    return mask_state

def print_gate_stats(mask_state, label):
    final_active, final_total = get_active_heads(mask_state)
    print(f"\n{label} has {final_active}/{final_total} active heads ({(final_active/final_total)*100:.1f}%).")
    for t in GATE_TYPES:
        a = sum(int(v.sum()) for k, v in mask_state.items() if gate_type(k) == t)
        n = sum(v.numel() for k, v in mask_state.items() if gate_type(k) == t)
        if n:
            print(f"  {t}: {a}/{n} ({100*a/n:.1f}%)")

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
    parser.add_argument("--components", nargs='+', default=GATE_TYPES,
                        help="Gate types to intersect (logical AND). Accepts individual types "
                             f"({', '.join(GATE_TYPES)}) or group aliases "
                             "(attention = head_gates + neuron_gates, mlp = hidden_gates + output_gates). "
                             "Excluded types are NOT intersected: neuron_gates inherit their head's state "
                             "(all neurons on if the head is on), all other excluded types are set fully on.")
    args = parser.parse_args()

    # Expand group aliases
    components = []
    for c in args.components:
        if c in COMPONENT_GROUPS:
            components.extend(COMPONENT_GROUPS[c])
        elif c in GATE_TYPES:
            components.append(c)
        else:
            parser.error(f"Unknown component '{c}'. Choose from {GATE_TYPES + list(COMPONENT_GROUPS)}")
    args.components = components

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
                if k in mask and gate_type(k) in args.components:
                    # Logical AND for boolean tensors (only for selected components)
                    intersected_state[k] = intersected_state[k] & mask[k]

    if args.mode == "difference":
        # Per-mask disjoint part: mask_i AND NOT (raw intersection over selected components)
        os.makedirs(args.output, exist_ok=True)
        for mask_path, mask in loaded:
            name = os.path.splitext(os.path.basename(mask_path))[0]
            diff_state = {k: v.clone() for k, v in mask.items()}
            for k in diff_state.keys():
                if k in intersected_state and gate_type(k) in args.components:
                    diff_state[k] = mask[k] & ~intersected_state[k]
            diff_state = handle_excluded_components(diff_state, args.components)
            print_gate_stats(diff_state, f"Disjoint part of {name}")
            out_path = os.path.join(args.output, f"{name}_minus_intersection.pt")
            torch.save(diff_state, out_path)
            print(f"Saved disjoint mask to {out_path}")
        return

    # Handle excluded components
    intersected_state = handle_excluded_components(intersected_state, args.components)

    print_gate_stats(intersected_state, "Final Intersected Mask")

    # Save the output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(intersected_state, args.output)
    print(f"Saved intersected mask to {args.output}")

if __name__ == "__main__":
    main()
