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

def main():
    parser = argparse.ArgumentParser(description="Compute the logical intersection of multiple pruning masks.")
    parser.add_argument("--masks", nargs='+', required=True, help="List of paths to the mask files.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the resulting intersected mask.")
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
                if k in mask and gate_type(k) in args.components:
                    # Logical AND for boolean tensors (only for selected components)
                    intersected_state[k] = intersected_state[k] & mask[k]

    # Handle excluded components
    for k in intersected_state.keys():
        t = gate_type(k)
        if t is None or t in args.components:
            continue
        if t == "neuron_gates":
            head_key = k.replace("neuron_gates", "head_gates")
            if "head_gates" in args.components and head_key in intersected_state:
                # Inherit head state: all neurons on inside surviving heads
                head_dim = intersected_state[k].numel() // intersected_state[head_key].numel()
                intersected_state[k] = intersected_state[head_key].repeat_interleave(head_dim)
            else:
                intersected_state[k] = torch.ones_like(intersected_state[k])
        else:
            intersected_state[k] = torch.ones_like(intersected_state[k])

    final_active, final_total = get_active_heads(intersected_state)
    print(f"\nFinal Intersected Mask has {final_active}/{final_total} active heads ({(final_active/final_total)*100:.1f}%).")
    for t in GATE_TYPES:
        a = sum(int(v.sum()) for k, v in intersected_state.items() if gate_type(k) == t)
        n = sum(v.numel() for k, v in intersected_state.items() if gate_type(k) == t)
        if n:
            print(f"  {t}: {a}/{n} ({100*a/n:.1f}%)")

    # Save the output
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(intersected_state, args.output)
    print(f"Saved intersected mask to {args.output}")

if __name__ == "__main__":
    main()
