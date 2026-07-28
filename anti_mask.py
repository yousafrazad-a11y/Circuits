import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Create an anti-circuit mask: logical NOT of every gate in the input mask.")
    parser.add_argument("--mask", type=str, required=True, help="Path to the input mask (.pt).")
    parser.add_argument("--output", type=str, required=True, help="Path to save the anti-circuit mask.")
    args = parser.parse_args()

    mask = torch.load(args.mask, weights_only=True)
    anti = {}
    for k, v in mask.items():
        if v.dtype == torch.bool:
            anti[k] = ~v
        else:
            anti[k] = 1 - v

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(anti, args.output)

    print(f"Anti-circuit of {args.mask}")
    for k, v in mask.items():
        short = k.split('.')[-1]
        print(f"  {k}: {int(v.sum())} -> {int(anti[k].sum())} active (of {v.numel()})")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
