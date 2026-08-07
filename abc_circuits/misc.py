import torch
heads = ["L1.H12","L1.H20","L10.H20","L10.H21","L13.H27","L14.H16","L15.H10",
         "L3.H20","L8.H17","L8.H25","L9.H9","L10.H26","L10.H9","L11.H27",
         "L13.H26","L13.H31","L2.H3","L4.H13","L5.H17"]
cks = {d: torch.load(f"masks_abc/{d}_checkpoint.pt", weights_only=True) for d in "ABC"}
for h in heads:
    L, H = h[1:].split(".H"); key = f"model.layers.{L}.attn.head_gates"
    vals = "  ".join(f"{d}:{cks[d][key][int(H)].item():+.2f}" for d in "ABC")
    print(f"{h:9s} {vals}")