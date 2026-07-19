import torch
mask = torch.tensor([1.0, 0.0])
try:
    res = torch.where(mask, torch.tensor(5.0), torch.tensor(-1e6))
    print(res)
except Exception as e:
    print(e)
