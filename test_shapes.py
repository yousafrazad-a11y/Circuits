import torch
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

with open("/home/exouser/pruning/induction_datasets/category_chains/fruits.jsonl") as f:
    item = json.loads(f.readline())
    c_seq = torch.tensor(tokenizer.encode(item["clean_prompt"], add_special_tokens=True))
    corr_seq = torch.tensor(tokenizer.encode(item["corr_prompt"] if "corr_prompt" in item else item.get("corrupted_prompt", ""), add_special_tokens=True))
    print("c_seq length:", len(c_seq))
    print("corr_seq length:", len(corr_seq))
