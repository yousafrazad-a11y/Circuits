import sys
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset

class CategoryDataset(Dataset):
    def __init__(self, cat, tokenizer):
        self.data = []
        with open(f"/home/exouser/pruning/induction_datasets/category_chains/{cat}.jsonl", 'r') as f:
            for line in f:
                item = json.loads(line)
                target_tok = tokenizer.encode(" " + item["target"], add_special_tokens=False)[-1]
                dist_tok = tokenizer.encode(" " + item["distractor"], add_special_tokens=False)[-1]
                self.data.append({
                    "clean_prompt": item["clean_prompt"],
                    "target_tok": target_tok,
                    "distractor_tok": dist_tok
                })

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch, tokenizer):
    clean_input_ids = []
    target_tokens, distractor_tokens, answer_positions = [], [], []
    for item in batch:
        c_seq = torch.tensor(tokenizer.encode(item["clean_prompt"], add_special_tokens=True))
        clean_input_ids.append(c_seq)
        target_tokens.append(item["target_tok"])
        distractor_tokens.append(item["distractor_tok"])
        answer_positions.append(len(c_seq) - 1)
        
    global_max_len = max(len(s) for s in clean_input_ids)
    clean_padded, attn_mask = [], []
    for c_seq in clean_input_ids:
        clean_padded.append(torch.cat([c_seq, torch.full((global_max_len - len(c_seq),), tokenizer.pad_token_id)]))
        attn_mask.append(torch.cat([torch.ones(len(c_seq)), torch.zeros(global_max_len - len(c_seq))]))
        
    return {
        "clean_input_ids": torch.stack(clean_padded).to("cuda"),
        "attention_mask": torch.stack(attn_mask).to("cuda"),
        "target_tokens": torch.tensor(target_tokens, device="cuda"),
        "distractor_tokens": torch.tensor(distractor_tokens, device="cuda"),
        "answer_positions": torch.tensor(answer_positions, device="cuda")
    }

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16).to("cuda")

for cat in ["fruits", "animals", "colors", "metals", "vehicles"]:
    ds = CategoryDataset(cat, tokenizer)
    dl = DataLoader(ds, batch_size=32, collate_fn=lambda b: collate_fn(b, tokenizer))
    correct = total = 0
    with torch.no_grad():
        for batch in dl:
            outputs = model(input_ids=batch["clean_input_ids"], attention_mask=batch["attention_mask"])
            b_idx = torch.arange(batch["clean_input_ids"].size(0), device="cuda")
            logit_good = outputs.logits[b_idx, batch["answer_positions"], batch["target_tokens"]]
            logit_bad = outputs.logits[b_idx, batch["answer_positions"], batch["distractor_tokens"]]
            correct += (logit_good > logit_bad).sum().item()
            total += batch["clean_input_ids"].size(0)
    print(f"{cat}: {correct/total*100:.1f}%")
