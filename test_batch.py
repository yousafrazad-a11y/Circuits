import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from induction_datasets.test_venn_induction import load_dataset
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16).to(device)
ds = load_dataset("induction_datasets/category_chains/fruits.jsonl")

# Test 1 unbatched
item = ds[0]
prompt = item["clean_prompt"] if "clean_prompt" in item else item["prompt"]
target_token = tokenizer.encode(item["target"], add_special_tokens=False)[0]
dist_token = tokenizer.encode(item["distractor"], add_special_tokens=False)[0]
inputs = tokenizer(prompt, return_tensors="pt").to(device)
logits1 = model(**inputs).logits[0, -1, :]
acc1 = logits1[target_token] > logits1[dist_token]

# Test 2 batched
item2 = ds[1]
prompt2 = item2["clean_prompt"] if "clean_prompt" in item2 else item2["prompt"]
target_token2 = tokenizer.encode(item2["target"], add_special_tokens=False)[0]
dist_token2 = tokenizer.encode(item2["distractor"], add_special_tokens=False)[0]

inputs2 = tokenizer([prompt, prompt2], return_tensors="pt", padding=True).to(device)
logits2 = model(**inputs2).logits

# manual padding like run_single_pruning
c_seq1 = torch.tensor(tokenizer.encode(prompt, add_special_tokens=True))
c_seq2 = torch.tensor(tokenizer.encode(prompt2, add_special_tokens=True))
max_len = max(len(c_seq1), len(c_seq2))
pad1 = torch.cat([c_seq1, torch.full((max_len - len(c_seq1),), tokenizer.pad_token_id)])
pad2 = torch.cat([c_seq2, torch.full((max_len - len(c_seq2),), tokenizer.pad_token_id)])
mask1 = torch.cat([torch.ones(len(c_seq1)), torch.zeros(max_len - len(c_seq1))])
mask2 = torch.cat([torch.ones(len(c_seq2)), torch.zeros(max_len - len(c_seq2))])

inputs3 = {
    "input_ids": torch.stack([pad1, pad2]).long().to(device),
    "attention_mask": torch.stack([mask1, mask2]).long().to(device)
}
logits3 = model(**inputs3).logits

print(f"Unbatched acc1: {acc1}")
print(f"Tokenizer batch acc1: {logits2[0, len(c_seq1)-1, target_token] > logits2[0, len(c_seq1)-1, dist_token]}")
print(f"Manual batch acc1: {logits3[0, len(c_seq1)-1, target_token] > logits3[0, len(c_seq1)-1, dist_token]}")
