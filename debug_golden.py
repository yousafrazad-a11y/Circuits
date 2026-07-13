import torch
import sys
from transformers import AutoTokenizer
sys.path.insert(0, "/home/exouser/pruning/circuit_pruning-argo")
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig
from induction_datasets.test_venn_induction import load_dataset
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token

config = PruningConfig()
model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)

ds = load_dataset("induction_datasets/category_chains/fruits.jsonl")

for i in range(2):
    item = ds[i]
    prompt = item["clean_prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tgt = tokenizer.encode(item["target"], add_special_tokens=False)[0]
    dst = tokenizer.encode(item["distractor"], add_special_tokens=False)[0]
    
    with torch.no_grad():
        out = model(**inputs).logits[0, -1]
    
    print(f"[{i}] Unbatched GT > GD? {out[tgt].item() > out[dst].item()}")

# Manual batch
prompt1 = ds[0]["clean_prompt"]
prompt2 = ds[1]["clean_prompt"]
inputs_b = tokenizer([prompt1, prompt2], return_tensors="pt", padding=True).to(device)
tgt = [tokenizer.encode(ds[0]["target"], add_special_tokens=False)[0], tokenizer.encode(ds[1]["target"], add_special_tokens=False)[0]]
dst = [tokenizer.encode(ds[0]["distractor"], add_special_tokens=False)[0], tokenizer.encode(ds[1]["distractor"], add_special_tokens=False)[0]]

with torch.no_grad():
    out_b = model(**inputs_b).logits

for i in range(2):
    seq_len = (inputs_b.input_ids[i] != tokenizer.pad_token_id).sum().item()
    last_idx = seq_len - 1
    out_t = out_b[i, last_idx, tgt[i]].item()
    out_d = out_b[i, last_idx, dst[i]].item()
    print(f"[{i}] Batched GT > GD? {out_t > out_d}")

