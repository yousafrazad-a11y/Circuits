import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import json
import random

token = "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"
model_name = "Qwen/Qwen2.5-32B"

objects = ["ball", "doll", "book", "pen", "toy", "card", "coin", "ring", "key", "watch", "shoe", "sock", "shirt", "hat", "belt", "cup", "mug", "fork", "spoon", "plate"]
places = ["basket", "box", "bag", "chest", "cart", "truck", "boat", "train", "plane", "tent", "house", "shop", "jar", "safe", "bowl", "drawer", "room", "car", "van", "case"]

v1_list = ["is put in the", "is placed in the", "is dropped in the", "is hidden in the", "is stored in the",
           "is left in the", "is tossed in the", "is tucked in the", "is deposited in the", "is locked in the",
           "is kept in the", "is stowed in the"]

v2_list = ["is moved to the", "is transferred to the", "is taken to the", "is shifted to the", "is carried to the",
           "is transported to the", "is relocated to the", "is conveyed to the", "is dragged to the", "is brought to the",
           "is pushed to the", "is passed to the"]

final_verbs = ["is in the", "is found in the"]

all_prompts = []
random.seed(42)

for v1 in v1_list:
    for v2 in v2_list:
        for fv in final_verbs:
            for i in range(40):
                # Sample 2 unique objects and 4 unique places
                obs = random.sample(objects, 2)
                pls = random.sample(places, 4)
                o1, o2 = obs[0], obs[1]
                p0, p1, p2, p3 = pls[0], pls[1], pls[2], pls[3]
                
                # Normal Clean
                n_clean = f"The {o1} {v1} {p0}. The {o2} {v1} {p1}. The {p0} {v2} {p2}. The {p2} {v2} {p3}. The {o1} {fv}"
                n_clean_target = p3
                
                # Normal Corrupted
                n_corr = f"The {o2} {v1} {p0}. The {o1} {v1} {p1}. The {p0} {v2} {p2}. The {p2} {v2} {p3}. The {o1} {fv}"
                n_corr_target = p1
                
                all_prompts.append({
                    "v1": v1, "v2": v2, "fv": fv, "example_idx": i,
                    "type": "normal_clean", "prompt": n_clean, "target": n_clean_target
                })
                all_prompts.append({
                    "v1": v1, "v2": v2, "fv": fv, "example_idx": i,
                    "type": "normal_corr", "prompt": n_corr, "target": n_corr_target
                })

results = []

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"\nLoading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto", token=token
)

for p in all_prompts:
    prompt = p["prompt"]
    target = p["target"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        top_k = torch.topk(probs, 10)
    
    top_pred = tokenizer.decode([top_k.indices[0].item()]).strip().lower()
    is_correct = (top_pred == target.lower())

    results.append({
        "model": model_name,
        "v1": p["v1"],
        "v2": p["v2"],
        "fv": p["fv"],
        "type": p["type"],
        "example_idx": p["example_idx"],
        "target": target,
        "is_correct": is_correct
    })
    
with open("/home/exouser/pruning/custom/llama_6hop/extended_verbs_results.json", "w") as f:
    json.dump(results, f, indent=4)
