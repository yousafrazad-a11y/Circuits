import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import json

token = "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"
models = [
    ("meta-llama/Llama-3.2-1B", False),
    ("meta-llama/Meta-Llama-3.1-8B", True),
    ("Qwen/Qwen2.5-32B", True)
]

living_actors = ["mom", "dad", "kitty", "puppy", "friend", "grandma", "uncle", "aunt", "boy", "girl", "teacher", "student"]
non_living = ["basket", "box", "bag", "chest", "cart", "truck", "boat", "train", "plane", "tent", "house", "shop"]
objects = ["ball", "doll", "book", "pen", "toy"]

# Generate 5 Fixed Living Pairs (3 Hops)
living_prompts = []
for i in range(5):
    o1, o2 = objects[i % 5], objects[(i+1) % 5]
    actors = living_actors[i:i+5] # We need 5 actors total: 2 initial + 3 passes
    
    # 3 hops: actors[0] -> actors[2] -> actors[3] -> actors[4]
    clean = f"The {o1} is given to the {actors[0]}. The {o2} is given to the {actors[1]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {actors[3]} hands it to the {actors[4]}. The {o1} is held by the"
    c_target = actors[4]
    
    corr = f"The {o2} is given to the {actors[0]}. The {o1} is given to the {actors[1]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {actors[3]} hands it to the {actors[4]}. The {o1} is held by the"
    corr_target = actors[1]
    
    living_prompts.append({
        "id": f"living_{i+1}",
        "clean": clean, "clean_target": c_target,
        "corr": corr, "corr_target": corr_target
    })

# Generate 5 Fixed Non-Living Pairs (3 Hops)
non_living_prompts = []
for i in range(5):
    o1, o2 = objects[i % 5], objects[(i+1) % 5]
    places = non_living[i:i+5]
    
    # 3 hops: places[0] -> places[2] -> places[3] -> places[4]
    clean = f"The {o1} is put in the {places[0]}. The {o2} is put in the {places[1]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {places[3]} is moved to the {places[4]}. The {o1} is located in the"
    c_target = places[4]
    
    corr = f"The {o2} is put in the {places[0]}. The {o1} is put in the {places[1]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {places[3]} is moved to the {places[4]}. The {o1} is located in the"
    corr_target = places[1]
    
    non_living_prompts.append({
        "id": f"non_living_{i+1}",
        "clean": clean, "clean_target": c_target,
        "corr": corr, "corr_target": corr_target
    })

all_prompts = living_prompts + non_living_prompts
results = {}

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

for model_name, use_4bit in models:
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    
    if use_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, quantization_config=quantization_config, device_map="auto", token=token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, token=token
        ).to("cuda")
    
    results[model_name] = []
    
    for p in all_prompts:
        for p_type in ["clean", "corr"]:
            prompt = p[p_type]
            target = p[f"{p_type}_target"]
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
                top_k = torch.topk(probs, 10)
            
            top_10 = []
            target_in_top_10 = False
            
            for rank in range(10):
                token_id = top_k.indices[rank].item()
                decoded = tokenizer.decode([token_id])
                prob = top_k.values[rank].item()
                top_10.append({"token": decoded, "prob": prob})
                
                if target.lower() in decoded.lower() and len(decoded.strip()) > 1:
                    target_in_top_10 = True
                        
            top_pred = tokenizer.decode([top_k.indices[0].item()]).strip().lower()
            is_correct = (top_pred == target.lower())

            results[model_name].append({
                "id": p["id"],
                "type": p_type,
                "target": target,
                "is_correct": is_correct,
                "target_in_top_10": target_in_top_10,
                "top_10": top_10
            })
            
    # Cleanup memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

with open("/home/exouser/pruning/custom/llama_6hop/3_hop_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print Summary
for model, data in results.items():
    print(f"\n======================================")
    print(f"MODEL: {model}")
    print(f"======================================")
    
    living_clean = 0
    living_corr = 0
    non_living_clean = 0
    non_living_corr = 0
    
    for item in data:
        cat = "living" if item["id"].startswith("living") else "non_living"
        t = item["type"]
        is_corr = item["is_correct"]
        
        if cat == "living" and t == "clean" and is_corr: living_clean += 1
        if cat == "living" and t == "corr" and is_corr: living_corr += 1
        if cat == "non_living" and t == "clean" and is_corr: non_living_clean += 1
        if cat == "non_living" and t == "corr" and is_corr: non_living_corr += 1
        
    print(f"Living Clean: {living_clean}/5")
    print(f"Living Corrupted: {living_corr}/5")
    print(f"Non-Living Clean: {non_living_clean}/5")
    print(f"Non-Living Corrupted: {non_living_corr}/5")
