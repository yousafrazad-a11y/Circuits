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

all_prompts = []

# Generate 2-Hop Normal Prompts
for i in range(5):
    o1, o2 = objects[i % 5], objects[(i+1) % 5]
    actors = living_actors[i:i+4] # 4 actors total for 2 hops: 2 initial + 2 passes
    places = non_living[i:i+4]
    
    # 2 hops: actors[0] -> actors[2] -> actors[3]
    l_clean = f"The {o1} is given to the {actors[0]}. The {o2} is given to the {actors[1]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {o1} is held by the"
    l_corr = f"The {o2} is given to the {actors[0]}. The {o1} is given to the {actors[1]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {o1} is held by the"
    
    nl_clean = f"The {o1} is put in the {places[0]}. The {o2} is put in the {places[1]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {o1} is located in the"
    nl_corr = f"The {o2} is put in the {places[0]}. The {o1} is put in the {places[1]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {o1} is located in the"
    
    all_prompts.append({"id": f"living_normal_{i+1}", "clean": l_clean, "clean_target": actors[3], "corr": l_corr, "corr_target": actors[1]})
    all_prompts.append({"id": f"non_living_normal_{i+1}", "clean": nl_clean, "clean_target": places[3], "corr": nl_corr, "corr_target": places[1]})

# Generate 2-Hop Shifted Prompts
for i in range(5):
    o1, o2 = objects[i % 5], objects[(i+1) % 5]
    actors = living_actors[i:i+4]
    places = non_living[i:i+4]
    
    l_clean = f"The {o1} is given to the {actors[0]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {o2} is given to the {actors[1]}. The {o1} is held by the"
    l_corr = f"The {o2} is given to the {actors[0]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {o1} is given to the {actors[1]}. The {o1} is held by the"
    
    nl_clean = f"The {o1} is put in the {places[0]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {o2} is put in the {places[1]}. The {o1} is located in the"
    nl_corr = f"The {o2} is put in the {places[0]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {o1} is put in the {places[1]}. The {o1} is located in the"
    
    all_prompts.append({"id": f"living_shifted_{i+1}", "clean": l_clean, "clean_target": actors[3], "corr": l_corr, "corr_target": actors[1]})
    all_prompts.append({"id": f"non_living_shifted_{i+1}", "clean": nl_clean, "clean_target": places[3], "corr": nl_corr, "corr_target": places[1]})

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
            
            top_pred = tokenizer.decode([top_k.indices[0].item()]).strip().lower()
            is_correct = (top_pred == target.lower())

            results[model_name].append({
                "id": p["id"],
                "type": p_type,
                "target": target,
                "is_correct": is_correct
            })
            
    # Cleanup memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

with open("/home/exouser/pruning/custom/llama_6hop/2_hop_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Print Summary
for model, data in results.items():
    print(f"\n======================================")
    print(f"MODEL: {model}")
    print(f"======================================")
    
    stats = {
        "living_normal_clean": 0, "living_normal_corr": 0,
        "non_living_normal_clean": 0, "non_living_normal_corr": 0,
        "living_shifted_clean": 0, "living_shifted_corr": 0,
        "non_living_shifted_clean": 0, "non_living_shifted_corr": 0,
    }
    
    for item in data:
        cat_id = item["id"] # e.g. living_normal_1
        # extract category
        cat_parts = cat_id.split("_")
        if cat_parts[0] == "non":
            base_cat = "non_living_" + cat_parts[2]
        else:
            base_cat = cat_parts[0] + "_" + cat_parts[1]
            
        key = base_cat + "_" + item["type"]
        if item["is_correct"]:
            stats[key] += 1
            
    print("--- Normal Prompts ---")
    print(f"Living Clean: {stats['living_normal_clean']}/5")
    print(f"Living Corrupted: {stats['living_normal_corr']}/5")
    print(f"Non-Living Clean: {stats['non_living_normal_clean']}/5")
    print(f"Non-Living Corrupted: {stats['non_living_normal_corr']}/5")
    
    print("\n--- Shifted Prompts ---")
    print(f"Living Clean: {stats['living_shifted_clean']}/5")
    print(f"Living Corrupted: {stats['living_shifted_corr']}/5")
    print(f"Non-Living Clean: {stats['non_living_shifted_clean']}/5")
    print(f"Non-Living Corrupted: {stats['non_living_shifted_corr']}/5")
