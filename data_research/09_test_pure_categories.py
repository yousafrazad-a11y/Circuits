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

# Generate 5 Fixed Living Pairs
living_prompts = []
for i in range(5):
    o1, o2 = objects[i % 5], objects[(i+1) % 5]
    actors = living_actors[i:i+6]
    
    clean = f"The {o1} is given to the {actors[0]}. The {o2} is given to the {actors[1]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {actors[3]} hands it to the {actors[4]}. The {actors[4]} hands it to the {actors[5]}. The {o1} is held by the"
    c_target = actors[5]
    
    corr = f"The {o2} is given to the {actors[0]}. The {o1} is given to the {actors[1]}. The {actors[0]} hands it to the {actors[2]}. The {actors[2]} hands it to the {actors[3]}. The {actors[3]} hands it to the {actors[4]}. The {actors[4]} hands it to the {actors[5]}. The {o1} is held by the"
    corr_target = actors[1]
    
    living_prompts.append({
        "id": f"living_{i+1}",
        "clean": clean, "clean_target": c_target,
        "corr": corr, "corr_target": corr_target
    })

# Generate 5 Fixed Non-Living Pairs
non_living_prompts = []
for i in range(5):
    o1, o2 = objects[i % 5], objects[(i+1) % 5]
    places = non_living[i:i+6]
    
    clean = f"The {o1} is put in the {places[0]}. The {o2} is put in the {places[1]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {places[3]} is moved to the {places[4]}. The {places[4]} is moved to the {places[5]}. The {o1} is located in the"
    c_target = places[5]
    
    corr = f"The {o2} is put in the {places[0]}. The {o1} is put in the {places[1]}. The {places[0]} is moved to the {places[2]}. The {places[2]} is moved to the {places[3]}. The {places[3]} is moved to the {places[4]}. The {places[4]} is moved to the {places[5]}. The {o1} is located in the"
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
            is_correct = False
            
            for rank in range(10):
                token_id = top_k.indices[rank].item()
                decoded = tokenizer.decode([token_id])
                prob = top_k.values[rank].item()
                top_10.append({"token": decoded, "prob": prob})
                
                if target.lower() in decoded.lower() and len(decoded.strip()) > 1:
                    target_in_top_10 = True
                    if rank == 0:
                        is_correct = True
                        
            # If the tokenizer adds a space before the word, we want to match it flexibly.
            # We did a simple substring match: target in decoded
            
            # Recalculate is_correct more robustly
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

with open("/home/exouser/pruning/custom/llama_6hop/pure_category_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("\nDone! Results saved to pure_category_results.json")
