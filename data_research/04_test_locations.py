import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

objects = ["ball", "doll", "book", "pen", "toy", "coin", "ring", "key", "card", "bone"]
containers = ["basket", "box", "bag", "chest", "purse", "backpack", "bucket", "jar", "case", "tray"]
distractors = ["shelf", "drawer", "desk", "closet", "cabinet", "safe", "bin", "locker", "fridge", "oven"]
locations = ["car", "van", "truck", "garage", "shed", "barn", "tent", "house", "shop", "park", "yard", "boat", "train", "plane"]

def generate_prompt(obj1, obj2, cont, distractor, locs, is_corrupted=False):
    if is_corrupted:
        # In corrupted, obj2 goes to container (which moves), obj1 goes to distractor (which stays)
        sent1 = f"The {obj2} is put in the {cont}. "
        sent2 = f"The {obj1} is put in the {distractor}. "
    else:
        # In clean, obj1 goes to container (which moves), obj2 goes to distractor
        sent1 = f"The {obj1} is put in the {cont}. "
        sent2 = f"The {obj2} is put in the {distractor}. "

    chain = f"The {cont} is moved to the {locs[0]}. "
    for i in range(len(locs)-1):
        chain += f"The {locs[i]} is moved to the {locs[i+1]}. "
        
    prompt = sent1 + sent2 + chain + f"The {obj1} is currently in the"
    return prompt

clean_acc = 0
corr_acc = 0
num_samples = 20

print("Testing Location-Based Prompts\n")

for i in range(num_samples):
    o1, o2 = random.sample(objects, 2)
    cont = random.choice(containers)
    dist = random.choice(distractors)
    locs = random.sample(locations, 6)

    clean_prompt = generate_prompt(o1, o2, cont, dist, locs, is_corrupted=False)
    corr_prompt = generate_prompt(o1, o2, cont, dist, locs, is_corrupted=True)

    clean_target = locs[-1]
    corr_target = dist

    inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    inputs_corr = tokenizer(corr_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_clean = model(**inputs_clean).logits[0, -1, :]
        out_corr = model(**inputs_corr).logits[0, -1, :]

    pred_clean = tokenizer.decode([torch.argmax(out_clean).item()]).strip()
    pred_corr = tokenizer.decode([torch.argmax(out_corr).item()]).strip()

    if pred_clean.lower() == clean_target.lower(): clean_acc += 1
    if pred_corr.lower() == corr_target.lower(): corr_acc += 1

    if i < 3: # Print first 3 for inspection
        print(f"--- Example {i+1} ---")
        print(f"Clean Prompt: {clean_prompt}")
        print(f"  Target: {clean_target} | Pred: '{pred_clean}'")
        print(f"Corr Prompt: {corr_prompt}")
        print(f"  Target: {corr_target} | Pred: '{pred_corr}'\n")

print(f"Clean Exact Match Accuracy: {clean_acc}/{num_samples} ({clean_acc/num_samples*100:.1f}%)")
print(f"Corrupted Exact Match Accuracy: {corr_acc}/{num_samples} ({corr_acc/num_samples*100:.1f}%)")
