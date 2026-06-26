import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import random

model_name = "meta-llama/Meta-Llama-3.1-70B"
token = "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"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto",
    token=token
)

print(f"--- Loaded {model_name} in 4-bit ---")

def get_top_k(prompt, question_target):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
        top_k = torch.topk(probs, 5)
        
        print(f"\nPrompt: {prompt}")
        print(f"Top 5 predictions:")
        for i in range(5):
            token_id = top_k.indices[i].item()
            token = tokenizer.decode([token_id])
            prob = top_k.values[i].item()
            print(f"  {i+1}: '{token}' (prob: {prob:.4f})")

# 1. Original Structure (Without Modifications)
print("\n=============================================")
print(" EXPERIMENT 1: Original 'Held By' Structure")
print("=============================================")
base_clean = "The ball is put in the basket. The doll is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The ball is held by the"
base_corrupted = "The doll is put in the basket. The ball is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The ball is held by the"

get_top_k(base_clean, "The ball is held by the")
get_top_k(base_corrupted, "The ball is held by the")

# 2. Location Structure (With Modifications)
print("\n=============================================")
print(" EXPERIMENT 2: Location 'Currently In' Structure")
print("=============================================")
objects = ["ball", "doll", "book", "pen", "toy"]
containers = ["basket", "box", "bag", "chest"]
distractors = ["shelf", "drawer", "desk", "closet"]
locations = ["car", "van", "truck", "garage", "shed", "barn", "tent", "house", "shop", "park"]

def generate_prompt(obj1, obj2, cont, distractor, locs, is_corrupted=False):
    if is_corrupted:
        sent1 = f"The {obj2} is put in the {cont}. "
        sent2 = f"The {obj1} is put in the {distractor}. "
    else:
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

print(f"\nLocation Clean Exact Match Accuracy: {clean_acc}/{num_samples} ({clean_acc/num_samples*100:.1f}%)")
print(f"Location Corrupted Exact Match Accuracy: {corr_acc}/{num_samples} ({corr_acc/num_samples*100:.1f}%)")
