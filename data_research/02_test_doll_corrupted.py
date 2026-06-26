import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

prompt = "The doll is put in the basket. The ball is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The ball is held by the"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    top_k = torch.topk(probs, 10)
    
    print("Corrupted Prompt:", prompt)
    print("\nTop 10 predictions for 'The ball is held by the':")
    for i in range(10):
        token_id = top_k.indices[i].item()
        token = tokenizer.decode([token_id])
        prob = top_k.values[i].item()
        print(f"  {i+1}: '{token}' (prob: {prob:.4f})")

    # Also check the clean prompt just to be sure
    clean_prompt = "The ball is put in the basket. The doll is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The ball is held by the"
    inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
    clean_outputs = model(**inputs_clean)
    clean_probs = torch.softmax(clean_outputs.logits[0, -1, :], dim=-1)
    clean_top_k = torch.topk(clean_probs, 5)
    
    print("\nClean Prompt:", clean_prompt)
    print("\nTop 5 predictions for 'The ball is held by the':")
    for i in range(5):
        token_id = clean_top_k.indices[i].item()
        token = tokenizer.decode([token_id])
        prob = clean_top_k.values[i].item()
        print(f"  {i+1}: '{token}' (prob: {prob:.4f})")
