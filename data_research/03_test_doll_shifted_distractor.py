import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Corrupted Prompt: The ball goes to the shelf, but we want to see if changing the question helps.
base_corrupted = "The doll is put in the basket. The ball is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. "

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

# Test various question endings
get_top_k(base_corrupted + "The ball is held by the", "The ball is held by the")
get_top_k(base_corrupted + "The ball is with the", "The ball is with the")
get_top_k(base_corrupted + "The ball is currently in the", "The ball is currently in the")
get_top_k(base_corrupted + "The ball is located in the", "The ball is located in the")
get_top_k(base_corrupted + "The ball was left in the", "The ball was left in the")


