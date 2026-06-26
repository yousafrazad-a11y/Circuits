import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# Clean prompt but asking for the DOLL
prompt = "The ball is put in the basket. The doll is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The doll is held by the"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    top_k = torch.topk(probs, 10)
    
    print("Prompt:", prompt)
    print("\nTop 10 predictions for 'The doll is held by the':")
    for i in range(10):
        token_id = top_k.indices[i].item()
        token = tokenizer.decode([token_id])
        prob = top_k.values[i].item()
        print(f"  {i+1}: '{token}' (prob: {prob:.4f})")
