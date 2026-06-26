import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-32B"
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

# Clean Prompt: Distractor sentence moved to the end
clean_prompt = "The ball is put in the basket. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The doll is put in the shelf. The ball is held by the"

# Corrupted Prompt: Distractor sentence moved to the end
corrupted_prompt = "The doll is put in the basket. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The ball is put in the shelf. The ball is held by the"

print("\n--- Shifted Prompts ---")
get_top_k(clean_prompt, "The ball is held by the")
get_top_k(corrupted_prompt, "The ball is held by the")

# Let's also check what happens if we ask about the doll
clean_prompt_doll = "The ball is put in the basket. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. The doll is put in the shelf. The doll is held by the"
get_top_k(clean_prompt_doll, "The doll is held by the")

print("\n--- Question Endings ---")
base_corrupted = "The doll is put in the basket. The ball is put in the shelf. The basket is given to the mom. The mom hands it to the dad. The dad hands it to the kitty. The kitty hands it to the puppy. The puppy hands it to the friend. The friend hands it to the grandma. "

get_top_k(base_corrupted + "The ball is held by the", "The ball is held by the")
get_top_k(base_corrupted + "The ball is with the", "The ball is with the")
get_top_k(base_corrupted + "The ball is currently in the", "The ball is currently in the")
get_top_k(base_corrupted + "The ball is located in the", "The ball is located in the")
get_top_k(base_corrupted + "The ball was left in the", "The ball was left in the")
