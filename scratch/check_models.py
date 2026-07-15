import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "circuit_pruning-argo"))
from models.llama_circuit import PrunableLlamaForCausalLM, PruningConfig

def main():
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    config = PruningConfig(prune_attention_heads=True, lambda_attention_heads=0.8)
    model = PrunableLlamaForCausalLM.from_pretrained_with_pruning("meta-llama/Llama-3.2-1B", pruning_config=config, torch_dtype=torch.bfloat16).to(device)
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.bfloat16).to(device)
    
    print(f"Model types: {type(model)} vs {type(base_model)}")
    
    prompt = "Sequence: apple banana mango apple banana"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    out_base = base_model(**inputs).logits[0, -1, :10]
    out_circ = model(**inputs).logits[0, -1, :10]
    
    print("Base logits:", out_base)
    print("Unpruned Circ logits:", out_circ)
    
    # Apply a mask that prunes almost everything
    model.set_final_circuit_mode(True)
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                module.log_alpha.data.fill_(-1e6)
                
    out_pruned = model(**inputs).logits[0, -1, :10]
    print("Fully Pruned Circ logits:", out_pruned)

if __name__ == "__main__":
    main()
