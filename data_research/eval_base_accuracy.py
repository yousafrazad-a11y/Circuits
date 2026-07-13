import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--task", type=str, default="induction_datasets/category_chains/fruits.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    print(f"Evaluating {args.task}...")
    
    with open(args.task, "r") as f:
        data = [json.loads(line) for line in f]

    clean_correct = 0
    corr_correct = 0
    total = len(data)

    for item in data:
        # Some datasets use clean_prompt/corr_prompt, some use prompt/corrupted_prompt
        clean_p = item.get("clean_prompt", item.get("prompt"))
        corr_p = item.get("corr_prompt", item.get("corrupted_prompt"))
        
        target = item.get("target", item.get("clean_target"))
        distractor = item.get("distractor", item.get("corrupted_target"))
        
        target_token = tokenizer.encode(target, add_special_tokens=False)[0]
        distractor_token = tokenizer.encode(distractor, add_special_tokens=False)[0]

        # Clean Evaluation
        inputs = tokenizer(clean_p, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        if logits[target_token] > logits[distractor_token]:
            clean_correct += 1

        # Corrupted Evaluation
        inputs = tokenizer(corr_p, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        if logits[target_token] > logits[distractor_token]:
            corr_correct += 1
            
    print(f"Total Samples: {total}")
    print(f"Clean Accuracy: {clean_correct/total*100:.2f}% ({clean_correct}/{total})")
    print(f"Corrupted Accuracy: {corr_correct/total*100:.2f}% ({corr_correct}/{total})")

if __name__ == "__main__":
    main()
