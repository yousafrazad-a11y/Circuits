"""Sample 10 random items from the arithmetic chain data and generate
model outputs for both the clean and corrupted prompts, so they can be
inspected side by side.

Usage:
    python generate_samples.py [--split test|train] [--n 10] [--seed 0]
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
HERE = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["test", "train"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args()

    data_path = HERE / f"arithmetic_chain_{args.split}.jsonl"
    out_path = Path(args.out) if args.out else HERE / f"generations_{args.split}.json"

    data = load_jsonl(data_path)
    rng = random.Random(args.seed)
    samples = rng.sample(data, k=min(args.n, len(data)))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    results = []
    for item in samples:
        clean_gen = generate(model, tokenizer, item["clean_prompt"], args.max_new_tokens)
        corrupt_gen = generate(model, tokenizer, item["corrupted_prompt"], args.max_new_tokens)
        results.append({
            "id": item["id"],
            "clean_prompt": item["clean_prompt"],
            "clean_generation": clean_gen,
            "corrupted_prompt": item["corrupted_prompt"],
            "corrupted_generation": corrupt_gen,
            "gold_output": item["gold_output"],
            "final_answer": item["final_answer"],
            "operand_mapping": item["operand_mapping"],
        })
        print(f"[{item['id']}] clean={clean_gen[:60]!r} corrupt={corrupt_gen[:60]!r}")

    with out_path.open("w") as f:
        json.dump({"model": MODEL_ID, "split": args.split, "seed": args.seed,
                   "samples": results}, f, indent=2)
    print(f"\nWrote {len(results)} samples to {out_path}")


if __name__ == "__main__":
    main()
