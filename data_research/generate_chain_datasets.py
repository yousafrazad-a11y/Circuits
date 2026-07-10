import json
import random
from pathlib import Path
import os

CATEGORIES = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "pear", "plum", "kiwi", "melon", "cherry", "lemon", "lime", "fig", "date", "papaya"],
    "animals": ["cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow", "pig", "sheep", "goat", "frog", "snake", "bird"],
    "colors": ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "gray", "brown", "cyan", "magenta", "teal", "navy", "maroon"],
    "metals": ["iron", "gold", "silver", "copper", "zinc", "lead", "tin", "nickel", "bronze", "brass", "steel", "aluminum", "platinum", "titanium", "chrome", "cobalt"],
    "vehicles": ["car", "bus", "truck", "train", "plane", "boat", "ship", "bike", "scooter", "van", "jeep", "taxi", "tram", "cart", "wagon", "jet"]
}

def generate_samples(words, num_samples=500):
    samples = []
    seen = set()
    while len(samples) < num_samples:
        # Pick 3 unique items for the chain: A, B, C
        a, b, c = random.sample(words, 3)
        # Pick 3 unique distractors for corruption: X, Y, Z
        x, y, z = random.sample(words, 3)
        
        # Ensure x, y, z don't accidentally reconstruct the chain
        if (a, b, c) in seen:
            continue
        seen.add((a, b, c))

        # Clean: Sequence: A B C A B -> C
        # Corrupted: Sequence: A B X Y Z -> ?
        clean_prompt = f"Sequence: {a} {b} {c} {a} {b}"
        corr_prompt = f"Sequence: {a} {b} {x} {y} {z}"

        samples.append({
            "clean_prompt": clean_prompt,
            "corr_prompt": corr_prompt,
            "target": c,
            "distractor": x  # The incorrect target for the corrupted chain
        })
    return samples

if __name__ == "__main__":
    out_dir = Path("/home/exouser/pruning/induction_datasets/category_chains")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat_name, words in CATEGORIES.items():
        samples = generate_samples(words, 500)
        out_path = out_dir / f"{cat_name}.jsonl"
        with open(out_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Generated 500 samples for {cat_name} -> {out_path}")
