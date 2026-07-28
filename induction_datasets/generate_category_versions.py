import json
import random
from pathlib import Path

# Same categories as data_research/generate_chain_datasets.py
CATEGORIES = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "pear", "plum", "kiwi", "melon", "cherry", "lemon", "lime", "fig", "date", "papaya"],
    "animals": ["cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow", "pig", "sheep", "goat", "frog", "snake", "bird"],
    "colors": ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "gray", "brown", "cyan", "magenta", "teal", "navy", "maroon"],
    "metals": ["iron", "gold", "silver", "copper", "zinc", "lead", "tin", "nickel", "bronze", "brass", "steel", "aluminum", "platinum", "titanium", "chrome", "cobalt"],
    "vehicles": ["car", "bus", "truck", "train", "plane", "boat", "ship", "bike", "scooter", "van", "jeep", "taxi", "tram", "cart", "wagon", "jet"],
    "instruments": ["piano", "guitar", "violin", "drums", "flute", "trumpet", "cello", "harp", "banjo", "clarinet", "oboe", "trombone", "tuba", "accordion", "mandolin", "saxophone"],
    "sports": ["soccer", "tennis", "golf", "hockey", "boxing", "skiing", "surfing", "cycling", "swimming", "baseball", "football", "cricket", "rugby", "volleyball", "badminton", "archery"],
    "professions": ["doctor", "teacher", "lawyer", "farmer", "baker", "nurse", "pilot", "chef", "writer", "singer", "dancer", "painter", "plumber", "carpenter", "mechanic", "scientist"],
    "clothing": ["shirt", "pants", "dress", "skirt", "jacket", "coat", "socks", "shoes", "boots", "hat", "gloves", "scarf", "sweater", "hoodie", "jeans", "shorts"],
    "furniture": ["table", "chair", "sofa", "bed", "desk", "lamp", "shelf", "stool", "bench", "dresser", "cabinet", "wardrobe", "couch", "mirror", "drawer", "bookcase"]
}

NUM_VERSIONS = 3
SAMPLES_PER_VERSION = 500

def generate_versions(words, num_versions=NUM_VERSIONS, samples_per_version=SAMPLES_PER_VERSION, seed=42):
    """
    Generate num_versions datasets of samples_per_version each.
    Chains (a, b, c) are deduplicated across ALL versions, so no clean_prompt
    (and therefore no example) appears in more than one version.
    16 words -> 16*15*14 = 3360 possible chains >= 1500 needed.
    """
    rng = random.Random(seed)
    seen = set()  # chains used in ANY version
    seen_prompts = set()  # prompt strings (clean or corr) used in ANY version
    versions = []
    for _ in range(num_versions):
        samples = []
        while len(samples) < samples_per_version:
            a, b, c = rng.sample(words, 3)
            x, y, z = rng.sample(words, 3)
            if (a, b, c) in seen:
                continue
            clean_prompt = f"Sequence: {a} {b} {c} {a} {b}"
            corr_prompt = f"Sequence: {a} {b} {x} {y} {z}"
            if clean_prompt in seen_prompts or corr_prompt in seen_prompts:
                continue
            seen.add((a, b, c))
            seen_prompts.add(clean_prompt)
            seen_prompts.add(corr_prompt)
            samples.append({
                "clean_prompt": clean_prompt,
                "corr_prompt": corr_prompt,
                "target": c,
                "distractor": x
            })
        versions.append(samples)
    return versions

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "category_chains"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat_name, words in CATEGORIES.items():
        versions = generate_versions(words)
        for i, samples in enumerate(versions, start=1):
            out_path = out_dir / f"{cat_name}_{i}.jsonl"
            with open(out_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
        print(f"Generated {NUM_VERSIONS}x{SAMPLES_PER_VERSION} samples for {cat_name} "
              f"-> {out_dir}/{cat_name}_1..{NUM_VERSIONS}.jsonl")
