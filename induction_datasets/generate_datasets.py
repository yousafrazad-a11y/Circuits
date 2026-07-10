import json
import random
from pathlib import Path

# Common English first and last names for Dataset 1
FIRST_NAMES = [
    "John", "James", "David", "Robert", "Michael", "William", "Richard", "Joseph", "Charles", "Thomas",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
    "Christopher", "Daniel", "Paul", "Mark", "Donald", "George", "Kenneth", "Steven", "Edward", "Brian",
    "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

# Fictional languages and nonsense words for Dataset 2
LANGUAGES = ["Zog", "Blip", "Krell", "Vorn", "Plor", "Snip", "Glar", "Mork", "Bort", "Quox"]
NONSENSE_WORDS = ["glib", "florp", "zazz", "plumb", "snarf", "blonk", "crim", "darg", "farn", "grom"]
REAL_WORDS = ["fast", "slow", "big", "small", "hot", "cold", "good", "bad", "happy", "sad", "red", "blue"]

def generate_dataset_1(num_samples: int, output_file: Path):
    """
    Dataset 1: Name Binding (Pure Induction)
    Clean: "Mr. John Smith went to the store. The manager asked for his name, and he replied 'John" -> Smith
    Corr A (Value corrupt): "Mr. John Miller went to the store. The manager asked for his name, and he replied 'John" -> (prob of Smith drops)
    Corr B (Query corrupt): "Mr. David Smith went to the store. The manager asked for his name, and he replied 'John" -> (prob of Smith drops)
    """
    samples = []
    
    # We need a stable random seed to reproduce the exact same datasets
    rng = random.Random(42)
    
    for _ in range(num_samples):
        # Pick names
        first = rng.choice(FIRST_NAMES)
        target_last = rng.choice(LAST_NAMES)
        
        # Corrupted names
        corr_a_last = rng.choice([n for n in LAST_NAMES if n != target_last])
        corr_b_first = rng.choice([n for n in FIRST_NAMES if n != first])
        
        # Decide title
        title = "Mr." if first in FIRST_NAMES[:10] + FIRST_NAMES[20:30] else "Ms."
        
        clean = f"{title} {first} {target_last} went to the store. The manager asked for a name, and the reply was '{first}"
        corr_a = f"{title} {first} {corr_a_last} went to the store. The manager asked for a name, and the reply was '{first}"
        corr_b = f"{title} {corr_b_first} {target_last} went to the store. The manager asked for a name, and the reply was '{first}"
        
        samples.append({
            "clean_prompt": clean,
            "corr_a_prompt": corr_a,
            "corr_b_prompt": corr_b,
            "target": target_last,
            "distractor_a": corr_a_last,
            "distractor_b": "N/A"
        })
        
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {num_samples} to {output_file}")


def generate_dataset_2(num_samples: int, output_file: Path):
    """
    Dataset 2: Nonsense Word Translation (Applied Induction)
    Clean: "In the fictional language of Zog, 'glib' translates to 'fast'. So the word 'glib' translates to '" -> fast
    Corr A (Value corrupt): "In the fictional language of Zog, 'glib' translates to 'slow'. So the word 'glib' translates to '" -> (prob of fast drops)
    Corr B (Query corrupt): "In the fictional language of Zog, 'florp' translates to 'fast'. So the word 'glib' translates to '" -> (prob of fast drops)
    """
    samples = []
    rng = random.Random(42)
    
    for _ in range(num_samples):
        lang = rng.choice(LANGUAGES)
        target_nonsense = rng.choice(NONSENSE_WORDS)
        target_real = rng.choice(REAL_WORDS)
        
        # For Corr A, change the real word
        corr_a_real = rng.choice([w for w in REAL_WORDS if w != target_real])
        
        # For Corr B, change the nonsense word in the context
        corr_b_nonsense = rng.choice([w for w in NONSENSE_WORDS if w != target_nonsense])
        
        clean = f"In the fictional language of {lang}, '{target_nonsense}' translates to '{target_real}'. So the word '{target_nonsense}' translates to '"
        corr_a = f"In the fictional language of {lang}, '{target_nonsense}' translates to '{corr_a_real}'. So the word '{target_nonsense}' translates to '"
        corr_b = f"In the fictional language of {lang}, '{corr_b_nonsense}' translates to '{target_real}'. So the word '{target_nonsense}' translates to '"
        
        samples.append({
            "clean_prompt": clean,
            "corr_a_prompt": corr_a,
            "corr_b_prompt": corr_b,
            "target": target_real,
            "distractor_a": corr_a_real,
            "distractor_b": "N/A"
        })
        
    with open(output_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {num_samples} to {output_file}")


if __name__ == "__main__":
    out_dir = Path(__file__).parent
    generate_dataset_1(500, out_dir / "dataset1_names.jsonl")
    generate_dataset_2(500, out_dir / "dataset2_nonsense.jsonl")
