import json
import glob
from pathlib import Path

def main():
    datasets_dir = Path("exp2full_heads_mlp_neurons/datasets")
    all_categories = {}
    
    for file_path in datasets_dir.glob("*.jsonl"):
        cat_name = file_path.stem
        targets = set()
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                targets.add(data["target"])
        all_categories[cat_name] = list(targets)
        
    with open("exp2full_heads_mlp_neurons/categories.json", "w") as f:
        json.dump(all_categories, f, indent=4)
        
    print("Categories saved to exp2full_heads_mlp_neurons/categories.json")

if __name__ == "__main__":
    main()
