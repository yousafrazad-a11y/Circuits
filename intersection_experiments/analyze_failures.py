import os
import json
from collections import defaultdict

CATEGORIES = {
    "fruits": ["apple", "banana", "mango", "orange", "grape", "peach", "pear", "plum", "kiwi", "melon", "cherry", "lemon", "lime", "fig", "date", "papaya"],
    "animals": ["cat", "dog", "lion", "tiger", "bear", "wolf", "fox", "deer", "horse", "cow", "pig", "sheep", "goat", "frog", "snake", "bird"],
    "colors": ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "gray", "brown", "cyan", "magenta", "teal", "navy", "maroon"],
    "metals": ["iron", "gold", "silver", "copper", "zinc", "lead", "tin", "nickel", "bronze", "brass", "steel", "aluminum", "platinum", "titanium", "chrome", "cobalt"],
    "vehicles": ["car", "bus", "truck", "train", "plane", "boat", "ship", "bike", "scooter", "van", "jeep", "taxi", "tram", "cart", "wagon", "jet"]
}

def analyze_dataset(cat_name):
    path = f"/home/exouser/pruning/intersection_experiments/results_5way_extreme/annotated_datasets/{cat_name}_annotated.jsonl"
    
    total = 0
    fail_intersect_pass_joint = 0
    
    target_fails = defaultdict(int)
    target_totals = defaultdict(int)
    
    length_fails = defaultdict(int)
    length_totals = defaultdict(int)
    
    word_fails = defaultdict(int)
    word_totals = defaultdict(int)
    
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            total += 1
            
            target = item["target"]
            words = item["clean_prompt"].lower().replace("sequence:", "").split()
            length = len(words)
            
            # Use Generative accuracy for analysis
            pass_intersect = item["passed_acc2_intersect"]
            pass_joint = item["passed_acc2_joint"]
            
            is_target_failure = (not pass_intersect) and pass_joint
            
            if is_target_failure:
                fail_intersect_pass_joint += 1
                
            target_totals[target] += 1
            if is_target_failure: target_fails[target] += 1
                
            length_totals[length] += 1
            if is_target_failure: length_fails[length] += 1
                
            unique_words = set(words)
            for w in unique_words:
                word_totals[w] += 1
                if is_target_failure: word_fails[w] += 1

    # Analysis Results
    print(f"\n{'='*50}")
    print(f"ANALYSIS FOR {cat_name.upper()}")
    print(f"{'='*50}")
    print(f"Total Examples: {total}")
    print(f"Failed Intersect BUT Passed Joint: {fail_intersect_pass_joint} ({fail_intersect_pass_joint/total*100:.1f}%)")
    
    print("\n[Target Item Vulnerability]")
    sorted_targets = sorted([(k, target_fails[k]/target_totals[k]*100, target_fails[k], target_totals[k]) for k in target_totals], key=lambda x: x[1], reverse=True)
    for t, pct, f, tot in sorted_targets[:5]:
        print(f"  - '{t}': {pct:.1f}% failure rate ({f}/{tot})")
        
    print("\n[Prompt Length Vulnerability]")
    sorted_lengths = sorted([(k, length_fails[k]/length_totals[k]*100, length_fails[k], length_totals[k]) for k in length_totals], key=lambda x: x[0])
    for l, pct, f, tot in sorted_lengths:
        print(f"  - Length {l}: {pct:.1f}% failure rate ({f}/{tot})")
        
    print("\n[Word Presence Vulnerability]")
    # Only consider words that appear at least 10 times to avoid noise
    valid_words = [k for k in word_totals if word_totals[k] >= 10]
    sorted_words = sorted([(k, word_fails[k]/word_totals[k]*100, word_fails[k], word_totals[k]) for k in valid_words], key=lambda x: x[1], reverse=True)
    for w, pct, f, tot in sorted_words[:10]:
        print(f"  - When prompt contains '{w}': {pct:.1f}% failure rate ({f}/{tot})")
        
def main():
    for cat in CATEGORIES.keys():
        analyze_dataset(cat)

if __name__ == "__main__":
    main()
