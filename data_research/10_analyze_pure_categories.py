import json

with open("/home/exouser/pruning/custom/llama_6hop/pure_category_results.json", "r") as f:
    results = json.load(f)

for model, data in results.items():
    print(f"\n======================================")
    print(f"MODEL: {model}")
    print(f"======================================")
    
    living_clean = 0
    living_corr = 0
    non_living_clean = 0
    non_living_corr = 0
    
    for item in data:
        cat = "living" if item["id"].startswith("living") else "non_living"
        t = item["type"]
        is_corr = item["is_correct"]
        
        if cat == "living" and t == "clean" and is_corr: living_clean += 1
        if cat == "living" and t == "corr" and is_corr: living_corr += 1
        if cat == "non_living" and t == "clean" and is_corr: non_living_clean += 1
        if cat == "non_living" and t == "corr" and is_corr: non_living_corr += 1
        
        # We can also print the specific result for insight
        if t == "corr":
            target = item['target']
            top1 = item['top_10'][0]['token']
            top1_prob = item['top_10'][0]['prob']
            top2 = item['top_10'][1]['token']
            top2_prob = item['top_10'][1]['prob']
            
            print(f"[{cat.upper()}] Target: {target} | Predicted: {top1} ({top1_prob:.2f}), {top2} ({top2_prob:.2f}) | Correct in top 10: {item['target_in_top_10']}")
            
    print(f"\n--- Accuracy ---")
    print(f"Living Clean: {living_clean}/5")
    print(f"Living Corrupted: {living_corr}/5")
    print(f"Non-Living Clean: {non_living_clean}/5")
    print(f"Non-Living Corrupted: {non_living_corr}/5")
