import json

with open('/home/exouser/pruning/datasets/dataset3_results_qwen_coder_32b.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

failures = [r for r in results if not r['combined_pass']]
print(f"Total failures: {len(failures)} out of {len(results)}")

for i in range(10):
    f = failures[i]
    print(f"CLEAN PROMPT: {f['clean_prompt']}")
    print(f"  TARGET: '{f['clean_target']}'")
    print(f"  PREDICTION: {f.get('clean_pred_top1', 'N/A')}")
    print(f"CORRUPT PROMPT: {f['corrupted_prompt']}")
    print(f"  TARGET: '{f['corrupted_target']}'")
    print(f"  PREDICTION: {f.get('corrupted_pred_top1', 'N/A')}")
    print("-" * 40)
