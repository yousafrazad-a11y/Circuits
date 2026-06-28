import json

with open('/home/exouser/pruning/datasets/dataset3_results_qwen_coder_32b.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

failures = [r for r in results if not r['combined_pass']]
print(f"Total failures: {len(failures)} out of {len(results)}")

for i in range(3):
    f = failures[i]
    print(f"CLEAN PROMPT: {f['clean_prompt']} -> TARGET: '{f['clean_target']}'")
    print(f"CORRUPT PROMPT: {f['corrupted_prompt']} -> TARGET: '{f['corrupted_target']}'")
    print("-" * 40)
