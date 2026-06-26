import json

file_path = "/home/exouser/pruning/custom/llama_6hop/detailed_prompts_and_logits.json"

with open(file_path, "r") as f:
    data = json.load(f)

# Flatten into a single list of objects, injecting the model name
flattened_data = []
for model_name, results_list in data.items():
    for entry in results_list:
        # Create a new dict with 'model' as the first key
        new_entry = {"model": model_name}
        new_entry.update(entry)
        flattened_data.append(new_entry)

with open(file_path, "w") as f:
    json.dump(flattened_data, f, indent=4)

print("Successfully flattened JSON and injected model names!")
