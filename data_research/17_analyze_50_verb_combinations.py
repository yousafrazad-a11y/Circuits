import json

with open("/home/exouser/pruning/custom/llama_6hop/internal_verbs_results.json", "r") as f:
    data = json.load(f)

# Structure to hold stats:
# stats[model][v1][v2][fv][type] = {"correct": 0, "total": 0}

stats = {}

for item in data:
    model = item["model"]
    v1 = item["v1"]
    v2 = item["v2"]
    fv = item["fv"]
    t = item["type"]
    
    if model not in stats:
        stats[model] = {}
    if v1 not in stats[model]:
        stats[model][v1] = {}
    if v2 not in stats[model][v1]:
        stats[model][v1][v2] = {}
    if fv not in stats[model][v1][v2]:
        stats[model][v1][v2][fv] = {}
    if t not in stats[model][v1][v2][fv]:
        stats[model][v1][v2][fv][t] = {"correct": 0, "total": 0}
        
    stats[model][v1][v2][fv][t]["total"] += 1
    if item["is_correct"]:
        stats[model][v1][v2][fv][t]["correct"] += 1

# Let's generate a markdown report
md_content = "# Internal Verbs Experiment Summary\n\n"
md_content += "In this experiment, we tested variations of internal movement verbs (`v1` and `v2`) while locking the final question verb (`fv`) to either `is in the` or `is found in the`. Only 2-hop chains for non-living objects were evaluated.\n\n"

for model in stats:
    md_content += f"## Model: {model}\n\n"
    
    for fv in stats[model][list(stats[model].keys())[0]][list(stats[model][list(stats[model].keys())[0]].keys())[0]]:
        md_content += f"### Final Verb: `{fv}`\n\n"
        md_content += "| `v1` (Initial Placement) | `v2` (Movement) | Normal Clean | Normal Corr | Shifted Clean | Shifted Corr |\n"
        md_content += "|:---|:---|:---|:---|:---|:---|\n"
        
        for v1 in stats[model]:
            for v2 in stats[model][v1]:
                res = stats[model][v1][v2][fv]
                
                def get_acc(t):
                    if res[t]["total"] == 0: return "0%"
                    return f"{(res[t]['correct'] / res[t]['total']) * 100:.0f}%"
                    
                nc = get_acc("normal_clean")
                nco = get_acc("normal_corr")
                sc = get_acc("shifted_clean")
                sco = get_acc("shifted_corr")
                
                md_content += f"| `{v1}` | `{v2}` | {nc} | {nco} | {sc} | {sco} |\n"
        md_content += "\n"

with open("/home/exouser/.gemini/antigravity-ide/brain/3b0d16b9-6980-4111-996d-ae6d2efb8502/internal_verbs_summary.md", "w") as f:
    f.write(md_content)

print("Analysis complete. Saved to internal_verbs_summary.md")
