import json

with open("/home/exouser/pruning/custom/llama_6hop/extended_verbs_results.json", "r") as f:
    data = json.load(f)

stats = {}

for item in data:
    v1 = item["v1"]
    v2 = item["v2"]
    fv = item["fv"]
    t = item["type"]
    
    key = (v1, v2, fv)
    if key not in stats:
        stats[key] = {"normal_clean": {"correct": 0, "total": 0}, "normal_corr": {"correct": 0, "total": 0}}
        
    stats[key][t]["total"] += 1
    if item["is_correct"]:
        stats[key][t]["correct"] += 1

summary_list = []
for key, vals in stats.items():
    v1, v2, fv = key
    
    nc_tot = vals["normal_clean"]["total"]
    nco_tot = vals["normal_corr"]["total"]
    
    nc_acc = vals["normal_clean"]["correct"] / nc_tot if nc_tot > 0 else 0
    nco_acc = vals["normal_corr"]["correct"] / nco_tot if nco_tot > 0 else 0
    
    avg_acc = (nc_acc + nco_acc) / 2
    
    summary_list.append({
        "v1": v1,
        "v2": v2,
        "fv": fv,
        "nc_acc": nc_acc,
        "nco_acc": nco_acc,
        "avg": avg_acc
    })

# Sort by avg descending
summary_list.sort(key=lambda x: x["avg"], reverse=True)

md_content = "# Extended Internal Verbs Summary (Sorted by Average Clean/Corrupted Acc)\n\n"
md_content += "Evaluated strictly on Qwen2.5-32B. Number of examples per template: 40.\n\n"
md_content += "| `v1` (Placement) | `v2` (Movement) | `fv` (Question) | Clean Acc | Corrupted Acc | **Average Acc** |\n"
md_content += "|:---|:---|:---|:---|:---|:---|\n"

for s in summary_list:
    md_content += f"| `{s['v1']}` | `{s['v2']}` | `{s['fv']}` | {s['nc_acc']*100:.0f}% | {s['nco_acc']*100:.0f}% | **{s['avg']*100:.1f}%** |\n"

with open("/home/exouser/.gemini/antigravity-ide/brain/3b0d16b9-6980-4111-996d-ae6d2efb8502/extended_verbs_sorted.md", "w") as f:
    f.write(md_content)

print("Analysis complete. Saved to extended_verbs_sorted.md")
