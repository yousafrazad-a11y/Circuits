import json

file_path = "/home/exouser/pruning/custom/llama_6hop/detailed_prompts_and_logits.json"
with open(file_path, "r") as f:
    data = json.load(f)

stats = {}
for item in data:
    key = (item['model'], item['hop'], item['category'], item['verb'])
    if key not in stats:
        stats[key] = {'normal_clean': {'correct': 0, 'total': 0}, 'normal_corr': {'correct': 0, 'total': 0}}
    
    t = item['type']
    if t in ['normal_clean', 'normal_corr']:
        stats[key][t]['total'] += 1
        if item['is_correct']:
            stats[key][t]['correct'] += 1

results = []
for key, vals in stats.items():
    model, hop, cat, verb = key
    nc_total = vals['normal_clean']['total']
    nco_total = vals['normal_corr']['total']
    
    if nc_total > 0 and nco_total > 0:
        nc_acc = vals['normal_clean']['correct'] / nc_total
        nco_acc = vals['normal_corr']['correct'] / nco_total
        
        # User requested 80% or more for both normal and corrupted
        if nc_acc >= 0.8 and nco_acc >= 0.8:
            results.append({
                'model': model,
                'hop': hop,
                'category': cat,
                'verb': verb,
                'nc_acc': nc_acc,
                'nco_acc': nco_acc
            })

print(f"Found {len(results)} combinations with >= 80% on both Normal Clean and Normal Corrupted:\n")
for r in results:
    print(f"Model: {r['model'].split('/')[-1]:<20} | Hops: {r['hop']} | Category: {r['category']:<10} | Verb: '{r['verb']:<20}' | Clean Acc: {r['nc_acc']*100:.0f}% | Corr Acc: {r['nco_acc']*100:.0f}%")
