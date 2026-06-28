import re

def build_target_regex(target):
    target = target.strip().lower()
    words = re.split(r'[\s\-\_\.]+', target)
    words = [re.escape(w) for w in words if w]
    if not words:
        return ""
    
    last_word = words[-1]
    last_word_pattern = last_word + r'(?:s|es)?'
    
    if len(words) > 1:
        core_pattern = r'[\s\-\_\.]*'.join(words[:-1]) + r'[\s\-\_\.]*' + last_word_pattern
    else:
        core_pattern = last_word_pattern
        
    pattern = r'^(?:a\s+|an\s+|the\s+)?' + core_pattern + r'(?:\b|\W|$)'
    return pattern

targets = ["supply yard", "metal_cabinet", "safe-deposit box", "central.bank", "yard"]
predictions = [
    " supply yard.",
    "a supply-yard",
    "the supply_yard",
    "supplyyard",
    "supply yards",
    "central bank",
    "central.bank",
    "safe deposit box",
    "the yardstick",
    " yard",
    " yards."
]

for t in targets:
    print(f"Target: {t}")
    pat = build_target_regex(t)
    for p in predictions:
        match = bool(re.match(pat, p.strip().lower()))
        if match:
            print(f"  Match: {p}")
