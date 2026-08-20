"""Template families for contextually-entangled patching datasets.

Each family produces (clean_prompt, corrupt_prompt, clean_target, corrupt_target)
where ONLY the control token differs between the two prompts. All control-token
pairs were probed to have EQUAL token counts under the Llama-3 tokenizer, so
P_clean and P_corrupt are near 1-to-1 token-positionally aligned.

Tokenizer-driven modifications from the original spec (Llama-3.1 tokenizer):
  T2: INACTIVE(2 tok) -> CLOSED(1 tok)        [ACTIVE is 1 tok]
  T6: COMBINE(3 tok)  -> MERGE(2 tok)         [HALT is 2 tok]
  T7: MUTE(2 tok) -> HOLD(1 tok)              [REPORT is 1 tok]
  T7: fallback "N/A"  -> "NULL"               [cleaner single-token fallback]
"""

import random

# --------------------------------------------------------------------------
# variation pools
# --------------------------------------------------------------------------
ITEMS = [
    "key", "wallet", "ring", "letter", "coin", "phone", "pen", "ticket",
    "card", "watch", "passport", "cookie", "book", "charger", "glasses",
    "remote", "umbrella", "hammer", "brush", "bottle", "laptop", "scarf",
    "battery", "map", "medicine", "receipt", "badge", "spoon", "towel",
    "flashlight", "dagger", "scroll", "lantern", "compass", "notebook",
    "candle", "mirror", "glove", "helmet", "rope",
]
CONTAINERS = [
    "drawer", "backpack", "box", "envelope", "jar", "locker", "case",
    "folder", "wallet", "safe", "shelf", "suitcase", "closet", "toolbox",
    "cabinet", "crate", "briefcase", "wardrobe", "compartment", "glovebox",
    "purse", "basket", "pouch", "cubby", "trunk", "chest", "bin",
    "satchel", "holster", "canister",
]

NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Irene", "Jack", "Karen", "Leo", "Mona", "Nathan", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Umar", "Vera", "Wendy", "Xander",
    "Yara", "Zack", "Liam", "Mia", "Noah", "Ava", "Ethan", "Luna",
    "Owen", "Ruby", "Caleb", "Nora", "Felix", "Iris", "Hugo",
]

PREFIXES = [
    "RED", "BLUE", "GREEN", "GOLD", "SILVER", "BLACK", "WHITE", "PURPLE",
    "ORANGE", "PINK", "BROWN", "GRAY", "CYAN", "MAGENTA", "TEAL",
]
SUFFIXES = [
    "STAR", "MOON", "SUN", "BIRD", "FISH", "TREE", "ROCK", "LEAF",
    "WAVE", "FLAME", "CLOUD", "STONE", "RIVER", "FIELD", "LIGHT",
]

ENTITIES_CATEGORIES = [
    ("Apple", "fruit"), ("Carrot", "vegetable"), ("Salmon", "fish"),
    ("Eagle", "bird"), ("Oak", "tree"), ("Rose", "flower"),
    ("Tiger", "mammal"), ("Cobra", "reptile"), ("Trout", "fish"),
    ("Potato", "vegetable"), ("Grape", "fruit"), ("Robin", "bird"),
    ("Pine", "tree"), ("Tulip", "flower"), ("Wolf", "mammal"),
    ("Lizard", "reptile"), ("Banana", "fruit"), ("Onion", "vegetable"),
    ("Hawk", "bird"), ("Maple", "tree"), ("Daisy", "flower"),
    ("Bear", "mammal"), ("Frog", "amphibian"), ("Melon", "fruit"),
    ("Spinach", "vegetable"), ("Sparrow", "bird"), ("Birch", "tree"),
    ("Lily", "flower"), ("Fox", "mammal"), ("Toad", "amphibian"),
    ("Cherry", "fruit"), ("Pepper", "vegetable"), ("Owl", "bird"),
    ("Cedar", "tree"), ("Violet", "flower"), ("Deer", "mammal"),
    ("Newt", "amphibian"), ("Plum", "fruit"), ("Garlic", "vegetable"),
    ("Finch", "bird"), ("Willow", "tree"), ("Orchid", "flower"),
    ("Moose", "mammal"), ("Salamander", "amphibian"), ("Peach", "fruit"),
    ("Turnip", "vegetable"), ("Crane", "bird"), ("Aspen", "tree"),
    ("Poppy", "flower"), ("Otter", "mammal"),
]

UNIT_CONVERSIONS = [
    ("kilometer", "kilometers", "meter", "meters", 1000),
    ("kilogram", "kilograms", "gram", "grams", 1000),
    ("liter", "liters", "milliliter", "milliliters", 1000),
]

LIST_WORDS = [
    "Alpha", "Beta", "Gamma", "Delta", "Omega", "Sigma", "Theta", "Zeta",
    "Kappa", "Lambda", "Nova", "Comet", "Orbit", "Quasar", "Nebula",
    "Vertex", "Prism", "Echo", "Falcon", "Viper", "Cobra", "Draco",
    "Lyra", "Orion", "Pegasus", "Phoenix", "Taurus", "Hydra", "Cygnus",
    "Vela", "Aquila", "Lupus", "Corvus", "Dorado", "Fornax", "Grus",
    "Indus", "Lacerta", "Mensa", "Norma", "Octans", "Pictor", "Reticulum",
    "Scutum", "Tucana", "Volans", "Auriga", "Bootes", "Carina",
]

COLORS = [
    "RED", "GREEN", "BLUE", "YELLOW", "ORANGE", "PURPLE", "WHITE", "BLACK",
    "PINK", "BROWN", "GRAY", "CYAN", "AMBER", "VIOLET", "INDIGO",
]


def _unique_pairs(rng, pool, n, k=2):
    combos = []
    seen = set()
    tries = 0
    while len(combos) < n:
        c = tuple(rng.sample(pool, k))
        if c not in seen:
            seen.add(c)
            combos.append(c)
        tries += 1
        if tries > 100 * n:
            raise RuntimeError(f"pool too small for {n} unique {k}-tuples")
    return combos


def _unique_combos(rng, pool_a, pool_b, n):
    """n unique (a, b) pairs drawn from two pools."""
    seen, out = set(), []
    tries = 0
    while len(out) < n:
        pair = (rng.choice(pool_a), rng.choice(pool_b))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
        tries += 1
        if tries > 100 * n:
            raise RuntimeError(f"pool too small for {n} unique pairs")
    return out


# --------------------------------------------------------------------------
# family definitions
# --------------------------------------------------------------------------
# Each family: dict with
#   key, name, control=(clean_token, corrupt_token), fallback,
#   samples(seed, n) -> list[dict], render(sample, corrupt) -> str,
#   clean_target(sample) -> str

def _t1_samples(rng, n):
    seen, out = set(), []
    while len(out) < n:
        b, a = rng.randint(10, 99), rng.randint(2, 9)
        if (b, a) not in seen:
            seen.add((b, a))
            out.append({"base": b, "add": a})
    return out

T1 = {
    "key": "t1_arithmetic_gate",
    "name": "Template 1: Single-Step Arithmetic with Gate",
    "control": ("ENABLED", "DISABLED"), "fallback": "OFF",
    "samples": _t1_samples,
    "render": lambda s, corrupt: (
        f"Data: Base value = {s['base']}. Add {s['add']}. "
        f"Gate: {'DISABLED' if corrupt else 'ENABLED'}. "
        f"If Gate is ENABLED, output final result. "
        f"If Gate is DISABLED, output \"OFF\".\nAnswer:"
    ),
    "clean_target": lambda s: str(s["base"] + s["add"]),
}

T2 = {
    "key": "t2_kv_lookup",
    "name": "Template 2: Single Key-Value Lookup with Output Mode (INACTIVE->CLOSED)",
    "control": ("ACTIVE", "CLOSED"), "fallback": "HIDDEN",
    "samples": lambda rng, n: [{"uid": u} for u in rng.sample(range(100, 999), n)],
    "render": lambda s, corrupt: (
        f"Record: User_ID is {s['uid']}. "
        f"Status: {'CLOSED' if corrupt else 'ACTIVE'}. "
        f"If Status is ACTIVE, print User_ID. "
        f"If Status is CLOSED, print \"HIDDEN\".\nResult:"
    ),
    "clean_target": lambda s: str(s["uid"]),
}

T3 = {
    "key": "t3_spatial_containment",
    "name": "Template 3: Direct Spatial Containment with Visibility Filter",
    "control": ("VISIBLE", "MASKED"), "fallback": "UNKNOWN",
    "samples": lambda rng, n: [
        {"item": it, "container": co}
        for it, co in _unique_combos(rng, ITEMS, CONTAINERS, n)
    ],
    "render": lambda s, corrupt: (
        f"Item: The {s['item']} is inside the {s['container']}. "
        f"Mode: {'MASKED' if corrupt else 'VISIBLE'}. "
        f"If Mode is VISIBLE, fill in the blank with the correct location. "
        f"If Mode is MASKED, fill in the blank with \"UNKNOWN\". "
        f"Blank: The {s['item']} is inside the ___\nAnswer:"
    ),
    "clean_target": lambda s: s["container"],
}


def _t4_samples(rng, n):
    # half PASS (score above thr), half FAIL (score just below thr, gap 1-10;
    # large-gap FAILs trigger a spurious BYPASS association in small models)
    out, seen = [], set()
    while len(out) < n:
        want_pass = len(out) % 2 == 0
        thr = rng.randint(30, 70)
        score = rng.randint(thr + 5, 100) if want_pass else rng.randint(max(0, thr - 10), thr - 1)
        if (score, thr) not in seen:
            seen.add((score, thr))
            out.append({"score": score, "thr": thr})
    return out

T4 = {
    "key": "t4_boolean_override",
    "name": "Template 4: Simple Boolean Comparison with Override (BYPASS->AUTO)",
    "control": ("NO", "YES"), "fallback": "AUTO",
    "samples": _t4_samples,
    "render": lambda s, corrupt: (
        f"Check: Score is {s['score']}. Passing threshold is {s['thr']}. "
        f"A score passes if it is above the threshold. "
        f"Override: {'YES' if corrupt else 'NO'}. "
        f"If Override is YES, output \"AUTO\". "
        f"If Override is NO, decide if the score passes. "
        f"Check the Override value before looking at the score. "
        f"Reply with exactly one word: AUTO, PASS, or FAIL.\nAnswer:"
    ),
    "clean_target": lambda s: "PASS" if s["score"] > s["thr"] else "FAIL",
}

T5 = {
    "key": "t5_relation_block",
    "name": "Template 5: Single-Hop Relation with Direct Block",
    "control": ("ALLOW", "BLOCK"), "fallback": "NONE",
    "samples": lambda rng, n: [
        {"a": a, "b": b} for a, b in _unique_pairs(rng, NAMES, n, k=2)
    ],
    "render": lambda s, corrupt: (
        f"Relation: {s['a']} is {s['b']}'s manager. "
        f"Filter: {'BLOCK' if corrupt else 'ALLOW'}. "
        f"If Filter is ALLOW, output the name of {s['b']}'s manager. "
        f"If Filter is BLOCK, output \"NONE\".\nManager:"
    ),
    "clean_target": lambda s: s["a"],
}

T6 = {
    "key": "t6_concat_directive",
    "name": "Template 6: String Concatenation with Output Directive (COMBINE->MERGE)",
    "control": ("MERGE", "HALT"), "fallback": "VOID",
    "samples": lambda rng, n: [
        {"p": p, "s": s} for p, s in _unique_combos(rng, PREFIXES, SUFFIXES, n)
    ],
    "render": lambda s, corrupt: (
        f"Tokens: Prefix = {s['p']}, Suffix = {s['s']}. "
        f"Directive: {'HALT' if corrupt else 'MERGE'}. "
        f"If Directive is MERGE, output joined token. "
        f"If Directive is HALT, output \"VOID\".\nResult:"
    ),
    "clean_target": lambda s: s["p"] + s["s"],
}

T7 = {
    "key": "t7_category_suppression",
    "name": "Template 7: Category Membership with Suppression Flag (MUTE->HOLD, N/A->NULL)",
    "control": ("REPORT", "HOLD"), "fallback": "NULL",
    "samples": lambda rng, n: [
        {"e": e, "cat": c} for e, c in ENTITIES_CATEGORIES[:n]
    ],
    "render": lambda s, corrupt: (
        f"Entity: {s['e']} is a {s['cat']}. "
        f"System: {'HOLD' if corrupt else 'REPORT'}. "
        f"If System is REPORT, category is {s['cat']}. "
        f"If System is HOLD, category is \"NULL\".\nOutput:"
    ),
    "clean_target": lambda s: s["cat"],
}


def _t8_samples(rng, n):
    out, seen = [], set()
    while len(out) < n:
        u = rng.choice(UNIT_CONVERSIONS)
        k = rng.randint(1, 20)
        if (u[0], k) not in seen:
            seen.add((u[0], k))
            out.append({"unit": u, "k": k})
        if len(seen) >= 3 * 20 and len(out) < n:
            raise RuntimeError("t8 pool exhausted")
    return out

T8 = {
    "key": "t8_unit_conversion",
    "name": "Template 8: Simple Unit Conversion with Bypass Switch",
    "control": ("RUN", "SKIP"), "fallback": "ZERO",
    "samples": _t8_samples,
    "render": lambda s, corrupt: (
        f"Input: {s['k']} {s['unit'][1] if s['k'] > 1 else s['unit'][0]} equals "
        f"{s['k'] * s['unit'][4]} {s['unit'][3]}. "
        f"Switch: {'SKIP' if corrupt else 'RUN'}. "
        f"If Switch is RUN, output {s['unit'][3]} count. "
        f"If Switch is SKIP, output \"ZERO\".\nValue:"
    ),
    "clean_target": lambda s: str(s["k"] * s["unit"][4]),
}

T9 = {
    "key": "t9_list_index",
    "name": "Template 9: List Index Lookup with Render Toggle",
    "control": ("ON", "OFF"), "fallback": "NULL",
    "samples": lambda rng, n: [
        {"a": a, "b": b} for a, b in _unique_pairs(rng, LIST_WORDS, n, k=2)
    ],
    "render": lambda s, corrupt: (
        f"List: First item is {s['a']}, Second item is {s['b']}. "
        f"Toggle: {'OFF' if corrupt else 'ON'}. "
        f"If Toggle is ON, return the First item from the list. "
        f"If Toggle is OFF, return \"NULL\".\nResult:"
    ),
    "clean_target": lambda s: s["a"],
}

T10 = {
    "key": "t10_state_tracking",
    "name": "Template 10: State Change Tracking with Audit Gate",
    "control": ("ENABLED", "DISABLED"), "fallback": "BLANK",
    "samples": lambda rng, n: [
        {"c1": a, "c2": b} for a, b in _unique_pairs(rng, COLORS, n, k=2)
    ],
    "render": lambda s, corrupt: (
        f"Log: The light changed from {s['c1']} to {s['c2']}. "
        f"Audit: {'DISABLED' if corrupt else 'ENABLED'}. "
        f"If Audit is ENABLED, output the color the light changed to. "
        f"If Audit is DISABLED, output \"BLANK\".\nAnswer:"
    ),
    "clean_target": lambda s: s["c2"],
}

FAMILIES = {t["key"]: t for t in [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]}


def _add_output_directive(p):
    """Insert control-routing + brevity directives before the final answer
    label. Applied identically to clean and corrupt prompts, preserving
    alignment."""
    head, sep, tail = p.rpartition("\n")
    return (head + " Follow only the rule that matches the stated control"
            " value. Output only the final answer." + sep + tail)


def build_dataset(family_key, n=50, seed=0, output_directive=True):
    """Returns list of dicts: clean_prompt, corrupt_prompt, clean_target,
    corrupt_target, meta."""
    fam = FAMILIES[family_key]
    rng = random.Random(seed)
    out = []
    for i, s in enumerate(fam["samples"](rng, n)):
        clean_p = fam["render"](s, corrupt=False)
        corrupt_p = fam["render"](s, corrupt=True)
        if output_directive:
            clean_p = _add_output_directive(clean_p)
            corrupt_p = _add_output_directive(corrupt_p)
        out.append({
            "id": i,
            "family": family_key,
            "clean_prompt": clean_p,
            "corrupt_prompt": corrupt_p,
            "clean_target": fam["clean_target"](s),
            "corrupt_target": fam["fallback"],
            "control": fam["control"],
        })
    return out
