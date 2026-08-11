"""Multi-hop swap-tracking task: construction, simulation, and prompt templates.

Task: N boxes each hold one distinct item. K pairwise swaps of box contents happen.
  - CLEAN:   query an item that was swapped >= min_hops times (multi-hop tracking).
  - CORRUPT: same init + same query item, but swaps are changed so the query item
             NEVER swaps -> answer is stated verbatim in the initial assignment
             (0 hops), and the corruption diverges mid-prompt.

All quantities are simulated exactly; ground truth never depends on rendering.
"""

import random

# Concrete, portable, single-word nouns. Filtered per-tokenizer at generation time
# (must be a single token with a leading space).
BOX_LABELS = "ABCDEFGH"

# Person names as "holders" — small LMs have strong name-binding machinery
# (cf. IOI circuits in GPT-2). Used by the person-* templates instead of boxes.
NAME_POOL = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
]

ITEM_POOL = [
    "ball", "key", "ring", "book", "cup", "pen", "hat", "coin", "card", "toy",
    "bell", "egg", "fork", "sock", "drum", "vase", "rope", "comb", "plate", "brush",
]

# ---------------------------------------------------------------------------
# Templates. Placeholders: {item}, {i1}, {i2} (items), {box}, {b1}, {b2} (boxes).
# Family "itemloc": track items, query an item's final box  -> answer " {box}"
# Family "boxcont": track contents, query a box's content   -> answer " {item}"
# ---------------------------------------------------------------------------
TEMPLATES = {
    # --- item-location family ---
    "itemloc_v1": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Now the {item} is in box",
        "answer": " {box}",
    },
    "itemloc_v2": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} are swapped.",
        "query": "Now the {item} is in box",
        "answer": " {box}",
    },
    "itemloc_v3": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "After all the swaps, the {item} is in box",
        "answer": " {box}",
    },
    "itemloc_v4": {
        "family": "itemloc",
        "init": "Box {box} has the {item}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Now the {item} is in box",
        "answer": " {box}",
    },
    "itemloc_nl": {  # newline-separated variant of v1
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Now the {item} is in box",
        "answer": " {box}",
        "sep": "\n",
    },
    # --- box-content family ---
    "boxcont_v1": {
        "family": "boxcont",
        "init": "Box {box} has the {item}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Now box {box} has the",
        "answer": " {item}",
    },
    "boxcont_v2": {
        "family": "boxcont",
        "init": "The {item} is in box {box}.",
        "swap": "The contents of box {b1} and box {b2} are swapped.",
        "query": "Now box {box} has the",
        "answer": " {item}",
    },
    "boxcont_v3": {
        "family": "boxcont",
        "init": "Box {box} contains the {item}.",
        "swap": "Box {b1} and box {b2} swap contents.",
        "query": "At the end, box {box} contains the",
        "answer": " {item}",
    },
    # --- swap-as-explicit-moves: each swap = two "moves to" sentences ---
    "swapmv_q": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_q2": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} is moved to box {b2}. The {i2} is moved to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_q3": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} goes to box {b2}. The {i2} goes to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_cont_q": {  # box-content query over explicit-move swaps
        "family": "boxcont",
        "init": "Box {box} has the {item}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "What does box {box} have now? Box {box} has the",
        "answer": " {item}",
    },
    # --- moves + sequence markers / final-state query cues / mixed few-shot ---
    "swapmv_then_q": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Then the {i1} moves to box {b2}. Then the {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_end_q": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} at the end? At the end, the {item} is in box",
        "answer": " {box}",
    },
    "swapmv_q_fs2m": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
        "shots": "mixed2",
    },
    "swapmv_q_fs4m": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
        "shots": "mixed4",
    },
    "swapmv_then_q_fs4m": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Then the {i1} moves to box {b2}. Then the {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
        "shots": "mixed4",
    },
    "swapmv_q_fs8m": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
        "shots": "mixed8",
    },
    # --- giving chain between people (natural narrative priors) ---
    "givechain_q": {
        "family": "itemloc",
        "holders": "names",
        "init": "The {item} is with {box}.",
        "swap": "{b1} gives the {i1} to {b2}. {b2} gives the {i2} to {b1}.",
        "query": "Who has the {item} now? The {item} is with",
        "answer": " {box}",
    },
    "givechain_q_fs4m": {
        "family": "itemloc",
        "holders": "names",
        "init": "The {item} is with {box}.",
        "swap": "{b1} gives the {i1} to {b2}. {b2} gives the {i2} to {b1}.",
        "query": "Who has the {item} now? The {item} is with",
        "answer": " {box}",
        "shots": "mixed4",
    },
    # --- temporal-contrast variants (old vs current state cues) ---
    "swapmv_start_q": {  # "started" marks init as past state
        "family": "itemloc",
        "init": "The {item} started in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_was_q": {  # was/at-first vs now tense contrast
        "family": "itemloc",
        "init": "At first, the {item} was in box {box}.",
        "swap": "Then the {i1} was moved to box {b2}. Then the {i2} was moved to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_nowstate_q": {  # moves stated directly as new states with "now"
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Now the {i1} is in box {b2}. Now the {i2} is in box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "swapmv_step_q": {  # numbered steps mark event order explicitly
        "family": "itemloc",
        "numbered": True,
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    # --- heavy mixed few-shot on the best zero-shot base (swapmv_end_q) ---
    "swapmv_end_q_fs8m_u": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} at the end? At the end, the {item} is in box",
        "answer": " {box}",
        "shots": "mixed8", "uniform_items": True,
    },
    "swapmv_end_q_fs12m_u": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} at the end? At the end, the {item} is in box",
        "answer": " {box}",
        "shots": "mixed12", "uniform_items": True,
    },
    "swapmv_end_q_fs16m_u": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} at the end? At the end, the {item} is in box",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True,
    },
    "swapmv_end_q_fs12m": {  # non-uniform control
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} at the end? At the end, the {item} is in box",
        "answer": " {box}",
        "shots": "mixed12",
    },
    # --- "Answer:" marker + uniform items & query item in shots ---
    "swapmv_ans_q": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
    },
    "swapmv_ans_q_fs8_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed8", "uniform_items": True, "uniform_query": True,
    },
    "swapmv_ans_q_fs12_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed12", "uniform_items": True, "uniform_query": True,
    },
    "swapmv_ans_q_fs20_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed20", "uniform_items": True, "uniform_query": True,
    },
    "swapmv_ans_q_fs20_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed20", "uniform_items": True, "uniform_query": True,
    },
    "swapmv_ans_q_fs16_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
    },
    "swapmv_ans_q_fs24_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed24", "uniform_items": True, "uniform_query": True,
    },
    "swapmv_ans_q_fs16_uq_l": {  # + query item's move stated last
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} moves to box {b2}. The {i2} moves to box {b1}.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
        "query_move_last": True,
    },
    # --- box-named swaps: items NEVER named after init; the answer is stated
    # nowhere and must be inferred (book in A; A<->C swapped => book in C).
    # A last-mention cheater outputs the init box = the corrupt answer, so
    # cheating is wrong on clean by construction. ---
    "swapbox_ans_q": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
    },
    "swapbox_ans_q_fs8_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed8", "uniform_items": True, "uniform_query": True,
    },
    "swapbox_ans_q_fs12_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed12", "uniform_items": True, "uniform_query": True,
    },
    "swapbox_ans_q_fs16_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
    },
    "swapbox_ans_q_fs24_uq": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed24", "uniform_items": True, "uniform_query": True,
    },
    "swapboxc_ans_q_fs16_uq": {  # "contents of" phrasing variant
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The contents of box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
    },
    # --- attempts to break the 3-box corrupt degeneracy ---
    "swapbox_ans_q_fs32_uq": {  # 4-box rescue attempt: more shots
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed32", "uniform_items": True, "uniform_query": True,
    },
    "swapbox_ans_q_fs16_uq_s7": {  # shot-set seed robustness
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
        "shot_seed": 7,
    },
    "swapbox_ans_q_fs16_uq_s777": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
        "shot_seed": 777,
    },
    "swapbox_ans_q_fs16_uq_c1": {  # corrupt with ONE swap: kills the
        "family": "itemloc",     # "swaps cancel out" tell
        "init": "The {item} is in box {box}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? Answer:",
        "answer": " {box}",
        "shots": "mixed16", "uniform_items": True, "uniform_query": True,
        "corrupt_n_swaps": 1,
    },
    # --- ledger format: uniform key-value stream, answer = value at the LAST
    # occurrence of the query key (induction-head-friendly, cf. repo dataset C) ---
    "ledger_v1": {
        "family": "itemloc",
        "init": "{item} {box}",
        "swap": "{i1} {b2}, {i2} {b1}",
        "query": "{item}",
        "answer": " {box}",
        "sep": ", ",
    },
    "ledger_v2": {
        "family": "itemloc",
        "init": "{item} in {box}",
        "swap": "{i1} in {b2}, {i2} in {b1}",
        "query": "{item} in",
        "answer": " {box}",
        "sep": ", ",
    },
    "ledger_v3": {
        "family": "itemloc",
        "init": "{item}: {box}",
        "swap": "{i1}: {b2}, {i2}: {b1}",
        "query": "{item}:",
        "answer": " {box}",
        "sep": ", ",
    },
    "ledger_nl": {
        "family": "itemloc",
        "init": "{item} {box}",
        "swap": "{i1} {b2}\n{i2} {b1}",
        "query": "{item}",
        "answer": " {box}",
        "sep": "\n",
    },
    # --- question-form queries (query echoes the init phrasing) ---
    "itemloc_q": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "itemloc_q_trade": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} trade places.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "itemloc_q_switch": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} switch places.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "itemloc_q_boxes": {  # swap stated box-centrically under item-location query
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The items in box {b1} and box {b2} are swapped.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
    },
    "itemloc_q_fs2": {  # 2-shot prefix version of itemloc_q
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
        "shots": 2,
    },
    "itemloc_q_fs3": {
        "family": "itemloc",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Where is the {item} now? The {item} is in box",
        "answer": " {box}",
        "shots": 3,
    },
    "boxcont_q": {
        "family": "boxcont",
        "init": "Box {box} has the {item}.",
        "swap": "Box {b1} and box {b2} are swapped.",
        "query": "What does box {box} have now? Box {box} has the",
        "answer": " {item}",
    },
    # --- preamble variants (task-establishing first line) ---
    "itemloc_pre": {
        "family": "itemloc",
        "pre": "Track where each item is.",
        "init": "The {item} is in box {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Now the {item} is in box",
        "answer": " {box}",
    },
    # --- person holders (names bind strongly in small LMs) ---
    "person_cont_v1": {
        "family": "boxcont",
        "holders": "names",
        "init": "{box} has the {item}.",
        "swap": "{b1} and {b2} swap their items.",
        "query": "Now {box} has the",
        "answer": " {item}",
    },
    "person_cont_v2": {
        "family": "boxcont",
        "holders": "names",
        "init": "{box} has the {item}.",
        "swap": "{b1} and {b2} swap their items.",
        "query": "After the swaps, {box} has the",
        "answer": " {item}",
    },
    "person_cont_q": {
        "family": "boxcont",
        "holders": "names",
        "init": "{box} has the {item}.",
        "swap": "{b1} and {b2} swap their items.",
        "query": "What does {box} have now? {box} has the",
        "answer": " {item}",
    },
    "person_itemloc": {
        "family": "itemloc",
        "holders": "names",
        "init": "The {item} is with {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Now the {item} is with",
        "answer": " {box}",
    },
    "person_itemloc_q": {
        "family": "itemloc",
        "holders": "names",
        "init": "The {item} is with {box}.",
        "swap": "The {i1} and the {i2} swap places.",
        "query": "Who has the {item} now? The {item} is with",
        "answer": " {box}",
    },
}


def simulate(n_boxes, init_contents, swaps):
    """init_contents[b] = item initially in box b. swaps = list of (b1, b2).
    Returns final contents list."""
    contents = list(init_contents)
    for b1, b2 in swaps:
        contents[b1], contents[b2] = contents[b2], contents[b1]
    return contents


def item_involvements(n_items_present, init_contents, swaps):
    """Count how many swaps each item participates in (item-location hops)."""
    counts = {it: 0 for it in init_contents}
    contents = list(init_contents)
    for b1, b2 in swaps:
        counts[contents[b1]] += 1
        counts[contents[b2]] += 1
        contents[b1], contents[b2] = contents[b2], contents[b1]
    return counts


def content_chain_length(query_box, swaps):
    """Backward trace of the content ending in query_box: number of swaps on the
    path (box-content hops) and the origin box."""
    box = query_box
    hops = 0
    for b1, b2 in reversed(swaps):
        if box == b1:
            box = b2
            hops += 1
        elif box == b2:
            box = b1
            hops += 1
    return hops, box


def _random_swaps(rng, n_boxes, n_swaps, avoid_box=None):
    boxes = [b for b in range(n_boxes) if b != avoid_box]
    swaps = []
    seen = set()
    tries = 0
    while len(swaps) < n_swaps and tries < 1000:
        tries += 1
        b1, b2 = rng.sample(boxes, 2)
        pair = (min(b1, b2), max(b1, b2))
        swaps.append(pair)  # consecutive duplicate pairs are allowed in clean
        seen.add(pair)
    return swaps


def make_sample(rng, n_boxes=5, n_swaps=3, min_hops=2, family="itemloc",
                labels=None, items=None, query_item=None,
                corrupt_n_swaps=None):
    """Build one clean/corrupt pair. Returns a dict of structured data."""
    labels = list(labels) if labels is not None else list(BOX_LABELS[:n_boxes])
    for _ in range(200):
        items = list(items) if items is not None else rng.sample(ITEM_POOL, n_boxes)
        init = list(items)  # init[b] = item initially in box b (boxes 0-indexed)
        swaps = _random_swaps(rng, n_boxes, n_swaps)
        final = simulate(n_boxes, init, swaps)

        if family == "itemloc":
            inv = item_involvements(n_boxes, init, swaps)
            # query item: swapped >= min_hops times AND ends somewhere else
            cands = [it for it in items
                     if inv[it] >= min_hops
                     and final.index(it) != init.index(it)]
            if query_item is not None:
                cands = [it for it in cands if it == query_item]
            if not cands:
                continue
            query_item = rng.choice(cands)
            query_box = init.index(query_item)  # its home box
            hops_clean = inv[query_item]
            clean_answer_box = final.index(query_item)
            corrupt_answer_box = query_box  # never moves in corrupt
        else:  # boxcont
            chains = [(*content_chain_length(b, swaps), b) for b in range(n_boxes)]
            cands = [b for h, origin, b in chains
                     if h >= min_hops and final[b] != init[b]]
            if not cands:
                continue
            query_box = rng.choice(cands)
            query_item = init[query_box]  # item starting in query box
            hops_clean = content_chain_length(query_box, swaps)[0]
            clean_answer_box = query_box  # answer is an item, see below
            corrupt_answer_box = query_box

        # corrupt swaps: never touch the query item's home box (itemloc) or the
        # query box (boxcont) -> 0 hops for the answer in both framings.
        n_cs = corrupt_n_swaps if corrupt_n_swaps is not None else n_swaps
        c_swaps = _random_swaps(rng, n_boxes, n_cs, avoid_box=query_box)
        if len(c_swaps) < n_cs:
            continue

        # answers
        if family == "itemloc":
            clean_ans, corrupt_ans = clean_answer_box, corrupt_answer_box
        else:
            clean_ans = final[query_box]   # an item
            corrupt_ans = init[query_box]  # an item
        if clean_ans == corrupt_ans:
            continue

        return {
            "family": family,
            "n_boxes": n_boxes,
            "n_swaps": n_swaps,
            "items": items,
            "init": init,                    # init[b] = item in box b
            "swaps": swaps,                  # clean swaps (box pairs)
            "corrupt_swaps": c_swaps,
            "query_item": query_item,
            "query_box": query_box,
            "hops_clean": hops_clean,
            "hops_corrupt": 0,
            "clean_answer": clean_ans,       # box idx (itemloc) or item (boxcont)
            "corrupt_answer": corrupt_ans,
            "labels": labels,
            "corrupt_n_swaps": n_cs,
        }
    raise RuntimeError("could not construct sample; relax constraints")


def _render_body(sample, t, swaps):
    sep = t.get("sep", " ")
    n_boxes = sample["n_boxes"]
    init = sample["init"]
    labels = sample.get("labels", list(BOX_LABELS[:n_boxes]))

    parts = []
    if t.get("pre"):
        parts.append(t["pre"])
    for b in range(n_boxes):
        parts.append(t["init"].format(item=init[b], box=labels[b]))
    # item-centric swap rendering needs the items sitting in the swapped boxes
    contents = list(init)
    for k, (b1, b2) in enumerate(swaps):
        if t.get("query_move_last") and contents[b1] == sample["query_item"]:
            b1, b2 = b2, b1  # query item's move sentence comes last
        sent = t["swap"].format(
            i1=contents[b1], i2=contents[b2], b1=labels[b1], b2=labels[b2])
        if t.get("numbered"):
            sent = f"Step {k + 1}: " + sent
        parts.append(sent)
        contents[b1], contents[b2] = contents[b2], contents[b1]
    parts.append(t["query"].format(item=sample["query_item"],
                                   box=labels[sample["query_box"]]))
    return sep.join(parts)


def _shot_str(s, t, view):
    """One solved example: prompt + answer. view='clean' (2-hop tracking) or
    'corrupt' (0-hop, query item never moves)."""
    swaps = s["swaps"] if view == "clean" else s["corrupt_swaps"]
    p = _render_body(s, t, swaps)
    val = s["clean_answer"] if view == "clean" else s["corrupt_answer"]
    if s["family"] == "itemloc":
        ans = t["answer"].format(box=s["labels"][val])
    else:
        ans = t["answer"].format(item=val)
    return p + ans


def _fewshot_prefix(sample, t, swaps):
    """Solved example(s) prepended to the prompt. Fixed seed -> identical for
    every sample, so clean/corrupt share the prefix exactly.
    shots: int N -> N clean-view (2-hop) examples.
           "mixedN" -> N examples alternating clean-view / corrupt-view, so both
           behaviors (tracked answer and never-moved answer) are demonstrated."""
    rng = random.Random(t.get("shot_seed", 12345))
    spec = t["shots"]
    if isinstance(spec, str) and spec.startswith("mixed"):
        n = int(spec[5:])
        views = ["clean" if i % 2 == 0 else "corrupt" for i in range(n)]
    else:
        views = ["clean"] * int(spec)
    shots = []
    for v in views:
        items = sample["items"] if t.get("uniform_items") else None
        q = sample["query_item"] if t.get("uniform_query") else None
        s = make_sample(rng, sample["n_boxes"], sample["n_swaps"],
                        max(1, min(2, sample["hops_clean"])),
                        sample["family"], sample.get("labels"),
                        items=items, query_item=q,
                        corrupt_n_swaps=sample.get("corrupt_n_swaps"))
        shots.append(_shot_str(s, t, v))
    return "\n\n".join(shots) + "\n\n"


def render(sample, template, swaps=None):
    """Render a prompt from structured sample data. `swaps` overrides which swap
    list is rendered (None -> clean swaps; pass [] to strip swaps entirely)."""
    t = TEMPLATES[template]
    swaps = sample["swaps"] if swaps is None else swaps
    body = _render_body(sample, t, swaps)
    if t.get("shots"):
        return _fewshot_prefix(sample, t, swaps) + body
    return body


def answer_str(sample, template, which="clean"):
    """The gold answer string (with leading space) for clean or corrupt."""
    t = TEMPLATES[template]
    val = sample[f"{which}_answer"]
    if sample["family"] == "itemloc":
        labels = sample.get("labels", list(BOX_LABELS[:sample["n_boxes"]]))
        return t["answer"].format(box=labels[val])
    return t["answer"].format(item=val)


def candidate_strs(sample, template):
    """All plausible answer strings (restricted-candidate eval set)."""
    t = TEMPLATES[template]
    if sample["family"] == "itemloc":
        labels = sample.get("labels", list(BOX_LABELS[:sample["n_boxes"]]))
        return [t["answer"].format(box=labels[b]) for b in range(sample["n_boxes"])]
    return [t["answer"].format(item=it) for it in sample["items"]]


def to_record(sample, template, idx, split):
    """Repo-convention JSONL record."""
    clean_p = render(sample, template, swaps=sample["swaps"])
    corrupt_p = render(sample, template, swaps=sample["corrupt_swaps"])
    clean_a = answer_str(sample, template, "clean")
    corrupt_a = answer_str(sample, template, "corrupt")
    return {
        "id": f"{idx:06d}",
        "dataset": "multihop",
        "split": split,
        "template": template,
        "clean_prompt": clean_p,
        "corrupt_prompt": corrupt_p,
        "clean_answer": clean_a,
        "corrupt_answer": corrupt_a,
        "ld_candidates": [clean_a, corrupt_a],
        "all_candidates": candidate_strs(sample, template),
        "family": sample["family"],
        "n_boxes": sample["n_boxes"],
        "n_swaps": sample["n_swaps"],
        "hops_clean": sample["hops_clean"],
        "hops_corrupt": sample["hops_corrupt"],
        "corrupt_n_swaps": sample.get("corrupt_n_swaps", sample["n_swaps"]),
        "query_item": sample["query_item"],
        "query_box": sample["labels"][sample["query_box"]],
        "labels": sample["labels"],
        "init": {sample["labels"][b]: it for b, it in enumerate(sample["init"])},
        "swaps": [[sample["labels"][a], sample["labels"][b]] for a, b in sample["swaps"]],
        "corrupt_swaps": [[sample["labels"][a], sample["labels"][b]] for a, b in sample["corrupt_swaps"]],
    }


def filter_single_token(tokenizer, pool):
    """Keep only words that are exactly one token with a leading space."""
    ok = []
    for it in pool:
        ids = tokenizer.encode(" " + it, add_special_tokens=False)
        if len(ids) == 1:
            ok.append(it)
    return ok
