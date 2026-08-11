"""Batch stateless chat: reads queries.json, writes answers.json.

Workflow:
  1. Edit queries.json - a JSON list. Each item is either:
       "a plain prompt string"
     or an object with optional overrides:
       {"prompt": "...", "system": "...", "max_new_tokens": 512, "raw": false}
  2. Run:  ../venv/bin/python batch_chat.py
  3. Read answers.json - list of {"query": ..., "answer": ...} aligned with
     queries.json. Unchanged queries reuse their previous answers; only new
     or edited queries are regenerated. Delete answers.json to force rerun.
"""
import json
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
QFILE = "queries.json"
AFILE = "answers.json"

queries = json.load(open(QFILE))
# normalize
norm = []
for q in queries:
    if isinstance(q, str):
        q = {"prompt": q}
    q.setdefault("system", None)
    q.setdefault("max_new_tokens", 256)
    q.setdefault("raw", False)
    norm.append(q)

prev = {}
try:
    for a in json.load(open(AFILE)):
        prev[json.dumps(a["query"], sort_keys=True)] = a["answer"]
except FileNotFoundError:
    pass

todo = [i for i, q in enumerate(norm) if json.dumps(q, sort_keys=True) not in prev]
print(f"{len(norm)} queries, {len(todo)} to generate "
      f"({len(norm) - len(todo)} reused from {AFILE})", flush=True)

answers = [None] * len(norm)
if todo:
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    for k, i in enumerate(todo):
        q = norm[i]
        if q["raw"]:
            ids = tok(q["prompt"], return_tensors="pt").to("cuda")
        else:
            msgs = ([{"role": "system", "content": q["system"]}] if q["system"] else []) \
                + [{"role": "user", "content": q["prompt"]}]
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=q["max_new_tokens"], do_sample=False,
                pad_token_id=tok.eos_token_id)
        answers[i] = tok.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        print(f"[{k + 1}/{len(todo)}] query #{i} done "
              f"({len(answers[i])} chars)", flush=True)

for i, q in enumerate(norm):
    if answers[i] is None:
        answers[i] = prev[json.dumps(q, sort_keys=True)]

with open(AFILE, "w") as f:
    json.dump([{"query": q, "answer": a} for q, a in zip(norm, answers)],
              f, indent=2)
print(f"wrote {AFILE}", flush=True)
