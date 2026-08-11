"""Stateless CLI chat with a local HF instruct model. Every prompt starts
from scratch - no conversation history is carried over.

Usage:
    ../venv/bin/python chat.py                              # default Llama-3.2-3B-Instruct
    ../venv/bin/python chat.py meta-llama/Llama-3.2-1B-Instruct
    HF_TOKEN=... ../venv/bin/python chat.py                 # if gated model not cached

Type your message, hit Enter, get the model's reply. 'quit' to exit.
Special prefixes:
    /n 200 <msg>        -> generate up to 200 tokens (default 256)
    /s <system prompt>  -> set a system prompt for the session
    /raw <msg>          -> completion mode (no chat template)
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"

print(f"loading {MODEL} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
print("ready. type a message, Enter to send, 'quit' to exit.")
print("prefixes: /n <tokens>  /s <system prompt>  /raw <completion mode>\n", flush=True)

system_prompt = None

while True:
    try:
        msg = input(">>> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if msg.strip().lower() in ("quit", "exit", "q"):
        break
    if not msg.strip():
        continue

    max_new = 256
    raw = False
    while True:
        if msg.startswith("/n "):
            parts = msg.split(" ", 2)
            if len(parts) == 3 and parts[1].isdigit():
                max_new = int(parts[1]); msg = parts[2]; continue
        if msg.startswith("/s "):
            system_prompt = msg[3:].strip()
            print(f"[system prompt set: {system_prompt!r}]")
            msg = ""
        if msg.startswith("/raw "):
            raw = True; msg = msg[5:]; continue
        break
    if not msg.strip():
        continue

    if raw:
        ids = tok(msg, return_tensors="pt").to("cuda")
    else:
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
            + [{"role": "user", "content": msg}]
        ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    print(text)
    print("-" * 60, flush=True)
