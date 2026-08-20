"""Create shared, model-correct IOI data for node pruning and PEAP.

The canonical representation is BOS-free.  PEAP CSV position columns are
shifted by one because TransformerLens prepends a BOS token; node pruning uses
the stored zero-based section starts directly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


PEAP_SPANS = (
    "prefix", "IO", "and", "S1", "S1+1", "action1", "S2", "action2", "to"
)

TEMPLATES = (
    "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to",
    "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to",
    "Then, {A} and {B} had a long argument, and afterwards {B} said to",
    "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to",
    "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to",
    "While {A} and {B} were working at the {PLACE}, {B} gave a {OBJECT} to",
    "While {A} and {B} were commuting to the {PLACE}, {B} gave a {OBJECT} to",
    "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to",
    "Then, {A} and {B} had a long argument. Afterwards {B} said to",
    "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to",
    "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {B} gave it to",
)

# The GPT-2 list from PEAP.  It is filtered below to names represented by one
# space-prefixed token, matching PEAP's IOI assumptions.
NAMES = tuple("""Michael Christopher Jessica Matthew Ashley Jennifer Joshua Amanda
Daniel David James Robert John Joseph Andrew Ryan Brandon Jason Justin Sarah William
Jonathan Stephanie Brian Nicole Nicholas Anthony Heather Eric Elizabeth Adam Megan
Melissa Kevin Steven Thomas Timothy Christina Kyle Rachel Laura Lauren Amber Brittany
Danielle Richard Kimberly Jeffrey Amy Crystal Michelle Tiffany Jeremy Benjamin Mark
Emily Aaron Charles Rebecca Jacob Stephen Patrick Sean Erin Jamie Kelly Samantha Nathan
Sara Dustin Paul Angela Tyler Scott Katherine Andrea Gregory Erica Mary Travis Lisa
Kenneth Bryan Lindsey Kristen Jose Alexander Jesse Katie Lindsay Shannon Vanessa
Courtney Christine Alicia Cody Allison Bradley Samuel""".split())
PLACES = ("store", "garden", "restaurant", "school", "hospital", "office", "house", "station")
OBJECTS = ("ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack")


def _single_token_names(tokenizer: GPT2TokenizerFast) -> list[str]:
    names = [n for n in NAMES if len(tokenizer.encode(" " + n, add_special_tokens=False)) == 1]
    if len(names) < 5:
        raise RuntimeError("Fewer than five PEAP names are single GPT-2 tokens.")
    return names


def _positions(tokenizer: GPT2TokenizerFast, prompt: str, io: str, subject: str) -> tuple[list[int], list[int]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    io_id = tokenizer.encode(" " + io, add_special_tokens=False)[0]
    subject_id = tokenizer.encode(" " + subject, add_special_tokens=False)[0]
    io_pos = ids.index(io_id)
    subject_positions = [i for i, token in enumerate(ids) if token == subject_id]
    if len(subject_positions) != 2:
        raise ValueError("Expected exactly two subject-name tokens.")
    s1, s2 = subject_positions
    starts = [0, io_pos, io_pos + 1, s1, s1 + 1, s1 + 2, s2, s2 + 1, len(ids) - 1]
    if starts != sorted(starts) or tokenizer.decode([ids[-1]]) != " to":
        raise ValueError("Prompt does not satisfy the PEAP ABBA schema.")
    lengths = [b - a for a, b in zip(starts, starts[1:] + [len(ids)])]
    return starts, lengths


def _candidate(rng: random.Random, tokenizer: GPT2TokenizerFast, names: list[str], candidate_id: int) -> dict:
    subject, indirect_object, counter_first, counter_second, counter_subject = rng.sample(names, 5)
    template_id = rng.randrange(len(TEMPLATES))
    place = rng.choice(PLACES)
    object_name = rng.choice(OBJECTS)
    clean = TEMPLATES[template_id].format(A=indirect_object, B=subject, PLACE=place, OBJECT=object_name)
    # PEAP's ABC counterfactual: all three visible name roles are unrelated.
    counter = TEMPLATES[template_id].format(
        A=counter_first, B=counter_second, PLACE=place, OBJECT=object_name
    )
    # Replace only the repeated subject occurrence with the third unrelated name.
    marker = " " + counter_second
    last = counter.rfind(marker)
    counter = counter[:last] + " " + counter_subject + counter[last + len(marker):]
    starts, lengths = _positions(tokenizer, clean, indirect_object, subject)
    counter_ids = tokenizer.encode(counter, add_special_tokens=False)
    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    if len(counter_ids) != len(clean_ids):
        raise ValueError("Clean/counterfactual token lengths differ.")
    return {
        "candidate_id": candidate_id,
        "template_order": "abba",
        "template_id": template_id,
        "prompt": clean,
        "counterfactual_prompt": counter,
        "correct_token": " " + indirect_object,
        "wrong_token": " " + subject,
        "IO_token": " " + indirect_object,
        "S1_token": " " + subject,
        "S2_token": " " + subject,
        "label": " " + indirect_object,
        "token_ids": json.dumps(clean_ids),
        "counterfactual_token_ids": json.dumps(counter_ids),
        "section_starts": json.dumps(starts),
        "section_lengths": json.dumps(lengths),
        "prompt_length": len(clean_ids),
    }


@torch.inference_mode()
def _screen(model, tokenizer, rows: list[dict], device: str, batch_size: int) -> list[dict]:
    accepted: list[dict] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        encoded = tokenizer(
            [row["prompt"] for row in batch], padding=True, return_tensors="pt"
        )
        bos = torch.full(
            (len(batch), 1), tokenizer.bos_token_id, dtype=encoded.input_ids.dtype
        )
        encoded["input_ids"] = torch.cat((bos, encoded.input_ids), dim=1)
        encoded["attention_mask"] = torch.cat(
            (torch.ones_like(bos), encoded.attention_mask), dim=1
        )
        encoded = encoded.to(device)
        logits = model(**encoded).logits
        final = encoded.attention_mask.sum(dim=1) - 1
        indices = torch.arange(len(batch), device=device)
        prediction_logits = logits[indices, final]
        predictions = prediction_logits.argmax(dim=-1)
        log_probs = prediction_logits.log_softmax(dim=-1)
        for i, row in enumerate(batch):
            correct_id = tokenizer.encode(row["correct_token"], add_special_tokens=False)[0]
            wrong_id = tokenizer.encode(row["wrong_token"], add_special_tokens=False)[0]
            row = dict(row)
            row["correct_token_id"] = correct_id
            row["wrong_token_id"] = wrong_id
            row["base_prediction_id"] = int(predictions[i])
            row["base_prediction"] = tokenizer.decode([int(predictions[i])])
            row["base_logit_diff"] = float(
                prediction_logits[i, correct_id] - prediction_logits[i, wrong_id]
            )
            row["base_correct_logprob"] = float(log_probs[i, correct_id])
            if int(predictions[i]) == correct_id:
                accepted.append(row)
    return accepted


def _write_outputs(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = pd.DataFrame(rows)
    canonical.to_csv(output_dir / "canonical_ioi_abba.csv", index=False)
    for split in ("discovery", "evaluation"):
        part = canonical[canonical["comparison_split"] == split].reset_index(drop=True)
        part.to_csv(output_dir / f"{split}.csv", index=False)

    # PEAP expects BOS-adjusted start positions and the literal split labels.
    peap_dir = output_dir / "peap"
    peap_dir.mkdir(exist_ok=True)
    peap = canonical.copy()
    peap["split"] = peap["comparison_split"].map(
        {"discovery": "circuit", "evaluation": "eval"}
    )
    starts = peap["section_starts"].map(json.loads)
    for index, span in enumerate(PEAP_SPANS):
        peap[span] = starts.map(lambda values, i=index: values[i] + 1)
    peap["length"] = peap["prompt_length"] + 1
    peap["prompt_id"] = peap["template_id"]
    peap["top_answer"] = peap["base_prediction"]
    clean_columns = [
        "example_id", "prompt", "prompt_id", *PEAP_SPANS, "length",
        "correct_token", "wrong_token", "IO_token", "S1_token", "S2_token",
        "label", "split", "top_answer", "base_logit_diff",
    ]
    peap[clean_columns].to_csv(peap_dir / "IOI_ABBA_data_clean.csv", index=False)
    counter = peap[clean_columns].copy()
    counter["prompt"] = canonical["counterfactual_prompt"]
    counter.to_csv(peap_dir / "IOI_ABBA_data_counter_abc.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discovery-size", type=int, default=500)
    parser.add_argument("--evaluation-size", type=int, default=500)
    parser.add_argument("--candidate-batch", type=int, default=512)
    parser.add_argument("--inference-batch", type=int, default=32)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    names = _single_token_names(tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    required = args.discovery_size + args.evaluation_size
    accepted: list[dict] = []
    seen: set[tuple[str, str]] = set()
    candidate_id = 0
    while len(accepted) < required:
        candidates: list[dict] = []
        while len(candidates) < args.candidate_batch:
            row = _candidate(rng, tokenizer, names, candidate_id)
            candidate_id += 1
            key = (row["prompt"], row["counterfactual_prompt"])
            if key not in seen:
                seen.add(key)
                candidates.append(row)
        accepted.extend(_screen(model, tokenizer, candidates, device, args.inference_batch))
        print(f"accepted {min(len(accepted), required)}/{required}", flush=True)

    accepted = accepted[:required]
    for index, row in enumerate(accepted):
        row["example_id"] = f"ioi_abba_{index:04d}"
        row["comparison_split"] = (
            "discovery" if index < args.discovery_size else "evaluation"
        )
    _write_outputs(accepted, args.output_dir)
    print(f"Wrote {args.discovery_size} discovery and {args.evaluation_size} evaluation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
