"""IOI data with PEAP's nine logical, variable-length position sections.

Each prompt is divided into a fixed sequence of semantic/structural sections.  A
section may contain a different number of GPT-2 tokens in each example, but all
tokens in a section share the same learned component mask.  Clean and corrupted
prompts are retained only when their corresponding section lengths match.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from comparison_experiments.common_metrics import CircuitMetricAccumulator


SECTION_NAMES = (
    "prefix",
    "IO",
    "and",
    "S1",
    "S1+1",
    "action1",
    "S2",
    "action2",
    "to",
)
NUM_SECTIONS = len(SECTION_NAMES)


BABA_TEMPLATES = [
    "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
    "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
    "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
    "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
    "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
    "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
    "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
    "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
    "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
    "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
]

ABBA_TEMPLATES = [template.replace("{B} and {A}", "{A} and {B}", 1) for template in BABA_TEMPLATES]

TEMPLATES_BY_ORDER = {"abba": ABBA_TEMPLATES, "baba": BABA_TEMPLATES}

NAMES = [
    "Mary", "John", "Alice", "Bob", "Sarah", "Michael", "Emma", "David",
    "Laura", "James", "Emily", "Robert", "Anna", "Daniel", "Jessica",
    "William", "Jennifer", "Thomas", "Linda", "Charles", "Susan", "Joseph",
    "Karen", "George", "Lisa", "Steven", "Nancy", "Edward", "Betty",
    "Andrew", "Helen", "Brian", "Sandra", "Kevin", "Donna", "Jason",
    "Carol", "Matthew", "Ruth", "Anthony", "Sharon", "Mark", "Michelle",
    "Donald", "Kimberly", "Paul", "Deborah", "Richard", "Crystal",
]
PLACES = [
    "store", "park", "office", "school", "hospital", "station", "library",
    "restaurant", "museum", "market", "theater", "garden",
]
OBJECTS = [
    "drink", "book", "gift", "letter", "ball", "ring", "snack", "ticket",
    "flower", "package", "kiss", "note",
]

PLACEHOLDER_PATTERN = re.compile(r"\{(A|B|PLACE|OBJECT)\}")


def _compile_template_regex(template: str) -> tuple[re.Pattern, list[str]]:
    labels: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for match in PLACEHOLDER_PATTERN.finditer(template):
        pieces.append(re.escape(template[cursor:match.start()]))
        pieces.append("(.+?)")
        labels.append(match.group(1))
        cursor = match.end()
    pieces.append(re.escape(template[cursor:]))
    return re.compile("^" + "".join(pieces) + "$"), labels


def _extract_occurrences(sentence: str, template: str) -> Optional[list[dict]]:
    regex, labels = _compile_template_regex(template)
    match = regex.fullmatch(sentence)
    if match is None:
        return None
    return [
        {
            "label": label,
            "value": match.group(index),
            "start": match.start(index),
            "end": match.end(index),
        }
        for index, label in enumerate(labels, start=1)
    ]


def find_template(sentence: str) -> Optional[dict]:
    """Find the template and enforce consistent repeated clean placeholders."""
    for order, templates in TEMPLATES_BY_ORDER.items():
        for template in templates:
            occurrences = _extract_occurrences(sentence, template)
            if occurrences is None:
                continue
            values: dict[str, str] = {}
            consistent = True
            for occurrence in occurrences:
                label = occurrence["label"]
                value = occurrence["value"]
                if label in values and values[label] != value:
                    consistent = False
                    break
                values[label] = value
            if consistent:
                return {
                    "template": template,
                    "order": order,
                    **{key.lower(): value for key, value in values.items()},
                }
    return None


def _move_boundary_whitespace(sections: list[str]) -> list[str]:
    """Attach inter-word whitespace to the following GPT-2 token section."""
    sections = list(sections)
    for index in range(len(sections) - 1):
        match = re.search(r"\s+$", sections[index])
        if match is None:
            continue
        whitespace = match.group(0)
        sections[index] = sections[index][:-len(whitespace)]
        sections[index + 1] = whitespace + sections[index + 1]
    return sections


def split_sentence_into_sections(sentence: str, template: str) -> Optional[list[str]]:
    """Split a clean or corrupted full IOI sentence into eight prompt sections."""
    occurrences = _extract_occurrences(sentence, template)
    if occurrences is None:
        return None
    name_occurrences = [item for item in occurrences if item["label"] in {"A", "B"}]
    if len(name_occurrences) != 4:
        return None

    first, second, repeated_subject, answer = name_occurrences
    between_subject_and_answer = sentence[repeated_subject["end"]:answer["start"]]
    relation_offset = between_subject_and_answer.rfind(" to ")
    if relation_offset < 0:
        return None

    action_end = repeated_subject["end"] + relation_offset
    sections = [
        sentence[:first["start"]],
        sentence[first["start"]:first["end"]],
        sentence[first["end"]:second["start"]],
        sentence[second["start"]:second["end"]],
        sentence[second["end"]:repeated_subject["start"]],
        sentence[repeated_subject["start"]:repeated_subject["end"]],
        sentence[repeated_subject["end"]:action_end],
        sentence[action_end:answer["start"]].rstrip(),
    ]
    sections = _move_boundary_whitespace(sections)
    expected_prompt = sentence[:answer["start"]].rstrip()
    if "".join(sections) != expected_prompt:
        return None
    return sections


def _tokenize_sections(
    sections: Sequence[str], tokenizer: PreTrainedTokenizerBase
) -> Optional[tuple[list[int], list[int]]]:
    section_token_ids = [
        tokenizer.encode(section, add_special_tokens=False) for section in sections
    ]
    flattened = [token for token_ids in section_token_ids for token in token_ids]
    full_prompt_ids = tokenizer.encode("".join(sections), add_special_tokens=False)
    if flattened != full_prompt_ids:
        return None
    return flattened, [len(token_ids) for token_ids in section_token_ids]


def _render_corrupted_template(
    template: str,
    names_by_occurrence: Sequence[str],
    place: str,
    object_name: str,
) -> str:
    name_index = 0
    output: list[str] = []
    cursor = 0
    for match in PLACEHOLDER_PATTERN.finditer(template):
        output.append(template[cursor:match.start()])
        label = match.group(1)
        if label in {"A", "B"}:
            output.append(names_by_occurrence[name_index])
            name_index += 1
        elif label == "PLACE":
            output.append(place)
        else:
            output.append(object_name)
        cursor = match.end()
    output.append(template[cursor:])
    return "".join(output)


def generate_ioi_data(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    template_order: str = "abba",
    seed: int = 0,
) -> List[Dict]:
    """Generate token-alignable IOI/counterfactual pairs from the paper templates."""
    if template_order not in TEMPLATES_BY_ORDER:
        raise ValueError(f"template_order must be one of {tuple(TEMPLATES_BY_ORDER)}")
    random_generator = random.Random(seed)

    names_by_token_length: dict[int, list[str]] = {}
    for name in NAMES:
        length = len(tokenizer.encode(" " + name, add_special_tokens=False))
        names_by_token_length.setdefault(length, []).append(name)
    viable_name_groups = [group for group in names_by_token_length.values() if len(group) >= 5]
    if not viable_name_groups:
        raise ValueError("No GPT-2 name token-length group contains at least five names.")
    name_group = max(viable_name_groups, key=len)

    generated: list[dict] = []
    templates = TEMPLATES_BY_ORDER[template_order]
    for _ in range(num_samples):
        target_name, subject_name, corr_first, corr_second, corr_subject = (
            random_generator.sample(name_group, 5)
        )
        place = random_generator.choice(PLACES)
        object_name = random_generator.choice(OBJECTS)
        template = random_generator.choice(templates)

        clean_sentence = template.format(
            A=target_name,
            B=subject_name,
            PLACE=place,
            OBJECT=object_name,
        )
        corrupted_sentence = _render_corrupted_template(
            template,
            names_by_occurrence=(corr_first, corr_second, corr_subject, corr_first),
            place=place,
            object_name=object_name,
        )
        generated.append(
            {
                "sentence": clean_sentence,
                "corrupted_sentence": corrupted_sentence,
                "ioi_sentences": clean_sentence,
                "corr_ioi_sentences": corrupted_sentence,
                "a": target_name,
                "b": subject_name,
                "template_order": template_order,
            }
        )
    return generated


def _convert_disk_sample(sample: Dict) -> Dict:
    converted = dict(sample)
    if "prompt" in sample and "counterfactual_prompt" in sample:
        return converted
    converted["sentence"] = sample.get("sentence", sample.get("ioi_sentences"))
    converted["corrupted_sentence"] = sample.get(
        "corrupted_sentence", sample.get("corr_ioi_sentences")
    )
    if converted["sentence"] is None or converted["corrupted_sentence"] is None:
        raise KeyError("Dataset rows need clean and corrupted IOI sentence fields.")
    return converted


def load_or_generate_ioi_data(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: Optional[str] = None,
    split: str = "train",
    num_samples: int = 500,
    template_order: str = "abba",
    seed: int = 0,
) -> List[Dict]:
    """Load shared comparison CSVs, a DatasetDict, or generate legacy data."""
    if dataset_path is not None and Path(dataset_path).exists():
        path = Path(dataset_path)
        if path.is_dir() and (path / "discovery.csv").exists():
            filename = "discovery.csv" if split in {"train", "validation"} else "evaluation.csv"
            rows = pd.read_csv(path / filename).to_dict("records")
            return rows[:num_samples]
        if path.suffix.lower() == ".csv":
            rows = pd.read_csv(path).to_dict("records")
            return rows[:num_samples]
        dataset_dict = load_from_disk(dataset_path)
        if split not in dataset_dict:
            raise ValueError(
                f"Split {split!r} is absent; available splits: {list(dataset_dict.keys())}"
            )
        rows = [_convert_disk_sample(sample) for sample in dataset_dict[split]]
        random_generator = random.Random(seed)
        if num_samples < len(rows):
            rows = random_generator.sample(rows, num_samples)
        return rows

    split_offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    return generate_ioi_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        template_order=template_order,
        seed=seed + split_offsets.get(split, 30_000),
    )


class IOIDataset(Dataset):
    """Tokenized IOI prompts using the exact nine-span PEAP human schema."""

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
        template_order: Optional[str] = "abba",
        require_single_token_answers: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.processed_data: list[dict] = []
        rejection_counts: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        for raw_index, raw_item in enumerate(data):
            item = _convert_disk_sample(raw_item)
            if "prompt" in item and "counterfactual_prompt" in item:
                bos_token_id = tokenizer.bos_token_id
                if bos_token_id is None:
                    raise ValueError("The comparison requires a tokenizer BOS token.")
                clean_ids = [bos_token_id] + tokenizer.encode(
                    item["prompt"], add_special_tokens=False
                )
                corrupted_ids = [bos_token_id] + tokenizer.encode(
                    item["counterfactual_prompt"], add_special_tokens=False
                )
                lengths = item["section_lengths"]
                if isinstance(lengths, str):
                    lengths = json.loads(lengths)
                lengths = list(lengths)
                lengths[0] += 1
                target_tokens = tokenizer.encode(
                    item["correct_token"], add_special_tokens=False
                )
                distractor_tokens = tokenizer.encode(
                    item["wrong_token"], add_special_tokens=False
                )
                if (
                    len(lengths) != NUM_SECTIONS
                    or sum(lengths) != len(clean_ids)
                    or len(clean_ids) != len(corrupted_ids)
                    or len(clean_ids) > max_length
                    or (require_single_token_answers and (
                        len(target_tokens) != 1 or len(distractor_tokens) != 1
                    ))
                ):
                    reject("invalid canonical comparison row")
                    continue
                self.processed_data.append(
                    {
                        **item,
                        "source_index": raw_index,
                        "template_order": "abba",
                        "section_lengths": lengths,
                        "clean_ids": clean_ids,
                        "corrupted_ids": corrupted_ids,
                        "target_tokens": target_tokens,
                        "distractor_tokens": distractor_tokens,
                    }
                )
                continue
            clean_sentence = item["sentence"]
            corrupted_sentence = item["corrupted_sentence"]
            template_info = find_template(clean_sentence)
            if template_info is None:
                reject("clean template mismatch")
                continue
            if template_order is not None and template_info["order"] != template_order:
                reject("template order")
                continue

            clean_sections = split_sentence_into_sections(
                clean_sentence, template_info["template"]
            )
            corrupted_sections = split_sentence_into_sections(
                corrupted_sentence, template_info["template"]
            )
            if clean_sections is None or corrupted_sections is None:
                reject("section split")
                continue

            clean_tokenization = _tokenize_sections(clean_sections, tokenizer)
            corrupted_tokenization = _tokenize_sections(corrupted_sections, tokenizer)
            if clean_tokenization is None or corrupted_tokenization is None:
                reject("section/token boundary mismatch")
                continue
            clean_ids, clean_lengths = clean_tokenization
            corrupted_ids, corrupted_lengths = corrupted_tokenization
            if clean_lengths != corrupted_lengths:
                reject("clean/corrupted section-length mismatch")
                continue
            if len(clean_ids) != len(corrupted_ids):
                reject("clean/corrupted prompt-length mismatch")
                continue
            if len(clean_ids) == 0 or len(clean_ids) > max_length:
                reject("prompt length")
                continue

            target = template_info["a"]
            distractor = template_info["b"]
            target_tokens = tokenizer.encode(" " + target, add_special_tokens=False)
            distractor_tokens = tokenizer.encode(" " + distractor, add_special_tokens=False)
            if not target_tokens or not distractor_tokens:
                reject("empty answer tokenization")
                continue
            if require_single_token_answers and (
                len(target_tokens) != 1 or len(distractor_tokens) != 1
            ):
                reject("multi-token answer")
                continue

            self.processed_data.append(
                {
                    **item,
                    "source_index": raw_index,
                    "template": template_info["template"],
                    "template_order": template_info["order"],
                    "clean_sections": clean_sections,
                    "corrupted_sections": corrupted_sections,
                    "section_lengths": clean_lengths,
                    "clean_ids": clean_ids,
                    "corrupted_ids": corrupted_ids,
                    "target": target,
                    "distractor": distractor,
                    "target_tokens": target_tokens,
                    "distractor_tokens": distractor_tokens,
                }
            )

        rejected = len(data) - len(self.processed_data)
        print(
            f"Processed {len(self.processed_data)}/{len(data)} section-aligned samples"
            + (f"; rejected {rejected}: {rejection_counts}" if rejected else "")
        )

    def __len__(self) -> int:
        return len(self.processed_data)

    def _pad(self, token_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        padding_length = self.max_length - len(token_ids)
        padded = list(token_ids) + [self.tokenizer.pad_token_id] * padding_length
        mask = [1] * len(token_ids) + [0] * padding_length
        return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.long)

    def __getitem__(self, index: int) -> Dict:
        item = self.processed_data[index]
        input_ids, attention_mask = self._pad(item["clean_ids"])
        corrupted_input_ids, corrupted_attention_mask = self._pad(item["corrupted_ids"])

        section_ids = torch.repeat_interleave(
            torch.arange(NUM_SECTIONS, dtype=torch.long),
            torch.tensor(item["section_lengths"], dtype=torch.long),
        )
        section_ids = F.pad(
            section_ids,
            (0, self.max_length - section_ids.numel()),
            value=0,
        )
        prompt_length = len(item["clean_ids"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "corrupted_input_ids": corrupted_input_ids,
            "corrupted_attention_mask": corrupted_attention_mask,
            "section_ids": section_ids,
            "section_lengths": torch.tensor(item["section_lengths"], dtype=torch.long),
            "target_tokens": torch.tensor(item["target_tokens"], dtype=torch.long),
            "distractor_tokens": torch.tensor(item["distractor_tokens"], dtype=torch.long),
            "T_Start": torch.tensor(prompt_length, dtype=torch.long),
            "T_End": torch.tensor(prompt_length + 1, dtype=torch.long),
            "D_Start": torch.tensor(prompt_length, dtype=torch.long),
            "D_End": torch.tensor(prompt_length + 1, dtype=torch.long),
            "T_len": torch.tensor(1, dtype=torch.long),
            "D_len": torch.tensor(1, dtype=torch.long),
            "template_order_id": torch.tensor(
                0 if item["template_order"] == "abba" else 1, dtype=torch.long
            ),
            "source_index": torch.tensor(item["source_index"], dtype=torch.long),
        }


def _is_circuit_model(model: nn.Module) -> bool:
    return bool(getattr(model, "supports_position_aware_pruning", False))


def _forward_for_batch(model: nn.Module, batch: Dict) -> object:
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if _is_circuit_model(model):
        kwargs.update(
            corrupted_input_ids=batch["corrupted_input_ids"],
            section_ids=batch["section_ids"],
        )
    return model(**kwargs)


def run_evaluation(
    model_to_eval: nn.Module,
    model_name: str,
    full_model_for_faithfulness: Optional[nn.Module],
    dataloader: DataLoader,
    device: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate IOI preference and faithfulness at the answer-prediction token."""
    model_to_eval.eval()
    if full_model_for_faithfulness is not None:
        full_model_for_faithfulness.eval()

    accumulator = CircuitMetricAccumulator()
    iterator = tqdm(dataloader, desc=f"Evaluating {model_name}", leave=False)
    with torch.no_grad():
        for batch in iterator:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = _forward_for_batch(model_to_eval, batch)
            batch_size = outputs.logits.size(0)
            indices = torch.arange(batch_size, device=device)
            positions = batch["T_Start"] - 1
            targets = batch["target_tokens"][:, 0]
            distractors = batch["distractor_tokens"][:, 0]

            prediction_logits = outputs.logits[indices, positions]
            if full_model_for_faithfulness is not None:
                full_outputs = full_model_for_faithfulness(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                full_prediction_logits = full_outputs.logits[indices, positions]
                accumulator.update(prediction_logits, full_prediction_logits, targets, distractors)
            else:
                accumulator.update(prediction_logits, prediction_logits, targets, distractors)

    results = accumulator.compute()
    if verbose:
        print(f"\n{model_name}: {results}")
    return results


def filter_dataset_by_model_correctness(
    data_list: List[Dict],
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    batch_size: int = 32,
    max_length: int = 64,
    template_order: Optional[str] = "abba",
) -> List[Dict]:
    """Keep raw rows where the base model prefers the IO over the distractor."""
    if not data_list:
        return []
    temporary_dataset = IOIDataset(
        data_list,
        tokenizer,
        max_length=max_length,
        template_order=template_order,
    )
    temporary_loader = DataLoader(temporary_dataset, batch_size=batch_size, shuffle=False)
    retained_source_indices: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(temporary_loader, desc="Filtering model-correct IOI prompts"):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            batch_size_current = outputs.logits.size(0)
            indices = torch.arange(batch_size_current, device=device)
            positions = batch["T_Start"] - 1
            targets = batch["target_tokens"][:, 0]
            distractors = batch["distractor_tokens"][:, 0]
            logits = outputs.logits[indices, positions]
            correct = logits[indices, targets] >= logits[indices, distractors]
            retained_source_indices.extend(
                batch["source_index"][correct].detach().cpu().tolist()
            )

    retained = [data_list[index] for index in retained_source_indices]
    print(f"Retained {len(retained)}/{len(data_list)} model-correct prompts")
    return retained
