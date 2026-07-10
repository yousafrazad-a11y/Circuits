import os
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from venn_circuit_discovery_v2 import VennCircuitDiscoverer, VennBatch

os.environ["HF_TOKEN"] = ""
MODEL_NAME = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def get_token_id(word):
    return tokenizer.encode(" " + word, add_special_tokens=False)[-1]

clean_a = ["Sequence: apple banana mango apple banana"]
corr_a = ["Sequence: apple banana plum pear kiwi"]
clean_b = ["Sequence: car truck plane car truck"]
corr_b = ["Sequence: car truck train bike boat"]

def pad_to_max(enc, max_len):
    if enc.input_ids.shape[1] < max_len:
        pad_shape = (enc.input_ids.shape[0], max_len - enc.input_ids.shape[1])
        pad_tensor = torch.full(pad_shape, tokenizer.pad_token_id, dtype=enc.input_ids.dtype)
        input_ids = torch.cat([pad_tensor, enc.input_ids], dim=1)
        mask_pad = torch.zeros(pad_shape, dtype=enc.attention_mask.dtype)
        attention_mask = torch.cat([mask_pad, enc.attention_mask], dim=1)
        return input_ids, attention_mask
    return enc.input_ids, enc.attention_mask

enc_clean_a = tokenizer(clean_a, return_tensors="pt")
enc_corr_a = tokenizer(corr_a, return_tensors="pt")
max_a = max(enc_clean_a.input_ids.shape[1], enc_corr_a.input_ids.shape[1])
clean_a_ids, clean_a_mask = pad_to_max(enc_clean_a, max_a)
corr_a_ids, _ = pad_to_max(enc_corr_a, max_a)

enc_clean_b = tokenizer(clean_b, return_tensors="pt")
enc_corr_b = tokenizer(corr_b, return_tensors="pt")
max_b = max(enc_clean_b.input_ids.shape[1], enc_corr_b.input_ids.shape[1])
clean_b_ids, clean_b_mask = pad_to_max(enc_clean_b, max_b)
corr_b_ids, _ = pad_to_max(enc_corr_b, max_b)

batch = VennBatch(
    clean_a_input_ids=clean_a_ids,
    clean_b_input_ids=clean_b_ids,
    corr_a_input_ids=corr_a_ids,
    corr_b_input_ids=corr_b_ids,
    answer_positions_a=torch.tensor([clean_a_ids.shape[1] - 1]),
    answer_positions_b=torch.tensor([clean_b_ids.shape[1] - 1]),
    target_a=torch.tensor([get_token_id("mango")]),
    distractor_a=torch.tensor([get_token_id("kiwi")]),
    target_b=torch.tensor([get_token_id("plane")]),
    distractor_b=torch.tensor([get_token_id("boat")]),
    attention_mask_a=clean_a_mask,
    attention_mask_b=clean_b_mask
)

discoverer = VennCircuitDiscoverer(model_name=MODEL_NAME, mode="intersection", target_kl=2.0)
discoverer.fit([batch], epochs=1)
print("SUCCESS!")
