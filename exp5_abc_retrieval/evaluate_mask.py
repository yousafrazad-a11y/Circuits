import os
import sys
import json
import torch
import re
import csv
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))
from pruning_manager import CircuitPruningManager

class MemoryDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, tokenizer):
    tokenizer.padding_side = 'left'
    clean_texts = [item['clean_prompt'] for item in batch]
    corr_texts = [item['corrupt_prompt'] for item in batch]

    clean_encoded = tokenizer(clean_texts, padding=True, return_tensors='pt', add_special_tokens=True)
    corr_encoded = tokenizer(corr_texts, padding=True, return_tensors='pt', add_special_tokens=True)

    max_len = max(clean_encoded['input_ids'].size(1), corr_encoded['input_ids'].size(1))

    def pad_left(tensor, pad_val, target_len):
        pad_len = target_len - tensor.size(1)
        if pad_len > 0:
            pads = torch.full((tensor.size(0), pad_len), pad_val, dtype=tensor.dtype)
            return torch.cat([pads, tensor], dim=1)
        return tensor

    input_ids = pad_left(clean_encoded['input_ids'], tokenizer.pad_token_id, max_len)
    attention_mask = pad_left(clean_encoded['attention_mask'], 0, max_len)
    corrupted_input_ids = pad_left(corr_encoded['input_ids'], tokenizer.pad_token_id, max_len)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'corrupted_input_ids': corrupted_input_ids,
        'ld_candidates': [item['ld_candidates'] for item in batch],
        'clean_answers': [item['clean_answer'] for item in batch],
    }

def evaluate_accuracies(model, dataloader, tokenizer, device="cuda"):
    """Logit-difference accuracy on ld_candidates + generative accuracy.

    prob: logit(clean_answer) > logit(corrupt_answer) at the last position.
    gen: greedy first generated token == clean_answer token.
    Returns (prob_acc, gen_acc, mean_logit_diff).
    """
    model.eval()
    prob_correct = 0
    gen_correct = 0
    total = 0
    total_ld = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            corrupted_input_ids = batch['corrupted_input_ids'].to(device)
            ld_cands = batch['ld_candidates']
            clean_answers = batch['clean_answers']

            if hasattr(model, 'set_final_circuit_mode'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    corrupted_input_ids=corrupted_input_ids,
                    use_cache=False
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )

            last_logits = outputs.logits[:, -1, :]

            for i in range(len(ld_cands)):
                clean_tok = tokenizer.encode(ld_cands[i][0], add_special_tokens=False)[0]
                corr_tok = tokenizer.encode(ld_cands[i][1], add_special_tokens=False)[0]
                ld = last_logits[i, clean_tok].item() - last_logits[i, corr_tok].item()
                total_ld += ld
                if ld > 0:
                    prob_correct += 1

            # generative: 1 greedy token
            next_toks = torch.argmax(last_logits, dim=-1)
            for i in range(len(clean_answers)):
                clean_tok = tokenizer.encode(clean_answers[i], add_special_tokens=False)[0]
                if next_toks[i].item() == clean_tok:
                    gen_correct += 1

            total += len(ld_cands)

    return prob_correct / total, gen_correct / total, total_ld / total

def get_active_heads(mask_path):
    masks = torch.load(mask_path, weights_only=True)
    total = 0
    active = 0
    for k, v in masks.items():
        if 'head_gates' in k:
            total += v.numel()
            active += v.sum().item()
    return active, total

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained pruning mask on an A/B/C retrieval dataset.")
    parser.add_argument("--mask", type=str, default=None, help="Path to the trained .pt mask file. Omit for base-model eval.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset file name in exp5_abc_retrieval/datasets/ (e.g., A_test.jsonl) or direct path.")
    parser.add_argument("--output", type=str, required=True, help="Name of the output CSV file to save in the results directory.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = args.dataset if args.dataset.endswith(".jsonl") and os.path.exists(args.dataset) \
        else f"exp5_abc_retrieval/datasets/{args.dataset}"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    ds = MemoryDataset(dataset_path)

    manager = CircuitPruningManager(model_name="meta-llama/Llama-3.2-1B", device=device)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, manager.tokenizer))

    active_heads, total_heads = (None, None)
    if args.mask:
        print(f"Loading mask from {args.mask}...")
        mask_state = torch.load(args.mask, weights_only=True)
        config = manager._get_default_config()
        if any('mlp_block_gate' in k for k in mask_state):
            config.prune_mlp_blocks = True
            print("Mask contains mlp_block_gate entries -> enabling MLP block pruning.")
        manager.initialize_model(config)
        manager.load_masks(args.mask)
        active_heads, total_heads = get_active_heads(args.mask)
        print(f"Active Heads in mask: {active_heads}/{total_heads}")
    else:
        manager.initialize_model(manager._get_default_config())

    ds_name = os.path.basename(dataset_path).replace(".jsonl", "")
    print(f"\n--- EVALUATING {ds_name.upper()} ---")

    print("Evaluating Base Model...")
    base_prob, base_gen, base_ld = evaluate_accuracies(manager.baseline_model, dl, manager.tokenizer, device)
    print(f"Base Prob: {base_prob:.4f} | Base Gen: {base_gen:.4f} | Base LD: {base_ld:.4f}")

    circ_prob = circ_gen = circ_ld = kl = None
    if args.mask:
        print("Evaluating Circuit Model...")
        manager.use_model(enable_masks=True)
        circ_prob, circ_gen, circ_ld = evaluate_accuracies(manager.model, dl, manager.tokenizer, device)
        print("Evaluating KL Divergence...")
        kl = manager.evaluate_kl_divergence(dl)
        print(f"Circ Prob: {circ_prob:.4f} | Circ Gen: {circ_gen:.4f} | Circ LD: {circ_ld:.4f} | KL: {kl:.4f}")

    os.makedirs("exp5_abc_retrieval/results", exist_ok=True)
    output_path = os.path.join("exp5_abc_retrieval/results", args.output)
    if not output_path.endswith('.csv'):
        output_path += '.csv'

    file_exists = os.path.isfile(output_path)
    keys = ["mask_path", "dataset", "base_prob_acc", "base_gen_acc", "base_logit_diff",
            "circ_prob_acc", "circ_gen_acc", "circ_logit_diff", "kl_divergence",
            "active_heads", "total_heads"]

    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "mask_path": args.mask or "BASE",
            "dataset": ds_name,
            "base_prob_acc": f"{base_prob:.4f}",
            "base_gen_acc": f"{base_gen:.4f}",
            "base_logit_diff": f"{base_ld:.4f}",
            "circ_prob_acc": f"{circ_prob:.4f}" if circ_prob is not None else "",
            "circ_gen_acc": f"{circ_gen:.4f}" if circ_gen is not None else "",
            "circ_logit_diff": f"{circ_ld:.4f}" if circ_ld is not None else "",
            "kl_divergence": f"{kl:.4f}" if kl is not None else "",
            "active_heads": active_heads if active_heads is not None else "",
            "total_heads": total_heads if total_heads is not None else "",
        })

    print(f"\nResults appended to {output_path}")

if __name__ == "__main__":
    main()
