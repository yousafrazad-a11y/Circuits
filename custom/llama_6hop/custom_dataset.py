import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Custom6HopDatasetLlama(Dataset):
    """Dataset adapted for Llama tokenizer loading from dataset.jsonl."""

    def __init__(self, data: list, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.processed_data = []

        for item in data:
            # Targets and distractors are provided in the dataset
            target = item['clean_target']
            distractor = item['corrupted_target']

            # Tokenize targets
            target_tokens = tokenizer.encode(target, add_special_tokens=False)
            distractor_tokens = tokenizer.encode(distractor, add_special_tokens=False)

            self.processed_data.append({
                **item,
                'sentence': item['clean_prompt'],
                'corrupted_sentence': item['corrupted_prompt'],
                'target': target,
                'distractor': distractor,
                'target_tokens': target_tokens,
                'distractor_tokens': distractor_tokens,
                'template_order': 'unknown',
            })

        print(f"Processed {len(self.processed_data)} valid samples from {len(data)} total")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        # Tokenize sentences
        inputs = self.tokenizer(
            item['sentence'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        corrupted_inputs = self.tokenizer(
            item['corrupted_sentence'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        # For our JSONL dataset, the clean_prompt ends exactly before the target.
        # Thus, T_Start is the length of the tokenized prompt including BOS token.
        T_Start = len(self.tokenizer.encode(item['sentence'], add_special_tokens=True))
        T_End = T_Start + len(item['target_tokens'])
        T_len = T_End - T_Start

        # D_Start is based on corrupted_sentence
        D_Start = len(self.tokenizer.encode(item['corrupted_sentence'], add_special_tokens=True))
        D_End = D_Start + len(item['distractor_tokens'])
        D_len = D_End - D_Start

        # Pad target/distractor tokens to fixed size (e.g., 5 tokens)
        target_tokens = item['target_tokens'][:5]
        distractor_tokens = item['distractor_tokens'][:5]
        target_tokens = target_tokens + [self.tokenizer.pad_token_id] * (5 - len(target_tokens))
        distractor_tokens = distractor_tokens + [self.tokenizer.pad_token_id] * (5 - len(distractor_tokens))

        return {
            "input_ids": inputs['input_ids'].squeeze(0),
            "attention_mask": inputs['attention_mask'].squeeze(0),
            "corrupted_input_ids": corrupted_inputs['input_ids'].squeeze(0),
            "corrupted_attention_mask": corrupted_inputs['attention_mask'].squeeze(0),
            "target_tokens": torch.tensor(target_tokens, dtype=torch.long),
            "distractor_tokens": torch.tensor(distractor_tokens, dtype=torch.long),
            "T_Start": torch.tensor(T_Start, dtype=torch.long),
            "T_End": torch.tensor(T_End, dtype=torch.long),
            "D_Start": torch.tensor(D_Start, dtype=torch.long),
            "D_End": torch.tensor(D_End, dtype=torch.long),
            "T_len": torch.tensor(T_len, dtype=torch.long),
            "D_len": torch.tensor(D_len, dtype=torch.long),
            "template_order": item['template_order'],
        }

def load_jsonl_dataset(path: str):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def filter_dataset_by_model_correctness(data_list, model, tokenizer, device, batch_size=32):
    """
    Filter JSONL data samples, keeping only those where the base model
    assigns a higher logit to the target than the distractor.
    """
    if not data_list:
        return []

    print(f"Filtering {len(data_list)} samples for base model correctness...")

    temp_dataset = Custom6HopDatasetLlama(data_list, tokenizer)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False)

    valid_indices = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(temp_loader, desc="Checking model predictions")):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            current_batch_size = outputs.logits.size(0)

            for i in range(current_batch_size):
                t_start = batch['T_Start'][i].item() - 1
                d_start = batch['D_Start'][i].item() - 1

                target_token_id = batch['target_tokens'][i][0].item()
                distractor_token_id = batch['distractor_tokens'][i][0].item()

                target_logit = outputs.logits[i, t_start, target_token_id].item()
                distractor_logit = outputs.logits[i, d_start, distractor_token_id].item()

                if target_logit >= distractor_logit:
                    global_idx = (batch_idx * batch_size) + i
                    valid_indices.append(global_idx)

    filtered_data = [data_list[i] for i in valid_indices]

    print(f"  -> Retained: {len(filtered_data)}/{len(data_list)} "
          f"({len(filtered_data) / len(data_list) * 100:.2f}%)")

    return filtered_data

import torch.nn.functional as F

def custom_run_evaluation(
    model_to_eval,
    model_name: str,
    full_model: torch.nn.Module,
    dataloader,
    device,
    tokenizer=None,
):
    print(f"\nCustom Evaluating: {model_name}")
    model_to_eval.eval()
    if full_model is not None:
        full_model.eval()

    metrics = {
        "base_clean_acc": 0.0, "base_clean_em": 0.0,
        "base_corr_acc": 0.0, "base_corr_em": 0.0,
        "pruned_clean_acc": 0.0, "pruned_clean_em": 0.0,
        "kl_div": 0.0
    }
    valid_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = val.to(device)

            batch_size = batch['input_ids'].size(0)

            # Determine the full model to use for clean/corrupted baselines
            actual_full_model = full_model if full_model is not None else model_to_eval

            # 1. Full Model on Clean Stream
            clean_full_out = actual_full_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            
            # 2. Full Model on Corrupted Stream
            corr_full_out = actual_full_model(input_ids=batch['corrupted_input_ids'], attention_mask=batch['corrupted_attention_mask'])

            # 3. Pruned Model (Dual Stream)
            if hasattr(model_to_eval, 'set_final_circuit_mode') or 'circuit' in model_name.lower() or 'val' in model_name.lower() or 'pre-final' in model_name.lower():
                pruned_out = model_to_eval(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    corrupted_input_ids=batch['corrupted_input_ids']
                )
            else:
                pruned_out = clean_full_out

            for i in range(batch_size):
                t_start = batch['T_Start'][i].item() - 1
                d_start = batch['D_Start'][i].item() - 1
                
                target_token_id = batch['target_tokens'][i][0].item()
                distractor_token_id = batch['distractor_tokens'][i][0].item()

                # --- Base Clean ---
                bc_target_logit = clean_full_out.logits[i, t_start, target_token_id].item()
                bc_dist_logit = clean_full_out.logits[i, t_start, distractor_token_id].item()
                if bc_target_logit > bc_dist_logit: metrics["base_clean_acc"] += 1
                if torch.argmax(clean_full_out.logits[i, t_start, :]).item() == target_token_id: metrics["base_clean_em"] += 1

                # --- Base Corrupted ---
                bcor_target_logit = corr_full_out.logits[i, d_start, target_token_id].item()
                bcor_dist_logit = corr_full_out.logits[i, d_start, distractor_token_id].item()
                if bcor_dist_logit > bcor_target_logit: metrics["base_corr_acc"] += 1
                if torch.argmax(corr_full_out.logits[i, d_start, :]).item() == distractor_token_id: metrics["base_corr_em"] += 1

                # --- Pruned Clean ---
                p_target_logit = pruned_out.logits[i, t_start, target_token_id].item()
                p_dist_logit = pruned_out.logits[i, t_start, distractor_token_id].item()
                if p_target_logit > p_dist_logit: metrics["pruned_clean_acc"] += 1
                if torch.argmax(pruned_out.logits[i, t_start, :]).item() == target_token_id: metrics["pruned_clean_em"] += 1

                # --- KL Div ---
                t_end = batch['T_End'][i].item() - 1
                valid_length = batch['attention_mask'][i].sum().item()
                end_pos = min(t_end, int(valid_length))
                if t_start < end_pos:
                    model_logits = pruned_out.logits[i, t_start:end_pos, :]
                    full_logits = clean_full_out.logits[i, t_start:end_pos, :]
                    
                    min_len = min(model_logits.size(0), full_logits.size(0))
                    if min_len > 0:
                        kl = F.kl_div(
                            F.log_softmax(model_logits[:min_len, :], dim=-1),
                            F.log_softmax(full_logits[:min_len, :], dim=-1),
                            log_target=True, reduction='batchmean'
                        ).item()
                        metrics["kl_div"] += kl

                valid_samples += 1

    # Normalize
    if valid_samples > 0:
        for k in metrics:
            metrics[k] /= valid_samples

    print(f"Metrics over {valid_samples} samples: {metrics}")
    return metrics
