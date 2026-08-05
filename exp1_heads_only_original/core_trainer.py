import torch
import torch.nn.functional as F
from torch.optim import AdamW

def collate_fn(batch, tokenizer):
    clean_input_ids = []
    corr_input_ids = []
    target_tokens = []
    distractor_tokens = []
    answer_positions = []
    
    for item in batch:
        prompt = item["clean_prompt"]
        clean_input_ids.append(torch.tensor(tokenizer.encode(prompt, add_special_tokens=True)))
        corr_input_ids.append(torch.tensor(tokenizer.encode(item["corr_prompt"] if "corr_prompt" in item else item.get("corrupted_prompt", ""), add_special_tokens=True)))
        
        target = item["target"]
        distractor = item["distractor"]
        target_token = tokenizer.encode(" " + target, add_special_tokens=False)[-1]
        distractor_token = tokenizer.encode(" " + distractor, add_special_tokens=False)[-1]
        
        target_tokens.append(target_token)
        distractor_tokens.append(distractor_token)
        
    max_len_clean = max(len(seq) for seq in clean_input_ids)
    max_len_corr = max(len(seq) for seq in corr_input_ids)
    max_len = max(max_len_clean, max_len_corr)
    
    clean_padded, corr_padded, attn_mask = [], [], []
    
    for c_seq, corr_seq in zip(clean_input_ids, corr_input_ids):
        pad_len_c = max_len - len(c_seq)
        pad_len_corr = max_len - len(corr_seq)
        clean_padded.append(torch.cat([c_seq, torch.full((pad_len_c,), tokenizer.pad_token_id)]))
        corr_padded.append(torch.cat([corr_seq, torch.full((pad_len_corr,), tokenizer.pad_token_id)]))
        attn_mask.append(torch.cat([torch.ones(len(c_seq)), torch.zeros(pad_len_c)]))
        answer_positions.append(len(c_seq) - 1)
        
    return {
        "clean_input_ids": torch.stack(clean_padded).long(),
        "corr_input_ids": torch.stack(corr_padded).long(),
        "attention_mask": torch.stack(attn_mask).long(),
        "target_tokens": torch.tensor(target_tokens).long(),
        "distractor_tokens": torch.tensor(distractor_tokens).long(),
        "answer_positions": torch.tensor(answer_positions).long(),
    }

def train_phase(model, dataloader, epochs, lr, device="cuda", compile_model=True):
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if "log_alpha" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float() 
            trainable_params += param.numel()
            
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    
    total_steps = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_kl = 0
        epoch_task = 0
        epoch_sparsity = 0
        
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_size_curr = batch["clean_input_ids"].size(0)
            
            model.eval()
            with torch.no_grad():
                golden_outputs = model(
                    input_ids=batch["clean_input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                golden_logits = golden_outputs.logits.detach()
            
            model.train()
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["clean_input_ids"],
                corrupted_input_ids=batch["corr_input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            
            pos = batch["answer_positions"]
            batch_indices = torch.arange(batch_size_curr, device=device)
            
            circuit_logits = logits[batch_indices, pos].float()
            target_logits = golden_logits[batch_indices, pos].float()
            
            kl_loss = F.kl_div(
                F.log_softmax(circuit_logits, dim=-1),
                F.log_softmax(target_logits, dim=-1),
                reduction='batchmean',
                log_target=True
            )
            
            logit_good = logits[batch_indices, pos, batch["target_tokens"]].float()
            logit_bad = logits[batch_indices, pos, batch["distractor_tokens"]].float()
            task_loss = F.relu(4.0 - (logit_good - logit_bad)).mean()
            
            if hasattr(model, 'get_sparsity_loss'):
                sparsity_loss = model.get_sparsity_loss(step=total_steps)["total_sparsity"]
            else:
                sparsity_loss = model._orig_mod.get_sparsity_loss(step=total_steps)["total_sparsity"]
            
            loss = kl_loss * 1.5 + sparsity_loss + task_loss
            loss.backward()
            optimizer.step()
            total_steps += 1
            
            with torch.no_grad():
                base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                for name, module in base_model.named_modules():
                    if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                        module.log_alpha.clamp_(-5.0, 5.0)
                        
            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_task += task_loss.item()
            epoch_sparsity += sparsity_loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.3f} | KL: {epoch_kl/len(dataloader):.3f} | Task: {epoch_task/len(dataloader):.3f} | Sparsity: {epoch_sparsity/len(dataloader):.3f}")
            
    return model._orig_mod if hasattr(model, '_orig_mod') else model

def extract_mask(model):
    model.eval()
    mask = {}
    with torch.no_grad():
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        for name, module in base_model.named_modules():
            if hasattr(module, 'log_alpha') and isinstance(module.log_alpha, torch.nn.Parameter):
                s = torch.sigmoid(module.log_alpha)
                s_stretched = s * 1.2 - 0.1
                mask[name] = (s_stretched > 0.5).bool().cpu()
    return mask

