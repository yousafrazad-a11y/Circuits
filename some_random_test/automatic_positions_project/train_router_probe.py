#!/usr/bin/env python3
"""Train only a token router over frozen positional pruning masks."""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from automatic_positions_project.src.metrics import routing_diagnostics
from automatic_positions_project.src.router import ROUTER_INPUTS, TokenRouter, extract_frozen_features, router_input_size
from automatic_positions_project.src.routed_circuit import routed_forward
from comparison_experiments.common_metrics import CircuitMetricAccumulator
from comparison_experiments.position_aware_node_pruning.dataset.ioi import IOIDataset, NUM_SECTIONS, SECTION_NAMES, filter_dataset_by_model_correctness, load_or_generate_ioi_data
from comparison_experiments.position_aware_node_pruning.models.gpt2_circuit import PrunableGPT2LMHeadModel, PruningConfig
from comparison_experiments.position_aware_node_pruning.utils import disable_dropout

ROOT = Path(__file__).resolve().parent

def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expert-checkpoint", type=Path, required=True)
    p.add_argument("--router-input", choices=ROUTER_INPUTS, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="gpt2"); p.add_argument("--data-path", type=Path, default=ROOT.parent/"comparison_experiments/data")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--residual-layer", type=int, default=2); p.add_argument("--temperature-start", type=float, default=2.0)
    p.add_argument("--temperature-end", type=float, default=.25); p.add_argument("--validation-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-routing", choices=("hard_gumbel", "soft"), default="hard_gumbel")
    p.add_argument("--load-balance-weight", type=float, default=0.0)
    p.add_argument("--no-model-correct-filter", action="store_true")
    return p.parse_args()

def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}

def load_experts(name, path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or not isinstance(saved.get("gate_state"), dict):
        raise ValueError("Expected a final or periodic positional gate checkpoint")
    allowed = PruningConfig.__dataclass_fields__
    cfg = {k:v for k,v in saved.get("pruning_config",{}).items() if k in allowed}; cfg["num_sections"] = NUM_SECTIONS
    model = PrunableGPT2LMHeadModel.from_pretrained_with_pruning(name, PruningConfig(**cfg))
    params = dict(model.named_parameters())
    with torch.no_grad():
        for key,value in saved["gate_state"].items():
            if key not in params or params[key].shape != value.shape: raise ValueError(f"Incompatible gate {key}")
            params[key].copy_(value)
    model.to(device).eval(); disable_dropout(model)
    for parameter in model.parameters(): parameter.requires_grad=False
    return model

def make_loaders(a, tokenizer, full, device):
    out={}
    for split,n in (("train",500),("validation",100),("test",500)):
        rows=load_or_generate_ioi_data(tokenizer,str(a.data_path),split,n,"abba",a.seed)
        if not a.no_model_correct_filter:
            rows=filter_dataset_by_model_correctness(rows,full,tokenizer,str(device),a.batch_size,64,"abba")
        ds=IOIDataset(rows,tokenizer,max_length=64,template_order="abba")
        out[split]=DataLoader(ds,batch_size=a.batch_size,shuffle=split=="train",num_workers=a.num_workers,
                              pin_memory=device.type=="cuda",persistent_workers=a.num_workers>0)
    return out

def answer_logits(output,batch):
    rows=torch.arange(output.logits.shape[0],device=output.logits.device)
    return output.logits[rows,batch["T_Start"]-1]

def aggregate_diagnostics(items):
    total = max(sum(x["valid_token_count"] for x in items), 1)
    result = {}
    for key in ("assignment_entropy", "mean_max_probability", "mean_top2_margin"):
        result[key] = sum(x[key] * x["valid_token_count"] for x in items) / total
    counts = torch.tensor([x["expert_counts"] for x in items]).sum(0)
    confusion = torch.tensor([x["section_to_expert_counts"] for x in items]).sum(0)
    result["expert_counts"] = counts.tolist()
    result["expert_usage"] = (counts.float() / total).tolist()
    result["experts_used"] = int((counts > 0).sum().item())
    result["valid_token_count"] = total
    result["raw_section_agreement"] = float(confusion.diag().sum().item() / total)
    result["section_to_expert_counts"] = confusion.tolist()
    row_totals = confusion.sum(-1, keepdim=True).clamp_min(1)
    result["section_to_expert_distribution"] = (confusion.float() / row_totals).tolist()
    return result

@torch.no_grad()
def evaluate(router,circuit,full,loader,a,device,routing_mode="hard"):
    router.eval(); acc=CircuitMetricAccumulator(); diag=[]
    for raw in loader:
        b=move(raw,device); base=full(input_ids=b["input_ids"],attention_mask=b["attention_mask"],use_cache=False,return_dict=True)
        if routing_mode == "oracle":
            cut=circuit(input_ids=b["input_ids"],corrupted_input_ids=b["corrupted_input_ids"],attention_mask=b["attention_mask"],
                        corrupted_attention_mask=b["corrupted_attention_mask"],section_ids=b["section_ids"],return_dict=True)
        else:
            f=extract_frozen_features(full,b["input_ids"],b["attention_mask"],a.router_input,a.residual_layer)
            soft,logits=router(f,temperature=a.temperature_end,hard=False)
            if routing_mode == "hard":
                weights=F.one_hot(logits.argmax(-1),NUM_SECTIONS).to(soft.dtype)
            elif routing_mode == "soft":
                weights=soft
            else:
                raise ValueError("Unknown routing mode {}".format(routing_mode))
            cut=routed_forward(circuit,b,weights)
            diag.append(routing_diagnostics(soft,b["attention_mask"],b["section_ids"],weights,len(SECTION_NAMES)))
        acc.update(answer_logits(cut,b),answer_logits(base,b),b["target_tokens"][:,0],b["distractor_tokens"][:,0])
    result=acc.compute()
    if diag:
        result.update(aggregate_diagnostics(diag))
    result["routing_mode"] = routing_mode
    return result

def main():
    a=args(); random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    a.output_dir.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok=GPT2TokenizerFast.from_pretrained(a.model); tok.pad_token=tok.pad_token or tok.eos_token
    full=GPT2LMHeadModel.from_pretrained(a.model).to(device).eval(); disable_dropout(full)
    for p in full.parameters(): p.requires_grad=False
    circuit=load_experts(a.model,a.expert_checkpoint,device); data=make_loaders(a,tok,full,device)
    router=TokenRouter(router_input_size(a.router_input,full.config.hidden_size),a.hidden_size,NUM_SECTIONS).to(device)
    opt=AdamW(router.parameters(),lr=a.learning_rate); log=a.output_dir/"metrics.jsonl"; log.write_text("",encoding="utf-8")
    oracle=evaluate(router,circuit,full,data["test"],a,device,"oracle"); print(f"Frozen human-section oracle: {oracle}",flush=True); best=float("inf")
    for epoch in range(a.epochs):
        router.train(); frac=epoch/max(a.epochs-1,1); temp=a.temperature_start*math.exp(math.log(a.temperature_end/a.temperature_start)*frac)
        sums={"loss":0.,"kl":0.,"task":0.,"balance":0.,"grad_norm":0.,"soft_entropy":0.,"mean_max_probability":0.}
        for raw in data["train"]:
            b=move(raw,device); opt.zero_grad(set_to_none=True)
            features=extract_frozen_features(full,b["input_ids"],b["attention_mask"],a.router_input,a.residual_layer)
            use_hard=a.train_routing == "hard_gumbel"
            weights,logits=router(features,temperature=temp,hard=use_hard)
            soft_probabilities=F.softmax(logits/temp,dim=-1)
            cut=routed_forward(circuit,b,weights)
            with torch.no_grad(): base=full(input_ids=b["input_ids"],attention_mask=b["attention_mask"])
            cl,fl=answer_logits(cut,b),answer_logits(base,b)
            kl=F.kl_div(F.log_softmax(cl,-1),F.log_softmax(fl,-1),reduction="batchmean",log_target=True)
            rows=torch.arange(cl.shape[0],device=device); margin=cl[rows,b["target_tokens"][:,0]]-cl[rows,b["distractor_tokens"][:,0]]
            task=F.relu(4.-margin).mean()
            valid=b["attention_mask"].bool(); valid_probabilities=soft_probabilities[valid]
            mean_usage=valid_probabilities.mean(0)
            balance=NUM_SECTIONS*(mean_usage.square().sum())-1.0
            loss=1.5*kl+task+a.load_balance_weight*balance
            loss.backward()
            grad_norm=math.sqrt(sum(float(p.grad.detach().float().square().sum()) for p in router.parameters() if p.grad is not None))
            opt.step()
            entropy=-(valid_probabilities.clamp_min(1e-12)*valid_probabilities.clamp_min(1e-12).log()).sum(-1).mean()
            maximum=valid_probabilities.max(-1).values.mean()
            for k,v in (("loss",loss),("kl",kl),("task",task),("balance",balance),("grad_norm",grad_norm),("soft_entropy",entropy),("mean_max_probability",maximum)):
                sums[k]+=float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
        record={"event":"epoch","epoch":epoch+1,"temperature":temp,"train_routing":a.train_routing,"load_balance_weight":a.load_balance_weight,**{k:v/max(len(data["train"]),1) for k,v in sums.items()}}
        if (epoch+1)%a.validation_interval==0 or epoch+1==a.epochs:
            val_hard=evaluate(router,circuit,full,data["validation"],a,device,"hard")
            val_soft=evaluate(router,circuit,full,data["validation"],a,device,"soft")
            record["validation_hard"]=val_hard; record["validation_soft"]=val_soft
            state={"epoch":epoch+1,"router_input":a.router_input,"router_state":router.state_dict(),"optimizer_state":opt.state_dict(),"args":vars(a),"validation":val_hard,"validation_hard":val_hard,"validation_soft":val_soft,"oracle_test":oracle}
            torch.save(state,a.output_dir/f"epoch_{epoch+1:04d}.pt")
            if val_hard["kl_div"]<best: best=val_hard["kl_div"]; torch.save(state,a.output_dir/"best.pt")
        with log.open("a",encoding="utf-8") as h: h.write(json.dumps(record,default=str,sort_keys=True)+"\n")
        print(json.dumps(record,default=str,sort_keys=True),flush=True)
    final_hard=evaluate(router,circuit,full,data["test"],a,device,"hard")
    final_soft=evaluate(router,circuit,full,data["test"],a,device,"soft")
    best_state=torch.load(a.output_dir/"best.pt",map_location=device,weights_only=False)
    router.load_state_dict(best_state["router_state"])
    best_hard=evaluate(router,circuit,full,data["test"],a,device,"hard")
    best_soft=evaluate(router,circuit,full,data["test"],a,device,"soft")
    final={"router_test":final_hard,"final_router_hard_test":final_hard,"final_router_soft_test":final_soft,"best_epoch":best_state["epoch"],"best_router_hard_test":best_hard,"best_router_soft_test":best_soft,"human_section_oracle_test":oracle,"section_names":SECTION_NAMES}
    (a.output_dir/"final_metrics.json").write_text(json.dumps(final,indent=2),encoding="utf-8")

if __name__=="__main__": main()
