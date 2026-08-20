"""Train an eight-section position-aware node circuit for GPT-2 IOI."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from .dataset.ioi import (
    IOIDataset,
    NUM_SECTIONS,
    filter_dataset_by_model_correctness,
    load_or_generate_ioi_data,
    run_evaluation,
)
from .models.gpt2_circuit import PrunableGPT2LMHeadModel, PruningConfig
from .models.l0 import HardConcreteGate
from .utils import analyze_and_finalize_circuit, disable_dropout, save_gate_state


MODEL_NAME = "gpt2"
NUM_EPOCHS = 500
VALIDATION_INTERVAL = 10
LEARNING_RATE = 3e-2
MAX_SEQ_LEN = 64
ACCURACY_BUDGET = 0.05
TEMPLATE_ORDER = "abba"  # Train BABA separately to preserve section semantics.
RANDOM_SEED = 42
DATASET_PATH = os.environ.get("IOI_DATASET_PATH")
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "outputs"
OUTPUT_PATH = OUTPUT_DIRECTORY / f"ioi_{TEMPLATE_ORDER}_section_gates.pt"
METRICS_PATH = OUTPUT_DIRECTORY / f"ioi_{TEMPLATE_ORDER}_training_metrics.jsonl"
CHECKPOINT_DIRECTORY: Path | None = None
INITIAL_GATE_CHECKPOINT: Path | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_batch_size(device: str) -> int:
    """Conservative FP32 sizing for two GPT-2 copies and a dual-stream forward."""
    if device == "cpu":
        return 2
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if memory_gib >= 20:
        return 32
    if memory_gib >= 12:
        return 24
    if memory_gib >= 8:
        return 16
    if memory_gib >= 6:
        return 8
    return 4


def move_batch(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=device != "cpu")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def circuit_density(model: torch.nn.Module) -> dict:
    expected_open = 0.0
    hard_open = 0
    total = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, HardConcreteGate):
                continue
            probability_open = torch.sigmoid(
                module.log_alpha
                - module.beta * torch.log(-module.gamma / module.zeta)
            )
            expected_open += probability_open.sum().item()
            hard_open += module.get_num_active()
            total += module.num_gates()
    return {
        "hard_active": hard_open,
        "total": total,
        "hard_percent": 100.0 * hard_open / max(total, 1),
        "expected_percent": 100.0 * expected_open / max(total, 1),
    }


def write_metric(record: dict) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, sort_keys=True) + "\n")


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_steps: int,
    epoch_metrics: dict,
    validation_metrics: dict,
) -> Path | None:
    """Save a resumable validation-cycle checkpoint when configured.

    Gate logits are deliberately saved before final hierarchy hardening so the
    optimizer can resume them.  The effective mask still obeys the hierarchy at
    inference because every parent gate controls its complete child pathway;
    final analysis propagates parent closure into the stored child masks.
    """
    if CHECKPOINT_DIRECTORY is None:
        return None

    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = CHECKPOINT_DIRECTORY / f"epoch_{epoch:04d}.pt"
    temporary = destination.with_suffix(".tmp")
    gate_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if "gate" in name
    }
    payload = {
        "checkpoint_type": "continuous_log_alpha",
        "epoch": epoch,
        "total_steps": total_steps,
        "pruning_config": vars(model.pruning_config),
        "gate_state": gate_state,
        "optimizer_state": optimizer.state_dict(),
        "epoch_metrics": epoch_metrics,
        "validation_metrics": validation_metrics,
        "random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "hierarchy_finalized": False,
    }
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_continuous_gate_initialization(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> None:
    """Load continuous gates, expanding one global row across all sections.

    Optimizer state is intentionally not loaded when expanding a global mask to
    positional masks: the parameter shapes change from ``(1, ...)`` to
    ``(num_sections, ...)``.  Every position begins with the same learned global
    log-alpha value and can then specialize independently.
    """
    source_path = Path(checkpoint_path)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source_state = payload.get("gate_state")
    if not isinstance(source_state, dict):
        raise ValueError(f"Checkpoint {source_path} has no gate_state mapping.")

    loaded = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "gate" not in name:
                continue
            if name not in source_state:
                raise KeyError(f"Gate {name!r} is missing from {source_path}.")
            source = source_state[name]
            if tuple(source.shape) == tuple(parameter.shape):
                expanded = source
            elif (
                source.ndim == parameter.ndim
                and source.shape[0] == 1
                and tuple(source.shape[1:]) == tuple(parameter.shape[1:])
            ):
                expanded = source.expand(parameter.shape).clone()
            else:
                raise ValueError(
                    f"Cannot initialize {name}: checkpoint shape "
                    f"{tuple(source.shape)} versus model shape "
                    f"{tuple(parameter.shape)}."
                )
            parameter.copy_(expanded.to(parameter.device, parameter.dtype))
            loaded += 1

    print(
        f"Loaded {loaded} continuous gate tensors from {source_path}; "
        f"expanded global rows to {model.pruning_config.num_sections} sections",
        flush=True,
    )


def build_pruning_config() -> PruningConfig:
    return PruningConfig(
        num_sections=NUM_SECTIONS,
        sparsity_warmup_steps=1000,
        depth_penalty_scaling=0.1,
        prune_attention_heads=True,
        lambda_attention_heads=0.8,
        prune_attention_neurons=True,
        lambda_attention_neurons=0.15,
        prune_mlp_hidden=True,
        lambda_mlp_hidden=1.0,
        prune_mlp_output=True,
        lambda_mlp_output=1.0,
        prune_attention_blocks=True,
        lambda_attention_blocks=0.5,
        prune_mlp_blocks=True,
        lambda_mlp_blocks=0.5,
        prune_full_layers=False,
        lambda_full_layers=0.0,
        prune_embedding=False,
        lambda_embedding=25.0,
    )


def main() -> None:
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = choose_batch_size(device)
    pruning_config = build_pruning_config()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("", encoding="utf-8")

    if device == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        print(
            f"Using {gpu.name} ({gpu.total_memory / 2**30:.1f} GiB) with "
            f"FP32 batch size {batch_size}",
            flush=True,
        )
    else:
        print(f"CUDA unavailable; using CPU batch size {batch_size}", flush=True)

    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    circuit_model = PrunableGPT2LMHeadModel.from_pretrained_with_pruning(
        MODEL_NAME, pruning_config
    ).to(device)
    if INITIAL_GATE_CHECKPOINT is not None:
        load_continuous_gate_initialization(circuit_model, INITIAL_GATE_CHECKPOINT)
    full_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device).eval()
    for parameter in full_model.parameters():
        parameter.requires_grad = False
    disable_dropout(circuit_model)

    total_parameters = 0
    trainable_parameters = 0
    for name, parameter in circuit_model.named_parameters():
        total_parameters += parameter.numel()
        parameter.requires_grad = "gate" in name
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
    print(
        f"Trainable section gates: {trainable_parameters:,}/{total_parameters:,} "
        f"({trainable_parameters / total_parameters:.4%})"
    )
    print(f"Gate slots by group: {circuit_model.gate_group_sizes()}")

    raw_train = load_or_generate_ioi_data(
        tokenizer,
        dataset_path=DATASET_PATH,
        split="train",
        num_samples=200,
        template_order=TEMPLATE_ORDER,
        seed=RANDOM_SEED,
    )
    raw_validation = load_or_generate_ioi_data(
        tokenizer,
        dataset_path=DATASET_PATH,
        split="validation",
        num_samples=200,
        template_order=TEMPLATE_ORDER,
        seed=RANDOM_SEED,
    )
    raw_test = load_or_generate_ioi_data(
        tokenizer,
        dataset_path=DATASET_PATH,
        split="test",
        num_samples=1000,
        template_order=TEMPLATE_ORDER,
        seed=RANDOM_SEED,
    )

    # As in the paper, discover and evaluate circuits only on prompts the base
    # model solves.  Filtering train as well avoids KL/task-loss disagreement.
    filter_kwargs = {
        "model": full_model,
        "tokenizer": tokenizer,
        "device": device,
        "batch_size": batch_size,
        "max_length": MAX_SEQ_LEN,
        "template_order": TEMPLATE_ORDER,
    }
    raw_train = filter_dataset_by_model_correctness(raw_train, **filter_kwargs)
    raw_validation = filter_dataset_by_model_correctness(
        raw_validation, **filter_kwargs
    )
    raw_test = filter_dataset_by_model_correctness(raw_test, **filter_kwargs)

    train_dataset = IOIDataset(
        raw_train, tokenizer, MAX_SEQ_LEN, template_order=TEMPLATE_ORDER
    )
    validation_dataset = IOIDataset(
        raw_validation, tokenizer, MAX_SEQ_LEN, template_order=TEMPLATE_ORDER
    )
    test_dataset = IOIDataset(
        raw_test, tokenizer, MAX_SEQ_LEN, template_order=TEMPLATE_ORDER
    )
    if not train_dataset or not validation_dataset or not test_dataset:
        raise RuntimeError("At least one aligned IOI split is empty after filtering.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 2,
        "pin_memory": device == "cuda",
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, **loader_kwargs)
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    baseline_results = run_evaluation(
        full_model, "Baseline GPT-2", None, test_loader, device
    )
    run_evaluation(
        circuit_model,
        "Initial section-aware circuit",
        full_model,
        validation_loader,
        device,
    )
    print(
        f"Target accuracy: at least "
        f"{max(0.0, baseline_results['accuracy'] - ACCURACY_BUDGET):.4f}"
    )

    gate_parameters = [
        parameter for parameter in circuit_model.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(gate_parameters, lr=LEARNING_RATE)
    total_steps = 0
    progress = tqdm(
        range(NUM_EPOCHS),
        desc="Training section-aware IOI circuit",
        disable=not sys.stdout.isatty(),
    )

    circuit_model.train()
    for epoch in progress:
        started = time.time()
        accumulated_loss = 0.0
        accumulated_kl = 0.0
        accumulated_task = 0.0
        accumulated_sparsity = 0.0

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            circuit_outputs = circuit_model(
                input_ids=batch["input_ids"],
                corrupted_input_ids=batch["corrupted_input_ids"],
                attention_mask=batch["attention_mask"],
                corrupted_attention_mask=batch["corrupted_attention_mask"],
                section_ids=batch["section_ids"],
            )
            with torch.no_grad():
                full_outputs = full_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            batch_size_current = circuit_outputs.logits.size(0)
            indices = torch.arange(batch_size_current, device=device)
            prediction_positions = batch["T_Start"] - 1
            circuit_prediction_logits = circuit_outputs.logits[
                indices, prediction_positions
            ]
            full_prediction_logits = full_outputs.logits[indices, prediction_positions]

            kl_loss = F.kl_div(
                F.log_softmax(circuit_prediction_logits, dim=-1),
                F.log_softmax(full_prediction_logits, dim=-1),
                reduction="batchmean",
                log_target=True,
            )
            target_tokens = batch["target_tokens"][:, 0]
            distractor_tokens = batch["distractor_tokens"][:, 0]
            target_logits = circuit_prediction_logits[indices, target_tokens]
            distractor_logits = circuit_prediction_logits[indices, distractor_tokens]
            task_loss = F.relu(4.0 - (target_logits - distractor_logits)).mean()
            sparsity_loss = circuit_model.get_sparsity_loss(total_steps)[
                "total_sparsity"
            ]
            loss = 1.5 * kl_loss + task_loss + sparsity_loss
            loss.backward()
            optimizer.step()

            accumulated_loss += loss.item()
            accumulated_kl += kl_loss.item()
            accumulated_task += task_loss.item()
            accumulated_sparsity += sparsity_loss.item()
            total_steps += 1

        batch_count = len(train_loader)
        density = circuit_density(circuit_model)
        epoch_record = {
            "event": "epoch",
            "epoch": epoch + 1,
            "steps": total_steps,
            "loss": accumulated_loss / batch_count,
            "kl_div": accumulated_kl / batch_count,
            "task_loss": accumulated_task / batch_count,
            "sparsity_loss": accumulated_sparsity / batch_count,
            "hard_circuit_percent": density["hard_percent"],
            "expected_circuit_percent": density["expected_percent"],
            "hard_active_gates": density["hard_active"],
            "total_gates": density["total"],
            "duration_seconds": time.time() - started,
            "batch_size": batch_size,
        }
        write_metric(epoch_record)
        print("EPOCH " + json.dumps(epoch_record, sort_keys=True), flush=True)
        progress.set_postfix(
            loss=f"{accumulated_loss / batch_count:.3f}",
            kl=f"{accumulated_kl / batch_count:.3f}",
            sparsity=f"{accumulated_sparsity / batch_count:.3f}",
            seconds=f"{time.time() - started:.1f}",
        )

        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            circuit_model.eval()
            validation_results = run_evaluation(
                circuit_model,
                f"Validation epoch {epoch + 1}",
                full_model,
                validation_loader,
                device,
                verbose=False,
            )
            validation_record = {
                "event": "validation",
                "epoch": epoch + 1,
                **validation_results,
                "hard_circuit_percent": density["hard_percent"],
                "expected_circuit_percent": density["expected_percent"],
            }
            write_metric(validation_record)
            print(
                "VALIDATION " + json.dumps(validation_record, sort_keys=True),
                flush=True,
            )
            checkpoint_path = save_training_checkpoint(
                circuit_model,
                optimizer,
                epoch=epoch + 1,
                total_steps=total_steps,
                epoch_metrics=epoch_record,
                validation_metrics=validation_record,
            )
            if checkpoint_path is not None:
                print(f"CHECKPOINT {checkpoint_path}", flush=True)
            circuit_model.train()

    analysis = analyze_and_finalize_circuit(circuit_model)
    save_gate_state(circuit_model, str(OUTPUT_PATH))
    final_results = run_evaluation(
        circuit_model,
        "Final section-aware circuit",
        full_model,
        test_loader,
        device,
    )
    write_metric({"event": "final", **final_results})
    print(f"Saved gate-only checkpoint to {OUTPUT_PATH}", flush=True)
    print(f"Structured metrics written to {METRICS_PATH}", flush=True)
    print(f"Final results: {final_results}")
    print(f"Final active slots: {analysis['category_totals']}")


if __name__ == "__main__":
    main()
