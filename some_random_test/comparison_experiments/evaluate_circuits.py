"""Unified held-out IOI evaluator for node circuits and PEAP."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "global-node": ROOT / "non_position_node_pruning/outputs/ioi_abba_global_gates.pt",
    "position-node": ROOT / "position_aware_node_pruning/outputs/ioi_abba_section_gates.pt",
    "peap": ROOT / "results/peap_monitor_extended_eval.topk_5000.circuit.pkl",
}


def evaluate_node(args: argparse.Namespace) -> dict:
    from torch.utils.data import DataLoader
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import comparison_experiments.position_aware_node_pruning.dataset.ioi as data_module
    from comparison_experiments.position_aware_node_pruning.dataset.ioi import (
        IOIDataset, load_or_generate_ioi_data, run_evaluation,
    )
    from comparison_experiments.position_aware_node_pruning.models.gpt2_circuit import (
        PrunableGPT2LMHeadModel, PruningConfig,
    )
    from comparison_experiments.position_aware_node_pruning.utils import disable_dropout
    from comparison_experiments.position_aware_node_pruning.utils import analyze_and_finalize_circuit
    from comparison_experiments.structural_metrics import node_structural_report

    payload = torch.load(args.circuit, map_location="cpu", weights_only=False)
    config = PruningConfig(**payload["pruning_config"])
    dataset = None
    if not args.structure_only:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        rows = load_or_generate_ioi_data(
            tokenizer, dataset_path=str(ROOT / "data"), split="test",
            num_samples=args.limit or 500, template_order="abba", seed=42,
        )
        dataset = IOIDataset(rows, tokenizer, max_length=64, template_order="abba")
    old_sections = None
    if args.method == "global-node":
        # The stored global gate has one row; map every token to that row.
        if dataset is not None:
            for item in dataset.processed_data:
                item["section_lengths"] = [sum(item["section_lengths"])]
        old_sections = data_module.NUM_SECTIONS
        data_module.NUM_SECTIONS = 1

    circuit = PrunableGPT2LMHeadModel.from_pretrained_with_pruning(args.model, config).to(args.device)
    missing, unexpected = circuit.load_state_dict(payload["gate_state"], strict=False)
    # Gate-only checkpoints persist learned log-alpha parameters, not constant
    # beta/gamma/zeta buffers. Require every learned tensor and retain freshly
    # initialized constants from HardConcreteGate.
    missing_gates = [name for name in missing if name.endswith("log_alpha")]
    if missing_gates or unexpected:
        raise RuntimeError(f"Gate checkpoint mismatch: missing={missing_gates}, unexpected={unexpected}")
    circuit.set_final_circuit_mode(True)
    disable_dropout(circuit)
    # Training-cycle checkpoints contain continuous gates before hierarchy
    # closure. Finalize in memory before both behavioral and structural
    # evaluation so they have exactly the same semantics as final gate files.
    analysis = analyze_and_finalize_circuit(circuit, verbose=False)
    if args.method == "global-node":
        analysis["section_names"] = ("global",)
    if args.structure_only:
        if not args.output.exists():
            raise FileNotFoundError(f"--structure-only needs an existing metrics file: {args.output}")
        metrics = json.loads(args.output.read_text())
        # main() restores these canonical envelope fields.
        for key in ("method", "circuit", "dataset"):
            metrics.pop(key, None)
    else:
        full = GPT2LMHeadModel.from_pretrained(args.model).to(args.device).eval()
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        metrics = run_evaluation(circuit, args.method, full, loader, args.device, verbose=False)
    metrics["structure"] = node_structural_report(circuit, analysis)
    if old_sections is not None:
        data_module.NUM_SECTIONS = old_sections
    return metrics


def evaluate_peap(args: argparse.Namespace) -> dict:
    import pandas as pd
    import transformers
    # TransformerLens 1.x reads this legacy constant at import time; current
    # Transformers releases removed it. This affects only cache-path lookup.
    if not hasattr(transformers, "TRANSFORMERS_CACHE"):
        transformers.TRANSFORMERS_CACHE = str(Path.home() / ".cache/huggingface/hub")
    from transformer_lens import HookedTransformer

    sys.path.insert(0, str(ROOT / "position_aware_eap/src"))
    from eval_utils import run_edges_with_mean_ablation, full_computation_graph_size
    from exp import IOI, logit_diff
    from comparison_experiments.structural_metrics import peap_structural_report

    with args.circuit.open("rb") as handle:
        payload = pickle.load(handle)
    graph = payload["circuit"] if isinstance(payload, dict) else payload
    clean_path = ROOT / "data/peap/IOI_ABBA_data_clean.csv"
    counter_path = ROOT / "data/peap/IOI_ABBA_data_counter_abc.csv"
    exp = IOI(
        exp_name="ioi", model_name=args.model, model_path=args.model,
        ablation_type="counterfactual", clean_dataset_path=str(clean_path),
        counter_dataset_path=str(counter_path),
        spans=["prefix", "IO", "and", "S1", "S1+1", "action1", "S2", "action2", "to", "length"],
        metric=logit_diff, seed=42,
    )
    frame = pd.read_csv(clean_path)
    frame = frame[(frame.split == "eval") & (frame.top_answer == frame.correct_token)]
    frame = frame.sort_values("example_id")
    if args.limit:
        frame = frame.head(args.limit)
    model = HookedTransformer.from_pretrained(
        args.model, center_writing_weights=False, center_unembed=False,
        fold_ln=False, device=args.device, dtype="fp32",
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)
    model.set_ungroup_grouped_query_attention(True)
    full_edges, full_nodes = full_computation_graph_size(
        model=model, exp=exp, df=frame, use_point_of_diff=False
    )
    if args.structure_only:
        if not args.output.exists():
            raise FileNotFoundError(
                f"--structure-only needs an existing metrics file: {args.output}"
            )
        metrics = json.loads(args.output.read_text())
        mean_active_edges = float(metrics["mean_circuit_edges"])
    else:
        result = run_edges_with_mean_ablation(
            model, frame, graph, exp, with_pos=True, sample_ablation_size=1,
            patch_q=True, return_shared_metrics=True,
        )
        metrics = result[-1]
        mean_active_edges = float(sum(result[4]) / max(len(result[4]), 1))
    metrics["mean_circuit_edges"] = mean_active_edges
    metrics["mean_full_model_edges"] = float(full_edges)
    metrics["structure"] = peap_structural_report(
        graph, mean_active_edges, float(full_edges), float(full_nodes)
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(DEFAULTS))
    parser.add_argument("--circuit", type=Path)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--structure-only", action="store_true",
        help="Backfill structural fields in an existing metrics JSON without rerunning inference.",
    )
    args = parser.parse_args()
    args.circuit = (args.circuit or DEFAULTS[args.method]).resolve()
    args.output = args.output or ROOT / f"results/unified_{args.method}.metrics.json"
    metrics = evaluate_peap(args) if args.method == "peap" else evaluate_node(args)
    result = {
        "method": args.method, "circuit": str(args.circuit),
        "dataset": str((ROOT / "data/evaluation.csv").resolve()), **metrics,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
