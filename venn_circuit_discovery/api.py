"""High-level interface for automated multi-task circuit discovery.

Example
-------
>>> discoverer = VennCircuitDiscoverer(
...     model_name="meta-llama/Llama-3.2-1B",
...     mode="intersection",
...     target_kl=0.1,
... )
>>> discoverer.fit(dataloader, epochs=5)
>>> circuit = discoverer.extract_circuit()   # per-region binary masks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union

import torch

from .gates import VennConcreteGate
from .models import LlamaVennCircuit, VennPruningConfig
from .scheduler import DualVennScheduler, DualVennSchedulerConfig, LogicMode
from .trainer import StepMetrics, TrainerConfig, VennBatch, VennTrainer


@dataclass
class VennHyperparameters:
    """User-facing hyper-parameters, flattened for convenience."""

    gate_lr: float = 0.05
    uncertainty_lr: float = 0.01
    margin: float = 4.0

    target_kl_a: float = 0.1
    target_kl_b: float = 0.1
    epsilon_discount: float = 0.5

    pid_kp: float = 0.5
    pid_ki: float = 0.05
    pid_kd: float = 0.1

    init_lambda: float = 1.0
    min_lambda: float = 1e-4
    max_lambda: float = 50.0


class VennCircuitDiscoverer:
    """One-stop object for loading a Llama model and discovering Venn circuits."""

    def __init__(
        self,
        model_name: str,
        mode: Union[str, LogicMode] = LogicMode.INTERSECTION,
        target_kl: Optional[float] = 0.1,
        hyperparameters: Optional[VennHyperparameters] = None,
        pruning_config: Optional[VennPruningConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        **model_kwargs,
    ) -> None:
        self.mode = LogicMode(mode)
        self.hp = hyperparameters or VennHyperparameters()
        if target_kl is not None:
            # A single target populates both tasks unless overridden individually.
            self.hp.target_kl_a = target_kl
            self.hp.target_kl_b = target_kl

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load & patch the Llama model (1B/8B/32B). `device_map`/`dtype` flow
        # through **model_kwargs so large variants can be sharded by the caller.
        model_kwargs.setdefault("torch_dtype", dtype)
        self.model: LlamaVennCircuit = LlamaVennCircuit.from_pretrained_with_venn(
            model_name, pruning_config=pruning_config, **model_kwargs
        )
        if "device_map" not in model_kwargs:
            self.model.to(self.device)

        self.scheduler = DualVennScheduler(
            DualVennSchedulerConfig(
                mode=self.mode,
                target_kl_a=self.hp.target_kl_a,
                target_kl_b=self.hp.target_kl_b,
                epsilon_discount=self.hp.epsilon_discount,
                kp=self.hp.pid_kp,
                ki=self.hp.pid_ki,
                kd=self.hp.pid_kd,
                init_lambda=self.hp.init_lambda,
                min_lambda=self.hp.min_lambda,
                max_lambda=self.hp.max_lambda,
            )
        )

        self.trainer = VennTrainer(
            model=self.model,
            scheduler=self.scheduler,
            config=TrainerConfig(
                gate_lr=self.hp.gate_lr,
                uncertainty_lr=self.hp.uncertainty_lr,
                margin=self.hp.margin,
            ),
            device=self.device,
        )

    # ------------------------------------------------------------------

    def fit(self, data: Iterable[VennBatch], epochs: int = 1) -> List[StepMetrics]:
        """Run the discovery training loop and return per-step metrics."""
        return self.trainer.train(data, epochs=epochs)

    @torch.no_grad()
    def extract_circuit(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return the deterministic binary masks for every gate location.

        For each gate the three regions (``core`` / ``a_only`` / ``b_only``) and
        the two effective per-task masks (``mask_a`` / ``mask_b``) are returned as
        boolean tensors, evaluated in deterministic (eval) mode.
        """
        was_training = self.model.training
        self.model.eval()
        circuit: Dict[str, Dict[str, torch.Tensor]] = {}
        try:
            for name, gate in self.model.iter_venn_gates().items():
                mask_a, mask_b = gate()
                circuit[name] = {
                    "core": (gate.g_core() > 0.5).bool().cpu(),
                    "a_only": (gate.g_a_only() > 0.5).bool().cpu(),
                    "b_only": (gate.g_b_only() > 0.5).bool().cpu(),
                    "mask_a": (mask_a > 0.5).bool().cpu(),
                    "mask_b": (mask_b > 0.5).bool().cpu(),
                }
        finally:
            self.model.train(was_training)
        return circuit

    @torch.no_grad()
    def circuit_summary(self) -> Dict[str, float]:
        """Aggregate density (fraction of open gates) per Venn region."""
        was_training = self.model.training
        self.model.eval()
        counts = {"core": 0, "a_only": 0, "b_only": 0, "mask_a": 0, "mask_b": 0}
        total = 0
        try:
            for gate in self.model.iter_venn_gates().values():
                mask_a, mask_b = gate()
                n = gate.num_gates()
                total += n
                counts["core"] += int((gate.g_core() > 0.5).sum().item())
                counts["a_only"] += int((gate.g_a_only() > 0.5).sum().item())
                counts["b_only"] += int((gate.g_b_only() > 0.5).sum().item())
                counts["mask_a"] += int((mask_a > 0.5).sum().item())
                counts["mask_b"] += int((mask_b > 0.5).sum().item())
        finally:
            self.model.train(was_training)
        total = max(total, 1)
        return {k: v / total for k, v in counts.items()}
