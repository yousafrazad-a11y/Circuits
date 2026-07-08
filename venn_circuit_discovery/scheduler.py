"""Dual PID control for the Venn sparsity lambdas.

Each task has its own PID controller that drives the sparsity penalty
(``lambda_a`` / ``lambda_b``) so that the task's KL divergence tracks a target
``epsilon``. The scheduler then derives ``lambda_core`` dynamically from the two
task lambdas according to the requested logic mode:

* Intersection (AND): the core must be at least as strong as the stronger task,
  with a discounted contribution from the weaker task.
* Union (OR): the core is bounded by the weaker task, minus a discount.

Design reference: ``pruning_scheduler_v2.py`` in the legacy codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class LogicMode(str, Enum):
    """Whether we discover the AND (intersection) or OR (union) of two circuits."""

    INTERSECTION = "intersection"
    UNION = "union"


@dataclass
class PIDController:
    """A discrete PID controller.

    The controller drives ``measurement`` towards ``setpoint``. Because a *higher*
    lambda increases pruning pressure and therefore *increases* the KL divergence,
    the error is defined as ``setpoint - measurement`` so that:

    * KL below target (faithful, room to prune)  -> positive error -> raise lambda.
    * KL above target (too much damage)          -> negative error -> lower lambda.
    """

    setpoint: float
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.1
    output: float = 1.0            # current control value (the lambda)
    out_min: float = 1e-4
    out_max: float = 50.0

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_error: float = field(default=0.0, init=False, repr=False)
    _initialised: bool = field(default=False, init=False, repr=False)

    def step(self, measurement: float) -> float:
        """Advance the controller one step and return the updated output."""
        error = self.setpoint - measurement

        self._integral += error
        # Clamp the integral to prevent wind-up beyond what output limits allow.
        self._integral = _clamp(self._integral, -self.out_max, self.out_max)

        derivative = 0.0 if not self._initialised else (error - self._prev_error)
        self._prev_error = error
        self._initialised = True

        control = self.kp * error + self.ki * self._integral + self.kd * derivative
        self.output = _clamp(self.output + control, self.out_min, self.out_max)
        return self.output


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class DualVennSchedulerConfig:
    """Hyper-parameters for :class:`DualVennScheduler`."""

    mode: LogicMode = LogicMode.INTERSECTION
    target_kl_a: float = 0.1
    target_kl_b: float = 0.1
    epsilon_discount: float = 0.5

    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.1

    init_lambda: float = 1.0
    min_lambda: float = 1e-4
    max_lambda: float = 50.0


class DualVennScheduler:
    """Maintains two PID controllers and links them into a shared core lambda."""

    def __init__(self, config: DualVennSchedulerConfig) -> None:
        self.config = config
        self.mode = LogicMode(config.mode)

        pid_kwargs = dict(
            kp=config.kp,
            ki=config.ki,
            kd=config.kd,
            output=config.init_lambda,
            out_min=config.min_lambda,
            out_max=config.max_lambda,
        )
        self.pid_a = PIDController(setpoint=config.target_kl_a, **pid_kwargs)
        self.pid_b = PIDController(setpoint=config.target_kl_b, **pid_kwargs)

        self.lambda_a: float = config.init_lambda
        self.lambda_b: float = config.init_lambda
        self.lambda_core: float = config.init_lambda

        self.history: List[Dict[str, float]] = []

    def _compute_core(self) -> float:
        """Derive ``lambda_core`` from the two task lambdas given the logic mode."""
        eps = self.config.epsilon_discount
        lo = min(self.lambda_a, self.lambda_b)
        hi = max(self.lambda_a, self.lambda_b)

        if self.mode is LogicMode.INTERSECTION:
            # Core is shared by both tasks -> penalise it at least as hard as the
            # more demanding task, plus a discounted share of the weaker one.
            core = hi + eps * lo
        else:  # UNION
            # Core is cheap to keep (it helps both tasks) -> relax below the
            # weaker task's pressure.
            core = lo - eps

        return _clamp(core, self.config.min_lambda, self.config.max_lambda)

    def step(self, kl_a: float, kl_b: float) -> Dict[str, float]:
        """Update both lambdas from measured KLs, then recompute the core lambda.

        Args:
            kl_a: Measured faithfulness (KL divergence) for task A.
            kl_b: Measured faithfulness (KL divergence) for task B.

        Returns:
            Dict with the current ``lambda_a`` / ``lambda_b`` / ``lambda_core``.
        """
        self.lambda_a = self.pid_a.step(kl_a)
        self.lambda_b = self.pid_b.step(kl_b)
        self.lambda_core = self._compute_core()

        lambdas = {
            "lambda_a": self.lambda_a,
            "lambda_b": self.lambda_b,
            "lambda_core": self.lambda_core,
        }
        self.history.append({"kl_a": kl_a, "kl_b": kl_b, **lambdas})
        return lambdas

    def current_lambdas(self) -> Dict[str, float]:
        return {
            "lambda_a": self.lambda_a,
            "lambda_b": self.lambda_b,
            "lambda_core": self.lambda_core,
        }
