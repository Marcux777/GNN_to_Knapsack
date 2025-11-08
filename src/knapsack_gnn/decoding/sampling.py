"""
Inference Sampler for Knapsack GNN
Vectorized sampling utilities, Lagrangian decoding and latency-oriented helpers.
"""

import math
import os
import time
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from knapsack_gnn.decoding.repair import SolutionRepairer
from knapsack_gnn.solvers.cp_sat import solve_knapsack_warm_start

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - optional dependency
    scipy_stats = None


class KnapsackSampler:
    """
    Sampler for generating Knapsack solutions from GNN probability outputs
    """

    @staticmethod
    def _get_safe_device() -> str:
        """Get a safe CUDA device or fallback to CPU if CUDA is incompatible."""
        if not torch.cuda.is_available():
            return "cpu"

        try:
            # Try to create a small tensor on CUDA to check compatibility
            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except (RuntimeError, AssertionError):
            # CUDA is available but incompatible (e.g., wrong compute capability)
            return "cpu"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | None = None,
        num_threads: int | None = None,
        compile_model: bool = False,
        quantize: bool = False,
        quantize_dtype: Any = torch.qint8,
    ):
        """
        Args:
            model: Trained KnapsackPNA model
            device: Device to run inference on (default: auto-detect safe device)
            num_threads: Optional cap for intra-op threads (latency tuning)
            compile_model: Whether to compile the model with torch.compile
            quantize: Apply dynamic quantization to Linear layers
            quantize_dtype: Quantized dtype (default: torch.qint8)
        """
        if device is None:
            device = self._get_safe_device()

        if num_threads is not None:
            torch.set_num_threads(int(num_threads))
            os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))

        if quantize:
            model = self._apply_dynamic_quantization(model, quantize_dtype)

        if compile_model:
            model = self._try_compile(model)

        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self._default_schedule = (32, 64, 128)
        self._sampling_tolerance = 1e-3

    @staticmethod
    def _apply_dynamic_quantization(model: torch.nn.Module, dtype: Any) -> torch.nn.Module:
        """Apply dynamic quantization if available."""
        try:
            from torch.ao.quantization import quantize_dynamic

            quantized = cast(
                torch.nn.Module, quantize_dynamic(model, {torch.nn.Linear}, dtype=dtype)
            )
            return quantized
        except Exception:  # pragma: no cover - optional feature
            return model

    @staticmethod
    def _try_compile(model: torch.nn.Module) -> torch.nn.Module:
        """Compile the model if torch.compile is available."""
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:  # pragma: no cover - optional feature
            return model
        try:
            compiled = cast(
                torch.nn.Module, compile_fn(model, mode="reduce-overhead", fullgraph=False)
            )
            return compiled
        except Exception:  # pragma: no cover - optional feature
            return model

    def get_probabilities(self, data: Data) -> torch.Tensor:
        """
        Get item selection probabilities from model

        Args:
            data: Graph data

        Returns:
            Probability vector [n_items]
        """
        data = data.to(self.device, non_blocking=True)
        with torch.inference_mode():
            probs: torch.Tensor = self.model(data)
        return probs.detach().cpu()

    def check_feasibility(self, solution: np.ndarray, weights: np.ndarray, capacity: float) -> bool:
        """
        Check if solution is feasible (doesn't exceed capacity)

        Args:
            solution: Binary solution vector
            weights: Item weights
            capacity: Knapsack capacity

        Returns:
            True if feasible, False otherwise
        """
        total_weight = np.sum(solution * weights)
        return bool(total_weight <= capacity)

    def compute_value(self, solution: np.ndarray, values: np.ndarray) -> float:
        """
        Compute total value of solution

        Args:
            solution: Binary solution vector
            values: Item values

        Returns:
            Total value
        """
        return float(np.sum(solution * values))

    def threshold_decode(
        self, probs: torch.Tensor, data: Data, threshold: float = 0.5
    ) -> tuple[np.ndarray, float, bool]:
        """
        Simple threshold-based decoding

        Args:
            probs: Item selection probabilities
            data: Graph data (for weights, values, capacity)
            threshold: Decision threshold

        Returns:
            Tuple of (solution, value, is_feasible)
        """
        solution = (probs.numpy() >= threshold).astype(np.int32)
        weights = data.item_weights.numpy()
        values = data.item_values.numpy()
        capacity = data.capacity

        is_feasible = self.check_feasibility(solution, weights, capacity)
        value = self.compute_value(solution, values)

        return solution, value, is_feasible

    def _temperature_scale(self, probs: Tensor, temperature: float) -> Tensor:
        """Apply temperature scaling to Bernoulli parameters."""
        temperature = max(float(temperature), 1e-6)
        probs = probs.clamp(1e-6, 1 - 1e-6)
        logits = torch.logit(probs)
        scaled = torch.sigmoid(logits / temperature)
        return scaled

    def sample_solutions(
        self, probs: torch.Tensor, data: Data, n_samples: int = 100, temperature: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized sampling of multiple solutions from probability distribution.

        Args:
            probs: Item selection probabilities
            data: Graph data
            n_samples: Number of solutions to sample
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Tuple of (solutions [n_samples, n_items], values, feasible_mask, weights)
        """
        if n_samples <= 0:
            return (
                np.zeros((0, probs.numel()), dtype=np.int32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.float32),
            )

        adjusted_probs = self._temperature_scale(probs.to(self.device), temperature)
        expanded = adjusted_probs.expand(n_samples, -1)

        with torch.inference_mode():
            samples = torch.bernoulli(expanded)

        weights_t = data.item_weights.to(self.device, dtype=adjusted_probs.dtype)
        values_t = data.item_values.to(self.device, dtype=adjusted_probs.dtype)

        sample_weights = torch.matmul(samples, weights_t)
        sample_values = torch.matmul(samples, values_t)

        feasible_mask = sample_weights <= (float(data.capacity) + 1e-6)

        return (
            samples.to(torch.int32).cpu().numpy(),
            sample_values.cpu().numpy(),
            feasible_mask.cpu().numpy(),
            sample_weights.cpu().numpy(),
        )

    def greedy_repair(
        self, solution: np.ndarray, weights: np.ndarray, values: np.ndarray, capacity: float
    ) -> np.ndarray:
        """
        Repair infeasible solution by greedily removing items

        Args:
            solution: Binary solution (potentially infeasible)
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity

        Returns:
            Feasible solution
        """
        solution = solution.copy()

        # If already feasible, return
        if self.check_feasibility(solution, weights, capacity):
            return solution

        # Get selected items
        selected = np.where(solution == 1)[0]

        # Sort by value/weight ratio (descending)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(weights[selected] > 0, values[selected] / weights[selected], np.inf)
        sorted_indices = selected[np.argsort(-ratios)]

        # Remove items until feasible
        current_weight = np.sum(solution * weights)
        for idx in sorted_indices:
            solution[idx] = 0
            current_weight -= weights[idx]
            if current_weight <= capacity:
                break

        # Greedily refill capacity with remaining items if space is left
        remaining = np.where(solution == 0)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            remaining_ratios = np.where(
                weights[remaining] > 0, values[remaining] / weights[remaining], -np.inf
            )
        for idx in remaining[np.argsort(-remaining_ratios)]:
            if weights[idx] <= 0:
                continue
            if current_weight + weights[idx] <= capacity:
                solution[idx] = 1
                current_weight += weights[idx]

        return solution

    def adaptive_threshold(
        self, probs: torch.Tensor, data: Data, n_trials: int = 20
    ) -> tuple[np.ndarray, float, float]:
        """
        Find best threshold adaptively

        Args:
            probs: Item selection probabilities
            data: Graph data
            n_trials: Number of thresholds to try

        Returns:
            Tuple of (best_solution, best_value, best_threshold)
        """
        weights = data.item_weights.numpy()
        values = data.item_values.numpy()
        capacity = data.capacity

        best_solution: np.ndarray = np.zeros_like(values, dtype=np.int32)
        best_value = -1.0
        best_threshold = 0.5

        # Try different thresholds
        thresholds = np.linspace(0.3, 0.7, n_trials)
        for threshold in thresholds:
            solution, value, is_feasible = self.threshold_decode(probs, data, threshold)

            # If not feasible, try to repair
            if not is_feasible:
                solution = self.greedy_repair(solution, weights, values, capacity)
                value = self.compute_value(solution, values)

            # Update best
            if value > best_value:
                best_value = value
                best_solution = solution
                best_threshold = threshold

        return best_solution, best_value, best_threshold

    @staticmethod
    def _select_lagrangian(
        values: np.ndarray,
        weights: np.ndarray,
        probs: np.ndarray | None,
        lam: float,
        bias: float,
    ) -> np.ndarray:
        """Select items given a Lagrange multiplier."""
        scores = values - lam * weights
        if probs is not None:
            scores = scores + bias * (probs - 0.5)

        solution = (scores > 0).astype(np.int32)
        if solution.sum() == 0:
            idx = int(np.argmax(scores))
            solution[idx] = 1
        return solution

    def lagrangian_decode(
        self,
        probs: torch.Tensor,
        data: Data,
        max_iter: int = 30,
        tol: float = 1e-4,
        bias: float = 0.0,
    ) -> tuple[np.ndarray, float, dict[str, float]]:
        """
        Decode solution via Lagrangian relaxation with bisection on lambda.

        Args:
            probs: Item probabilities (torch tensor)
            data: Graph data
            max_iter: Maximum number of bisection iterations
            tol: Relative tolerance on capacity satisfaction
            bias: Tie-breaking bias using model probabilities

        Returns:
            Tuple of (solution, value, diagnostics)
        """
        weights = data.item_weights.cpu().numpy().astype(np.float64)
        values = data.item_values.cpu().numpy().astype(np.float64)
        capacity = float(data.capacity)
        probs_np = probs.cpu().numpy() if probs is not None else None

        if capacity <= 0:
            return np.zeros_like(values, dtype=np.int32), 0.0, {"lambda": 0.0, "iterations": 0}

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(weights > 0, values / weights, values)
        lam_low = 0.0
        lam_high = float(np.max(ratio) + 1.0)
        lam_high = max(lam_high, 1.0)

        # Ensure lam_high yields feasible solution
        for _ in range(20):
            trial = self._select_lagrangian(values, weights, probs_np, lam_high, bias)
            if np.dot(trial, weights) <= capacity:
                break
            lam_high *= 2.0

        best_solution = None
        best_value = -np.inf
        best_weight = np.inf
        fallback_solution = None
        fallback_value = -np.inf

        iterations = 0
        for iteration in range(1, max_iter + 1):
            iterations = iteration
            lam = 0.5 * (lam_low + lam_high)
            solution = self._select_lagrangian(values, weights, probs_np, lam, bias)
            weight = float(np.dot(solution, weights))
            value = float(np.dot(solution, values))

            if weight <= capacity + 1e-6:
                if value > best_value or (np.isclose(value, best_value) and weight < best_weight):
                    best_solution = solution.copy()
                    best_value = value
                    best_weight = weight
                lam_high = lam
            else:
                lam_low = lam
                if value > fallback_value:
                    fallback_solution = solution.copy()
                    fallback_value = value

            if abs(weight - capacity) / max(capacity, 1.0) <= tol:
                break
            if lam_high - lam_low <= tol:
                break

        if best_solution is None:
            best_solution = np.zeros_like(values, dtype=np.int32)

        if best_weight > capacity or best_weight == np.inf:
            candidate = fallback_solution if fallback_solution is not None else best_solution
            best_solution = self.greedy_repair(candidate, weights, values, capacity)
            best_value = self.compute_value(best_solution, values)
        else:
            # Final feasibility check
            if not self.check_feasibility(best_solution, weights, capacity):
                best_solution = self.greedy_repair(best_solution, weights, values, capacity)
                best_value = self.compute_value(best_solution, values)

        diagnostics = {
            "lambda": 0.5 * (lam_low + lam_high),
            "iterations": iterations,
            "initial_ratio_max": float(np.max(ratio)) if ratio.size else 0.0,
        }

        return best_solution.astype(np.int32), float(best_value), diagnostics

    def anytime_sampling(
        self,
        probs: torch.Tensor,
        data: Data,
        temperature: float = 1.0,
        schedule: Sequence[int] | None = None,
        tolerance: float = 1e-3,
        max_samples: int | None = None,
        target_value: float | None = None,
    ) -> dict:
        """
        Vectorized sampling with adaptive stopping.

        Args:
            probs: Item probabilities
            data: Graph data
            temperature: Sampling temperature
            schedule: Sequence of batch sizes (default: 32, 64, 128)
            tolerance: Early stopping tolerance on gap/improvement
            max_samples: Optional hard cap on samples
            target_value: Optional known optimum for early exit

        Returns:
            Dict containing solution, value, samples_used and metadata
        """
        schedule = tuple(schedule) if schedule is not None else self._default_schedule
        if not schedule:
            schedule = self._default_schedule

        tolerance = max(tolerance, 0.0)
        total_samples = 0
        feasible_samples = 0
        best_solution = None
        best_value = -np.inf
        best_weight = np.inf
        previous_best = -np.inf
        best_infeasible = None
        best_infeasible_value = -np.inf

        weights_np = data.item_weights.cpu().numpy().astype(np.float64)
        values_np = data.item_values.cpu().numpy().astype(np.float64)
        capacity = float(data.capacity)

        for batch_size in schedule:
            batch_size = int(batch_size)
            if batch_size <= 0:
                continue

            if max_samples is not None:
                remaining = max_samples - total_samples
                print(f"remaining: {remaining}")
                if remaining <= 0:
                    break
                batch_size = min(batch_size, remaining)
                print(f"batch_size: {batch_size}")

            sols, vals, feas_mask, weights = self.sample_solutions(
                probs=probs, data=data, n_samples=batch_size, temperature=temperature
            )

            total_samples += batch_size
            if feas_mask.size:
                feasible_samples += int(feas_mask.sum())

            if feas_mask.any():
                feasible_values = vals[feas_mask]
                feasible_solutions = sols[feas_mask]
                feasible_weights = weights[feas_mask]

                idx = int(np.argmax(feasible_values))
                candidate_value = float(feasible_values[idx])
                candidate_solution = feasible_solutions[idx]
                candidate_weight = float(feasible_weights[idx])

                if candidate_value > best_value or (
                    np.isclose(candidate_value, best_value) and candidate_weight < best_weight
                ):
                    best_value = candidate_value
                    best_solution = candidate_solution.copy()
                    best_weight = candidate_weight

            if vals.size and np.max(vals) > best_infeasible_value:
                best_infeasible_value = float(np.max(vals))
                best_infeasible = sols[int(np.argmax(vals))].copy()

            if target_value is not None and best_value >= target_value - 1e-6:
                break

            if best_value > -np.inf:
                improvement = best_value - previous_best
                previous_best = best_value
                if target_value is not None and target_value > 0:
                    gap = (target_value - best_value) / target_value
                    if gap <= tolerance:
                        break
                else:
                    if improvement <= max(tolerance * max(abs(best_value), 1.0), 1e-6):
                        break

        if best_solution is None:
            if best_infeasible is not None:
                best_solution = self.greedy_repair(best_infeasible, weights_np, values_np, capacity)
                best_value = self.compute_value(best_solution, values_np)
            else:
                best_solution = np.zeros_like(values_np, dtype=np.int32)
                best_value = 0.0
        else:
            if best_weight > capacity or not self.check_feasibility(
                best_solution, weights_np, capacity
            ):
                best_solution = self.greedy_repair(best_solution, weights_np, values_np, capacity)
                best_value = self.compute_value(best_solution, values_np)

        return {
            "solution": best_solution.astype(np.int32),
            "value": float(best_value),
            "samples_used": int(total_samples),
            "n_feasible_samples": int(feasible_samples),
            "schedule": tuple(schedule),
        }

    def _prepare_warm_start(
        self,
        probs: torch.Tensor,
        data: Data,
        temperature: float,
        schedule: Sequence[int] | None,
        tolerance: float,
        max_samples: int | None,
        target_value: float | None,
        fix_threshold: float,
    ) -> dict:
        """
        Generate an initial solution, hints, and fixed variables for warm-start ILP.
        """
        sampling_result = self.anytime_sampling(
            probs=probs,
            data=data,
            temperature=temperature,
            schedule=schedule,
            tolerance=tolerance,
            max_samples=max_samples,
            target_value=target_value,
        )

        initial_solution = sampling_result["solution"]
        probs_np = probs.cpu().numpy()

        fix_threshold = float(fix_threshold)
        fix_map: dict[int, int] = {}
        if fix_threshold > 0:
            high_cut = min(max(fix_threshold, 0.5), 1.0)
            low_cut = 1.0 - high_cut
            for idx, p in enumerate(probs_np):
                if p >= high_cut:
                    fix_map[idx] = 1
                elif p <= low_cut:
                    fix_map[idx] = 0

        return {
            "initial_solution": initial_solution,
            "fix_map": fix_map,
            "sampling_result": sampling_result,
        }

    def solve(
        self,
        data: Data,
        strategy: str = "sampling",
        n_samples: int = 100,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> dict:
        """
        Solve Knapsack instance using specified strategy

        Args:
            data: Graph data
            strategy: Solution strategy ('threshold', 'sampling', 'adaptive')
            n_samples: Number of samples for sampling strategy
            temperature: Temperature for sampling
            **kwargs: Additional strategy-specific arguments

        Returns:
            Dictionary with solution, value, and metadata
        """
        # Get probabilities
        probs = self.get_probabilities(data)

        weights = data.item_weights.cpu().numpy().astype(np.float64)
        values = data.item_values.cpu().numpy().astype(np.float64)
        capacity = float(data.capacity)

        result = {
            "probabilities": probs.numpy(),
            "strategy": strategy,
        }

        if strategy == "threshold":
            # Simple threshold decoding
            threshold = kwargs.get("threshold", 0.5)
            solution, value, is_feasible = self.threshold_decode(probs, data, threshold)

            # Repair if needed
            if not is_feasible:
                solution = self.greedy_repair(solution, weights, values, capacity)
                value = self.compute_value(solution, values)
                is_feasible = True

            result.update(
                {
                    "solution": solution,
                    "value": float(value),
                    "is_feasible": is_feasible,
                    "threshold": threshold,
                }
            )

        elif strategy == "sampling":
            schedule = kwargs.get("schedule", kwargs.get("sampling_schedule", None))
            tolerance = kwargs.get("sampling_tolerance", self._sampling_tolerance)
            max_samples = kwargs.get("max_samples", n_samples)
            target_value = kwargs.get("target_value", getattr(data, "optimal_value", None))
            if target_value is not None and target_value <= 0:
                target_value = None

            sampling_result = self.anytime_sampling(
                probs=probs,
                data=data,
                temperature=temperature,
                schedule=schedule,
                tolerance=tolerance,
                max_samples=max_samples,
                target_value=target_value,
            )

            result.update(
                {
                    "solution": sampling_result["solution"],
                    "value": sampling_result["value"],
                    "is_feasible": True,
                    "n_samples": max_samples,
                    "samples_used": sampling_result["samples_used"],
                    "n_feasible_samples": sampling_result["n_feasible_samples"],
                    "schedule": sampling_result["schedule"],
                    "temperature": temperature,
                }
            )

        elif strategy == "adaptive":
            # Adaptive threshold search
            n_trials = kwargs.get("n_trials", 20)
            solution, value, threshold = self.adaptive_threshold(probs, data, n_trials)

            result.update(
                {
                    "solution": solution,
                    "value": float(value),
                    "is_feasible": True,
                    "best_threshold": threshold,
                    "n_trials": n_trials,
                }
            )

        elif strategy == "lagrangian":
            lagrangian_iters = kwargs.get("lagrangian_iters", 30)
            lagrangian_tol = kwargs.get("lagrangian_tol", 1e-4)
            lagrangian_bias = kwargs.get("lagrangian_bias", 0.0)

            solution, value, diagnostics = self.lagrangian_decode(
                probs=probs,
                data=data,
                max_iter=lagrangian_iters,
                tol=lagrangian_tol,
                bias=lagrangian_bias,
            )

            result.update(
                {
                    "solution": solution,
                    "value": float(value),
                    "is_feasible": self.check_feasibility(solution, weights, capacity),
                    "samples_used": 0,
                    "lagrangian_lambda": diagnostics["lambda"],
                    "lagrangian_iterations": diagnostics["iterations"],
                    "lagrangian_bias": lagrangian_bias,
                }
            )

        elif strategy == "warm_start":
            warm_schedule = kwargs.get("sampling_schedule", kwargs.get("schedule", None))
            warm_tolerance = kwargs.get("sampling_tolerance", self._sampling_tolerance)
            max_samples = kwargs.get("max_samples", n_samples)
            target_value = kwargs.get("target_value", getattr(data, "optimal_value", None))
            if target_value is not None and target_value <= 0:
                target_value = None

            fix_threshold = kwargs.get("fix_threshold", 0.9)
            ilp_time_limit = kwargs.get("ilp_time_limit", 1.0)
            ilp_threads = kwargs.get("ilp_threads", None)
            max_hint_items = kwargs.get("max_hint_items", None)

            warm_inputs = self._prepare_warm_start(
                probs=probs,
                data=data,
                temperature=temperature,
                schedule=warm_schedule,
                tolerance=warm_tolerance,
                max_samples=max_samples,
                target_value=target_value,
                fix_threshold=fix_threshold,
            )

            sampling_result = warm_inputs["sampling_result"]
            initial_solution = warm_inputs["initial_solution"]
            fix_map = warm_inputs["fix_map"]

            warm_result = solve_knapsack_warm_start(
                weights=weights,
                values=values,
                capacity=capacity,
                initial_solution=initial_solution,
                fixed_variables=fix_map,
                time_limit=ilp_time_limit,
                num_threads=ilp_threads,
                max_hint_items=max_hint_items,
            )

            warm_status = warm_result["status_name"]
            warm_solution = (
                warm_result["solution"]
                if warm_result["objective"] is not None
                else initial_solution
            )
            warm_value = self.compute_value(warm_solution, values)
            is_feasible = self.check_feasibility(warm_solution, weights, capacity)
            ilp_success = warm_status in ("OPTIMAL", "FEASIBLE")

            result.update(
                {
                    "solution": warm_solution,
                    "value": float(warm_value),
                    "is_feasible": is_feasible,
                    "n_samples": max_samples,
                    "samples_used": sampling_result["samples_used"],
                    "n_feasible_samples": sampling_result["n_feasible_samples"],
                    "schedule": sampling_result["schedule"],
                    "temperature": temperature,
                    "ilp_wall_time": warm_result["wall_time"],
                    "ilp_status": warm_status,
                    "ilp_success": ilp_success,
                    "ilp_objective": warm_result["objective"],
                    "ilp_fixed_count": warm_result["fixed_count"],
                    "ilp_hint_count": warm_result["hint_count"],
                    "ilp_branches": warm_result["branches"],
                    "ilp_conflicts": warm_result["conflicts"],
                    "ilp_best_bound": warm_result["best_bound"],
                    "initial_sampling_value": float(sampling_result["value"]),
                    "fixed_variables": fix_map,
                }
            )

        elif strategy == "sampling_repair":
            # Sampling + greedy repair + local search
            schedule = kwargs.get("schedule", kwargs.get("sampling_schedule", None))
            tolerance = kwargs.get("sampling_tolerance", self._sampling_tolerance)
            max_samples = kwargs.get("max_samples", n_samples)
            target_value = kwargs.get("target_value", getattr(data, "optimal_value", None))
            if target_value is not None and target_value <= 0:
                target_value = None

            # First, do regular sampling
            sampling_result = self.anytime_sampling(
                probs=probs,
                data=data,
                temperature=temperature,
                schedule=schedule,
                tolerance=tolerance,
                max_samples=max_samples,
                target_value=target_value,
            )

            initial_solution = sampling_result["solution"]
            initial_value = sampling_result["value"]

            # Apply repair + local search
            repairer = SolutionRepairer(
                n_restarts=kwargs.get("repair_restarts", 3),
                ratio_jitter=kwargs.get("repair_ratio_jitter", 0.05),
                random_state=kwargs.get("repair_seed"),
            )
            final_solution, repair_metadata = repairer.hybrid_repair_and_search(
                initial_solution, weights, values, capacity, use_1swap=True, use_2opt=False
            )

            # SAFETY: If repair made things worse, revert to sampling solution
            if repair_metadata["final_value"] < initial_value:
                final_solution = initial_solution
                final_value = initial_value
                repair_metadata["value_improvement"] = 0.0
                repair_metadata["value_improvement_pct"] = 0.0
                repair_metadata["reverted"] = True
            else:
                final_value = repair_metadata["final_value"]
                repair_metadata["reverted"] = False

            result.update(
                {
                    "solution": final_solution,
                    "value": final_value,
                    "is_feasible": repair_metadata["final_feasible"],
                    "n_samples": max_samples,
                    "samples_used": sampling_result["samples_used"],
                    "n_feasible_samples": sampling_result["n_feasible_samples"],
                    "schedule": sampling_result["schedule"],
                    "temperature": temperature,
                    "initial_sampling_value": initial_value,
                    "repair_improvement": repair_metadata["value_improvement"],
                    "repair_improvement_pct": repair_metadata["value_improvement_pct"],
                    "n_improvements_1swap": repair_metadata["n_improvements_1swap"],
                    "repair_reverted": repair_metadata["reverted"],
                }
            )

        elif strategy == "warm_start_repair":
            # Warm-start ILP + greedy repair + local search (applied to result)
            warm_schedule = kwargs.get("sampling_schedule", kwargs.get("schedule", None))
            warm_tolerance = kwargs.get("sampling_tolerance", self._sampling_tolerance)
            max_samples = kwargs.get("max_samples", n_samples)
            target_value = kwargs.get("target_value", getattr(data, "optimal_value", None))
            if target_value is not None and target_value <= 0:
                target_value = None

            fix_threshold = kwargs.get("fix_threshold", 0.9)
            ilp_time_limit = kwargs.get("ilp_time_limit", 1.0)
            ilp_threads = kwargs.get("ilp_threads", None)
            max_hint_items = kwargs.get("max_hint_items", None)

            warm_inputs = self._prepare_warm_start(
                probs=probs,
                data=data,
                temperature=temperature,
                schedule=warm_schedule,
                tolerance=warm_tolerance,
                max_samples=max_samples,
                target_value=target_value,
                fix_threshold=fix_threshold,
            )

            sampling_result = warm_inputs["sampling_result"]
            initial_solution = warm_inputs["initial_solution"]
            fix_map = warm_inputs["fix_map"]

            warm_result = solve_knapsack_warm_start(
                weights=weights,
                values=values,
                capacity=capacity,
                initial_solution=initial_solution,
                fixed_variables=fix_map,
                time_limit=ilp_time_limit,
                num_threads=ilp_threads,
                max_hint_items=max_hint_items,
            )

            warm_status = warm_result["status_name"]
            ilp_solution = (
                warm_result["solution"]
                if warm_result["objective"] is not None
                else initial_solution
            )
            ilp_value = self.compute_value(ilp_solution, values)
            ilp_success = warm_status in ("OPTIMAL", "FEASIBLE")

            # Apply repair + local search to ILP result
            repairer = SolutionRepairer(
                n_restarts=kwargs.get("repair_restarts", 3),
                ratio_jitter=kwargs.get("repair_ratio_jitter", 0.05),
                random_state=kwargs.get("repair_seed"),
            )
            final_solution, repair_metadata = repairer.hybrid_repair_and_search(
                ilp_solution, weights, values, capacity, use_1swap=True, use_2opt=False
            )

            # SAFETY: If repair made things worse, revert to ILP solution
            if repair_metadata["final_value"] < ilp_value:
                final_solution = ilp_solution
                final_value = ilp_value
                repair_metadata["value_improvement"] = 0.0
                repair_metadata["value_improvement_pct"] = 0.0
                repair_metadata["reverted"] = True
            else:
                final_value = repair_metadata["final_value"]
                repair_metadata["reverted"] = False

            result.update(
                {
                    "solution": final_solution,
                    "value": final_value,
                    "is_feasible": repair_metadata["final_feasible"],
                    "n_samples": max_samples,
                    "samples_used": sampling_result["samples_used"],
                    "n_feasible_samples": sampling_result["n_feasible_samples"],
                    "schedule": sampling_result["schedule"],
                    "temperature": temperature,
                    "ilp_wall_time": warm_result["wall_time"],
                    "ilp_status": warm_status,
                    "ilp_success": ilp_success,
                    "ilp_objective": warm_result["objective"],
                    "ilp_fixed_count": warm_result["fixed_count"],
                    "ilp_hint_count": warm_result["hint_count"],
                    "ilp_branches": warm_result["branches"],
                    "ilp_conflicts": warm_result["conflicts"],
                    "ilp_best_bound": warm_result["best_bound"],
                    "initial_sampling_value": float(sampling_result["value"]),
                    "ilp_value": ilp_value,
                    "repair_improvement": repair_metadata["value_improvement"],
                    "repair_improvement_pct": repair_metadata["value_improvement_pct"],
                    "n_improvements_1swap": repair_metadata["n_improvements_1swap"],
                    "repair_reverted": repair_metadata["reverted"],
                    "fixed_variables": fix_map,
                }
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute optimality gap if ground truth available
        if hasattr(data, "optimal_value") and data.optimal_value > 0:
            gap = (data.optimal_value - result["value"]) / data.optimal_value * 100
            result["optimality_gap"] = gap
            result["optimal_value"] = data.optimal_value

        return result


def evaluate_model(
    model: Any,
    dataset: Any,
    strategy: str = "sampling",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    sampler_kwargs: dict | None = None,
    **kwargs: Any,
) -> dict:
    """
    Evaluate model on a dataset

    Args:
        model: Trained model
        dataset: Test dataset
        strategy: Solution strategy
        device: Device
        **kwargs: Strategy-specific arguments

    Returns:
        Evaluation results
    """
    sampler_kwargs = sampler_kwargs or {}
    sampler = KnapsackSampler(model, device, **sampler_kwargs)

    results: dict[str, Any] = {
        "values": [],
        "optimal_values": [],
        "gaps": [],
        "approx_ratios": [],
        "inference_times": [],
        "solver_times": [],
        "speedups": [],
        "feasible_count": 0,
        "total_count": 0,
        "samples_used": [],
        "feasible_samples": [],
        "ilp_wall_times": [],
        "ilp_objectives": [],
        "ilp_fixed_counts": [],
        "ilp_hint_counts": [],
        "ilp_statuses": [],
    }

    print(f"Evaluating on {len(dataset)} instances using {strategy} strategy...")

    for i, data in enumerate(dataset):
        start_time = time.perf_counter()
        result = sampler.solve(data, strategy=strategy, **kwargs)
        inference_time = time.perf_counter() - start_time

        results["inference_times"].append(inference_time)

        results["values"].append(result["value"])
        if "optimal_value" in result:
            results["optimal_values"].append(result["optimal_value"])
            results["gaps"].append(result["optimality_gap"])
            if result["optimal_value"] > 0:
                results["approx_ratios"].append(result["value"] / result["optimal_value"])

        solver_time = getattr(data, "solve_time", None)
        if solver_time is not None:
            solver_time = float(solver_time)
            results["solver_times"].append(solver_time)
            if inference_time > 0:
                results["speedups"].append(solver_time / inference_time)

        if result["is_feasible"]:
            results["feasible_count"] += 1
        results["total_count"] += 1

        if "samples_used" in result:
            results["samples_used"].append(result["samples_used"])
        if "n_feasible_samples" in result:
            results["feasible_samples"].append(result["n_feasible_samples"])
        if "ilp_wall_time" in result:
            results["ilp_wall_times"].append(result["ilp_wall_time"])
        if "ilp_objective" in result and result["ilp_objective"] is not None:
            results["ilp_objectives"].append(result["ilp_objective"])
        if "ilp_fixed_count" in result:
            results["ilp_fixed_counts"].append(result["ilp_fixed_count"])
        if "ilp_hint_count" in result:
            results["ilp_hint_counts"].append(result["ilp_hint_count"])
        if "ilp_status" in result:
            results["ilp_statuses"].append(result["ilp_status"])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(dataset)}")

    # Compute statistics
    if results["gaps"]:
        gap_array = np.array(results["gaps"], dtype=np.float64)
        results["mean_gap"] = float(np.mean(gap_array))
        results["std_gap"] = float(np.std(gap_array))
        results["median_gap"] = float(np.median(gap_array))
        results["max_gap"] = float(np.max(gap_array))

        n = gap_array.size
        if n == 1:
            results["gap_t_stat"] = None
            results["gap_p_value"] = None
            results["gap_mean_ci_95"] = [float(gap_array[0]), float(gap_array[0])]
        else:
            sample_std = gap_array.std(ddof=1)
            if sample_std == 0:
                results["gap_t_stat"] = None
                results["gap_p_value"] = None
                results["gap_mean_ci_95"] = [float(gap_array[0]), float(gap_array[0])]
            else:
                se = sample_std / math.sqrt(n)
                mean_gap = results["mean_gap"]
                t_stat = mean_gap / se
                if scipy_stats is not None:
                    p_value = float(scipy_stats.t.sf(abs(t_stat), df=n - 1) * 2)
                    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
                else:
                    z = abs(t_stat)
                    # Normal approximation fallback
                    p_value = float(2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))))
                    t_crit = 1.96
                half_width = t_crit * se
                results["gap_t_stat"] = float(t_stat)
                results["gap_p_value"] = p_value
                results["gap_mean_ci_95"] = [
                    float(mean_gap - half_width),
                    float(mean_gap + half_width),
                ]
    else:
        results["mean_gap"] = None
        results["std_gap"] = None
        results["median_gap"] = None
        results["max_gap"] = None
        results["gap_t_stat"] = None
        results["gap_p_value"] = None
        results["gap_mean_ci_95"] = None

    if results["approx_ratios"]:
        approx_array = np.array(results["approx_ratios"], dtype=np.float64)
        results["mean_approx_ratio"] = float(np.mean(approx_array))
        results["std_approx_ratio"] = float(np.std(approx_array))
        results["median_approx_ratio"] = float(np.median(approx_array))
        results["min_approx_ratio"] = float(np.min(approx_array))
        results["max_approx_ratio"] = float(np.max(approx_array))
    else:
        results["mean_approx_ratio"] = None
        results["std_approx_ratio"] = None
        results["median_approx_ratio"] = None
        results["min_approx_ratio"] = None
        results["max_approx_ratio"] = None

    inference_array = np.array(results["inference_times"], dtype=np.float64)
    if inference_array.size:
        results["mean_inference_time"] = float(np.mean(inference_array))
        results["median_inference_time"] = float(np.median(inference_array))
        results["std_inference_time"] = float(np.std(inference_array))
        results["min_inference_time"] = float(np.min(inference_array))
        results["max_inference_time"] = float(np.max(inference_array))
        results["p50_inference_time"] = float(np.percentile(inference_array, 50))
        results["p90_inference_time"] = float(np.percentile(inference_array, 90))
        results["p99_inference_time"] = float(np.percentile(inference_array, 99))
        total_inference_time = float(np.sum(inference_array))
        results["total_inference_time"] = total_inference_time
        results["throughput"] = (
            results["total_count"] / total_inference_time if total_inference_time > 0 else None
        )
    else:
        results["mean_inference_time"] = None
        results["median_inference_time"] = None
        results["std_inference_time"] = None
        results["min_inference_time"] = None
        results["max_inference_time"] = None
        results["p50_inference_time"] = None
        results["p90_inference_time"] = None
        results["p99_inference_time"] = None
        results["total_inference_time"] = None
        results["throughput"] = None

    if results["solver_times"]:
        solver_array = np.array(results["solver_times"], dtype=np.float64)
        results["mean_solver_time"] = float(np.mean(solver_array))
        results["median_solver_time"] = float(np.median(solver_array))
        results["std_solver_time"] = float(np.std(solver_array))
        results["min_solver_time"] = float(np.min(solver_array))
        results["max_solver_time"] = float(np.max(solver_array))
    else:
        results["mean_solver_time"] = None
        results["median_solver_time"] = None
        results["std_solver_time"] = None
        results["min_solver_time"] = None
        results["max_solver_time"] = None

    if results["speedups"]:
        speedup_array = np.array(results["speedups"], dtype=np.float64)
        results["mean_speedup"] = float(np.mean(speedup_array))
        results["median_speedup"] = float(np.median(speedup_array))
        results["min_speedup"] = float(np.min(speedup_array))
        results["max_speedup"] = float(np.max(speedup_array))
    else:
        results["mean_speedup"] = None
        results["median_speedup"] = None
        results["min_speedup"] = None
        results["max_speedup"] = None

    if results["samples_used"]:
        samples_array = np.array(results["samples_used"], dtype=np.float64)
        results["mean_samples_used"] = float(np.mean(samples_array))
        results["median_samples_used"] = float(np.median(samples_array))
    else:
        results["mean_samples_used"] = None
        results["median_samples_used"] = None

    if results["ilp_wall_times"]:
        ilp_array = np.array(results["ilp_wall_times"], dtype=np.float64)
        results["mean_ilp_time"] = float(np.mean(ilp_array))
        results["median_ilp_time"] = float(np.median(ilp_array))
        results["p90_ilp_time"] = float(np.percentile(ilp_array, 90))
        results["p99_ilp_time"] = float(np.percentile(ilp_array, 99))
    else:
        results["mean_ilp_time"] = None
        results["median_ilp_time"] = None
        results["p90_ilp_time"] = None
        results["p99_ilp_time"] = None

    if results["ilp_statuses"]:
        status_counts: dict[str, int] = {}
        for status in results["ilp_statuses"]:
            status_counts[status] = status_counts.get(status, 0) + 1
        results["ilp_status_counts"] = status_counts
        total_ilp_runs = sum(status_counts.values())
        success = sum(status_counts.get(s, 0) for s in ("OPTIMAL", "FEASIBLE"))
        results["ilp_success_rate"] = success / total_ilp_runs if total_ilp_runs else None
    else:
        results["ilp_status_counts"] = {}
        results["ilp_success_rate"] = None

    results["feasibility_rate"] = results["feasible_count"] / results["total_count"]

    print("\n=== Evaluation Results ===")
    print(f"Feasibility Rate: {results['feasibility_rate'] * 100:.2f}%")
    if results["mean_gap"] is not None:
        print(f"Mean Optimality Gap: {results['mean_gap']:.2f}%")
        print(f"Median Optimality Gap: {results['median_gap']:.2f}%")
        print(f"Std Optimality Gap: {results['std_gap']:.2f}%")
        print(f"Max Optimality Gap: {results['max_gap']:.2f}%")
        if results["gap_mean_ci_95"]:
            ci_low, ci_high = results["gap_mean_ci_95"]
            print(f"95% CI (Mean Gap): [{ci_low:.2f}%, {ci_high:.2f}%]")
        if results["gap_p_value"] is not None:
            print(f"T-test vs 0 Gap: t={results['gap_t_stat']:.3f}, p={results['gap_p_value']:.4f}")

    if results["mean_approx_ratio"] is not None:
        print("\nApproximation Ratio:")
        print(f"  Mean:   {results['mean_approx_ratio']:.4f}")
        print(f"  Median: {results['median_approx_ratio']:.4f}")
        print(f"  Min:    {results['min_approx_ratio']:.4f}")
        print(f"  Max:    {results['max_approx_ratio']:.4f}")

    if results["mean_inference_time"] is not None:
        print("\nInference Timing:")
        print(f"  Mean:   {results['mean_inference_time'] * 1000:.2f} ms")
        print(f"  Median: {results['median_inference_time'] * 1000:.2f} ms")
        print(f"  P90:    {results['p90_inference_time'] * 1000:.2f} ms")
        print(f"  P99:    {results['p99_inference_time'] * 1000:.2f} ms")
        if results["throughput"] is not None:
            print(f"  Throughput: {results['throughput']:.2f} inst/s")
        else:
            print("  Throughput: N/A")

    if results["mean_solver_time"] is not None:
        print("\nExact Solver Timing:")
        print(f"  Mean:   {results['mean_solver_time'] * 1000:.2f} ms")
        print(f"  Median: {results['median_solver_time'] * 1000:.2f} ms")

    if results["mean_speedup"] is not None:
        print("\nSpeedup vs Exact Solver:")
        print(f"  Mean:   {results['mean_speedup']:.2f}x")
        print(f"  Median: {results['median_speedup']:.2f}x")
        print(f"  Min:    {results['min_speedup']:.2f}x")
        print(f"  Max:    {results['max_speedup']:.2f}x")

    if results["samples_used"]:
        samples_array = np.array(results["samples_used"], dtype=np.float64)
        results["mean_samples_used"] = float(np.mean(samples_array))
        results["median_samples_used"] = float(np.median(samples_array))
    else:
        results["mean_samples_used"] = None
        results["median_samples_used"] = None

    if results["mean_samples_used"] is not None:
        print("\nSampling Stats:")
        print(f"  Mean samples used: {results['mean_samples_used']:.2f}")
        print(f"  Median samples used: {results['median_samples_used']:.2f}")

    if results["mean_ilp_time"] is not None:
        print("\nWarm-Start ILP Timing:")
        print(f"  Mean:   {results['mean_ilp_time'] * 1000:.2f} ms")
        print(f"  Median: {results['median_ilp_time'] * 1000:.2f} ms")
        print(f"  P90:    {results['p90_ilp_time'] * 1000:.2f} ms")
        print(f"  P99:    {results['p99_ilp_time'] * 1000:.2f} ms")
    if results["ilp_status_counts"]:
        success_rate = results["ilp_success_rate"]
        if success_rate is not None:
            print(f"  ILP success rate: {success_rate * 100:.2f}%")
        else:
            print("  ILP success rate: N/A")
        print(f"  Status counts: {results['ilp_status_counts']}")

    return results


# Convenience wrapper functions for backward compatibility
def sample_solutions(
    logits: torch.Tensor,
    n_samples: int = 100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample multiple binary solutions from logits.

    Args:
        logits: Raw model outputs (before sigmoid)
        n_samples: Number of solutions to sample
        temperature: Temperature for sampling (higher = more random)

    Returns:
        Tensor of shape [n_samples, n_items] with binary solutions
    """
    probs = torch.sigmoid(logits)
    adjusted_probs = probs.clamp(1e-6, 1 - 1e-6)

    if temperature != 1.0:
        temperature = max(float(temperature), 1e-6)
        logits_temp = torch.logit(adjusted_probs) / temperature
        adjusted_probs = torch.sigmoid(logits_temp)

    expanded = adjusted_probs.expand(n_samples, -1)
    samples = torch.bernoulli(expanded)

    return samples


def vectorized_sampling(
    logits: torch.Tensor,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float,
    n_samples: int = 100,
    temperature: float = 1.0,
) -> tuple[np.ndarray, float, bool]:
    """
    Vectorized sampling with feasibility checking.

    Args:
        logits: Raw model outputs
        weights: Item weights
        values: Item values
        capacity: Knapsack capacity
        n_samples: Number of solutions to sample
        temperature: Temperature for sampling

    Returns:
        Tuple of (best_solution, best_value, is_feasible)
    """
    samples = sample_solutions(logits, n_samples, temperature)
    samples_np = samples.cpu().numpy()

    # Compute values and weights for all samples
    sample_values = samples_np @ values
    sample_weights = samples_np @ weights

    # Find feasible samples
    feasible_mask = sample_weights <= capacity

    if np.any(feasible_mask):
        # Select best feasible solution
        feasible_values = np.where(feasible_mask, sample_values, -np.inf)
        best_idx = np.argmax(feasible_values)
        return samples_np[best_idx], sample_values[best_idx], True
    else:
        # No feasible solution found, return best by value (even if infeasible)
        best_idx = np.argmax(sample_values)
        return samples_np[best_idx], sample_values[best_idx], False


if __name__ == "__main__":
    print("This module provides inference utilities.")
    print("Use evaluate.py script to evaluate models.")
