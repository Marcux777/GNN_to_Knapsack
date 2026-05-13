"""
Advanced baseline solvers for the knapsack problem.

Includes:
    - FPTASSolver: Fully Polynomial Time Approximation Scheme.
    - MeetInTheMiddleSolver: Horowitz–Sahni style exact solver with optional fallback.
"""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from knapsack_gnn.data.generator import KnapsackInstance

_EPS = 1e-9


def _compute_opt_gap(value: float, optimal_value: float | None) -> float | None:
    if optimal_value is None or optimal_value <= 0:
        return None
    return 100.0 * (optimal_value - value) / optimal_value


def _build_result(
    instance: KnapsackInstance,
    solution: np.ndarray,
    solve_time: float,
) -> dict:
    values = instance.values
    weights = instance.weights

    total_value = float(np.dot(solution, values))
    total_weight = float(np.dot(solution, weights))
    gap = _compute_opt_gap(total_value, instance.optimal_value)

    return {
        "solution": solution.astype(np.int32),
        "value": total_value,
        "is_feasible": bool(total_weight <= instance.capacity + _EPS),
        "solve_time": solve_time,
        "optimality_gap": gap,
        "weight_used": total_weight,
        "capacity": instance.capacity,
        "optimal_value": instance.optimal_value,
    }


class FPTASSolver:
    """Fully Polynomial Time Approximation Scheme for knapsack."""

    def __init__(self, epsilon: float = 0.05) -> None:
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError("epsilon must be in (0, 1).")
        self.epsilon = float(epsilon)

    def solve(self, instance: KnapsackInstance) -> dict:
        start = time.perf_counter()
        solution = self._solve_solution(instance)
        solve_time = time.perf_counter() - start
        return _build_result(instance, solution, solve_time)

    def solve_batch(self, instances: Iterable[KnapsackInstance]) -> list[dict]:
        return [self.solve(instance) for instance in instances]

    def _solve_solution(self, instance: KnapsackInstance) -> np.ndarray:
        n_items = instance.n_items
        if n_items == 0:
            return np.zeros(0, dtype=np.int32)

        values = instance.values.astype(np.float64)
        weights = instance.weights.astype(np.float64)

        max_value = float(np.max(values)) if values.size else 0.0
        if max_value <= 0:
            return np.zeros(n_items, dtype=np.int32)

        scale = self.epsilon * max_value / max(1, n_items)
        scale = max(scale, 1e-9)

        scaled_values = np.floor(values / scale).astype(int)
        scaled_values = np.where((scaled_values <= 0) & (values > 0), 1, scaled_values)
        capacity = float(instance.capacity)

        total_scaled = int(np.sum(scaled_values))
        if total_scaled <= 0:
            return np.zeros(n_items, dtype=np.int32)

        dp = np.full(total_scaled + 1, np.inf, dtype=np.float64)
        trace: list[dict[int, int]] = [dict() for _ in range(n_items)]
        dp[0] = 0.0

        for idx in range(n_items):
            sv = int(scaled_values[idx])
            if sv <= 0:
                continue
            weight = float(weights[idx])
            for value_key in range(total_scaled, sv - 1, -1):
                candidate_weight = dp[value_key - sv] + weight
                if candidate_weight + _EPS < dp[value_key]:
                    dp[value_key] = candidate_weight
                    trace[idx][value_key] = value_key - sv

        feasible_values = [v for v in range(total_scaled + 1) if dp[v] <= capacity + _EPS]
        if not feasible_values:
            return np.zeros(n_items, dtype=np.int32)

        best_value = max(feasible_values)
        solution = np.zeros(n_items, dtype=np.int32)

        curr_value = best_value
        for idx in range(n_items - 1, -1, -1):
            prev = trace[idx].get(curr_value)
            if prev is None:
                continue
            solution[idx] = 1
            curr_value = prev
            if curr_value <= 0:
                break

        return solution


class MeetInTheMiddleSolver:
    """
    Horowitz–Sahni meet-in-the-middle solver with optional FPTAS fallback.

    For instances larger than `max_exact_items` a FPTAS fallback is used to avoid
    exponential blow-up while still providing a reasonable baseline.
    """

    def __init__(self, max_exact_items: int = 32, fallback_epsilon: float = 0.02) -> None:
        if max_exact_items <= 0:
            raise ValueError("max_exact_items must be positive.")
        self.max_exact_items = int(max_exact_items)
        self._fallback = FPTASSolver(epsilon=fallback_epsilon)

    def solve(self, instance: KnapsackInstance) -> dict:
        start = time.perf_counter()
        if instance.n_items == 0:
            solution = np.zeros(0, dtype=np.int32)
        elif instance.n_items <= self.max_exact_items:
            solution = self._meet_in_middle(instance)
        else:
            solution = self._fallback._solve_solution(instance)
        solve_time = time.perf_counter() - start
        return _build_result(instance, solution, solve_time)

    def solve_batch(self, instances: Iterable[KnapsackInstance]) -> list[dict]:
        return [self.solve(instance) for instance in instances]

    def _meet_in_middle(self, instance: KnapsackInstance) -> np.ndarray:
        weights = instance.weights.astype(np.float64)
        values = instance.values.astype(np.float64)
        capacity = float(instance.capacity)
        n_items = instance.n_items

        mid = n_items // 2
        left_states = self._enumerate_half(weights[:mid], values[:mid], capacity)
        right_states = self._enumerate_half(weights[mid:], values[mid:], capacity)

        if not left_states:
            left_states = [(0.0, 0.0, 0)]
        if not right_states:
            right_states = [(0.0, 0.0, 0)]

        left_states.sort(key=lambda item: item[0])
        left_weights = [w for w, _, _ in left_states]
        best_left_value = []
        best_left_mask = []
        current_val = -math.inf
        current_mask = 0
        for _, value, mask in left_states:
            if value > current_val + _EPS:
                current_val = value
                current_mask = mask
            best_left_value.append(current_val)
            best_left_mask.append(current_mask)

        from bisect import bisect_right

        best_total = 0.0
        best_left = 0
        best_right = 0

        for weight_r, value_r, mask_r in right_states:
            if weight_r > capacity + _EPS:
                continue
            remaining = capacity - weight_r
            idx = bisect_right(left_weights, remaining) - 1
            if idx < 0:
                continue
            total_value = value_r + best_left_value[idx]
            if total_value > best_total + _EPS:
                best_total = total_value
                best_left = best_left_mask[idx]
                best_right = mask_r

        solution = np.zeros(n_items, dtype=np.int32)
        for i in range(mid):
            if best_left & (1 << i):
                solution[i] = 1
        for j in range(n_items - mid):
            if best_right & (1 << j):
                solution[mid + j] = 1
        return solution

    def _enumerate_half(
        self, weights: np.ndarray, values: np.ndarray, capacity: float
    ) -> list[tuple[float, float, int]]:
        m = len(weights)
        combos: list[tuple[float, float, int]] = []
        for mask in range(1 << m):
            total_weight = 0.0
            total_value = 0.0
            feasible = True
            for i in range(m):
                if mask & (1 << i):
                    total_weight += float(weights[i])
                    if total_weight > capacity + _EPS:
                        feasible = False
                        break
                    total_value += float(values[i])
            if feasible:
                combos.append((total_weight, total_value, mask))

        combos.sort(key=lambda item: (item[0], -item[1]))
        pruned: list[tuple[float, float, int]] = []
        best_value = -math.inf
        for weight, value, mask in combos:
            if value > best_value + _EPS:
                pruned.append((weight, value, mask))
                best_value = value
        return pruned
