"""
Tests for the advanced baseline solvers (FPTAS and meet-in-the-middle).
"""

import numpy as np

from knapsack_gnn.baselines.advanced import FPTASSolver, MeetInTheMiddleSolver
from knapsack_gnn.data.generator import KnapsackInstance


def _brute_force_opt(values: np.ndarray, weights: np.ndarray, capacity: float) -> tuple[float, np.ndarray]:
    """Compute optimal value/solution via brute force (used for assertions)."""
    n_items = len(values)
    best_value = -1.0
    best_solution = np.zeros(n_items, dtype=np.int32)

    for mask in range(1 << n_items):
        total_weight = 0.0
        total_value = 0.0
        feasible = True
        for idx in range(n_items):
            if mask & (1 << idx):
                total_weight += float(weights[idx])
                if total_weight > capacity:
                    feasible = False
                    break
                total_value += float(values[idx])
        if feasible and total_value > best_value:
            best_value = total_value
            for idx in range(n_items):
                best_solution[idx] = 1 if (mask & (1 << idx)) else 0

    return best_value, best_solution.copy()


def _make_instance(values, weights, capacity) -> KnapsackInstance:
    return KnapsackInstance(
        weights=np.array(weights, dtype=np.float32),
        values=np.array(values, dtype=np.float32),
        capacity=int(capacity),
    )


def test_fptas_respects_approximation_guarantee():
    """FPTAS should stay within (1 - ε) of the optimal value."""
    values = np.array([12, 7, 20, 15, 9], dtype=np.float32)
    weights = np.array([4, 3, 8, 6, 4], dtype=np.float32)
    capacity = 15
    optimal_value, _ = _brute_force_opt(values, weights, capacity)

    instance = _make_instance(values, weights, capacity)
    instance.optimal_value = optimal_value

    solver = FPTASSolver(epsilon=0.1)
    result = solver.solve(instance)

    assert result["is_feasible"]
    assert result["value"] >= (1 - solver.epsilon) * optimal_value - 1e-6


def test_meet_in_middle_matches_optimal_small_instances():
    """Meet-in-the-middle should recover the exact optimum for small n."""
    values = np.array([10, 40, 30, 50], dtype=np.float32)
    weights = np.array([5, 4, 6, 3], dtype=np.float32)
    capacity = 10
    optimal_value, _ = _brute_force_opt(values, weights, capacity)

    instance = _make_instance(values, weights, capacity)
    instance.optimal_value = optimal_value

    solver = MeetInTheMiddleSolver(max_exact_items=10)
    result = solver.solve(instance)

    assert result["is_feasible"]
    assert abs(result["value"] - optimal_value) <= 1e-6


def test_meet_in_middle_fallback_handles_large_instances():
    """When n > max_exact_items the solver should fall back gracefully."""
    rng = np.random.default_rng(0)
    values = rng.integers(5, 20, size=8).astype(np.float32)
    weights = rng.integers(2, 10, size=8).astype(np.float32)
    capacity = int(weights.sum() * 0.6)
    optimal_value, _ = _brute_force_opt(values, weights, capacity)

    instance = _make_instance(values, weights, capacity)
    instance.optimal_value = optimal_value

    solver = MeetInTheMiddleSolver(max_exact_items=3, fallback_epsilon=0.005)
    result = solver.solve(instance)

    assert result["is_feasible"]
    assert result["value"] >= optimal_value * (1 - 0.01)
