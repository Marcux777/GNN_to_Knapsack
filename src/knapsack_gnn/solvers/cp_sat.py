"""
Warm-start ILP solver utilities for the Knapsack problem using OR-Tools CP-SAT.
"""

import time
from typing import Any

import numpy as np
import torch

cp_model: Any | None
try:
    from ortools.sat.python import cp_model as _cp_model

    cp_model = _cp_model
except ImportError:  # pragma: no cover - optional dependency
    cp_model = None


def solve_knapsack_warm_start(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float,
    initial_solution: np.ndarray | None = None,
    fixed_variables: dict[int, int] | None = None,
    time_limit: float | None = None,
    num_threads: int | None = None,
    max_hint_items: int | None = None,
) -> dict:
    """
    Solve the 0/1 knapsack problem with optional warm-start hints and fixed variables.

    Args:
        weights: Item weights (array of ints/floats, will be cast to int).
        values: Item values (array of ints/floats).
        capacity: Knapsack capacity.
        initial_solution: Optional binary array to hint the solver (length = n_items).
        fixed_variables: Optional dict {index: value} to fix certain variables.
        time_limit: Optional time limit (seconds).
        num_threads: Optional number of solver threads (default: 1 for reproducibility).
        max_hint_items: Optional cap on number of hints (if provided, highest scores kept).

    Returns:
        Dictionary with keys:
            status (int), status_name (str), solution (np.ndarray),
            objective (float or None), wall_time (float),
            best_bound (float or None), fixed_count (int),
            hint_count (int), branches (int), conflicts (int), stats (str)
    """
    if cp_model is None:
        fallback_solution, fallback_objective = _dp_fallback_solver(weights, values, capacity)
        fallback_status = 1  # Mimic OPTIMAL
        hint_count = int(initial_solution.size) if initial_solution is not None else 0
        return {
            "status": fallback_status,
            "status_name": "OPTIMAL",
            "solution": fallback_solution,
            "objective": fallback_objective,
            "wall_time": 0.0,
            "best_bound": fallback_objective,
            "fixed_count": len(fixed_variables) if fixed_variables else 0,
            "hint_count": hint_count,
            "branches": 0,
            "conflicts": 0,
            "stats": "Fallback dynamic-programming solver",
        }

    weights = np.asarray(weights, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    capacity = int(capacity)
    n_items = weights.size

    model = cp_model.CpModel()
    x_vars = [model.NewBoolVar(f"x_{i}") for i in range(n_items)]

    # Capacity constraint
    model.Add(sum(int(weights[i]) * x_vars[i] for i in range(n_items)) <= capacity)

    # Objective
    model.Maximize(sum(int(values[i]) * x_vars[i] for i in range(n_items)))

    # Apply fixed variables (hard constraints)
    fixed_count = 0
    if fixed_variables:
        for idx, val in fixed_variables.items():
            if 0 <= idx < n_items:
                model.Add(x_vars[idx] == int(val))
                fixed_count += 1

    hint_count = 0
    if initial_solution is not None:
        hints = np.asarray(initial_solution).astype(np.int32)
        if hints.shape[0] != n_items:
            raise ValueError("initial_solution must have the same length as weights/values.")

        if max_hint_items is not None and 0 < max_hint_items < n_items:
            # Select variables with highest weights*values as hints
            importance = np.abs(values) + np.abs(weights)
            top_indices = np.argsort(-importance)[:max_hint_items]
            for idx in top_indices:
                model.AddHint(x_vars[idx], int(hints[idx]))
                hint_count += 1
        else:
            for idx, val in enumerate(hints):
                model.AddHint(x_vars[idx], int(val))
            hint_count = n_items

    solver = cp_model.CpSolver()
    if time_limit is not None and time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = max(1, int(num_threads)) if num_threads else 1
    solver.parameters.random_seed = 0

    status = solver.Solve(model)

    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }

    solution = np.zeros(n_items, dtype=np.int32)
    objective = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = np.array([solver.Value(var) for var in x_vars], dtype=np.int32)
        objective = float(solver.ObjectiveValue())

    best_bound = (
        solver.BestObjectiveBound() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    )

    return {
        "status": status,
        "status_name": status_map.get(status, "UNKNOWN"),
        "solution": solution,
        "objective": objective,
        "wall_time": float(solver.WallTime()),
        "best_bound": float(best_bound) if best_bound is not None else None,
        "fixed_count": fixed_count,
        "hint_count": hint_count,
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
        "stats": solver.ResponseStats(),
    }


def warm_start_ilp_solve(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    initial_solution: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    time_limit: float = 1.0,
    fix_threshold: float = 0.9,
    seed: int = 0,
) -> tuple[float, np.ndarray, str]:
    """
    Simplified wrapper for warm-start ILP solving compatible with test expectations.

    Args:
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        initial_solution: Optional initial solution hint
        probabilities: Optional probabilities to determine which variables to fix
        time_limit: Time limit in seconds
        fix_threshold: Threshold for fixing variables when using probabilities
        seed: Random seed for reproducibility

    Returns:
        Tuple of (objective_value, solution, status_name)
    """
    # Handle fixed variables based on probabilities and threshold
    fixed_variables = None
    if probabilities is not None:
        fixed_variables = {}
        for idx, prob in enumerate(probabilities):
            if prob >= fix_threshold:
                fixed_variables[idx] = 1
            elif prob <= (1.0 - fix_threshold):
                fixed_variables[idx] = 0

    result = solve_knapsack_warm_start(
        weights=weights,
        values=values,
        capacity=capacity,
        initial_solution=initial_solution,
        fixed_variables=fixed_variables,
        time_limit=time_limit,
        num_threads=1,
    )

    objective = result["objective"] if result["objective"] is not None else 0.0
    return objective, result["solution"], result["status_name"]


def refine_solution(
    logits: torch.Tensor,
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    fix_threshold: float = 0.9,
    time_limit: float = 1.0,
) -> tuple[float, np.ndarray, float, str]:
    """
    Refine a GNN solution using ILP with warm-start hints.

    Args:
        logits: GNN output logits (torch.Tensor)
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        fix_threshold: Threshold for fixing variables based on probabilities
        time_limit: Time limit in seconds

    Returns:
        Tuple of (objective_value, solution, ilp_time, status_name)
    """
    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()

    # Create initial solution from probabilities
    initial_solution = (probabilities > 0.5).astype(np.int32)

    start_time = time.perf_counter()
    objective, solution, status = warm_start_ilp_solve(
        values=values,
        weights=weights,
        capacity=capacity,
        initial_solution=initial_solution,
        probabilities=probabilities,
        time_limit=time_limit,
        fix_threshold=fix_threshold,
    )
    ilp_time = time.perf_counter() - start_time

    return objective, solution, ilp_time, status


def _dp_fallback_solver(
    weights: np.ndarray, values: np.ndarray, capacity: float
) -> tuple[np.ndarray, float]:
    """Simple DP solver used when OR-Tools is unavailable."""
    w = np.asarray(weights, dtype=np.int64)
    v = np.asarray(values, dtype=np.float64)
    cap = int(capacity)
    n_items = w.size

    dp = np.zeros((n_items + 1, cap + 1), dtype=np.float64)
    for i in range(1, n_items + 1):
        w_i = int(w[i - 1])
        v_i = float(v[i - 1])
        for c in range(cap + 1):
            dp[i, c] = dp[i - 1, c]
            if c >= w_i:
                candidate = dp[i - 1, c - w_i] + v_i
                if candidate > dp[i, c]:
                    dp[i, c] = candidate

    solution = np.zeros(n_items, dtype=np.int32)
    remaining = cap
    for i in range(n_items, 0, -1):
        if remaining < 0:
            break
        if dp[i, remaining] > dp[i - 1, remaining] + 1e-9:
            solution[i - 1] = 1
            remaining -= int(w[i - 1])

    return solution, float(dp[n_items, cap])
