"""
Tests for solve_knapsack_warm_start to ensure fixed variables and timing work.
"""

import time

import numpy as np
import pytest

from knapsack_gnn.solvers import cp_sat
from knapsack_gnn.solvers.cp_sat import solve_knapsack_warm_start


def _skip_if_no_cp_sat() -> None:
    if cp_sat.cp_model is None:  # pragma: no cover - dependency missing
        pytest.skip("OR-Tools CP-SAT is not available in this environment")


def test_fixed_variables_are_respected():
    """Fixed variables should be enforced in the MILP model."""
    _skip_if_no_cp_sat()
    weights = np.array([2, 3, 4, 5], dtype=np.float32)
    values = np.array([4, 5, 6, 7], dtype=np.float32)
    capacity = 7.0

    result = solve_knapsack_warm_start(
        weights=weights,
        values=values,
        capacity=capacity,
        fixed_variables={0: 1, 1: 0},  # force include item 0 and exclude item 1
        time_limit=0.5,
    )

    solution = result["solution"]
    assert solution[0] == 1, "Item 0 should be fixed to 1"
    assert solution[1] == 0, "Item 1 should be fixed to 0"
    assert np.dot(solution, weights) <= capacity + 1e-6


def test_time_limit_is_honored_within_margin():
    """Solver wall time should stay close to the imposed limit."""
    _skip_if_no_cp_sat()
    weights = np.array([3, 5, 7, 9, 11, 13, 17], dtype=np.float32)
    values = np.array([9, 12, 14, 15, 19, 23, 31], dtype=np.float32)
    capacity = 30.0
    time_limit = 0.05

    start = time.perf_counter()
    result = solve_knapsack_warm_start(
        weights=weights,
        values=values,
        capacity=capacity,
        time_limit=time_limit,
        num_threads=1,
    )
    elapsed = time.perf_counter() - start

    assert result["wall_time"] <= time_limit * 5, (
        f"Reported wall time {result['wall_time']:.3f}s exceeds budget margin"
    )
    assert elapsed <= time_limit * 10, (
        f"Solver elapsed time {elapsed:.3f}s exceeds acceptable margin"
    )
