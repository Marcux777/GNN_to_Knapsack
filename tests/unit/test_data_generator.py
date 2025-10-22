"""
Tests for knapsack instance generation and exact solver.
"""

import numpy as np

from knapsack_gnn.data.generator import (
    generate_knapsack_instance,
    solve_knapsack_dp,
    solve_with_ortools,
)

class TestKnapsackGenerator:
    """Test suite for knapsack instance generation."""

    def test_generate_instance_shape(self):
        """Test that generated instances have correct shapes."""
        n_items = 10
        instance = generate_knapsack_instance(n_items, seed=42)

        assert len(instance["values"]) == n_items
        assert len(instance["weights"]) == n_items
        assert isinstance(instance["capacity"], (int, float))

    def test_generate_instance_positive_values(self):
        """Test that values and weights are positive."""
        instance = generate_knapsack_instance(20, seed=42)

        assert np.all(instance["values"] > 0), "All values must be positive"
        assert np.all(instance["weights"] > 0), "All weights must be positive"
        assert instance["capacity"] > 0, "Capacity must be positive"

    def test_generate_instance_capacity_reasonable(self):
        """Test that capacity is reasonable (typically ~50% of total weight)."""
        instance = generate_knapsack_instance(50, seed=42)
        total_weight = np.sum(instance["weights"])

        # Capacity should be between 30% and 70% of total weight
        assert 0.3 * total_weight <= instance["capacity"] <= 0.7 * total_weight

    def test_generate_instance_deterministic(self):
        """Test that same seed produces same instance."""
        inst1 = generate_knapsack_instance(15, seed=42)
        inst2 = generate_knapsack_instance(15, seed=42)

        assert np.array_equal(inst1["values"], inst2["values"])
        assert np.array_equal(inst1["weights"], inst2["weights"])
        assert inst1["capacity"] == inst2["capacity"]

    def test_generate_instance_different_seeds(self):
        """Test that different seeds produce different instances."""
        inst1 = generate_knapsack_instance(15, seed=42)
        inst2 = generate_knapsack_instance(15, seed=123)

        assert not np.array_equal(inst1["values"], inst2["values"])

class TestExactSolvers:
    """Test suite for exact knapsack solvers."""

    def test_dp_solver_small_instance(self):
        """Test DP solver on a small known instance."""
        values = np.array([10, 20, 15], dtype=np.int32)
        weights = np.array([2, 5, 3], dtype=np.int32)
        capacity = 7

        max_value, solution = solve_knapsack_dp(values, weights, capacity)

        assert max_value == 35, f"Expected value 35, got {max_value}"
        assert np.sum(solution) <= capacity, "Solution exceeds capacity"
        assert np.dot(solution, values) == max_value, "Solution value mismatch"

    def test_ortools_solver_small_instance(self):
        """Test OR-Tools solver on a small known instance."""
        values = np.array([10.0, 20.0, 15.0], dtype=np.float32)
        weights = np.array([2.0, 5.0, 3.0], dtype=np.float32)
        capacity = 7.0

        max_value, solution = solve_with_ortools(values, weights, capacity)

        assert max_value == 35.0, f"Expected value 35, got {max_value}"
        assert np.sum(solution * weights) <= capacity, "Solution exceeds capacity"
        assert np.dot(solution, values) == max_value, "Solution value mismatch"

    def test_dp_ortools_consistency(self):
        """Test that DP and OR-Tools produce same results."""
        values = np.array([12, 8, 15, 6, 18], dtype=np.float32)
        weights = np.array([3, 2, 4, 1, 5], dtype=np.float32)
        capacity = 10.0

        dp_value, dp_sol = solve_knapsack_dp(
            values.astype(np.int32), weights.astype(np.int32), int(capacity)
        )
        ort_value, ort_sol = solve_with_ortools(values, weights, capacity)

        # Both solvers should find optimal value
        assert dp_value == ort_value, "DP and OR-Tools should produce same optimal value"

    def test_solver_feasibility(self, small_knapsack_instance):
        """Test that solver produces feasible solution."""
        inst = small_knapsack_instance
        _, solution = solve_with_ortools(inst["values"], inst["weights"], inst["capacity"])

        total_weight = np.dot(solution, inst["weights"])
        assert total_weight <= inst["capacity"], "Solution violates capacity constraint"

    def test_solver_deterministic_with_seed(self):
        """Test that solver produces same result with same seed."""
        instance = generate_knapsack_instance(10, seed=42)

        val1, sol1 = solve_with_ortools(
            instance["values"], instance["weights"], instance["capacity"], seed=42
        )

        val2, sol2 = solve_with_ortools(
            instance["values"], instance["weights"], instance["capacity"], seed=42
        )

        assert val1 == val2, "Same seed should produce same optimal value"
        assert np.array_equal(sol1, sol2), "Same seed should produce same solution"

class TestInstanceDistribution:
    """Test statistical properties of generated instances."""

    def test_value_distribution(self):
        """Test that values follow expected distribution."""
        n_instances = 100
        all_values = []

        for i in range(n_instances):
            inst = generate_knapsack_instance(20, seed=i)
            all_values.extend(inst["values"].tolist())

        all_values = np.array(all_values)

        # Values should be in expected range (e.g., 1-1000)
        assert np.min(all_values) >= 1
        assert np.max(all_values) <= 2500

        # Mean should be reasonable
        assert 100 <= np.mean(all_values) <= 2000

    def test_weight_distribution(self):
        """Test that weights follow expected distribution."""
        n_instances = 100
        all_weights = []

        for i in range(n_instances):
            inst = generate_knapsack_instance(20, seed=i)
            all_weights.extend(inst["weights"].tolist())

        all_weights = np.array(all_weights)

        # Weights should be in expected range
        assert np.min(all_weights) >= 1
        assert np.max(all_weights) <= 2500

        # Mean should be reasonable
        assert 100 <= np.mean(all_weights) <= 2000
