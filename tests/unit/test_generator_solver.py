"""
Tests for the KnapsackGenerator + KnapsackSolver integration.
"""

import numpy as np

from knapsack_gnn.data.generator import (
    KnapsackGenerator,
    KnapsackInstance,
    KnapsackSolver,
)


class TestGeneratorSolver:
    """Ensure generated instances are solved correctly by the exact solver."""

    def test_solver_returns_feasible_solutions_for_batch(self):
        """Solve a batch of instances and ensure feasibility + metadata."""
        generator = KnapsackGenerator(seed=123)
        instances = [generator.generate_instance(n_items=12) for _ in range(20)]

        for instance in instances:
            solved = KnapsackSolver.solve(instance, time_limit=2.0)
            assert solved.solution is not None, "Solver must attach a solution"
            assert solved.optimal_value is not None, "Solver must set optimal value"

            total_weight = np.dot(solved.solution, solved.weights)
            assert total_weight <= solved.capacity + 1e-6, "Solution must satisfy capacity"

            # Optimal value should match solution·values
            solution_value = float(np.dot(solved.solution, solved.values))
            assert np.isclose(solution_value, solved.optimal_value), (
                f"Optimal value mismatch: {solution_value} vs {solved.optimal_value}"
            )

    def test_optimal_value_increases_with_high_value_item(self):
        """Adding a high-value light item should strictly improve the optimum."""
        weights = np.array([4, 6, 5], dtype=np.float32)
        values = np.array([10, 14, 12], dtype=np.float32)
        base_capacity = 10.0

        base_instance = KnapsackInstance(weights=weights, values=values, capacity=int(base_capacity))
        base_solution = KnapsackSolver.solve(base_instance)
        assert base_solution.optimal_value is not None

        bonus_weight = np.array([1], dtype=np.float32)
        bonus_value = np.array([100], dtype=np.float32)  # Dominant item

        augmented_instance = KnapsackInstance(
            weights=np.concatenate([weights, bonus_weight]),
            values=np.concatenate([values, bonus_value]),
            capacity=int(base_capacity + bonus_weight[0]),
        )
        augmented_solution = KnapsackSolver.solve(augmented_instance)

        assert augmented_solution.optimal_value is not None
        assert augmented_solution.optimal_value > base_solution.optimal_value, (
            "Adding a dominant item should improve the optimal objective"
        )
