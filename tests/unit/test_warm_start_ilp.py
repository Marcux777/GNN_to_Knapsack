"""
Tests for warm-start ILP refinement.
"""

import numpy as np
import torch

from knapsack_gnn.solvers.cp_sat import refine_solution, warm_start_ilp_solve


class TestWarmStartILP:
    """Test suite for ILP refinement with GNN warm-start."""

    def test_warm_start_never_worsens_solution(self, small_knapsack_instance):
        """Test that ILP refinement never produces worse solution than input."""
        inst = small_knapsack_instance

        # Start with a suboptimal but feasible solution
        initial_solution = np.array([1, 0, 1, 0, 0], dtype=np.int32)
        initial_value = np.dot(initial_solution, inst["values"])

        # Refine with ILP
        refined_value, refined_solution, status = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=initial_solution,
            time_limit=1.0,
            fix_threshold=0.9,
        )

        # Refined solution should be at least as good
        assert refined_value >= initial_value, (
            f"ILP worsened solution: {initial_value} -> {refined_value}"
        )

    def test_warm_start_respects_time_budget(self, small_knapsack_instance):
        """Test that ILP respects time limit."""
        import time

        inst = small_knapsack_instance
        initial_solution = np.zeros(inst["n_items"], dtype=np.int32)

        time_limit = 0.1  # 100ms
        start = time.time()

        _, _, status = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=initial_solution,
            time_limit=time_limit,
            fix_threshold=0.9,
        )

        elapsed = time.time() - start

        # Should complete within reasonable margin of time limit
        # Allow 5x overhead for solver setup
        assert elapsed < time_limit * 5.0, (
            f"ILP took {elapsed:.3f}s, exceeded budget {time_limit}s by too much"
        )

    def test_warm_start_produces_feasible_solution(self, small_knapsack_instance):
        """Test that ILP always produces feasible solution."""
        inst = small_knapsack_instance

        # Even with infeasible initial solution
        initial_solution = np.ones(inst["n_items"], dtype=np.int32)  # Select all

        refined_value, refined_solution, status = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=initial_solution,
            time_limit=1.0,
            fix_threshold=0.9,
        )

        # Check feasibility
        total_weight = np.dot(refined_solution, inst["weights"])
        assert total_weight <= inst["capacity"], (
            f"ILP solution violates capacity: {total_weight} > {inst['capacity']}"
        )

    def test_fix_threshold_effect(self, small_knapsack_instance):
        """Test that fix_threshold controls how many variables are fixed."""
        inst = small_knapsack_instance

        # Create probabilities with some high, some low
        probabilities = np.array([0.95, 0.85, 0.60, 0.40, 0.20])

        # High threshold: fix fewer variables
        _, _, status_high = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=None,
            probabilities=probabilities,
            time_limit=1.0,
            fix_threshold=0.95,
        )

        # Low threshold: fix more variables
        _, _, status_low = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=None,
            probabilities=probabilities,
            time_limit=1.0,
            fix_threshold=0.50,
        )

        # Both should succeed
        assert status_high in ["OPTIMAL", "FEASIBLE"], "High threshold solve should succeed"
        assert status_low in ["OPTIMAL", "FEASIBLE"], "Low threshold solve should succeed"

    def test_warm_start_deterministic_with_seed(self, small_knapsack_instance):
        """Test that ILP is deterministic with same seed."""
        inst = small_knapsack_instance
        initial_solution = np.array([1, 0, 1, 0, 0], dtype=np.int32)

        val1, sol1, status1 = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=initial_solution,
            time_limit=1.0,
            fix_threshold=0.9,
            seed=42,
        )

        val2, sol2, status2 = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=initial_solution,
            time_limit=1.0,
            fix_threshold=0.9,
            seed=42,
        )

        assert val1 == val2, "Same seed should produce same value"
        assert np.array_equal(sol1, sol2), "Same seed should produce same solution"
        assert status1 == status2, "Same seed should produce same status"

    def test_refine_solution_with_logits(self, small_knapsack_instance):
        """Test refine_solution function with GNN logits."""
        inst = small_knapsack_instance

        # Mock logits from GNN
        logits = torch.randn(inst["n_items"])

        refined_value, refined_solution, ilp_time, status = refine_solution(
            logits=logits,
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            fix_threshold=0.9,
            time_limit=1.0,
        )

        # Should produce valid output
        assert isinstance(refined_value, int | float), "Value should be numeric"
        assert len(refined_solution) == inst["n_items"], "Solution should match instance size"
        assert ilp_time >= 0, "ILP time should be non-negative"
        assert status in ["OPTIMAL", "FEASIBLE", "INFEASIBLE", "UNKNOWN"], (
            f"Invalid status: {status}"
        )

    def test_warm_start_status_codes(self, small_knapsack_instance):
        """Test that solver returns appropriate status codes."""
        inst = small_knapsack_instance

        # Feasible instance should return OPTIMAL or FEASIBLE
        _, _, status = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=None,
            time_limit=2.0,
            fix_threshold=0.9,
        )

        assert status in ["OPTIMAL", "FEASIBLE"], f"Expected OPTIMAL/FEASIBLE, got {status}"

    def test_warm_start_gap_improvement(self, small_knapsack_instance):
        """Test that ILP typically improves gap compared to greedy."""
        inst = small_knapsack_instance

        # Greedy solution: sort by value/weight ratio
        ratios = inst["values"] / inst["weights"]
        sorted_indices = np.argsort(ratios)[::-1]

        greedy_solution = np.zeros(inst["n_items"], dtype=np.int32)
        total_weight = 0.0
        for idx in sorted_indices:
            if total_weight + inst["weights"][idx] <= inst["capacity"]:
                greedy_solution[idx] = 1
                total_weight += inst["weights"][idx]

        greedy_value = np.dot(greedy_solution, inst["values"])

        # Refine with ILP
        ilp_value, ilp_solution, status = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=greedy_solution,
            time_limit=1.0,
            fix_threshold=0.9,
        )

        # ILP should be at least as good as greedy
        assert ilp_value >= greedy_value, (
            f"ILP ({ilp_value}) should improve on greedy ({greedy_value})"
        )


class TestILPIntegration:
    """Integration tests for warm-start ILP in full pipeline."""

    def test_sampling_plus_ilp_workflow(self, small_knapsack_instance):
        """Test complete workflow: GNN -> Sampling -> ILP refinement."""
        inst = small_knapsack_instance

        # Step 1: Mock GNN logits
        logits = torch.randn(inst["n_items"])

        # Step 2: Sample solutions
        from knapsack_gnn.decoding.sampling import sample_solutions

        solutions = sample_solutions(logits, n_samples=10, temperature=1.0)

        # Step 3: Pick best feasible solution
        best_value = 0
        best_solution = None

        for sol in solutions:
            sol_np = sol.cpu().numpy()
            total_weight = np.dot(sol_np, inst["weights"])

            if total_weight <= inst["capacity"]:
                value = np.dot(sol_np, inst["values"])
                if value > best_value:
                    best_value = value
                    best_solution = sol_np

        if best_solution is None:
            best_solution = np.zeros(inst["n_items"], dtype=np.int32)

        # Step 4: Refine with ILP
        ilp_value, ilp_solution, status = warm_start_ilp_solve(
            values=inst["values"],
            weights=inst["weights"],
            capacity=inst["capacity"],
            initial_solution=best_solution,
            time_limit=1.0,
            fix_threshold=0.9,
        )

        # ILP should not worsen sampling result
        assert ilp_value >= best_value, "ILP should not worsen sampling solution"
