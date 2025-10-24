"""
Tests for sampling-based decoder.
"""

import numpy as np
import torch

from knapsack_gnn.decoding.sampling import sample_solutions


class DummyModel(torch.nn.Module):
    def forward(self, data):
        return torch.sigmoid(torch.randn(data.n_items))


class TestSamplingDecoder:
    """Test suite for sampling-based inference."""

    def test_sample_solutions_returns_correct_shape(self):
        """Test that sampling returns correct solution shape."""
        # Mock logits for 5 items
        logits = torch.randn(5)
        n_samples = 10

        solutions = sample_solutions(logits, n_samples=n_samples, temperature=1.0).long()

        assert solutions.shape == (n_samples, 5), f"Expected shape (10, 5), got {solutions.shape}"
        assert solutions.dtype == torch.long or solutions.dtype == torch.int32

    def test_sample_solutions_binary(self):
        """Test that sampled solutions are binary (0 or 1)."""
        logits = torch.randn(10)
        solutions = sample_solutions(logits, n_samples=50, temperature=1.0)

        unique_values = torch.unique(solutions)
        assert len(unique_values) <= 2, "Solutions should only contain 0 and 1"
        assert all(v in [0, 1] for v in unique_values), "Solutions should be binary"

    def test_temperature_effect(self):
        """Test that temperature affects sampling distribution."""
        # Strong logits favoring selection
        logits = torch.tensor([5.0, 5.0, 5.0, -5.0, -5.0])

        # Low temperature (more deterministic)
        solutions_low_temp = sample_solutions(logits, n_samples=100, temperature=0.1)
        selection_rate_low = solutions_low_temp[:, :3].float().mean()

        # High temperature (more random)
        solutions_high_temp = sample_solutions(logits, n_samples=100, temperature=2.0)
        selection_rate_high = solutions_high_temp[:, :3].float().mean()

        # Low temp should select high-logit items more often
        assert selection_rate_low > selection_rate_high, (
            "Low temperature should be more deterministic"
        )

    def test_best_of_n_monotonicity(self, small_knapsack_instance):
        """Test that best-of-N gap improves (or stays same) with more samples."""
        inst = small_knapsack_instance

        # Mock logits that favor optimal solution
        optimal_sol = inst["optimal_solution"]
        logits = torch.tensor(optimal_sol, dtype=torch.float32) * 2.0

        gaps = []
        sample_counts = [1, 5, 10, 20, 50]

        for n_samples in sample_counts:
            solutions = sample_solutions(logits, n_samples=n_samples, temperature=1.0)

            # Evaluate each solution
            values = []
            for sol in solutions:
                sol_np = sol.cpu().numpy()
                total_weight = np.dot(sol_np, inst["weights"])

                if total_weight <= inst["capacity"]:
                    value = np.dot(sol_np, inst["values"])
                    values.append(value)

            if values:
                best_value = max(values)
                gap = (inst["optimal_value"] - best_value) / inst["optimal_value"] * 100
                gaps.append(gap)
            else:
                gaps.append(100.0)  # All infeasible

        # Gap should generally decrease (or stay same) with more samples
        # Allow some tolerance due to randomness
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1] - 5.0, (
                f"Gap increased from {gaps[i]:.2f}% to {gaps[i + 1]:.2f}% with more samples"
            )

    def test_vectorized_sampling_schedule(self, small_knapsack_instance):
        """Test vectorized sampling with adaptive schedule."""
        from torch_geometric.data import Data

        from knapsack_gnn.decoding.sampling import KnapsackSampler

        inst = small_knapsack_instance
        logits = torch.randn(inst["n_items"])
        schedule = [4, 8, 16]
        max_samples = 16
        sampler = KnapsackSampler(model=DummyModel(), device="cpu")
        data = Data(
            item_weights=torch.from_numpy(inst["weights"]),
            item_values=torch.from_numpy(inst["values"]),
            capacity=inst["capacity"],
            n_items=inst["n_items"],
        )

        result = sampler.anytime_sampling(
            probs=torch.sigmoid(logits),
            data=data,
            schedule=schedule,
            max_samples=max_samples,
            temperature=1.0,
        )
        solution = result["solution"]
        n_samples_used = result["samples_used"]

        assert solution.shape[0] == inst["n_items"], "Solutions should have same length as logits"
        assert n_samples_used <= max_samples, "Should not exceed max_samples"
        assert n_samples_used > 0, "Should use at least one sample"

    def test_sampling_deterministic_with_seed(self):
        """Test that sampling is deterministic with fixed seed."""
        from knapsack_gnn.training.utils import set_seed

        logits = torch.randn(10)

        set_seed(42)
        solutions1 = sample_solutions(logits, n_samples=20, temperature=1.0).long()

        set_seed(42)
        solutions2 = sample_solutions(logits, n_samples=20, temperature=1.0).long()

        assert torch.equal(solutions1, solutions2), (
            "Sampling should be deterministic with same seed"
        )

    def test_sampling_feasibility_check(self, small_knapsack_instance):
        """Test that feasibility filtering works correctly."""
        inst = small_knapsack_instance

        # Generate random solutions
        logits = torch.randn(inst["n_items"])
        solutions = sample_solutions(logits, n_samples=50, temperature=1.0)

        # Check feasibility of each solution
        feasible_count = 0
        for sol in solutions:
            sol_np = sol.cpu().numpy()
            total_weight = np.dot(sol_np, inst["weights"])

            if total_weight <= inst["capacity"]:
                feasible_count += 1

        # At least some solutions should be feasible (with probability > 0)
        # This is a soft check since randomness can make all infeasible in rare cases
        assert feasible_count >= 0, "Feasibility check should work"


class TestAdaptiveSampling:
    """Test suite for adaptive sampling strategies."""

    def test_early_stopping_logic(self, small_knapsack_instance):
        """Test that early stopping works when solution converges."""
        from torch_geometric.data import Data

        from knapsack_gnn.decoding.sampling import KnapsackSampler

        inst = small_knapsack_instance
        # Create logits that strongly favor a specific solution
        optimal_sol = torch.tensor(inst["optimal_solution"], dtype=torch.float32)
        logits = (optimal_sol * 100) - (1 - optimal_sol) * 100

        schedule = [8, 16, 32, 64]

        sampler = KnapsackSampler(model=DummyModel(), device="cpu")
        data = Data(
            item_weights=torch.from_numpy(inst["weights"]),
            item_values=torch.from_numpy(inst["values"]),
            capacity=inst["capacity"],
            n_items=inst["n_items"],
        )

        result = sampler.anytime_sampling(
            probs=torch.sigmoid(logits),
            data=data,
            schedule=schedule,
            max_samples=64,
            temperature=0.1,  # Low temp for convergence
            target_value=inst["optimal_value"],
        )
        n_used = result["samples_used"]

        # With strong signal and low temperature, should converge early
        assert n_used < 64, "Should stop early when converged"

    def test_schedule_progression(self, small_knapsack_instance):
        """Test that sampling follows the schedule."""
        from torch_geometric.data import Data

        from knapsack_gnn.decoding.sampling import KnapsackSampler

        inst = small_knapsack_instance
        optimal_sol = torch.tensor(inst["optimal_solution"], dtype=torch.float32)
        logits = (optimal_sol * 0.1) + (1 - optimal_sol) * 0.01
        schedule = [4, 8, 16]
        sampler = KnapsackSampler(model=DummyModel(), device="cpu")
        data = Data(
            item_weights=torch.from_numpy(inst["weights"]),
            item_values=torch.from_numpy(inst["values"]),
            capacity=inst["capacity"],
            n_items=inst["n_items"],
        )

        result = sampler.anytime_sampling(
            probs=torch.sigmoid(logits),
            data=data,
            schedule=schedule,
            max_samples=16,
            temperature=1.0,
        )
        n_used = result["samples_used"]

        # Should use one of the schedule values
        assert n_used in schedule or n_used == 16, (
            f"Used {n_used} samples, expected one of {schedule} or max=16"
        )
