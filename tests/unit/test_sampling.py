"""
Tests for the KnapsackSampler decoding strategies.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from knapsack_gnn.decoding.sampling import KnapsackSampler


class _StubModel(torch.nn.Module):
    """Simple deterministic model that returns preset probabilities."""

    def __init__(self, probs: list[float]) -> None:
        super().__init__()
        self.register_buffer("probs", torch.tensor(probs, dtype=torch.float32))

    def forward(self, data: Data) -> torch.Tensor:  # type: ignore[override]
        n_items = int(data.item_values.shape[0])
        return self.probs[:n_items]


def _make_data(values: np.ndarray, weights: np.ndarray, capacity: float) -> Data:
    """Create a minimal PyG Data object for decoding tests."""
    n_items = len(values)
    x = torch.zeros((n_items + 1, 8), dtype=torch.float32)

    edge_pairs = []
    constraint_idx = n_items
    for idx in range(n_items):
        edge_pairs.append([idx, constraint_idx])
        edge_pairs.append([constraint_idx, idx])
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    y = torch.zeros(n_items, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.item_weights = torch.tensor(weights, dtype=torch.float32)
    data.item_values = torch.tensor(values, dtype=torch.float32)
    data.capacity = float(capacity)
    data.node_types = torch.zeros(n_items + 1, dtype=torch.long)
    data.node_types[-1] = 1
    data.batch = torch.zeros(n_items + 1, dtype=torch.long)
    data.n_items = n_items
    data.optimal_value = float(values.max())
    return data


class TestKnapsackSamplerStrategies:
    """Validate threshold, sampling, and lagrangian decoding paths."""

    def setup_method(self) -> None:
        self.values = np.array([12.0, 7.0, 9.0, 6.0], dtype=np.float32)
        self.weights = np.array([4.0, 3.0, 5.0, 2.0], dtype=np.float32)
        self.capacity = 10.0
        self.data = _make_data(self.values, self.weights, self.capacity)

        probs = [0.8, 0.2, 0.65, 0.55]
        self.sampler = KnapsackSampler(model=_StubModel(probs), device="cpu")

    def test_threshold_strategy_produces_feasible_solution(self):
        """Threshold decoding should always return a feasible solution."""
        result = self.sampler.solve(self.data, strategy="threshold", threshold=0.5)
        solution = result["solution"]

        total_weight = np.dot(solution, self.weights)
        assert total_weight <= self.capacity + 1e-6
        assert result["is_feasible"] is True

    def test_sampling_strategy_respects_max_samples(self):
        """Sampling should obey the max_samples budget."""
        max_samples = 5
        result = self.sampler.solve(
            self.data,
            strategy="sampling",
            n_samples=32,
            max_samples=max_samples,
            schedule=(8, 16),
        )

        assert result["samples_used"] <= max_samples
        solution = result["solution"]
        assert np.dot(solution, self.weights) <= self.capacity + 1e-6

    def test_lagrangian_strategy_returns_feasible_solution(self):
        """Lagrangian decoding must respect capacity constraints."""
        result = self.sampler.solve(
            self.data,
            strategy="lagrangian",
            lagrangian_iters=20,
            lagrangian_tol=1e-3,
        )

        solution = result["solution"]
        assert result["is_feasible"] is True
        assert np.dot(solution, self.weights) <= self.capacity + 1e-6
