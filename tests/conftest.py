"""
Pytest configuration and shared fixtures for testing.
"""

import os
import warnings

import numpy as np
import pytest
import torch

from knapsack_gnn.training.utils import set_seed

os.environ.setdefault("APP_DISABLE_WATCHER", "1")
warnings.filterwarnings(
    "ignore",
    message="`torch_geometric.distributed` has been deprecated",
    category=DeprecationWarning,
    module="torch_geometric.llm.utils.backend_utils",
)


@pytest.fixture(scope="session", autouse=True)
def set_test_seed():
    """Set a fixed seed for all tests to ensure reproducibility."""
    set_seed(42, deterministic=True)


@pytest.fixture
def small_knapsack_instance():
    """
    Create a small knapsack instance for fast testing.

    Returns:
        dict with keys: values, weights, capacity, optimal_value, optimal_solution
    """
    # Small instance with 5 items
    values = np.array([10, 20, 15, 25, 18], dtype=np.float32)
    weights = np.array([2, 5, 3, 7, 4], dtype=np.float32)
    capacity = 10.0

    # Known optimal solution: items [0, 2, 4] with total value 43
    optimal_solution = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    optimal_value = 43.0

    return {
        "values": values,
        "weights": weights,
        "capacity": capacity,
        "optimal_value": optimal_value,
        "optimal_solution": optimal_solution,
        "n_items": 5,
    }


@pytest.fixture
def tiny_knapsack_batch():
    """
    Create a batch of tiny knapsack instances for testing.

    Returns:
        list of knapsack instance dicts
    """
    instances = []

    # Instance 1: 3 items
    instances.append(
        {
            "values": np.array([5, 10, 8], dtype=np.float32),
            "weights": np.array([1, 3, 2], dtype=np.float32),
            "capacity": 4.0,
            "optimal_value": 18.0,  # items [1, 2]
            "n_items": 3,
        }
    )

    # Instance 2: 4 items
    instances.append(
        {
            "values": np.array([12, 8, 15, 6], dtype=np.float32),
            "weights": np.array([3, 2, 4, 1], dtype=np.float32),
            "capacity": 6.0,
            "optimal_value": 29.0,  # items [0, 1, 3]
            "n_items": 4,
        }
    )

    return instances


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """
    Create a temporary checkpoint directory for testing.

    Returns:
        Path to temporary checkpoint directory
    """
    checkpoint_dir = tmp_path / "checkpoints" / "test_run"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def mock_model_weights():
    """
    Generate mock model weights for testing without full training.

    Returns:
        dict of mock weights compatible with PNA model
    """
    return {
        "node_encoder.weight": torch.randn(64, 2),
        "node_encoder.bias": torch.randn(64),
        # Add minimal weights to make model loadable
    }
