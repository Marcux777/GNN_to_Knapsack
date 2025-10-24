"""
Tests for seed management and determinism.
"""

import random

import numpy as np
import pytest
import torch

from knapsack_gnn.training.utils import get_checkpoint_name, set_seed, validate_seed


class TestSeedManager:
    """Test suite for seed management utilities."""

    def test_set_seed_deterministic(self):
        """Test that set_seed produces deterministic results."""
        # Set seed twice and verify same random numbers
        set_seed(42, deterministic=True)
        random_val_1 = random.random()
        numpy_val_1 = np.random.rand()
        torch_val_1 = torch.rand(1).item()

        set_seed(42, deterministic=True)
        random_val_2 = random.random()
        numpy_val_2 = np.random.rand()
        torch_val_2 = torch.rand(1).item()

        assert random_val_1 == random_val_2, "Python random not deterministic"
        assert numpy_val_1 == numpy_val_2, "NumPy random not deterministic"
        assert torch_val_1 == torch_val_2, "PyTorch random not deterministic"

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42, deterministic=True)
        val_42 = torch.rand(10)

        set_seed(123, deterministic=True)
        val_123 = torch.rand(10)

        assert not torch.allclose(val_42, val_123), (
            "Different seeds should produce different values"
        )

    def test_get_checkpoint_name_with_seed(self):
        """Test checkpoint name generation with seed."""
        name = get_checkpoint_name("run_20251020_104533", seed=42)
        assert name == "run_20251020_104533_seed42"

    def test_get_checkpoint_name_without_seed(self):
        """Test checkpoint name generation without seed."""
        name = get_checkpoint_name("run_20251020_104533", seed=None)
        assert name == "run_20251020_104533"

    def test_validate_seed_valid(self):
        """Test validation of valid seeds."""
        validate_seed(0)
        validate_seed(42)
        validate_seed(2**32 - 1)
        # Should not raise

    def test_validate_seed_invalid(self):
        """Test validation of invalid seeds."""
        with pytest.raises(ValueError):
            validate_seed(-1)

        with pytest.raises(ValueError):
            validate_seed(2**32)

        with pytest.raises(ValueError):
            validate_seed(2**33)


class TestCUDADeterminism:
    """Test CUDA-specific determinism if GPU is available."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_deterministic(self):
        """Test that CUDA operations are deterministic with set_seed."""
        try:
            set_seed(42, deterministic=True)
            cuda_val_1 = torch.rand(10, device="cuda")

            set_seed(42, deterministic=True)
            cuda_val_2 = torch.rand(10, device="cuda")

            assert torch.allclose(cuda_val_1, cuda_val_2), "CUDA random not deterministic"
        except RuntimeError as e:
            if "no kernel image" in str(e) or "not compatible" in str(e):
                pytest.skip(f"CUDA GPU not compatible with PyTorch: {e}")
            raise
