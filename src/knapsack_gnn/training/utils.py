"""
Centralized seed management for reproducible experiments.

This module provides utilities to set random seeds across all common libraries
used in the project (Python, NumPy, PyTorch, CUDA) to ensure deterministic
and reproducible results.
"""

import os
import random
from typing import Optional

import numpy as np
import torch

def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value (typically in range 0-9999)
        deterministic: If True, enables deterministic algorithms in PyTorch.
                      This may reduce performance but ensures exact reproducibility.

    Example:
        >>> from utils.seed_manager import set_seed
        >>> set_seed(42, deterministic=True)
        >>> # All random operations are now reproducible

    Notes:
        - Sets PYTHONHASHSEED environment variable for Python's hash randomization
        - Sets seeds for: random, numpy, torch, torch.cuda
        - When deterministic=True, sets torch.use_deterministic_algorithms(True)
        - For complete reproducibility, also set CUBLAS_WORKSPACE_CONFIG=:4096:8
          in your environment before running the script
    """
    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Python hash seed (for dictionary ordering, etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic algorithms (may impact performance)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        # CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For CUDA >= 10.2, set workspace config for deterministic operations
        # User should set this in their environment: export CUBLAS_WORKSPACE_CONFIG=:4096:8
        if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_checkpoint_name(base_name: str, seed: Optional[int] = None) -> str:
    """
    Generate checkpoint directory name with optional seed suffix.

    Args:
        base_name: Base name for checkpoint (e.g., "run_20251020_104533")
        seed: Random seed used for the run (if None, no seed suffix added)

    Returns:
        Checkpoint name with seed suffix if provided

    Example:
        >>> get_checkpoint_name("run_20251020_104533", seed=42)
        'run_20251020_104533_seed42'
        >>> get_checkpoint_name("run_20251020_104533")
        'run_20251020_104533'
    """
    if seed is not None:
        return f"{base_name}_seed{seed}"
    return base_name

def validate_seed(seed: int) -> None:
    """
    Validate that seed is in acceptable range.

    Args:
        seed: Random seed value

    Raises:
        ValueError: If seed is not in range [0, 2^32 - 1]

    Example:
        >>> validate_seed(42)  # OK
        >>> validate_seed(-1)  # Raises ValueError
    """
    if not (0 <= seed < 2**32):
        raise ValueError(f"Seed must be in range [0, {2**32 - 1}], got {seed}")
