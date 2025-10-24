"""
Common type definitions for Knapsack GNN.

Provides type aliases and protocols for type checking.
"""

from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray

# Type aliases
FloatArray: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int32]
Tensor: TypeAlias = torch.Tensor

# Common return types
KnapsackInstance: TypeAlias = tuple[FloatArray, FloatArray, float]  # (values, weights, capacity)
Solution: TypeAlias = tuple[float, IntArray]  # (objective_value, binary_solution)
GraphData: TypeAlias = object  # torch_geometric.data.Data (avoid circular import)

# Config types
ConfigDict: TypeAlias = dict[str, int | float | str | bool | list | dict]
MetricsDict: TypeAlias = dict[str, float]

# Path types
PathLike: TypeAlias = str | Path
