"""
Common type definitions for Knapsack GNN.

Provides type aliases and protocols for type checking.
"""

from typing import Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

# Type aliases
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int32]
Tensor = torch.Tensor

# Common return types
KnapsackInstance = Tuple[FloatArray, FloatArray, float]  # (values, weights, capacity)
Solution = Tuple[float, IntArray]  # (objective_value, binary_solution)
GraphData = object  # torch_geometric.data.Data (avoid circular import)

# Config types
ConfigDict = Dict[str, Union[int, float, str, bool, List, Dict]]
MetricsDict = Dict[str, float]

# Path types
PathLike = Union[str, Path]
