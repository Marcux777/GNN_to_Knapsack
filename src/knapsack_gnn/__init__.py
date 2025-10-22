"""
Knapsack GNN - Learning to Optimize
====================================

A Graph Neural Network library for solving the 0-1 Knapsack Problem
using the Learning to Optimize (L2O) approach.

Main modules:
- data: Problem generation and graph construction
- models: GNN architectures (PNA, GCN, GAT)
- training: Training loops and utilities
- decoding: Solution decoding strategies
- solvers: Integration with exact solvers
- baselines: Classical heuristics
"""

__version__ = "1.0.0"

# Public API exports
from knapsack_gnn import data, models, training, decoding, solvers, baselines
from knapsack_gnn.types import (
    FloatArray,
    IntArray,
    KnapsackInstance,
    Solution,
    ConfigDict,
    MetricsDict,
)

__all__ = [
    "data",
    "models",
    "training",
    "decoding",
    "solvers",
    "baselines",
    "__version__",
    # Types
    "FloatArray",
    "IntArray",
    "KnapsackInstance",
    "Solution",
    "ConfigDict",
    "MetricsDict",
]
