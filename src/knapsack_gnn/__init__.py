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
from knapsack_gnn import baselines, data, decoding, models, solvers, training
from knapsack_gnn.types import (
    ConfigDict,
    FloatArray,
    IntArray,
    KnapsackInstance,
    MetricsDict,
    Solution,
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
