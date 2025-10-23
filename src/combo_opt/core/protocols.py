"""
Protocols for structural typing.

Protocols define interfaces using duck typing, allowing flexible
composition without requiring inheritance.
"""

from typing import Any, Protocol

import torch
from torch_geometric.data import Data


class Trainable(Protocol):
    """Protocol for trainable objects."""

    def train_step(self, batch: Data, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch of training data
            optimizer: PyTorch optimizer

        Returns:
            Dictionary with metrics (loss, accuracy, etc.)
        """
        ...

    def val_step(self, batch: Data) -> dict[str, float]:
        """
        Perform one validation step.

        Args:
            batch: Batch of validation data

        Returns:
            Dictionary with validation metrics
        """
        ...


class Evaluable(Protocol):
    """Protocol for evaluable objects."""

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """
        Evaluate on a dataset.

        Args:
            dataset: Dataset to evaluate on

        Returns:
            Dictionary with evaluation metrics
        """
        ...


class GraphConvertible(Protocol):
    """Protocol for objects convertible to graphs."""

    def to_graph(self) -> Data:
        """
        Convert to PyTorch Geometric Data.

        Returns:
            torch_geometric.data.Data object
        """
        ...


class Solvable(Protocol):
    """Protocol for solvable optimization problems."""

    def solve(self) -> tuple[Any, float]:
        """
        Solve the problem (exact or heuristic).

        Returns:
            Tuple of (solution, objective_value)
        """
        ...


class Repairable(Protocol):
    """Protocol for solution repair strategies."""

    def repair(self, solution: Any, problem_instance: Any) -> Any:
        """
        Repair infeasible solution.

        Args:
            solution: Infeasible solution
            problem_instance: Problem instance

        Returns:
            Feasible solution
        """
        ...
