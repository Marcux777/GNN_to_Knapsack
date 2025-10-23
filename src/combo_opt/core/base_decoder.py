"""
Abstract base class for solution decoders.

Defines the interface for decoding GNN outputs into problem solutions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class DecodingResult:
    """
    Structured result from decoder.

    Attributes:
        solution: Binary/discrete solution (numpy array or list)
        objective_value: Objective function value
        is_feasible: Whether solution satisfies all constraints
        metadata: Optional metadata (time, num_samples, etc.)
    """

    solution: np.ndarray
    objective_value: float
    is_feasible: bool
    metadata: dict[str, Any] | None = None


class AbstractDecoder(ABC):
    """
    Abstract base class for solution decoders.

    Decoders convert GNN model outputs (probabilities/logits) into
    discrete solutions for combinatorial optimization problems.

    Example:
        >>> class GreedyDecoder(AbstractDecoder):
        ...     def decode(self, model_output, problem_data):
        ...         # Select items by probability
        ...         sorted_indices = torch.argsort(model_output, descending=True)
        ...         solution = self._greedy_select(sorted_indices, problem_data)
        ...         return DecodingResult(
        ...             solution=solution,
        ...             objective_value=np.dot(solution, problem_data.values),
        ...             is_feasible=self.validate_solution(solution, problem_data)
        ...         )
    """

    @abstractmethod
    def decode(self, model_output: torch.Tensor, problem_data: Any) -> DecodingResult:
        """
        Decode model output into solution.

        Args:
            model_output: Model predictions (probabilities/logits)
                Shape: [num_decision_variables] or [num_decision_variables, classes]
            problem_data: Problem instance data (graph, constraints, etc.)

        Returns:
            DecodingResult containing solution and metadata

        Example:
            >>> decoder = GreedyDecoder()
            >>> result = decoder.decode(probs, problem_instance)
            >>> print(f"Objective: {result.objective_value}")
            >>> print(f"Feasible: {result.is_feasible}")
        """
        pass

    @abstractmethod
    def validate_solution(self, solution: np.ndarray, problem_data: Any) -> bool:
        """
        Check if solution is feasible.

        Args:
            solution: Candidate solution
            problem_data: Problem instance

        Returns:
            True if solution satisfies all constraints

        Example:
            >>> is_valid = decoder.validate_solution(solution, instance)
        """
        pass

    def get_statistics(self) -> dict[str, Any]:
        """
        Get decoder statistics (optional).

        Returns:
            Dictionary with stats (e.g., avg decode time, success rate)

        Example:
            >>> stats = decoder.get_statistics()
            >>> print(f"Avg time: {stats['avg_time_ms']:.2f}ms")
        """
        return {}
