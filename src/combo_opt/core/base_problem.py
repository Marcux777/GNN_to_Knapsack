"""
Abstract base class for optimization problems.

Defines the interface for representing and solving combinatorial optimization problems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from torch_geometric.data import Data

# Type variables for problem instance and solution
T_Instance = TypeVar("T_Instance")
T_Solution = TypeVar("T_Solution")


@dataclass
class ProblemInstance:
    """
    Base class for problem instances.

    Subclasses should define specific fields for their problem.

    Example:
        >>> @dataclass
        >>> class KnapsackInstance(ProblemInstance):
        ...     weights: np.ndarray
        ...     values: np.ndarray
        ...     capacity: float
        ...     n_items: int
    """

    pass


@dataclass
class Solution:
    """
    Base class for solutions.

    Attributes:
        variables: Solution variables (binary vector, permutation, etc.)
        objective_value: Objective function value
        is_feasible: Whether solution satisfies constraints
    """

    variables: Any
    objective_value: float
    is_feasible: bool


class OptimizationProblem(ABC, Generic[T_Instance, T_Solution]):
    """
    Abstract base class for combinatorial optimization problems.

    Defines the interface that problem implementations must follow.

    Example:
        >>> class KnapsackProblem(OptimizationProblem):
        ...     def to_graph(self, instance):
        ...         # Convert to bipartite graph
        ...         return Data(x=..., edge_index=...)
        ...
        ...     def compute_objective(self, solution, instance):
        ...         return np.dot(solution, instance.values)
        ...
        ...     def is_feasible(self, solution, instance):
        ...         return np.dot(solution, instance.weights) <= instance.capacity
    """

    @abstractmethod
    def to_graph(self, instance: T_Instance) -> Data:
        """
        Convert problem instance to graph representation.

        Args:
            instance: Problem instance

        Returns:
            torch_geometric.data.Data object with:
                - x: Node features
                - edge_index: Graph connectivity
                - (optional) edge_attr, y, etc.

        Example:
            >>> problem = KnapsackProblem()
            >>> instance = KnapsackInstance(weights=[...], values=[...], capacity=...)
            >>> graph = problem.to_graph(instance)
        """
        pass

    @abstractmethod
    def compute_objective(self, solution: T_Solution, instance: T_Instance) -> float:
        """
        Compute objective function value.

        Args:
            solution: Solution to evaluate
            instance: Problem instance

        Returns:
            Objective value (higher is better for maximization)

        Example:
            >>> value = problem.compute_objective(solution, instance)
        """
        pass

    @abstractmethod
    def is_feasible(self, solution: T_Solution, instance: T_Instance) -> bool:
        """
        Check if solution satisfies all constraints.

        Args:
            solution: Solution to check
            instance: Problem instance

        Returns:
            True if solution is feasible

        Example:
            >>> if problem.is_feasible(solution, instance):
            ...     print("Solution is valid!")
        """
        pass

    def evaluate_solution(self, solution: T_Solution, instance: T_Instance) -> Solution:
        """
        Full evaluation: objective + feasibility.

        Args:
            solution: Solution to evaluate
            instance: Problem instance

        Returns:
            Solution object with all information

        Example:
            >>> result = problem.evaluate_solution(solution, instance)
            >>> print(f"Value: {result.objective_value}, Feasible: {result.is_feasible}")
        """
        return Solution(
            variables=solution,
            objective_value=self.compute_objective(solution, instance),
            is_feasible=self.is_feasible(solution, instance),
        )
