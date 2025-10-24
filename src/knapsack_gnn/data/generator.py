"""
Knapsack Problem Instance Generator and Exact Solver
Generates random Knapsack instances and solves them using OR-Tools for generating training labels
"""

import os
import pickle
import time
from typing import Any

import numpy as np

try:
    # OR-Tools <= 9.13
    from ortools.algorithms import pywrapknapsack_solver as knapsack_solver
except ImportError:
    # OR-Tools >= 9.14 reorganized algorithms into python submodule
    from ortools.algorithms.python import knapsack_solver


class KnapsackInstance:
    """Represents a single Knapsack problem instance"""

    def __init__(self, weights: np.ndarray, values: np.ndarray, capacity: int):
        self.weights: np.ndarray = weights
        self.values: np.ndarray = values
        self.capacity: int = capacity
        self.n_items: int = len(weights)
        self.solution: np.ndarray | None = None
        self.optimal_value: float | None = None
        self.solve_time: float | None = None

    def __repr__(self) -> str:
        return f"KnapsackInstance(n_items={self.n_items}, capacity={self.capacity})"


class KnapsackGenerator:
    """Generates random Knapsack problem instances"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_instance(
        self,
        n_items: int,
        weight_range: tuple[int, int] = (1, 100),
        value_range: tuple[int, int] = (1, 100),
        capacity_ratio: float = 0.5,
    ) -> KnapsackInstance:
        """
        Generate a random Knapsack instance

        Args:
            n_items: Number of items
            weight_range: (min_weight, max_weight) for items
            value_range: (min_value, max_value) for items
            capacity_ratio: Capacity as a fraction of total weight (default: 0.5)

        Returns:
            KnapsackInstance object
        """
        weights = self.rng.randint(weight_range[0], weight_range[1] + 1, size=n_items)
        values = self.rng.randint(value_range[0], value_range[1] + 1, size=n_items)

        # Set capacity as a fraction of total weight
        total_weight = np.sum(weights)
        capacity = int(total_weight * capacity_ratio)

        return KnapsackInstance(weights, values, capacity)

    def generate_dataset(
        self, n_instances: int, n_items_range: tuple[int, int], **kwargs: Any
    ) -> list[KnapsackInstance]:
        """
        Generate multiple instances with varying sizes

        Args:
            n_instances: Number of instances to generate
            n_items_range: (min_items, max_items) range
            **kwargs: Additional arguments passed to generate_instance

        Returns:
            List of KnapsackInstance objects
        """
        instances = []
        for _ in range(n_instances):
            n_items = self.rng.randint(n_items_range[0], n_items_range[1] + 1)
            instance = self.generate_instance(n_items, **kwargs)
            instances.append(instance)
        return instances


class KnapsackSolver:
    """Exact solver for Knapsack problem using OR-Tools"""

    @staticmethod
    def solve(instance: KnapsackInstance, time_limit: float = 60.0) -> KnapsackInstance:
        """
        Solve Knapsack instance using exact algorithm

        Args:
            instance: KnapsackInstance to solve
            time_limit: Time limit in seconds (default: 60.0)

        Returns:
            Same instance with solution and optimal_value filled
        """
        # Create the solver
        solver_type = getattr(
            knapsack_solver,
            "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER",
            getattr(knapsack_solver.KnapsackSolver, "KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER", None),
        )
        if solver_type is None:
            raise ImportError("Could not locate OR-Tools knapsack solver type constant.")

        solver = knapsack_solver.KnapsackSolver(solver_type, "KnapsackSolver")

        # Convert to lists (OR-Tools requirement)
        values = [int(round(v)) for v in instance.values.tolist()]
        weights = [[int(round(w)) for w in instance.weights.tolist()]]
        capacities = [int(round(instance.capacity))]

        # Initialize and solve
        init_fn = getattr(solver, "Init", solver.init)
        init_fn(values, weights, capacities)

        set_time_limit_fn = getattr(solver, "SetTimeLimit", getattr(solver, "set_time_limit", None))
        if set_time_limit_fn:
            set_time_limit_fn(time_limit)

        solve_fn = getattr(solver, "Solve", solver.solve)
        start_time = time.perf_counter()
        optimal_value = solve_fn()
        instance.solve_time = time.perf_counter() - start_time

        # Extract solution (binary vector)
        solution = np.zeros(instance.n_items, dtype=np.int32)
        best_contains_fn = getattr(solver, "BestSolutionContains", solver.best_solution_contains)
        for i in range(instance.n_items):
            if best_contains_fn(i):
                solution[i] = 1

        instance.solution = solution
        instance.optimal_value = optimal_value

        return instance

    @staticmethod
    def solve_batch(
        instances: list[KnapsackInstance], time_limit: float = 60.0, verbose: bool = True
    ) -> list[KnapsackInstance]:
        """
        Solve multiple instances

        Args:
            instances: List of KnapsackInstance objects
            time_limit: Time limit per instance
            verbose: Print progress

        Returns:
            List of solved instances
        """
        solved = []
        for i, instance in enumerate(instances):
            if verbose and (i + 1) % 10 == 0:
                print(f"Solved {i + 1}/{len(instances)} instances")
            solved.append(KnapsackSolver.solve(instance, time_limit))
        return solved


class KnapsackDataset:
    """Dataset manager for Knapsack instances"""

    def __init__(self, instances: list[KnapsackInstance]):
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> KnapsackInstance:
        return self.instances[idx]

    def save(self, filepath: str) -> None:
        """Save dataset to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.instances, f)
        print(f"Dataset saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> "KnapsackDataset":
        """Load dataset from file"""
        with open(filepath, "rb") as f:
            instances = pickle.load(f)
        for inst in instances:
            if not hasattr(inst, "solve_time"):
                inst.solve_time = None
        print(f"Dataset loaded from {filepath} ({len(instances)} instances)")
        return KnapsackDataset(instances)

    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        n_items = [inst.n_items for inst in self.instances]
        capacities = [inst.capacity for inst in self.instances]

        stats = {
            "n_instances": len(self.instances),
            "n_items_mean": np.mean(n_items),
            "n_items_std": np.std(n_items),
            "n_items_min": np.min(n_items),
            "n_items_max": np.max(n_items),
            "capacity_mean": np.mean(capacities),
            "capacity_std": np.std(capacities),
        }

        # Check if solved
        if self.instances[0].solution is not None:
            optimal_values = [inst.optimal_value for inst in self.instances]
            stats["optimal_value_mean"] = np.mean(optimal_values)
            stats["optimal_value_std"] = np.std(optimal_values)

        return stats


def create_datasets(
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 200,
    n_items_range: tuple[int, int] = (10, 50),
    seed: int = 42,
    output_dir: str = "data/datasets",
) -> tuple[KnapsackDataset, KnapsackDataset, KnapsackDataset]:
    """
    Create train, validation, and test datasets

    Args:
        train_size: Number of training instances
        val_size: Number of validation instances
        test_size: Number of test instances
        n_items_range: Range of items per instance
        seed: Random seed
        output_dir: Directory to save datasets

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("Generating datasets...")

    # Generate instances
    generator = KnapsackGenerator(seed=seed)

    print(f"Generating {train_size} training instances...")
    train_instances = generator.generate_dataset(train_size, n_items_range)

    print(f"Generating {val_size} validation instances...")
    val_instances = generator.generate_dataset(val_size, n_items_range)

    print(f"Generating {test_size} test instances...")
    test_instances = generator.generate_dataset(test_size, n_items_range)

    # Solve all instances
    print("\nSolving training instances...")
    train_instances = KnapsackSolver.solve_batch(train_instances)

    print("Solving validation instances...")
    val_instances = KnapsackSolver.solve_batch(val_instances)

    print("Solving test instances...")
    test_instances = KnapsackSolver.solve_batch(test_instances)

    # Create datasets
    train_dataset = KnapsackDataset(train_instances)
    val_dataset = KnapsackDataset(val_instances)
    test_dataset = KnapsackDataset(test_instances)

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.save(f"{output_dir}/train.pkl")
    val_dataset.save(f"{output_dir}/val.pkl")
    test_dataset.save(f"{output_dir}/test.pkl")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print("Train:", train_dataset.get_statistics())
    print("Val:", val_dataset.get_statistics())
    print("Test:", test_dataset.get_statistics())

    return train_dataset, val_dataset, test_dataset


# Convenience wrapper functions for backward compatibility
def generate_knapsack_instance(
    n_items: int,
    weight_range: tuple[int, int] = (1, 100),
    value_range: tuple[int, int] = (1, 100),
    capacity_ratio: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Generate a single random knapsack instance.

    Args:
        n_items: Number of items
        weight_range: (min_weight, max_weight) for items
        value_range: (min_value, max_value) for items
        capacity_ratio: Capacity as a fraction of total weight
        seed: Random seed

    Returns:
        Dictionary with keys: weights, values, capacity, n_items
    """
    generator = KnapsackGenerator(seed=seed)
    instance = generator.generate_instance(
        n_items=n_items,
        weight_range=weight_range,
        value_range=value_range,
        capacity_ratio=capacity_ratio,
    )

    # Return as dictionary for backward compatibility
    return {
        "weights": instance.weights,
        "values": instance.values,
        "capacity": instance.capacity,
        "n_items": instance.n_items,
    }


def solve_knapsack_dp(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: int,
    time_limit: float = 60.0,
    seed: int = 0,
) -> tuple[float | None, np.ndarray | None]:
    """
    Solve knapsack instance using dynamic programming (via OR-Tools).

    Args:
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        time_limit: Time limit in seconds
        seed: Random seed (unused, for compatibility)

    Returns:
        Tuple of (optimal_value, solution)
    """
    # Create instance with correct parameter order (weights, values, capacity)
    instance = KnapsackInstance(weights=weights, values=values, capacity=capacity)
    solved = KnapsackSolver.solve(instance, time_limit=time_limit)
    return solved.optimal_value, solved.solution


def solve_with_ortools(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: int,
    time_limit: float = 60.0,
    seed: int = 0,
) -> tuple[float | None, np.ndarray | None]:
    """
    Solve knapsack problem using OR-Tools.

    Args:
        values: Item values
        weights: Item weights
        capacity: Knapsack capacity
        time_limit: Time limit in seconds
        seed: Random seed (unused, for compatibility)

    Returns:
        Tuple of (optimal_value, solution)
    """
    # Create instance with correct parameter order (weights, values, capacity)
    instance = KnapsackInstance(weights=weights, values=values, capacity=capacity)
    solved = KnapsackSolver.solve(instance, time_limit=time_limit)
    return solved.optimal_value, solved.solution


if __name__ == "__main__":
    # Example usage
    print("Creating Knapsack datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        train_size=100,  # Small dataset for testing
        val_size=20,
        test_size=20,
        n_items_range=(10, 30),
        seed=42,
    )
    print("\nDatasets created successfully!")
