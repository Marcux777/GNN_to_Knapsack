"""
Greedy Solver for Knapsack Problem
Uses value-to-weight ratio heuristic
"""

import numpy as np
import time
from typing import Dict, Tuple
from knapsack_gnn.data.generator import KnapsackInstance


class GreedySolver:
    """
    Greedy algorithm for 0-1 Knapsack Problem

    Algorithm:
    1. Compute value/weight ratio for each item
    2. Sort items by ratio in descending order
    3. Greedily add items until capacity is reached
    """

    def __init__(self):
        pass

    def solve(self, instance: KnapsackInstance) -> Dict:
        """
        Solve knapsack instance using greedy heuristic

        Args:
            instance: KnapsackInstance object

        Returns:
            Dictionary with solution, value, and metrics
        """
        start_time = time.perf_counter()

        weights = instance.weights
        values = instance.values
        capacity = instance.capacity
        n_items = instance.n_items

        # Compute value-to-weight ratios
        ratios = values / weights

        # Sort items by ratio (descending)
        sorted_indices = np.argsort(ratios)[::-1]

        # Greedily pack items
        solution = np.zeros(n_items, dtype=np.int32)
        current_weight = 0

        for idx in sorted_indices:
            if current_weight + weights[idx] <= capacity:
                solution[idx] = 1
                current_weight += weights[idx]

        # Compute solution value
        solution_value = np.sum(solution * values)

        # Check feasibility
        is_feasible = np.sum(solution * weights) <= capacity

        solve_time = time.perf_counter() - start_time

        # Compute optimality gap if optimal value is available
        gap = None
        if instance.optimal_value is not None and instance.optimal_value > 0:
            gap = 100.0 * (instance.optimal_value - solution_value) / instance.optimal_value

        return {
            "solution": solution,
            "value": int(solution_value),
            "is_feasible": is_feasible,
            "solve_time": solve_time,
            "optimality_gap": gap,
            "weight_used": int(np.sum(solution * weights)),
            "capacity": capacity,
            "optimal_value": instance.optimal_value,
        }

    def solve_batch(self, instances: list, verbose: bool = False) -> list:
        """
        Solve multiple instances

        Args:
            instances: List of KnapsackInstance objects
            verbose: Print progress

        Returns:
            List of result dictionaries
        """
        results = []
        for i, instance in enumerate(instances):
            if verbose and (i + 1) % 50 == 0:
                print(f"Solved {i + 1}/{len(instances)} instances")
            result = self.solve(instance)
            results.append(result)

        return results

    @staticmethod
    def evaluate_results(results: list) -> Dict:
        """
        Compute aggregate statistics from results

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary with aggregate metrics
        """
        gaps = [r["optimality_gap"] for r in results if r["optimality_gap"] is not None]
        times = [r["solve_time"] for r in results]
        feasible = [r["is_feasible"] for r in results]

        stats = {
            "mean_gap": float(np.mean(gaps)) if gaps else None,
            "median_gap": float(np.median(gaps)) if gaps else None,
            "std_gap": float(np.std(gaps)) if gaps else None,
            "max_gap": float(np.max(gaps)) if gaps else None,
            "min_gap": float(np.min(gaps)) if gaps else None,
            "mean_time": float(np.mean(times)),
            "median_time": float(np.median(times)),
            "feasibility_rate": float(np.mean(feasible)),
            "throughput": len(results) / np.sum(times) if np.sum(times) > 0 else 0,
            "n_instances": len(results),
        }

        return stats


class RandomSolver:
    """
    Random baseline for Knapsack Problem
    Randomly selects items until capacity is reached
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def solve(self, instance: KnapsackInstance, max_attempts: int = 100) -> Dict:
        """
        Generate random feasible solution

        Args:
            instance: KnapsackInstance object
            max_attempts: Maximum attempts to find feasible solution

        Returns:
            Dictionary with solution, value, and metrics
        """
        start_time = time.perf_counter()

        weights = instance.weights
        values = instance.values
        capacity = instance.capacity
        n_items = instance.n_items

        best_solution = np.zeros(n_items, dtype=np.int32)
        best_value = 0

        # Try multiple random solutions
        for _ in range(max_attempts):
            # Random permutation of items
            perm = self.rng.permutation(n_items)
            solution = np.zeros(n_items, dtype=np.int32)
            current_weight = 0

            # Greedily add items in random order
            for idx in perm:
                if current_weight + weights[idx] <= capacity:
                    solution[idx] = 1
                    current_weight += weights[idx]

            # Check if better
            solution_value = np.sum(solution * values)
            if solution_value > best_value:
                best_solution = solution
                best_value = solution_value

        solve_time = time.perf_counter() - start_time

        # Check feasibility
        is_feasible = np.sum(best_solution * weights) <= capacity

        # Compute optimality gap
        gap = None
        if instance.optimal_value is not None and instance.optimal_value > 0:
            gap = 100.0 * (instance.optimal_value - best_value) / instance.optimal_value

        return {
            "solution": best_solution,
            "value": int(best_value),
            "is_feasible": is_feasible,
            "solve_time": solve_time,
            "optimality_gap": gap,
            "weight_used": int(np.sum(best_solution * weights)),
            "capacity": capacity,
            "optimal_value": instance.optimal_value,
        }

    def solve_batch(self, instances: list, verbose: bool = False) -> list:
        """Solve multiple instances"""
        results = []
        for i, instance in enumerate(instances):
            if verbose and (i + 1) % 50 == 0:
                print(f"Solved {i + 1}/{len(instances)} instances")
            result = self.solve(instance)
            results.append(result)
        return results


if __name__ == "__main__":
    # Test greedy solver
    from knapsack_gnn.data.generator import KnapsackGenerator, KnapsackSolver

    print("Testing Greedy Solver...")

    # Generate test instance
    generator = KnapsackGenerator(seed=42)
    instance = generator.generate_instance(n_items=20)
    instance = KnapsackSolver.solve(instance)

    print(f"\nInstance: {instance.n_items} items, capacity={instance.capacity}")
    print(f"Optimal value: {instance.optimal_value}")

    # Solve with greedy
    greedy_solver = GreedySolver()
    result = greedy_solver.solve(instance)

    print(f"\nGreedy Solution:")
    print(f"  Value: {result['value']}")
    print(f"  Optimality gap: {result['optimality_gap']:.2f}%")
    print(f"  Feasible: {result['is_feasible']}")
    print(f"  Time: {result['solve_time'] * 1000:.2f} ms")

    # Solve with random
    random_solver = RandomSolver(seed=42)
    result_random = random_solver.solve(instance)

    print(f"\nRandom Solution:")
    print(f"  Value: {result_random['value']}")
    print(f"  Optimality gap: {result_random['optimality_gap']:.2f}%")
    print(f"  Feasible: {result_random['is_feasible']}")
    print(f"  Time: {result_random['solve_time'] * 1000:.2f} ms")


# Convenience wrapper functions for backward compatibility
def greedy_knapsack(instance: KnapsackInstance) -> Dict:
    """
    Solve knapsack instance using greedy heuristic.

    Args:
        instance: KnapsackInstance object

    Returns:
        Dictionary with solution and metrics
    """
    solver = GreedySolver()
    return solver.solve(instance)


def random_knapsack(instance: KnapsackInstance, seed: int = 42, max_attempts: int = 100) -> Dict:
    """
    Solve knapsack instance using random sampling.

    Args:
        instance: KnapsackInstance object
        seed: Random seed
        max_attempts: Maximum number of random attempts

    Returns:
        Dictionary with solution and metrics
    """
    solver = RandomSolver(seed=seed)
    return solver.solve(instance, max_attempts=max_attempts)
