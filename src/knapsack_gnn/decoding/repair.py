"""
Solution Repair and Local Search Strategies

Implements greedy repair and local search methods to improve infeasible or suboptimal solutions:
    - Greedy repair with reinsertion
    - 1-swap local search
    - 2-opt local search
    - Hybrid repair + local search

These methods are designed to:
    1. Fix infeasible solutions (capacity violations)
    2. Improve feasible solutions via local search
    3. Reduce tail of optimality gaps (target: p95 ≤ 2%)
"""

import numpy as np
from typing import Tuple, Optional


class SolutionRepairer:
    """
    Repair and improve knapsack solutions using greedy heuristics and local search.
    """

    def __init__(self, max_repair_iterations: int = 100, max_local_search_iterations: int = 50):
        """
        Args:
            max_repair_iterations: Maximum iterations for repair phase
            max_local_search_iterations: Maximum iterations for local search
        """
        self.max_repair_iterations = max_repair_iterations
        self.max_local_search_iterations = max_local_search_iterations

    def check_feasibility(self, solution: np.ndarray, weights: np.ndarray, capacity: float) -> bool:
        """Check if solution is feasible (doesn't exceed capacity)."""
        total_weight = np.sum(solution * weights)
        return total_weight <= capacity + 1e-6  # Small tolerance for floating point

    def compute_value(self, solution: np.ndarray, values: np.ndarray) -> float:
        """Compute total value of solution."""
        return float(np.sum(solution * values))

    def greedy_repair(
        self,
        solution: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        capacity: float,
    ) -> np.ndarray:
        """
        Repair infeasible solution by greedily removing items.

        Args:
            solution: Binary solution (potentially infeasible)
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity

        Returns:
            Feasible solution
        """
        solution = solution.copy()

        # If already feasible, return
        if self.check_feasibility(solution, weights, capacity):
            return solution

        # Get selected items
        selected = np.where(solution == 1)[0]

        if len(selected) == 0:
            return solution

        # Sort by value/weight ratio (ascending - remove worst first)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(weights[selected] > 0, values[selected] / weights[selected], -np.inf)

        sorted_indices = selected[np.argsort(ratios)]  # Ascending order

        # Remove items until feasible
        current_weight = np.sum(solution * weights)
        for idx in sorted_indices:
            if current_weight <= capacity:
                break
            solution[idx] = 0
            current_weight -= weights[idx]

        return solution

    def greedy_repair_with_reinsertion(
        self,
        solution: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        capacity: float,
    ) -> np.ndarray:
        """
        Repair infeasible solution, then greedily refill with remaining items.

        This is the PRIMARY repair method - it not only fixes infeasibility
        but also tries to improve the solution by reinserting items.

        Args:
            solution: Binary solution (potentially infeasible)
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity

        Returns:
            Improved feasible solution
        """
        solution = solution.copy()

        # Step 1: Remove items until feasible
        if not self.check_feasibility(solution, weights, capacity):
            selected = np.where(solution == 1)[0]

            if len(selected) > 0:
                # Sort by value/weight ratio (ascending)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = np.where(
                        weights[selected] > 0,
                        values[selected] / weights[selected],
                        -np.inf,
                    )

                sorted_indices = selected[np.argsort(ratios)]

                current_weight = np.sum(solution * weights)
                for idx in sorted_indices:
                    if current_weight <= capacity:
                        break
                    solution[idx] = 0
                    current_weight -= weights[idx]

        # Step 2: Greedily refill capacity with remaining items
        remaining = np.where(solution == 0)[0]

        if len(remaining) > 0:
            # Sort remaining by value/weight ratio (descending - add best first)
            with np.errstate(divide="ignore", invalid="ignore"):
                remaining_ratios = np.where(
                    weights[remaining] > 0,
                    values[remaining] / weights[remaining],
                    -np.inf,
                )

            sorted_remaining = remaining[np.argsort(-remaining_ratios)]

            current_weight = np.sum(solution * weights)
            for idx in sorted_remaining:
                if weights[idx] <= 0:
                    continue
                if current_weight + weights[idx] <= capacity:
                    solution[idx] = 1
                    current_weight += weights[idx]

        return solution

    def local_search_1swap(
        self,
        solution: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        capacity: float,
        max_iterations: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        1-swap local search: try swapping each item in/out.

        Explores the neighborhood of swapping one item at a time.

        Args:
            solution: Current solution
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity
            max_iterations: Maximum iterations (default: self.max_local_search_iterations)

        Returns:
            Tuple of (improved_solution, n_improvements)
        """
        if max_iterations is None:
            max_iterations = self.max_local_search_iterations

        solution = solution.copy()
        current_value = self.compute_value(solution, values)
        n_improvements = 0

        for iteration in range(max_iterations):
            improved = False

            # Try removing each selected item
            selected = np.where(solution == 1)[0]
            for idx in selected:
                candidate = solution.copy()
                candidate[idx] = 0
                candidate_value = self.compute_value(candidate, values)

                if candidate_value > current_value and self.check_feasibility(
                    candidate, weights, capacity
                ):
                    solution = candidate
                    current_value = candidate_value
                    improved = True
                    n_improvements += 1
                    break

            if improved:
                continue

            # Try adding each unselected item
            unselected = np.where(solution == 0)[0]
            for idx in unselected:
                candidate = solution.copy()
                candidate[idx] = 1

                if not self.check_feasibility(candidate, weights, capacity):
                    continue

                candidate_value = self.compute_value(candidate, values)
                if candidate_value > current_value:
                    solution = candidate
                    current_value = candidate_value
                    improved = True
                    n_improvements += 1
                    break

            # No improvement found
            if not improved:
                break

        return solution, n_improvements

    def local_search_2opt(
        self,
        solution: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        capacity: float,
        max_iterations: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        2-opt local search: try swapping pairs of items (1-in, 1-out).

        Args:
            solution: Current solution
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity
            max_iterations: Maximum iterations

        Returns:
            Tuple of (improved_solution, n_improvements)
        """
        if max_iterations is None:
            max_iterations = self.max_local_search_iterations

        solution = solution.copy()
        current_value = self.compute_value(solution, values)
        n_improvements = 0

        selected = np.where(solution == 1)[0]
        unselected = np.where(solution == 0)[0]

        for iteration in range(max_iterations):
            improved = False

            # Try swapping each (in, out) pair
            for idx_out in selected:
                for idx_in in unselected:
                    candidate = solution.copy()
                    candidate[idx_out] = 0
                    candidate[idx_in] = 1

                    if not self.check_feasibility(candidate, weights, capacity):
                        continue

                    candidate_value = self.compute_value(candidate, values)
                    if candidate_value > current_value:
                        solution = candidate
                        current_value = candidate_value
                        improved = True
                        n_improvements += 1
                        # Update selected/unselected
                        selected = np.where(solution == 1)[0]
                        unselected = np.where(solution == 0)[0]
                        break

                if improved:
                    break

            if not improved:
                break

        return solution, n_improvements

    def hybrid_repair_and_search(
        self,
        solution: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        capacity: float,
        use_1swap: bool = True,
        use_2opt: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Full pipeline: repair + local search.

        This is the RECOMMENDED method for improving solutions.

        Args:
            solution: Initial solution (may be infeasible)
            weights: Item weights
            values: Item values
            capacity: Knapsack capacity
            use_1swap: Whether to apply 1-swap search
            use_2opt: Whether to apply 2-opt search (slower)

        Returns:
            Tuple of (final_solution, metadata)
        """
        initial_value = self.compute_value(solution, values)
        initial_feasible = self.check_feasibility(solution, weights, capacity)

        # Step 1: Repair
        solution = self.greedy_repair_with_reinsertion(solution, weights, values, capacity)
        after_repair_value = self.compute_value(solution, values)

        # Step 2: Local search (1-swap)
        n_improvements_1swap = 0
        if use_1swap:
            solution, n_improvements_1swap = self.local_search_1swap(
                solution, weights, values, capacity
            )

        after_1swap_value = self.compute_value(solution, values)

        # Step 3: Local search (2-opt) - optional
        n_improvements_2opt = 0
        if use_2opt:
            solution, n_improvements_2opt = self.local_search_2opt(
                solution, weights, values, capacity
            )

        final_value = self.compute_value(solution, values)
        final_feasible = self.check_feasibility(solution, weights, capacity)

        metadata = {
            "initial_value": float(initial_value),
            "initial_feasible": bool(initial_feasible),
            "after_repair_value": float(after_repair_value),
            "after_1swap_value": float(after_1swap_value),
            "final_value": float(final_value),
            "final_feasible": bool(final_feasible),
            "n_improvements_1swap": int(n_improvements_1swap),
            "n_improvements_2opt": int(n_improvements_2opt),
            "value_improvement": float(final_value - initial_value),
            "value_improvement_pct": float(
                (final_value - initial_value) / max(initial_value, 1) * 100
            ),
        }

        return solution, metadata


# Convenience functions
def repair_solution(
    solution: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float,
    use_local_search: bool = True,
) -> np.ndarray:
    """
    Convenience function to repair and optionally improve a solution.

    Args:
        solution: Binary solution (may be infeasible)
        weights: Item weights
        values: Item values
        capacity: Knapsack capacity
        use_local_search: Whether to apply local search after repair

    Returns:
        Repaired (and optionally improved) solution
    """
    repairer = SolutionRepairer()

    if use_local_search:
        solution, _ = repairer.hybrid_repair_and_search(
            solution, weights, values, capacity, use_1swap=True, use_2opt=False
        )
    else:
        solution = repairer.greedy_repair_with_reinsertion(solution, weights, values, capacity)

    return solution


def improve_solution(
    solution: np.ndarray,
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float,
    max_iterations: int = 50,
) -> Tuple[np.ndarray, int]:
    """
    Improve a feasible solution using local search.

    Args:
        solution: Feasible binary solution
        weights: Item weights
        values: Item values
        capacity: Knapsack capacity
        max_iterations: Maximum local search iterations

    Returns:
        Tuple of (improved_solution, n_improvements)
    """
    repairer = SolutionRepairer(max_local_search_iterations=max_iterations)
    return repairer.local_search_1swap(solution, weights, values, capacity)


if __name__ == "__main__":
    # Test the repairer
    print("Testing Solution Repairer")
    print("=" * 80)

    # Create a simple test instance
    np.random.seed(42)
    n_items = 20
    weights = np.random.randint(1, 10, size=n_items)
    values = np.random.randint(1, 20, size=n_items)
    capacity = int(np.sum(weights) * 0.5)

    print(f"Test instance: {n_items} items, capacity={capacity}")
    print(f"Total weight: {np.sum(weights)}, Total value: {np.sum(values)}")
    print()

    # Create an infeasible solution (select all items)
    infeasible_solution = np.ones(n_items, dtype=np.int32)

    repairer = SolutionRepairer()
    print(f"Initial solution: {np.sum(infeasible_solution)} items selected")
    print(f"  Weight: {np.sum(infeasible_solution * weights)} (capacity: {capacity})")
    print(f"  Value: {repairer.compute_value(infeasible_solution, values)}")
    print(f"  Feasible: {repairer.check_feasibility(infeasible_solution, weights, capacity)}")
    print()

    # Repair
    repaired, metadata = repairer.hybrid_repair_and_search(
        infeasible_solution, weights, values, capacity, use_1swap=True, use_2opt=False
    )

    print("After hybrid repair + 1-swap:")
    print(f"  Items selected: {np.sum(repaired)}")
    print(f"  Weight: {np.sum(repaired * weights)} (capacity: {capacity})")
    print(f"  Value: {repairer.compute_value(repaired, values)}")
    print(f"  Feasible: {repairer.check_feasibility(repaired, weights, capacity)}")
    print()

    print("Metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    print()

    print("=" * 80)
