"""
Core evaluation metrics for knapsack solutions.

Pure functions with no side effects - suitable for library use.
"""

import numpy as np
from typing import Dict, Tuple

def compute_optimality_gap(predicted_value: float, optimal_value: float) -> float:
    """
    Compute optimality gap as percentage.

    Args:
        predicted_value: Value from predicted solution
        optimal_value: Optimal value

    Returns:
        Gap percentage: (optimal - predicted) / optimal * 100

    Example:
        >>> compute_optimality_gap(98.0, 100.0)
        2.0
    """
    if optimal_value == 0:
        return 0.0 if predicted_value == 0 else 100.0
    return (optimal_value - predicted_value) / optimal_value * 100.0

def compute_accuracy(predicted_solution: np.ndarray, optimal_solution: np.ndarray) -> float:
    """
    Compute binary accuracy between solutions.

    Args:
        predicted_solution: Predicted binary solution
        optimal_solution: Optimal binary solution

    Returns:
        Accuracy (fraction of correct predictions)

    Example:
        >>> pred = np.array([1, 0, 1, 0])
        >>> opt = np.array([1, 0, 0, 0])
        >>> compute_accuracy(pred, opt)
        0.75
    """
    return float(np.mean(predicted_solution == optimal_solution))

def compute_precision_recall_f1(
    predicted_solution: np.ndarray, optimal_solution: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        predicted_solution: Predicted binary solution
        optimal_solution: Optimal binary solution

    Returns:
        Tuple of (precision, recall, f1)

    Example:
        >>> pred = np.array([1, 1, 0, 0])
        >>> opt = np.array([1, 0, 1, 0])
        >>> p, r, f1 = compute_precision_recall_f1(pred, opt)
        >>> round(p, 2), round(r, 2), round(f1, 2)
        (0.5, 0.5, 0.5)
    """
    # True positives, false positives, false negatives
    tp = np.sum((predicted_solution == 1) & (optimal_solution == 1))
    fp = np.sum((predicted_solution == 1) & (optimal_solution == 0))
    fn = np.sum((predicted_solution == 0) & (optimal_solution == 1))

    # Precision and recall
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_solution(
    predicted_solution: np.ndarray,
    optimal_solution: np.ndarray,
    predicted_value: float,
    optimal_value: float,
    weights: np.ndarray,
    capacity: float,
) -> Dict[str, float]:
    """
    Comprehensive solution evaluation.

    Args:
        predicted_solution: Predicted binary solution
        optimal_solution: Optimal binary solution
        predicted_value: Predicted solution value
        optimal_value: Optimal value
        weights: Item weights
        capacity: Knapsack capacity

    Returns:
        Dictionary of metrics including:
        - optimality_gap: Gap percentage
        - accuracy: Binary accuracy
        - precision, recall, f1_score: Classification metrics
        - is_feasible: Whether solution respects capacity
        - total_weight: Total weight of selected items
        - capacity_utilization: Fraction of capacity used

    Example:
        >>> weights = np.array([2, 3, 4])
        >>> pred_sol = np.array([1, 1, 0])
        >>> opt_sol = np.array([1, 0, 1])
        >>> result = evaluate_solution(pred_sol, opt_sol, 5.0, 6.0, weights, 6.0)
        >>> result['is_feasible']
        True
    """
    # Check feasibility
    total_weight = float(np.sum(predicted_solution * weights))
    is_feasible = total_weight <= capacity

    # Compute metrics
    gap = compute_optimality_gap(predicted_value, optimal_value)
    accuracy = compute_accuracy(predicted_solution, optimal_solution)
    precision, recall, f1 = compute_precision_recall_f1(predicted_solution, optimal_solution)

    return {
        "optimality_gap": gap,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "is_feasible": is_feasible,
        "total_weight": total_weight,
        "capacity": float(capacity),
        "capacity_utilization": total_weight / capacity if capacity > 0 else 0.0,
        "predicted_value": float(predicted_value),
        "optimal_value": float(optimal_value),
    }

def compute_solution_stats(metrics_list: list[Dict]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple solutions.

    Args:
        metrics_list: List of metric dictionaries from evaluate_solution()

    Returns:
        Dictionary of aggregate statistics

    Example:
        >>> metrics = [
        ...     {"optimality_gap": 0.5, "is_feasible": True},
        ...     {"optimality_gap": 1.0, "is_feasible": True}
        ... ]
        >>> stats = compute_solution_stats(metrics)
        >>> stats["mean_gap"]
        0.75
    """
    if not metrics_list:
        return {}

    gaps = [m["optimality_gap"] for m in metrics_list]
    feasible = [m["is_feasible"] for m in metrics_list]

    return {
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "std_gap": float(np.std(gaps)),
        "max_gap": float(np.max(gaps)),
        "min_gap": float(np.min(gaps)),
        "feasibility_rate": float(np.mean(feasible)),
        "n_instances": len(metrics_list),
    }
