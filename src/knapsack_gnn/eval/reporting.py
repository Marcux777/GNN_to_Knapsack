"""
Evaluation result reporting and I/O utilities.

Handles export of results to CSV, JSON, and console output.
"""

import csv
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class EvalResult:
    """Single instance evaluation result."""

    instance_id: int
    n_items: int
    strategy: str
    gap: float
    feasible: bool
    time_ms: float
    samples_used: Optional[int] = None
    ilp_time_ms: Optional[float] = None
    seed: int = 0
    commit: str = "unknown"


@dataclass
class SummaryMetrics:
    """Aggregate metrics across instances."""

    strategy: str
    mean_gap: float
    median_gap: float
    std_gap: float
    max_gap: float
    feasibility_rate: float
    mean_time_ms: float
    p90_time_ms: float
    p50_time_ms: float
    throughput: float  # instances/second


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def export_results_to_csv(
    results: list[dict], filepath: str, include_metadata: bool = True
) -> None:
    """
    Export evaluation results to CSV format.

    Args:
        results: List of result dictionaries (one per instance)
        filepath: Path to save CSV file
        include_metadata: If True, include seed, commit hash, timestamp

    Example:
        >>> results = [
        ...     {"instance_id": 0, "gap": 0.05, "time_ms": 12.3, "feasible": True},
        ...     {"instance_id": 1, "gap": 0.12, "time_ms": 15.1, "feasible": True},
        ... ]
        >>> export_results_to_csv(results, "results.csv")
    """
    if not results:
        print("No results to export")
        return

    # Get git commit hash if available
    commit_hash = get_git_commit() if include_metadata else "unknown"

    # Determine fieldnames from first result
    base_fields = list(results[0].keys())

    if include_metadata and "commit" not in base_fields:
        base_fields.extend(["commit", "timestamp"])

    # Ensure Path
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()

        timestamp = datetime.now().isoformat()

        for result in results:
            row = result.copy()
            if include_metadata:
                if "commit" not in row:
                    row["commit"] = commit_hash
                if "timestamp" not in row:
                    row["timestamp"] = timestamp
            writer.writerow(row)

    print(f"Results exported to CSV: {filepath}")


def export_summary_to_csv(summary: Dict, filepath: str, strategy: str = "unknown") -> None:
    """
    Export summary metrics to CSV format.

    Args:
        summary: Dictionary with aggregate metrics
        filepath: Path to save CSV file
        strategy: Strategy name (e.g., "sampling", "warm_start")

    Example:
        >>> summary = {
        ...     "mean_gap": 0.070,
        ...     "median_gap": 0.0,
        ...     "mean_time_ms": 13.45,
        ...     "feasibility_rate": 1.0
        ... }
        >>> export_summary_to_csv(summary, "summary.csv", strategy="sampling")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "strategy", "timestamp"])

        timestamp = datetime.now().isoformat()

        for key, value in summary.items():
            writer.writerow([key, value, strategy, timestamp])

    print(f"Summary exported to CSV: {filepath}")


def save_results_to_json(results: Dict, filepath: str) -> None:
    """
    Save results to JSON file with proper type conversion.

    Args:
        results: Results dictionary
        filepath: Output filepath

    Example:
        >>> results = {"mean_gap": 0.05, "gaps": np.array([0.1, 0.2])}
        >>> save_results_to_json(results, "results.json")
    """

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    serializable_results = {k: convert(v) for k, v in results.items()}

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {filepath}")


def print_evaluation_summary(results: Dict, title: str = "Evaluation Results") -> None:
    """
    Print formatted evaluation summary to console.

    Args:
        results: Results dictionary with aggregate metrics
        title: Title for the summary

    Example:
        >>> results = {
        ...     "mean_gap": 0.070,
        ...     "feasibility_rate": 1.0,
        ...     "mean_inference_time": 0.01345
        ... }
        >>> print_evaluation_summary(results, "Test Results")
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60 + "\n")

    # Gap statistics
    if "mean_gap" in results:
        print(f"Optimality Gap:")
        print(f"  Mean:   {results['mean_gap']:.4f}%")
        print(f"  Median: {results.get('median_gap', 0):.4f}%")
        print(f"  Std:    {results.get('std_gap', 0):.4f}%")
        print(f"  Max:    {results.get('max_gap', 0):.4f}%")

    # Feasibility
    if "feasibility_rate" in results:
        print(f"\nFeasibility:")
        print(f"  Rate: {results['feasibility_rate'] * 100:.2f}%")
        if "feasible_count" in results:
            print(f"  Count: {results['feasible_count']}/{results.get('total_count', 0)}")

    # Timing
    if "mean_inference_time" in results:
        print(f"\nInference Timing:")
        print(f"  Mean:   {results['mean_inference_time'] * 1000:.2f} ms")
        print(f"  Median: {results.get('median_inference_time', 0) * 1000:.2f} ms")
        print(f"  P90:    {results.get('p90_inference_time', 0) * 1000:.2f} ms")

    # Solver timing (if available)
    if "mean_solver_time" in results:
        print(f"\nExact Solver Timing:")
        print(f"  Mean:   {results['mean_solver_time'] * 1000:.2f} ms")
        print(f"  Median: {results.get('median_solver_time', 0) * 1000:.2f} ms")

    # Speedup
    if "mean_speedup" in results:
        print(f"\nSpeedup vs Exact Solver:")
        print(f"  Mean:   {results['mean_speedup']:.2f}x")
        print(f"  Median: {results.get('median_speedup', 0):.2f}x")

    # Sampling stats
    if "mean_samples_used" in results:
        print(f"\nSampling:")
        print(f"  Mean samples: {results['mean_samples_used']:.1f}")
        print(f"  Median samples: {results.get('median_samples_used', 0):.1f}")

    # ILP stats
    if "mean_ilp_time" in results and results["mean_ilp_time"] is not None:
        print(f"\nILP Refinement:")
        print(f"  Mean time: {results['mean_ilp_time'] * 1000:.2f} ms")
        print(f"  Success rate: {results.get('ilp_success_rate', 0) * 100:.1f}%")

    print("=" * 60 + "\n")


def create_results_dataframe(results: list[dict]):
    """
    Convert results to pandas DataFrame for analysis.

    Args:
        results: List of result dictionaries

    Returns:
        DataFrame with results

    Note:
        Requires pandas to be installed

    Example:
        >>> results = [{"gap": 0.05, "time_ms": 12.3}, {"gap": 0.12, "time_ms": 15.1}]
        >>> df = create_results_dataframe(results)
        >>> print(df.describe())
    """
    try:
        import pandas as pd

        return pd.DataFrame(results)
    except ImportError as err:
        raise ImportError(
            "pandas is required for DataFrame export. Install with: pip install pandas"
        ) from err


def save_run_metadata(
    output_dir: Path,
    config: Dict,
    seed: int,
    device: str,
) -> None:
    """
    Save run metadata (config, seed, commit, hardware) to run.json.

    Args:
        output_dir: Directory to save run.json
        config: Configuration dictionary
        seed: Random seed used
        device: Device used (cpu/cuda)

    Example:
        >>> save_run_metadata(
        ...     Path("results/run_001"),
        ...     {"lr": 0.002, "epochs": 50},
        ...     seed=42,
        ...     device="cpu"
        ... )
    """
    import platform
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "commit": get_git_commit(),
        "seed": seed,
        "config": config,
        "hardware": {
            "device": device,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
    }

    filepath = output_dir / "run.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Run metadata saved to {filepath}")
