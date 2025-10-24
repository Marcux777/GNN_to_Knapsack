# mypy: ignore-errors
"""
In-Distribution Analysis Script

Analyzes model performance on in-distribution test sets with rigorous statistical validation.
Generates:
    - Gap statistics by size (mean, median, p50/p90/p95/p99)
    - CDF plots by size
    - Percentile plots
    - Violin plots
    - Bootstrap confidence intervals
    - Sample size adequacy checks
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.visualization import (
    plot_gap_cdf_by_size,
    plot_gap_percentiles_by_size,
    plot_gap_violin_by_size,
)
from knapsack_gnn.analysis.stats import (
    check_sample_size_adequacy,
    compute_gap_statistics_by_size,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_results_from_json(filepath: Path) -> dict:
    """Load evaluation results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def extract_gaps_and_sizes(results: dict) -> tuple[list[float], list[int]]:
    """
    Extract gaps and sizes from evaluation results.

    Args:
        results: Results dictionary with 'gaps' and optionally 'sizes'

    Returns:
        Tuple of (gaps, sizes)
    """
    gaps = results.get("gaps", [])

    # Try to extract sizes from results
    # Option 1: explicit 'sizes' field
    if "sizes" in results:
        sizes = results["sizes"]
    # Option 2: extract from per-instance results
    elif "instance_results" in results:
        sizes = [inst.get("n_items", None) for inst in results["instance_results"]]
    else:
        # If no size information, return empty
        sizes = []

    return gaps, sizes


def analyze_distribution(
    gaps: list[float],
    sizes: list[int],
    output_dir: Path,
    strategy_name: str = "Unknown",
    size_bins: list[int] = None,
):
    """
    Perform comprehensive distribution analysis.

    Args:
        gaps: List of optimality gap percentages
        sizes: List of problem sizes
        output_dir: Output directory for results
        strategy_name: Name of the strategy being analyzed
        size_bins: Optional list of size bins (default: unique sizes)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"DISTRIBUTION ANALYSIS: {strategy_name}")
    print("=" * 80)
    print(f"Total instances: {len(gaps)}")
    print(f"Size range: {min(sizes)} - {max(sizes)}")
    print()

    # Compute statistics by size
    print("Computing gap statistics by size...")
    stats_by_size = compute_gap_statistics_by_size(gaps, sizes, size_bins)

    # Print summary table
    print("\n" + "=" * 100)
    print("GAP STATISTICS BY SIZE")
    print("=" * 100)
    print(
        f"{'Size':<6} | {'Count':<6} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | "
        f"{'p50':<8} | {'p90':<8} | {'p95':<8} | {'p99':<8} | {'Max':<8}"
    )
    print("-" * 100)

    for size in sorted(stats_by_size.keys()):
        s = stats_by_size[size]
        if s["count"] == 0:
            continue
        print(
            f"{size:<6} | {s['count']:<6} | {s['mean']:<8.2f} | {s['median']:<8.2f} | "
            f"{s['std']:<8.2f} | {s['p50']:<8.2f} | {s['p90']:<8.2f} | {s['p95']:<8.2f} | "
            f"{s['p99']:<8.2f} | {s['max']:<8.2f}"
        )

    print("=" * 100)

    # Save statistics to CSV
    df = pd.DataFrame.from_dict(stats_by_size, orient="index")
    df.index.name = "size"
    csv_path = output_dir / f"{strategy_name}_stats_by_size.csv"
    df.to_csv(csv_path)
    print(f"\nStatistics saved to: {csv_path}")

    # Save to JSON (convert numpy int64 keys to int)
    json_path = output_dir / f"{strategy_name}_stats_by_size.json"
    stats_serializable = {int(k): v for k, v in stats_by_size.items()}
    with open(json_path, "w") as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"Statistics saved to: {json_path}")

    # Sample size adequacy check
    print("\n" + "=" * 80)
    print("SAMPLE SIZE ADEQUACY CHECK (target error: ±0.5%)")
    print("=" * 80)
    print(
        f"{'Size':<6} | {'n':<6} | {'Std':<8} | {'Required n':<12} | {'Margin':<10} | {'Status':<8}"
    )
    print("-" * 80)

    adequacy_results = {}
    for size in sorted(stats_by_size.keys()):
        if stats_by_size[size]["count"] == 0:
            continue

        size_gaps = np.array([g for g, sz in zip(gaps, sizes, strict=False) if sz == size])
        adequacy = check_sample_size_adequacy(size_gaps, target_error=0.5, confidence=0.95)
        adequacy_results[size] = adequacy

        status = "✓" if adequacy["adequate"] else "✗ INCREASE"
        print(
            f"{size:<6} | {adequacy['current_n']:<6} | {adequacy['current_std']:<8.2f} | "
            f"{adequacy['required_n']:<12} | ±{adequacy['margin_of_error']:<9.2f} | {status}"
        )

    print("=" * 80)

    # Save adequacy results (convert numpy int64 keys to int)
    adequacy_path = output_dir / f"{strategy_name}_sample_adequacy.json"
    adequacy_serializable = {int(k): v for k, v in adequacy_results.items()}
    with open(adequacy_path, "w") as f:
        json.dump(adequacy_serializable, f, indent=2)
    print(f"\nAdequacy results saved to: {adequacy_path}")

    # Check p95 criterion
    print("\n" + "=" * 80)
    print("CRITERION CHECK: p95 ≤ 1% for small sizes (10-50)")
    print("=" * 80)

    small_sizes = [s for s in stats_by_size.keys() if s <= 50 and stats_by_size[s]["count"] > 0]
    criterion_met = True
    for size in small_sizes:
        p95 = stats_by_size[size]["p95"]
        status = "✓ PASS" if p95 <= 1.0 else "✗ FAIL"
        if p95 > 1.0:
            criterion_met = False
        print(f"Size {size}: p95 = {p95:.2f}% {status}")

    print("-" * 80)
    if criterion_met:
        print("✓ CRITERION MET: All small sizes have p95 ≤ 1%")
    else:
        print("✗ CRITERION NOT MET: Some sizes exceed p95 threshold")
    print("=" * 80)

    # Generate plots
    print("\nGenerating plots...")

    # CDF plot
    cdf_path = output_dir / f"{strategy_name}_cdf_by_size.png"
    plot_gap_cdf_by_size(
        gaps,
        sizes,
        size_bins=size_bins,
        title=f"CDF of Optimality Gaps by Size ({strategy_name})",
        save_path=str(cdf_path),
    )

    # Percentile plot
    percentile_path = output_dir / f"{strategy_name}_percentiles_by_size.png"
    plot_gap_percentiles_by_size(
        stats_by_size,
        title=f"Gap Percentiles vs Problem Size ({strategy_name})",
        save_path=str(percentile_path),
    )

    # Violin plot
    violin_path = output_dir / f"{strategy_name}_violin_by_size.png"
    plot_gap_violin_by_size(
        gaps,
        sizes,
        size_bins=size_bins,
        title=f"Gap Distribution by Size ({strategy_name})",
        save_path=str(violin_path),
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()

    return stats_by_size, adequacy_results, criterion_met


def parse_args():
    parser = argparse.ArgumentParser(description="In-distribution analysis script")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="Unknown",
        help="Strategy name for labeling",
    )
    parser.add_argument(
        "--size-bins",
        nargs="+",
        type=int,
        default=None,
        help="Optional size bins to group by (e.g., 10 25 50 100)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    results = load_results_from_json(results_path)

    # Extract gaps and sizes
    gaps, sizes = extract_gaps_and_sizes(results)

    if len(gaps) == 0:
        print("Error: No gap data found in results")
        sys.exit(1)

    if len(sizes) == 0:
        print("Warning: No size information found. Using dummy sizes.")
        sizes = [50] * len(gaps)  # Default size if not available

    # Run analysis
    output_dir = Path(args.output_dir)
    analyze_distribution(
        gaps=gaps,
        sizes=sizes,
        output_dir=output_dir,
        strategy_name=args.strategy,
        size_bins=args.size_bins,
    )


if __name__ == "__main__":
    main()
