"""
Large-Scale Performance Analysis
Analyzes model performance across different problem sizes
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_results(checkpoint_dir: str) -> dict:
    """Load all available evaluation results"""
    results = {}

    # Load n=200 (in-distribution, upper bound)
    ood_path = Path(checkpoint_dir) / "evaluation" / "ood" / "ood_results_n200.json"
    if ood_path.exists():
        with open(ood_path) as f:
            results[200] = json.load(f)
            print(f"✓ Loaded n=200: {results[200]['n_instances']} instances")

    # Load n=500 (moderate extrapolation)
    n500_path = Path(checkpoint_dir) / "evaluation" / "large_instances" / "results_n500.json"
    if n500_path.exists():
        with open(n500_path) as f:
            results[500] = json.load(f)
            print(f"✓ Loaded n=500: {results[500]['n_instances']} instances")

    # Load n=2000 (extreme extrapolation)
    ood_2000_path = Path(checkpoint_dir) / "evaluation" / "ood" / "ood_results_n2000.json"
    if ood_2000_path.exists():
        with open(ood_2000_path) as f:
            results[2000] = json.load(f)
            print(f"✓ Loaded n=2000: {results[2000]['n_instances']} instances")

    return results


def print_comprehensive_table(results: dict, training_range: tuple):
    """Print comprehensive comparison table"""
    print("\n" + "=" * 90)
    print("LARGE-SCALE PERFORMANCE ANALYSIS")
    print("=" * 90)
    print(f"\nTraining Range: n={training_range[0]}-{training_range[1]}")
    print("\n" + "-" * 90)
    print(
        f"{'Size':>6} | {'N':>4} | {'Extrap':>7} | {'Mean Gap':>10} | {'Median Gap':>12} | "
        f"{'Std':>8} | {'Max Gap':>9} | {'Feasible':>9}"
    )
    print("-" * 90)

    for size in sorted(results.keys()):
        r = results[size]
        extrap_factor = size / training_range[1]

        print(
            f"{size:>6} | {r['n_instances']:>4} | {extrap_factor:>6.1f}x | "
            f"{r['mean_gap']:>9.2f}% | {r['median_gap']:>11.2f}% | "
            f"{r['std_gap']:>7.2f}% | {r['max_gap']:>8.2f}% | "
            f"{r['feasibility_rate'] * 100:>8.1f}%"
        )

    print("-" * 90)


def analyze_degradation(results: dict, training_range: tuple):
    """Analyze performance degradation"""
    print("\n" + "=" * 90)
    print("DEGRADATION ANALYSIS")
    print("=" * 90)

    sizes = sorted(results.keys())

    for size in sizes:
        r = results[size]
        extrap_factor = size / training_range[1]
        gap = r["mean_gap"]
        median_gap = r["median_gap"]

        print(f"\nn={size} (extrapolation {extrap_factor:.1f}x):")
        print(f"  Instances: {r['n_instances']}")
        print(f"  Mean gap: {gap:.2f}%")
        print(f"  Median gap: {median_gap:.2f}%")
        print(f"  Std gap: {r['std_gap']:.2f}%")
        print(f"  Max gap: {r['max_gap']:.2f}%")

        # Quality assessment
        if median_gap < 1.0:
            quality = "✅ EXCELLENT"
        elif median_gap < 5.0:
            quality = "✓ GOOD"
        elif median_gap < 10.0:
            quality = "⚠ MODERATE"
        else:
            quality = "❌ POOR"

        print(f"  Quality (median): {quality}")

        # Check for outliers
        if gap > 2 * median_gap:
            print("  ⚠️ WARNING: High variance detected (mean >> median)")
            print("     → Likely has outliers pulling the mean up")

            # Calculate percentage of good solutions
            if "gaps" in r:
                gaps = np.array(r["gaps"])
                good_pct = (gaps < 5.0).sum() / len(gaps) * 100
                excellent_pct = (gaps < 1.0).sum() / len(gaps) * 100
                print(f"     → {excellent_pct:.1f}% with gap < 1%")
                print(f"     → {good_pct:.1f}% with gap < 5%")


def statistical_comparison(results: dict):
    """Statistical comparison between sizes"""
    print("\n" + "=" * 90)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 90)

    sizes = sorted(results.keys())

    for i in range(len(sizes) - 1):
        size1, size2 = sizes[i], sizes[i + 1]
        r1, r2 = results[size1], results[size2]

        if "gaps" in r1 and "gaps" in r2 and len(r1["gaps"]) > 1 and len(r2["gaps"]) > 1:
            gaps1 = np.array(r1["gaps"])
            gaps2 = np.array(r2["gaps"])

            # t-test
            t_stat, p_value = stats.ttest_ind(gaps1, gaps2)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(gaps1) - 1) * np.var(gaps1) + (len(gaps2) - 1) * np.var(gaps2))
                / (len(gaps1) + len(gaps2) - 2)
            )
            cohens_d = (np.mean(gaps2) - np.mean(gaps1)) / pooled_std if pooled_std > 0 else 0

            print(f"\nn={size1} vs n={size2}:")
            print(f"  Mean gap difference: {r2['mean_gap'] - r1['mean_gap']:+.2f}%")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f}", end="")

            if abs(cohens_d) < 0.2:
                print(" (negligible)")
            elif abs(cohens_d) < 0.5:
                print(" (small)")
            elif abs(cohens_d) < 0.8:
                print(" (medium)")
            else:
                print(" (large)")

            if p_value < 0.05:
                if r2["mean_gap"] > r1["mean_gap"]:
                    print("  → ❌ Significant degradation (p < 0.05)")
                else:
                    print("  → ✅ Significant improvement (p < 0.05)")
            else:
                print("  → No significant difference (p >= 0.05)")


def create_visualization(results: dict, training_range: tuple, output_path: str):
    """Create comprehensive visualization"""
    sizes = sorted(results.keys())

    # Extract metrics
    mean_gaps = [results[s]["mean_gap"] for s in sizes]
    median_gaps = [results[s]["median_gap"] for s in sizes]
    std_gaps = [results[s]["std_gap"] for s in sizes]
    [results[s]["max_gap"] for s in sizes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Gap comparison (mean vs median)
    ax = axes[0, 0]
    x = np.arange(len(sizes))
    width = 0.35

    ax.bar(x - width / 2, mean_gaps, width, label="Mean Gap", color="#E63946", alpha=0.8)
    ax.bar(x + width / 2, median_gaps, width, label="Median Gap", color="#2A9D8F", alpha=0.8)

    ax.set_xlabel("Problem Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Optimality Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title("Mean vs Median Gap by Size", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={s}" for s in sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add training range indicator
    ax.axvline(
        x=-0.5,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"Training: n={training_range[0]}-{training_range[1]}",
    )

    # 2. Gap with error bars (mean ± std)
    ax = axes[0, 1]
    ax.errorbar(
        sizes,
        mean_gaps,
        yerr=std_gaps,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#457B9D",
        label="Mean ± Std",
    )
    ax.plot(
        sizes,
        median_gaps,
        marker="s",
        linewidth=2,
        markersize=8,
        color="#F77F00",
        linestyle="--",
        label="Median",
    )

    ax.axvspan(
        training_range[0], training_range[1], alpha=0.2, color="green", label="Training Range"
    )

    ax.set_xlabel("Problem Size (# items)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Optimality Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title("Generalization Performance", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # 3. Box plot of gap distribution
    ax = axes[1, 0]
    gap_distributions = []
    labels = []

    for size in sizes:
        if "gaps" in results[size]:
            gap_distributions.append(results[size]["gaps"])
            labels.append(f"n={size}\n(N={results[size]['n_instances']})")

    if gap_distributions:
        bp = ax.boxplot(
            gap_distributions, labels=labels, patch_artist=True, showfliers=True, notch=True
        )

        # Color boxes
        colors = ["#06D6A0", "#FFD166", "#EF476F"]
        for _i, (patch, color) in enumerate(zip(bp["boxes"], colors, strict=False)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Highlight outliers
        for flier in bp["fliers"]:
            flier.set(marker="o", color="red", alpha=0.5, markersize=6)

    ax.set_ylabel("Optimality Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title("Gap Distribution (with outliers)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 4. Extrapolation factor vs performance
    ax = axes[1, 1]
    extrap_factors = [s / training_range[1] for s in sizes]

    ax.scatter(
        extrap_factors,
        median_gaps,
        s=200,
        alpha=0.6,
        c=range(len(sizes)),
        cmap="viridis",
        edgecolors="black",
        linewidth=2,
    )

    for _i, (x, y, size) in enumerate(zip(extrap_factors, median_gaps, sizes, strict=False)):
        ax.annotate(
            f"n={size}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Extrapolation Factor", fontsize=12, fontweight="bold")
    ax.set_ylabel("Median Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance vs Extrapolation", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()


def main():
    checkpoint_dir = "checkpoints/run_20251020_104533"
    output_dir = Path(checkpoint_dir) / "evaluation" / "large_scale_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("LOADING RESULTS")
    print("=" * 90)

    results = load_results(checkpoint_dir)

    if not results:
        print("\n❌ No results found!")
        return

    # Get training range from first result
    training_range = results[list(results.keys())[0]].get("training_size_range", (50, 200))

    # Analysis
    print_comprehensive_table(results, training_range)
    analyze_degradation(results, training_range)
    statistical_comparison(results)

    # Visualization
    viz_path = output_dir / "large_scale_performance.png"
    create_visualization(results, training_range, str(viz_path))

    # Save summary
    summary = {
        "training_range": training_range,
        "sizes_tested": sorted(results.keys()),
        "summary": {
            str(size): {
                "n_instances": r["n_instances"],
                "extrapolation_factor": size / training_range[1],
                "mean_gap": r["mean_gap"],
                "median_gap": r["median_gap"],
                "std_gap": r["std_gap"],
                "max_gap": r["max_gap"],
                "feasibility_rate": r["feasibility_rate"],
            }
            for size, r in results.items()
        },
    }

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_path}")

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
