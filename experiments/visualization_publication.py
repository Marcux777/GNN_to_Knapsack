# mypy: ignore-errors
"""
Publication-Quality Figures for Scientific Validation

Creates publication-ready multi-panel figures that demonstrate:
    - Panel A: Gap vs problem size with confidence intervals
    - Panel B: CDF of gaps by size range
    - Panel C: Violin plots comparing strategies
    - Panel D: Reliability diagram (calibration)

This is the "Figure 1" that silences critics.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)


def create_publication_figure(
    stats_by_size: dict[int, dict],
    gaps_all: list[float],
    sizes_all: list[int],
    strategy_gaps: dict[str, list[float]],
    calibration_results: dict,
    save_path: str | Path,
    title: str = "GNN-based Knapsack Solver: Comprehensive Validation",
):
    """
    Create 4-panel publication figure.

    Args:
        stats_by_size: Dictionary from compute_gap_statistics_by_size()
        gaps_all: All gap values
        sizes_all: Corresponding sizes
        strategy_gaps: Dict mapping strategy name -> list of gaps
        calibration_results: Calibration evaluation results
        save_path: Path to save figure
        title: Overall figure title
    """
    # Create figure with GridSpec for custom layout
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Panel A: Gap vs Size with CI
    ax_a = fig.add_subplot(gs[0, 0])
    plot_gap_vs_size_with_ci(ax_a, stats_by_size)
    ax_a.text(-0.1, 1.05, "A", transform=ax_a.transAxes, fontsize=20, fontweight="bold")

    # Panel B: CDF by Size Range
    ax_b = fig.add_subplot(gs[0, 1])
    plot_cdf_by_size_range(ax_b, gaps_all, sizes_all)
    ax_b.text(-0.1, 1.05, "B", transform=ax_b.transAxes, fontsize=20, fontweight="bold")

    # Panel C: Strategy Comparison (Violin)
    ax_c = fig.add_subplot(gs[1, 0])
    plot_strategy_violin(ax_c, strategy_gaps)
    ax_c.text(-0.1, 1.05, "C", transform=ax_c.transAxes, fontsize=20, fontweight="bold")

    # Panel D: Reliability Diagram
    ax_d = fig.add_subplot(gs[1, 1])
    plot_reliability_panel(ax_d, calibration_results)
    ax_d.text(-0.1, 1.05, "D", transform=ax_d.transAxes, fontsize=20, fontweight="bold")

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Save
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Publication figure saved to: {save_path}")
    plt.close(fig)


def plot_gap_vs_size_with_ci(ax: plt.Axes, stats_by_size: dict[int, dict]):
    """
    Panel A: Gap vs problem size with percentiles and confidence intervals.
    """
    sizes = sorted([s for s in stats_by_size.keys() if stats_by_size[s]["count"] > 0])

    means = [stats_by_size[s]["mean"] for s in sizes]
    medians = [stats_by_size[s]["median"] for s in sizes]
    p95 = [stats_by_size[s]["p95"] for s in sizes]
    p99 = [stats_by_size[s]["p99"] for s in sizes]

    # CI bands
    ci_lower = []
    ci_upper = []
    for s in sizes:
        ci = stats_by_size[s]["ci_95"]
        if ci[0] is not None:
            ci_lower.append(ci[0])
            ci_upper.append(ci[1])
        else:
            ci_lower.append(means[sizes.index(s)])
            ci_upper.append(means[sizes.index(s)])

    # Plot mean with CI
    ax.plot(
        sizes, means, "o-", linewidth=2.5, markersize=8, label="Mean", color="#2E86AB", zorder=3
    )
    ax.fill_between(sizes, ci_lower, ci_upper, alpha=0.2, color="#2E86AB", label="95% CI")

    # Plot percentiles
    ax.plot(sizes, medians, "s--", linewidth=2, markersize=6, label="Median (p50)", color="#A23B72")
    ax.plot(sizes, p95, "^-", linewidth=2, markersize=6, label="p95", color="#C73E1D")
    ax.plot(sizes, p99, "v-", linewidth=2, markersize=6, label="p99", color="#6A994E")

    # Target line
    ax.axhline(y=1.0, color="red", linestyle=":", linewidth=2, label="Target (p95 ≤ 1%)", alpha=0.7)

    ax.set_xlabel("Problem Size (# items)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Optimality Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title("Gap Statistics vs Problem Size", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_cdf_by_size_range(ax: plt.Axes, gaps_all: list[float], sizes_all: list[int]):
    """
    Panel B: CDF of gaps grouped by size ranges.
    """
    from knapsack_gnn.analysis.stats import compute_cdf

    gaps_all = np.array(gaps_all)
    sizes_all = np.array(sizes_all)

    # Define size ranges
    ranges = [
        ("10-25", 10, 25),
        ("26-50", 26, 50),
        ("51-100", 51, 100),
        ("101-200", 101, 200),
    ]

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ranges)))

    for (label, size_min, size_max), color in zip(ranges, colors, strict=False):
        mask = (sizes_all >= size_min) & (sizes_all <= size_max)
        if not mask.any():
            continue

        range_gaps = gaps_all[mask]
        x, cdf = compute_cdf(range_gaps)

        ax.plot(x, cdf, linewidth=2.5, color=color, label=f"n={label} ({len(range_gaps)} inst.)")

    ax.set_xlabel("Optimality Gap (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Probability", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Distribution by Size Range", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1.05])


def plot_strategy_violin(ax: plt.Axes, strategy_gaps: dict[str, list[float]]):
    """
    Panel C: Violin plots comparing strategies.
    """
    strategies = list(strategy_gaps.keys())
    gaps_lists = [strategy_gaps[s] for s in strategies]

    # Filter out empty lists
    valid_strategies = []
    valid_gaps = []
    for s, g in zip(strategies, gaps_lists, strict=False):
        if len(g) > 0:
            valid_strategies.append(s)
            valid_gaps.append(g)

    if not valid_strategies:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Violin plot
    parts = ax.violinplot(
        valid_gaps,
        positions=range(len(valid_strategies)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    # Customize colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(valid_strategies)))
    for pc, color in zip(parts["bodies"], colors, strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1)

    # Add mean annotations
    for i, (_strategy, gaps) in enumerate(zip(valid_strategies, valid_gaps, strict=False)):
        mean_val = np.mean(gaps)
        np.median(gaps)
        p95_val = np.percentile(gaps, 95)

        # Annotate above violin
        y_pos = max(gaps) * 1.05
        ax.text(
            i,
            y_pos,
            f"μ={mean_val:.2f}%\np95={p95_val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "gray",
            },
        )

    ax.set_xticks(range(len(valid_strategies)))
    ax.set_xticklabels(valid_strategies, rotation=15, ha="right")
    ax.set_ylabel("Optimality Gap (%)", fontsize=12, fontweight="bold")
    ax.set_title("Strategy Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)


def plot_reliability_panel(ax: plt.Axes, calibration_results: dict):
    """
    Panel D: Reliability diagram (calibration).
    """
    # Extract reliability curve from best method
    # Try temperature_scaled first, then uncalibrated
    if "temperature_scaled" in calibration_results:
        results = calibration_results["temperature_scaled"]
        label = f"Temperature Scaled (T={results.get('optimal_temperature', 1.0):.2f})"
        color = "#2E86AB"
    else:
        results = calibration_results.get("uncalibrated", {})
        label = "Uncalibrated"
        color = "#C73E1D"

    rel_curve = results.get("reliability_curve", {})
    mean_pred = np.array(rel_curve.get("mean_predicted", []))
    frac_pos = np.array(rel_curve.get("fraction_positive", []))
    counts = np.array(rel_curve.get("counts", []))

    # Filter NaN
    valid = ~np.isnan(mean_pred) & ~np.isnan(frac_pos)
    mean_pred = mean_pred[valid]
    frac_pos = frac_pos[valid]
    counts = counts[valid]

    if len(mean_pred) > 0:
        # Scatter with size proportional to count
        sizes = (counts / counts.max()) * 300 + 30
        ax.scatter(
            mean_pred,
            frac_pos,
            s=sizes,
            alpha=0.6,
            color=color,
            edgecolors="black",
            linewidths=1,
            label=label,
            zorder=3,
        )

        # Connect with line
        sorted_idx = np.argsort(mean_pred)
        ax.plot(
            mean_pred[sorted_idx],
            frac_pos[sorted_idx],
            linestyle="--",
            alpha=0.5,
            color=color,
            linewidth=2,
        )

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2.5, label="Perfect Calibration", alpha=0.7, zorder=2)

    # ECE annotation
    ece = results.get("ece", 0.0)
    status = "✓" if ece < 0.1 else "✗"
    ax.text(
        0.05,
        0.95,
        f"ECE = {ece:.4f} {status}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12, fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontsize=12, fontweight="bold")
    ax.set_title("Probability Calibration", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")


def create_results_table_latex(
    stats_by_size: dict[int, dict],
    strategy_results: dict[str, dict],
    save_path: str | Path,
):
    """
    Create LaTeX table of results.

    Args:
        stats_by_size: Gap statistics by size
        strategy_results: Results by strategy
        save_path: Path to save .tex file
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Optimality Gap Statistics by Problem Size and Strategy}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Size & Count & Mean (\%) & Median (\%) & p95 (\%) & p99 (\%) & Max (\%) \\")
    lines.append(r"\midrule")

    for size in sorted(stats_by_size.keys()):
        s = stats_by_size[size]
        if s["count"] == 0:
            continue
        lines.append(
            f"{size} & {s['count']} & {s['mean']:.2f} & {s['median']:.2f} & "
            f"{s['p95']:.2f} & {s['p99']:.2f} & {s['max']:.2f} \\\\"
        )

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textbf{Strategy Comparison}} \\")
    lines.append(r"\midrule")

    for strategy, results in strategy_results.items():
        mean_gap = results.get("mean_gap", 0.0)
        median_gap = results.get("median_gap", 0.0)
        p95 = np.percentile(results.get("gaps", [0]), 95) if results.get("gaps") else 0.0
        mean_time = results.get("mean_inference_time_ms", 0.0)

        lines.append(
            f"{strategy} & -- & {mean_gap:.2f} & {median_gap:.2f} & "
            f"{p95:.2f} & -- & {mean_time:.2f} ms \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))

    print(f"LaTeX table saved to: {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Publication figure module loaded.")
    print("Use create_publication_figure() to generate 4-panel figure.")
