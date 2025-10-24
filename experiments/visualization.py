# mypy: ignore-errors
"""
Visualization utilities for experiment analysis.

Plotting functions for gaps, training curves, and comparisons.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300  # High DPI for publication


def plot_optimality_gaps(
    gaps: list[float],
    title: str = "Optimality Gap Distribution",
    save_path: str | Path | None = None,
):
    """
    Plot distribution of optimality gaps with histogram and box plot.

    Args:
        gaps: List of optimality gap percentages
        title: Plot title
        save_path: Path to save figure (PNG recommended)

    Example:
        >>> gaps = [0.05, 0.12, 0.03, 0.08, 0.15]
        >>> plot_optimality_gaps(gaps, "Test Gaps", "gaps.png")
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(gaps, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Optimality Gap (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title} - Histogram")
    axes[0].axvline(
        np.mean(gaps), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(gaps):.2f}%"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    bp = axes[1].boxplot(gaps, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    axes[1].set_ylabel("Optimality Gap (%)")
    axes[1].set_title(f"{title} - Box Plot")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_performance_vs_size(
    instance_sizes: list[int],
    metrics: list[float],
    metric_name: str = "Optimality Gap (%)",
    title: str = "Performance vs Problem Size",
    save_path: str | Path | None = None,
):
    """
    Plot metric performance vs instance size with scatter and trend line.

    Args:
        instance_sizes: List of instance sizes (number of items)
        metrics: Corresponding metric values
        metric_name: Name of the metric (for y-axis label)
        title: Plot title
        save_path: Path to save figure

    Example:
        >>> sizes = [10, 20, 30, 40, 50]
        >>> gaps = [0.05, 0.08, 0.12, 0.15, 0.18]
        >>> plot_performance_vs_size(sizes, gaps, save_path="size_scaling.png")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    ax.scatter(instance_sizes, metrics, alpha=0.5, s=50, color="steelblue")

    # Compute average per size
    size_to_metrics = defaultdict(list)
    for size, metric in zip(instance_sizes, metrics, strict=False):
        size_to_metrics[size].append(metric)

    avg_sizes = sorted(size_to_metrics.keys())
    avg_metrics = [np.mean(size_to_metrics[s]) for s in avg_sizes]
    std_metrics = [np.std(size_to_metrics[s]) for s in avg_sizes]

    # Plot average line with error bars
    ax.errorbar(
        avg_sizes,
        avg_metrics,
        yerr=std_metrics,
        fmt="o-",
        color="red",
        linewidth=2,
        markersize=8,
        label="Mean ± Std",
        capsize=5,
    )

    ax.set_xlabel("Number of Items", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: str | Path | None = None,
):
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Dictionary with keys like "train_loss", "val_loss", "train_acc", "val_acc"
        title: Plot title
        save_path: Path to save figure

    Example:
        >>> history = {
        ...     "train_loss": [0.5, 0.4, 0.3, 0.2],
        ...     "val_loss": [0.6, 0.5, 0.4, 0.3],
        ...     "train_acc": [0.7, 0.8, 0.85, 0.9],
        ...     "val_acc": [0.65, 0.75, 0.8, 0.85]
        ... }
        >>> plot_training_history(history, save_path="training.png")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss plot
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Loss Curve", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if "train_acc" in history:
        axes[1].plot(epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=2)
    if "val_acc" in history:
        axes[1].plot(epochs, history["val_acc"], "r-", label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Accuracy Curve", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_strategy_comparison(
    strategies: list[str],
    gaps: list[list[float]],
    title: str = "Strategy Comparison",
    save_path: str | Path | None = None,
):
    """
    Compare multiple strategies side-by-side with violin plots.

    Args:
        strategies: List of strategy names
        gaps: List of gap lists (one per strategy)
        title: Plot title
        save_path: Path to save figure

    Example:
        >>> strategies = ["Sampling", "Warm-start", "Greedy"]
        >>> gaps = [[0.05, 0.08], [0.12, 0.15], [0.5, 0.6]]
        >>> plot_strategy_comparison(strategies, gaps, save_path="comparison.png")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for violin plot
    parts = ax.violinplot(gaps, positions=range(len(strategies)), showmeans=True, showmedians=True)

    # Customize colors
    colors = plt.cm.Set3(range(len(strategies)))
    for pc, color in zip(parts["bodies"], colors, strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha="right")
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean values as text
    for i, (_strategy, gap_list) in enumerate(zip(strategies, gaps, strict=False)):
        mean_val = np.mean(gap_list)
        ax.text(i, max(gap_list) * 1.05, f"{mean_val:.2f}%", ha="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_gap_cdf_by_size(
    gaps: list[float],
    sizes: list[int],
    size_bins: list[int] | None = None,
    title: str = "CDF of Optimality Gaps by Problem Size",
    save_path: str | Path | None = None,
):
    """
    Plot cumulative distribution function of gaps grouped by problem size.

    Args:
        gaps: List of optimality gap percentages
        sizes: List of problem sizes (number of items) corresponding to gaps
        size_bins: Optional list of size bins to plot (default: unique sizes)
        title: Plot title
        save_path: Path to save figure

    Example:
        >>> gaps = [0.05, 0.12, 0.03, 0.08, 0.15, 0.02, 0.10]
        >>> sizes = [10, 10, 10, 25, 25, 50, 50]
        >>> plot_gap_cdf_by_size(gaps, sizes, save_path="cdf.png")
    """
    from knapsack_gnn.analysis.stats import compute_cdf_by_size

    gaps = np.asarray(gaps)
    sizes = np.asarray(sizes)

    if size_bins is None:
        size_bins = sorted(set(sizes))

    cdfs = compute_cdf_by_size(gaps, sizes, size_bins)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(size_bins)))

    for size, color in zip(size_bins, colors, strict=False):
        cdf_data = cdfs[size]
        if len(cdf_data["x"]) == 0:
            continue
        ax.plot(
            cdf_data["x"],
            cdf_data["cdf"],
            marker="o",
            linewidth=2,
            color=color,
            label=f"n={size}",
            markersize=4,
        )

    ax.set_xlabel("Optimality Gap (%)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_gap_percentiles_by_size(
    stats_by_size: dict,
    title: str = "Gap Percentiles vs Problem Size",
    save_path: str | Path | None = None,
):
    """
    Plot gap percentiles (p50, p90, p95, p99) vs problem size.

    Args:
        stats_by_size: Dictionary from compute_gap_statistics_by_size()
        title: Plot title
        save_path: Path to save figure

    Example:
        >>> from knapsack_gnn.analysis.stats import compute_gap_statistics_by_size
        >>> stats = compute_gap_statistics_by_size(gaps, sizes)
        >>> plot_gap_percentiles_by_size(stats, save_path="percentiles.png")
    """
    sizes = sorted(stats_by_size.keys())
    p50 = [stats_by_size[s]["p50"] for s in sizes if stats_by_size[s]["p50"] is not None]
    p90 = [stats_by_size[s]["p90"] for s in sizes if stats_by_size[s]["p90"] is not None]
    p95 = [stats_by_size[s]["p95"] for s in sizes if stats_by_size[s]["p95"] is not None]
    p99 = [stats_by_size[s]["p99"] for s in sizes if stats_by_size[s]["p99"] is not None]
    mean_vals = [stats_by_size[s]["mean"] for s in sizes if stats_by_size[s]["mean"] is not None]

    # Filter sizes to match available data
    valid_sizes = [s for s in sizes if stats_by_size[s]["p50"] is not None]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(valid_sizes, mean_vals, "o-", linewidth=2, markersize=8, label="Mean", color="#2E86AB")
    ax.plot(
        valid_sizes, p50, "s-", linewidth=2, markersize=6, label="p50 (Median)", color="#A23B72"
    )
    ax.plot(valid_sizes, p90, "^-", linewidth=2, markersize=6, label="p90", color="#F18F01")
    ax.plot(valid_sizes, p95, "d-", linewidth=2, markersize=6, label="p95", color="#C73E1D")
    ax.plot(valid_sizes, p99, "v-", linewidth=2, markersize=6, label="p99", color="#6A994E")

    ax.set_xlabel("Problem Size (# items)", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def plot_gap_violin_by_size(
    gaps: list[float],
    sizes: list[int],
    size_bins: list[int] | None = None,
    title: str = "Gap Distribution by Problem Size",
    save_path: str | Path | None = None,
):
    """
    Plot violin plots of gap distribution grouped by problem size.

    Args:
        gaps: List of optimality gap percentages
        sizes: List of problem sizes corresponding to gaps
        size_bins: Optional list of size bins (default: unique sizes)
        title: Plot title
        save_path: Path to save figure

    Example:
        >>> gaps = [0.05, 0.12, 0.03, 0.08, 0.15, 0.02, 0.10]
        >>> sizes = [10, 10, 10, 25, 25, 50, 50]
        >>> plot_gap_violin_by_size(gaps, sizes, save_path="violin.png")
    """
    gaps = np.asarray(gaps)
    sizes = np.asarray(sizes)

    if size_bins is None:
        size_bins = sorted(set(sizes))

    # Group gaps by size
    gaps_by_size = []
    labels = []
    for size in size_bins:
        mask = sizes == size
        size_gaps = gaps[mask]
        if len(size_gaps) > 0:
            gaps_by_size.append(size_gaps)
            labels.append(f"n={size}")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Violin plot
    parts = ax.violinplot(
        gaps_by_size,
        positions=range(len(gaps_by_size)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    # Customize colors
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(gaps_by_size)))
    for pc, color in zip(parts["bodies"], colors, strict=False):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig
