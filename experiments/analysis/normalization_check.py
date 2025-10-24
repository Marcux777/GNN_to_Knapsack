"""
Normalization and Size Invariance Check

Diagnostic script to verify that the GNN model is properly normalized
and exhibits size invariance properties.

Checks:
1. Feature normalization in graph_builder.py
2. Degree histogram across different problem sizes
3. PNA aggregator activation statistics by size
4. Gap variance consistency across sizes

Goal: Ensure std(gap) is similar across sizes, no saturation in small sizes.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from knapsack_gnn.data.generator import KnapsackDataset, KnapsackGenerator, KnapsackSolver
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sns.set_style("whitegrid")


def check_feature_normalization(dataset: KnapsackGraphDataset, output_dir: Path):
    """
    Check feature normalization statistics.

    Verifies:
    - Item weights normalized by capacity
    - Item values have reasonable scale
    - No extreme outliers
    """
    print("=" * 80)
    print("FEATURE NORMALIZATION CHECK")
    print("=" * 80)

    all_item_weights = []
    all_item_values = []
    all_capacities = []

    for data in dataset:
        all_item_weights.append(data.item_weights.numpy())
        all_item_values.append(data.item_values.numpy())
        all_capacities.append(float(data.capacity))

    item_weights = np.concatenate(all_item_weights)
    item_values = np.concatenate(all_item_values)
    capacities = np.array(all_capacities)

    print(f"\nFeature Statistics (n={len(item_weights)} items):")
    print(
        f"  Item weights: μ={np.mean(item_weights):.4f}, σ={np.std(item_weights):.4f}, "
        f"min={np.min(item_weights):.4f}, max={np.max(item_weights):.4f}"
    )
    print(
        f"  Item values:  μ={np.mean(item_values):.4f}, σ={np.std(item_values):.4f}, "
        f"min={np.min(item_values):.4f}, max={np.max(item_values):.4f}"
    )
    print(
        f"  Capacities:   μ={np.mean(capacities):.4f}, σ={np.std(capacities):.4f}, "
        f"min={np.min(capacities):.4f}, max={np.max(capacities):.4f}"
    )

    # Check if weights are normalized by capacity
    print("\nNormalization Check:")
    if np.max(item_weights) <= 1.0 and np.min(item_weights) >= 0.0:
        print("  ✓ Item weights appear normalized to [0, 1] (likely by capacity)")
    else:
        print("  ✗ Item weights NOT normalized to [0, 1]")
        print(f"    Range: [{np.min(item_weights):.4f}, {np.max(item_weights):.4f}]")

    # Plot histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(item_weights, bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("Normalized Weight")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Item Weights Distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(item_values, bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Item Values Distribution")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(capacities, bins=30, alpha=0.7, edgecolor="black")
    axes[2].set_xlabel("Capacity")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Capacities Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "feature_normalization.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)

    # Save statistics
    stats = {
        "item_weights": {
            "mean": float(np.mean(item_weights)),
            "std": float(np.std(item_weights)),
            "min": float(np.min(item_weights)),
            "max": float(np.max(item_weights)),
        },
        "item_values": {
            "mean": float(np.mean(item_values)),
            "std": float(np.std(item_values)),
            "min": float(np.min(item_values)),
            "max": float(np.max(item_values)),
        },
        "capacities": {
            "mean": float(np.mean(capacities)),
            "std": float(np.std(capacities)),
            "min": float(np.min(capacities)),
            "max": float(np.max(capacities)),
        },
        "normalized_to_01": bool(np.max(item_weights) <= 1.0 and np.min(item_weights) >= 0.0),
    }

    stats_path = output_dir / "feature_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("=" * 80)

    return stats


def check_degree_histogram(datasets_by_size: dict, output_dir: Path):
    """
    Check degree histogram across different problem sizes.

    For knapsack graphs (bipartite: items ↔ capacity node):
    - Item nodes have degree 1 (connected to capacity)
    - Capacity node has degree = n_items
    """
    print("\n" + "=" * 80)
    print("DEGREE HISTOGRAM CHECK")
    print("=" * 80)

    degree_by_size = {}

    for size, dataset in datasets_by_size.items():
        degrees = []
        for data in dataset:
            # PyG edge_index is [2, num_edges]
            edge_index = data.edge_index
            n_nodes = data.x.size(0)

            # Count degree for each node
            node_degrees = torch.zeros(n_nodes, dtype=torch.long)
            for i in range(edge_index.size(1)):
                src = edge_index[0, i].item()
                dst = edge_index[1, i].item()
                node_degrees[src] += 1
                node_degrees[dst] += 1

            degrees.extend(node_degrees.numpy().tolist())

        degree_by_size[size] = degrees

    # Plot degree distribution by size
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(degree_by_size)))

    for (size, degrees), color in zip(sorted(degree_by_size.items()), colors, strict=False):
        unique_degrees, counts = np.unique(degrees, return_counts=True)
        ax.bar(
            unique_degrees + (size - min(degree_by_size.keys())) * 0.1,
            counts,
            width=0.8,
            alpha=0.6,
            label=f"n={size}",
            color=color,
        )

    ax.set_xlabel("Node Degree", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("Degree Distribution by Problem Size", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = output_dir / "degree_histogram.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)

    # Statistics
    print("\nDegree Statistics by Size:")
    for size in sorted(degree_by_size.keys()):
        degrees = np.array(degree_by_size[size])
        print(
            f"  Size {size}: mean={np.mean(degrees):.2f}, "
            f"std={np.std(degrees):.2f}, "
            f"min={np.min(degrees)}, max={np.max(degrees)}"
        )

    print("=" * 80)

    return degree_by_size


def check_aggregator_activations(
    model,
    datasets_by_size: dict,
    device: str,
    output_dir: Path,
):
    """
    Check PNA aggregator activations by problem size.

    Verifies that aggregators don't saturate on small instances.
    """
    print("\n" + "=" * 80)
    print("PNA AGGREGATOR ACTIVATION CHECK")
    print("=" * 80)

    model = model.to(device)
    model.eval()

    activations_by_size = defaultdict(list)

    with torch.inference_mode():
        for size, dataset in datasets_by_size.items():
            print(f"Processing size {size}...")

            for data in dataset[:10]:  # Sample 10 instances per size
                data = data.to(device)

                # Forward pass
                output = model(data)

                # Get activations (simplified - just output statistics)
                activations_by_size[size].append(output.cpu().numpy())

    # Plot activation statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sizes = sorted(activations_by_size.keys())

    # Mean activation
    means = []
    stds = []
    mins = []
    maxs = []

    for size in sizes:
        acts = np.concatenate(activations_by_size[size])
        means.append(np.mean(acts))
        stds.append(np.std(acts))
        mins.append(np.min(acts))
        maxs.append(np.max(acts))

    # Plot 1: Mean activation
    axes[0, 0].plot(sizes, means, "o-", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Problem Size")
    axes[0, 0].set_ylabel("Mean Activation")
    axes[0, 0].set_title("Mean Output Activation by Size")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Std activation
    axes[0, 1].plot(sizes, stds, "s-", linewidth=2, markersize=8, color="orange")
    axes[0, 1].set_xlabel("Problem Size")
    axes[0, 1].set_ylabel("Std Activation")
    axes[0, 1].set_title("Activation Std by Size")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Min/Max range
    axes[1, 0].plot(sizes, mins, "v-", linewidth=2, markersize=6, label="Min", color="blue")
    axes[1, 0].plot(sizes, maxs, "^-", linewidth=2, markersize=6, label="Max", color="red")
    axes[1, 0].fill_between(sizes, mins, maxs, alpha=0.2)
    axes[1, 0].set_xlabel("Problem Size")
    axes[1, 0].set_ylabel("Activation")
    axes[1, 0].set_title("Activation Range by Size")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Distribution violin
    acts_list = [np.concatenate(activations_by_size[s]).flatten() for s in sizes]
    axes[1, 1].violinplot(acts_list, positions=range(len(sizes)), showmeans=True)
    axes[1, 1].set_xticks(range(len(sizes)))
    axes[1, 1].set_xticklabels([f"n={s}" for s in sizes])
    axes[1, 1].set_ylabel("Activation")
    axes[1, 1].set_title("Activation Distribution by Size")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = output_dir / "aggregator_activations.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)

    # Check for saturation
    print("\nSaturation Check:")
    for size in sizes:
        acts = np.concatenate(activations_by_size[size]).flatten()
        # Check if activations are near 0 or 1 (sigmoid saturation)
        near_zero = np.mean(acts < 0.1)
        near_one = np.mean(acts > 0.9)
        print(f"  Size {size}: {near_zero * 100:.1f}% near 0, {near_one * 100:.1f}% near 1")

        if near_zero > 0.5 or near_one > 0.5:
            print("    ⚠️  WARNING: Potential saturation detected")
        else:
            print("    ✓  No saturation")

    print("=" * 80)


def check_gap_variance_consistency(datasets_by_size: dict, model, device: str, output_dir: Path):
    """
    Check if gap variance is consistent across sizes.

    Goal: std(gap) should be similar across sizes (size invariance).
    """
    print("\n" + "=" * 80)
    print("GAP VARIANCE CONSISTENCY CHECK")
    print("=" * 80)

    from knapsack_gnn.decoding.sampling import KnapsackSampler

    sampler = KnapsackSampler(model, device)

    gap_stats_by_size = {}

    for size, dataset in datasets_by_size.items():
        print(f"Evaluating size {size}...")

        gaps = []
        for data in dataset[:20]:  # Sample 20 instances per size
            result = sampler.solve(data, strategy="sampling", n_samples=64)
            if "optimality_gap" in result:
                gaps.append(result["optimality_gap"])

        if gaps:
            gap_stats_by_size[size] = {
                "mean": float(np.mean(gaps)),
                "std": float(np.std(gaps)),
                "min": float(np.min(gaps)),
                "max": float(np.max(gaps)),
                "count": len(gaps),
            }

    # Plot variance by size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sizes = sorted(gap_stats_by_size.keys())
    means = [gap_stats_by_size[s]["mean"] for s in sizes]
    stds = [gap_stats_by_size[s]["std"] for s in sizes]

    # Mean gap
    axes[0].plot(sizes, means, "o-", linewidth=2, markersize=8, color="#2E86AB")
    axes[0].set_xlabel("Problem Size", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean Gap (%)", fontsize=12, fontweight="bold")
    axes[0].set_title("Mean Gap by Size", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Std gap
    axes[1].plot(sizes, stds, "s-", linewidth=2, markersize=8, color="#F18F01")
    axes[1].axhline(
        y=np.mean(stds),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean Std = {np.mean(stds):.2f}%",
        alpha=0.7,
    )
    axes[1].set_xlabel("Problem Size", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Std Gap (%)", fontsize=12, fontweight="bold")
    axes[1].set_title("Gap Variance by Size (Size Invariance)", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "gap_variance_by_size.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)

    # Consistency check
    print("\nVariance Consistency:")
    mean_std = np.mean(stds)
    for size in sizes:
        std_gap = gap_stats_by_size[size]["std"]
        deviation = abs(std_gap - mean_std) / mean_std * 100
        status = "✓" if deviation < 50 else "⚠️"
        print(f"  Size {size}: std={std_gap:.2f}%, deviation from mean={deviation:.1f}% {status}")

    # Save stats
    stats_path = output_dir / "gap_variance_stats.json"
    with open(stats_path, "w") as f:
        json.dump(gap_stats_by_size, f, indent=2)

    print("=" * 80)

    return gap_stats_by_size


def main():
    parser = argparse.ArgumentParser(description="Normalization and size invariance check")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--n-instances-per-size", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NORMALIZATION AND SIZE INVARIANCE CHECK")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Sizes: {args.sizes}")
    print(f"Instances per size: {args.n_instances_per_size}")
    print("=" * 80)
    print()

    # Generate datasets for different sizes
    print("Generating datasets...")
    generator = KnapsackGenerator(seed=args.seed)
    datasets_by_size = {}

    for size in args.sizes:
        instances = generator.generate_dataset(
            n_instances=args.n_instances_per_size,
            n_items_range=(size, size),
        )
        instances = KnapsackSolver.solve_batch(instances, verbose=False)
        dataset = KnapsackDataset(instances)
        datasets_by_size[size] = KnapsackGraphDataset(dataset, normalize_features=True)

    print("Datasets generated\n")

    # Load model
    print("Loading model...")
    checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
    sample_dataset = datasets_by_size[args.sizes[0]]

    model = create_model(
        dataset=sample_dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    state = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print("Model loaded\n")

    # Run checks
    check_feature_normalization(sample_dataset, output_dir)
    check_degree_histogram(datasets_by_size, output_dir)
    check_aggregator_activations(model, datasets_by_size, args.device, output_dir)
    check_gap_variance_consistency(datasets_by_size, model, args.device, output_dir)

    print("\n" + "=" * 80)
    print("NORMALIZATION CHECK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
