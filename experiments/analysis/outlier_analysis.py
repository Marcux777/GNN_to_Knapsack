"""
Outlier Analysis Script for Knapsack GNN
Investigates instances with high optimality gaps to diagnose issues
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import KnapsackSampler
from knapsack_gnn.models.pna import create_model


def load_results(results_path: str) -> dict:
    """Load evaluation results from JSON"""
    with open(results_path) as f:
        results = json.load(f)
    return results


def find_outliers(results: dict, threshold: float = 10.0) -> list:
    """
    Find instances with gap > threshold

    Returns:
        List of (index, gap) tuples
    """
    outliers = []
    for i, gap in enumerate(results["gaps"]):
        if gap > threshold:
            outliers.append((i, gap))
    return sorted(outliers, key=lambda x: x[1], reverse=True)


def analyze_instance(
    instance_idx: int, dataset, model, sampler, strategy: str = "sampling", device: str = "cpu"
) -> dict:
    """
    Perform detailed analysis of a single instance

    Returns:
        Dictionary with analysis results
    """
    data = dataset[instance_idx]

    # Get model predictions
    probs = sampler.get_probabilities(data)
    result = sampler.solve(data, strategy=strategy, n_samples=100)

    # Extract instance info
    weights = data.item_weights.numpy()
    values = data.item_values.numpy()
    capacity = data.capacity
    optimal_solution = data.y.numpy()
    predicted_solution = result["solution"]
    optimal_value = data.optimal_value
    predicted_value = result["value"]

    # Compute metrics
    optimal_weight = np.sum(optimal_solution * weights)
    predicted_weight = np.sum(predicted_solution * weights)

    # Item-level analysis
    value_weight_ratios = values / weights

    # Find swapped items
    different_items = optimal_solution != predicted_solution
    wrong_included = (predicted_solution == 1) & (optimal_solution == 0)
    wrong_excluded = (predicted_solution == 0) & (optimal_solution == 1)

    # Sanity check: recalculate optimal value
    recalculated_optimal = np.sum(optimal_solution * values)

    analysis = {
        "instance_idx": instance_idx,
        "n_items": len(weights),
        "capacity": capacity,
        # Solutions
        "optimal_solution": optimal_solution,
        "predicted_solution": predicted_solution,
        "probabilities": probs.numpy(),
        # Values
        "optimal_value": optimal_value,
        "predicted_value": predicted_value,
        "recalculated_optimal": recalculated_optimal,
        "value_difference": optimal_value - predicted_value,
        "gap": result.get("optimality_gap", 0),
        # Weights
        "optimal_weight": optimal_weight,
        "predicted_weight": predicted_weight,
        "weight_difference": optimal_weight - predicted_weight,
        # Capacity usage
        "optimal_capacity_usage": optimal_weight / capacity,
        "predicted_capacity_usage": predicted_weight / capacity,
        # Item analysis
        "weights": weights,
        "values": values,
        "value_weight_ratios": value_weight_ratios,
        # Errors
        "n_different_items": np.sum(different_items),
        "n_wrong_included": np.sum(wrong_included),
        "n_wrong_excluded": np.sum(wrong_excluded),
        "wrong_included_indices": np.where(wrong_included)[0],
        "wrong_excluded_indices": np.where(wrong_excluded)[0],
        # Sanity checks
        "optimal_value_matches": np.isclose(optimal_value, recalculated_optimal),
        "predicted_is_feasible": predicted_weight <= capacity,
        "optimal_is_feasible": optimal_weight <= capacity,
        # Strategy info
        "strategy": strategy,
        "n_feasible_samples": result.get("n_feasible_samples", None),
    }

    return analysis


def print_analysis(analysis: dict):
    """Print formatted analysis"""
    print("\n" + "=" * 80)
    print(f"INSTANCE #{analysis['instance_idx']} ANALYSIS")
    print("=" * 80)

    print("\n📊 Instance Characteristics:")
    print(f"  Items: {analysis['n_items']}")
    print(f"  Capacity: {analysis['capacity']}")
    print(
        f"  Value/Weight ratio range: [{np.min(analysis['value_weight_ratios']):.2f}, {np.max(analysis['value_weight_ratios']):.2f}]"
    )

    print("\n🎯 Performance:")
    print(f"  Optimality Gap: {analysis['gap']:.2f}%")
    print(f"  Optimal Value: {analysis['optimal_value']}")
    print(f"  Predicted Value: {analysis['predicted_value']}")
    print(f"  Value Difference: {analysis['value_difference']}")

    print("\n⚖️ Capacity Usage:")
    print(
        f"  Optimal: {analysis['optimal_weight']:.0f}/{analysis['capacity']} ({analysis['optimal_capacity_usage'] * 100:.1f}%)"
    )
    print(
        f"  Predicted: {analysis['predicted_weight']:.0f}/{analysis['capacity']} ({analysis['predicted_capacity_usage'] * 100:.1f}%)"
    )
    print(f"  Weight Difference: {analysis['weight_difference']:.0f}")

    print("\n🔄 Item Selection Errors:")
    print(f"  Total Different Items: {analysis['n_different_items']}")
    print(f"  Wrong Included (should be 0): {analysis['n_wrong_included']}")
    print(f"  Wrong Excluded (should be 1): {analysis['n_wrong_excluded']}")

    if analysis["n_wrong_included"] > 0:
        print(f"\n  Items wrongly included: {analysis['wrong_included_indices']}")
        for idx in analysis["wrong_included_indices"]:
            print(
                f"    Item {idx}: w={analysis['weights'][idx]:.0f}, v={analysis['values'][idx]:.0f}, "
                + f"ratio={analysis['value_weight_ratios'][idx]:.2f}, prob={analysis['probabilities'][idx]:.3f}"
            )

    if analysis["n_wrong_excluded"] > 0:
        print(f"\n  Items wrongly excluded: {analysis['wrong_excluded_indices']}")
        for idx in analysis["wrong_excluded_indices"]:
            print(
                f"    Item {idx}: w={analysis['weights'][idx]:.0f}, v={analysis['values'][idx]:.0f}, "
                + f"ratio={analysis['value_weight_ratios'][idx]:.2f}, prob={analysis['probabilities'][idx]:.3f}"
            )

    print("\n✅ Sanity Checks:")
    print(f"  Optimal value matches recalculation: {analysis['optimal_value_matches']}")
    if not analysis["optimal_value_matches"]:
        print(
            f"    ⚠️ MISMATCH: Stored={analysis['optimal_value']}, Recalculated={analysis['recalculated_optimal']}"
        )
    print(f"  Predicted solution is feasible: {analysis['predicted_is_feasible']}")
    print(f"  Optimal solution is feasible: {analysis['optimal_is_feasible']}")

    if analysis["n_feasible_samples"] is not None:
        print("\n🎲 Sampling Info:")
        print(f"  Strategy: {analysis['strategy']}")
        print(f"  Feasible samples: {analysis['n_feasible_samples']}/100")

    print("=" * 80)


def visualize_instance(analysis: dict, save_path: str = None):
    """Create comprehensive visualization of instance"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    n_items = analysis["n_items"]
    x = np.arange(n_items)

    # 1. Solution comparison
    ax1 = fig.add_subplot(gs[0, :2])
    width = 0.35
    ax1.bar(
        x - width / 2,
        analysis["optimal_solution"],
        width,
        label="Optimal",
        alpha=0.7,
        color="green",
    )
    ax1.bar(
        x + width / 2,
        analysis["predicted_solution"],
        width,
        label="Predicted",
        alpha=0.7,
        color="blue",
    )
    ax1.set_xlabel("Item Index")
    ax1.set_ylabel("Selected (1) / Not Selected (0)")
    ax1.set_title(f"Solution Comparison (Gap: {analysis['gap']:.2f}%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Probabilities
    ax2 = fig.add_subplot(gs[1, :2])
    colors = ["green" if o == 1 else "red" for o in analysis["optimal_solution"]]
    ax2.bar(x, analysis["probabilities"], color=colors, alpha=0.6)
    ax2.axhline(0.5, color="black", linestyle="--", label="Threshold", linewidth=1)
    ax2.set_xlabel("Item Index")
    ax2.set_ylabel("Selection Probability")
    ax2.set_title("Model Probabilities (Green=Should Select, Red=Should Not)")
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Weight vs Value scatter
    ax3 = fig.add_subplot(gs[2, :2])
    # Color code: correct decisions, wrong inclusion, wrong exclusion
    colors_scatter = []
    for i in range(n_items):
        if analysis["optimal_solution"][i] == analysis["predicted_solution"][i]:
            colors_scatter.append("green")  # Correct
        elif analysis["predicted_solution"][i] == 1:
            colors_scatter.append("red")  # Wrong inclusion
        else:
            colors_scatter.append("orange")  # Wrong exclusion

    ax3.scatter(analysis["weights"], analysis["values"], c=colors_scatter, s=100, alpha=0.6)
    ax3.set_xlabel("Weight")
    ax3.set_ylabel("Value")
    ax3.set_title("Items (Green=Correct, Red=Wrong Include, Orange=Wrong Exclude)")
    ax3.grid(True, alpha=0.3)

    # 4. Probability distribution
    ax4 = fig.add_subplot(gs[0, 2])
    ax4.hist(analysis["probabilities"], bins=20, edgecolor="black", alpha=0.7)
    ax4.axvline(0.5, color="red", linestyle="--", label="Threshold")
    ax4.set_xlabel("Probability")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Probability Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Value/Weight ratios
    ax5 = fig.add_subplot(gs[1, 2])
    colors_ratio = ["green" if o == 1 else "gray" for o in analysis["optimal_solution"]]
    ax5.bar(x, analysis["value_weight_ratios"], color=colors_ratio, alpha=0.6)
    ax5.set_xlabel("Item Index")
    ax5.set_ylabel("Value/Weight Ratio")
    ax5.set_title("Item Efficiency (Green=Optimal Selection)")
    ax5.grid(True, alpha=0.3)

    # 6. Summary metrics
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis("off")

    summary_text = f"""
SUMMARY METRICS

Gap: {analysis["gap"]:.2f}%
Optimal: {analysis["optimal_value"]}
Predicted: {analysis["predicted_value"]}

Capacity: {analysis["capacity"]}
Optimal Weight: {analysis["optimal_weight"]:.0f}
Predicted Weight: {analysis["predicted_weight"]:.0f}

Items Different: {analysis["n_different_items"]}/{n_items}
Wrong Included: {analysis["n_wrong_included"]}
Wrong Excluded: {analysis["n_wrong_excluded"]}

Feasibility:
  Optimal: {"✓" if analysis["optimal_is_feasible"] else "✗"}
  Predicted: {"✓" if analysis["predicted_is_feasible"] else "✗"}

Value Check:
  {"✓" if analysis["optimal_value_matches"] else "⚠️ MISMATCH"}
    """

    ax6.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
        transform=ax6.transAxes,
    )

    fig.suptitle(
        f"Outlier Instance #{analysis['instance_idx']} - Detailed Analysis",
        fontsize=14,
        fontweight="bold",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Visualization saved to {save_path}")

    return fig


def compare_with_normal(
    outlier_analysis: dict, normal_indices: list, dataset, sampler, strategy: str = "sampling"
) -> dict:
    """Compare outlier with normal instances"""

    print("\n" + "=" * 80)
    print("COMPARISON WITH NORMAL INSTANCES")
    print("=" * 80)

    normal_analyses = []
    for idx in normal_indices[:5]:  # Max 5 for comparison
        analysis = analyze_instance(idx, dataset, None, sampler, strategy)
        normal_analyses.append(analysis)

    # Aggregate statistics
    outlier_probs = outlier_analysis["probabilities"]
    normal_probs = [a["probabilities"] for a in normal_analyses]

    print("\n📈 Probability Statistics:")
    print(f"  Outlier mean prob: {np.mean(outlier_probs):.3f}")
    print(f"  Normal mean prob: {np.mean([np.mean(p) for p in normal_probs]):.3f}")
    print(f"  Outlier prob std: {np.std(outlier_probs):.3f}")
    print(f"  Normal prob std: {np.mean([np.std(p) for p in normal_probs]):.3f}")

    print("\n⚖️ Capacity Usage:")
    print(f"  Outlier: {outlier_analysis['optimal_capacity_usage'] * 100:.1f}%")
    print(
        f"  Normal avg: {np.mean([a['optimal_capacity_usage'] for a in normal_analyses]) * 100:.1f}%"
    )

    print("\n🎲 Item Selection Errors:")
    print(f"  Outlier: {outlier_analysis['n_different_items']} items")
    print(f"  Normal avg: {np.mean([a['n_different_items'] for a in normal_analyses]):.1f} items")

    return {"outlier": outlier_analysis, "normal": normal_analyses}


def main():
    parser = argparse.ArgumentParser(description="Analyze Knapsack GNN Outliers")

    parser.add_argument(
        "--results_json", type=str, required=True, help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/datasets", help="Path to datasets directory"
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=10.0,
        help="Gap threshold for outlier detection (default: 10.0)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive"],
        help="Inference strategy (default: sampling)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outlier_analysis",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("KNAPSACK GNN OUTLIER ANALYSIS")
    print("=" * 80)

    # Load results
    print(f"\n1. Loading results from {args.results_json}...")
    results = load_results(args.results_json)

    # Find outliers
    print(f"\n2. Finding outliers (gap > {args.gap_threshold}%)...")
    outliers = find_outliers(results, args.gap_threshold)

    print(f"\nFound {len(outliers)} outlier(s):")
    for idx, gap in outliers:
        print(f"  Instance #{idx}: Gap = {gap:.2f}%")

    if len(outliers) == 0:
        print("\n✅ No outliers found! All instances have gap <", args.gap_threshold, "%")
        return

    # Load dataset and model
    print(f"\n3. Loading dataset from {args.data_dir}...")
    test_dataset_raw = KnapsackDataset.load(f"{args.data_dir}/test.pkl")
    test_dataset = KnapsackGraphDataset(test_dataset_raw, normalize_features=True)

    print(f"\n4. Loading model from {args.checkpoint_dir}...")
    train_dataset_raw = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    train_dataset = KnapsackGraphDataset(train_dataset_raw, normalize_features=True)

    model = create_model(train_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    checkpoint = torch.load(f"{args.checkpoint_dir}/best_model.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    sampler = KnapsackSampler(model, device=args.device)

    # Analyze each outlier
    print("\n5. Analyzing outliers...")
    for outlier_idx, outlier_gap in outliers:
        print(f"\n{'=' * 80}")
        print(f"Analyzing Instance #{outlier_idx} (Gap: {outlier_gap:.2f}%)")
        print(f"{'=' * 80}")

        analysis = analyze_instance(
            outlier_idx, test_dataset, model, sampler, args.strategy, args.device
        )
        print_analysis(analysis)

        # Visualize
        viz_path = f"{args.output_dir}/outlier_instance_{outlier_idx}.png"
        visualize_instance(analysis, save_path=viz_path)

        # Compare with normal instances
        normal_indices = [i for i, gap in enumerate(results["gaps"]) if gap < 2.0]
        if normal_indices:
            compare_with_normal(analysis, normal_indices, test_dataset, sampler, args.strategy)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
