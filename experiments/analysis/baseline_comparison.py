"""
Baseline Comparison Script
Compares GNN, Greedy, Random, and OR-Tools on the same test set
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knapsack_gnn.baselines.greedy import GreedySolver, RandomSolver
from knapsack_gnn.data.generator import KnapsackDataset, KnapsackGenerator, KnapsackSolver
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import KnapsackSampler
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.metrics import save_results_to_json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Compare baseline methods")

    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory containing trained GNN model"
    )
    parser.add_argument(
        "--checkpoint_name", type=str, default="best_model.pt", help="Checkpoint filename"
    )

    # Test data parameters
    parser.add_argument(
        "--data_dir", type=str, default="data/datasets", help="Directory containing test dataset"
    )
    parser.add_argument(
        "--test_size", type=int, default=100, help="Number of test instances (default: 100)"
    )
    parser.add_argument("--n_items_min", type=int, default=10, help="Minimum items per instance")
    parser.add_argument("--n_items_max", type=int, default=50, help="Maximum items per instance")
    parser.add_argument(
        "--generate_test",
        action="store_true",
        help="Generate new test set instead of using existing",
    )

    # GNN inference parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive"],
        help="GNN inference strategy",
    )
    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of samples for GNN sampling strategy"
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    # Random baseline parameters
    parser.add_argument(
        "--random_attempts", type=int, default=100, help="Attempts for random baseline"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="baselines", help="Output directory for results"
    )
    parser.add_argument("--visualize", action="store_true", help="Generate comparison plots")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    parser.add_argument("--seed", type=int, default=999, help="Random seed for reproducibility")

    return parser.parse_args()


def load_or_generate_test_data(args):
    """Load existing test set or generate new one"""
    if args.generate_test:
        print(f"Generating new test set ({args.test_size} instances)...")
        generator = KnapsackGenerator(seed=args.seed)  # Use provided seed
        instances = generator.generate_dataset(
            n_instances=args.test_size, n_items_range=(args.n_items_min, args.n_items_max)
        )
        print("Solving instances with OR-Tools...")
        instances = KnapsackSolver.solve_batch(instances, verbose=True)
        test_dataset = KnapsackDataset(instances)
    else:
        print(f"Loading test set from {args.data_dir}/test.pkl...")
        test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")
        # Limit to test_size
        if len(test_dataset) > args.test_size:
            test_dataset.instances = test_dataset.instances[: args.test_size]

    print(f"Test set: {len(test_dataset)} instances")
    return test_dataset


def evaluate_gnn(model, test_graph_dataset, args) -> dict:
    """Evaluate GNN model"""
    print("\n" + "=" * 70)
    print("EVALUATING GNN (PNA)")
    print("=" * 70)

    sampler = KnapsackSampler(model, device=args.device)

    results = []
    times = []

    for i, data in enumerate(test_graph_dataset):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(test_graph_dataset)} instances")

        start = time.perf_counter()
        result = sampler.solve(
            data, strategy=args.strategy, n_samples=args.n_samples, temperature=args.temperature
        )
        elapsed = time.perf_counter() - start

        times.append(elapsed)

        # Handle optimal_value (can be int or tensor)
        opt_val = data.optimal_value
        if torch.is_tensor(opt_val):
            opt_val = opt_val.item()

        results.append(
            {
                "value": result["value"],
                "optimality_gap": result.get("optimality_gap", 0),
                "is_feasible": result["is_feasible"],
                "solve_time": elapsed,
                "optimal_value": opt_val,
            }
        )

    # Aggregate statistics
    gaps = [r["optimality_gap"] for r in results]
    stats = {
        "method": f"GNN-PNA ({args.strategy})",
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "std_gap": float(np.std(gaps)),
        "max_gap": float(np.max(gaps)),
        "min_gap": float(np.min(gaps)),
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "feasibility_rate": float(np.mean([r["is_feasible"] for r in results])),
        "throughput": len(results) / np.sum(times),
        "results": results,
    }

    print("\nGNN Results:")
    print(f"  Mean Gap: {stats['mean_gap']:.2f}%")
    print(f"  Median Gap: {stats['median_gap']:.2f}%")
    print(f"  Mean Time: {stats['mean_time'] * 1000:.2f} ms")
    print(f"  Throughput: {stats['throughput']:.1f} inst/s")

    return stats


def evaluate_greedy(test_dataset) -> dict:
    """Evaluate greedy heuristic"""
    print("\n" + "=" * 70)
    print("EVALUATING GREEDY HEURISTIC")
    print("=" * 70)

    solver = GreedySolver()
    results = solver.solve_batch(test_dataset.instances, verbose=True)
    stats = solver.evaluate_results(results)
    stats["method"] = "Greedy"
    stats["results"] = results

    print("\nGreedy Results:")
    print(f"  Mean Gap: {stats['mean_gap']:.2f}%")
    print(f"  Median Gap: {stats['median_gap']:.2f}%")
    print(f"  Mean Time: {stats['mean_time'] * 1000:.4f} ms")
    print(f"  Throughput: {stats['throughput']:.1f} inst/s")

    return stats


def evaluate_random(test_dataset, max_attempts: int = 100, seed: int = 42) -> dict:
    """Evaluate random baseline"""
    print("\n" + "=" * 70)
    print("EVALUATING RANDOM BASELINE")
    print("=" * 70)

    solver = RandomSolver(seed=seed)
    results = []

    for i, instance in enumerate(test_dataset.instances):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(test_dataset)} instances")
        result = solver.solve(instance, max_attempts=max_attempts)
        results.append(result)

    # Aggregate statistics
    gaps = [r["optimality_gap"] for r in results if r["optimality_gap"] is not None]
    times = [r["solve_time"] for r in results]

    stats = {
        "method": "Random",
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "std_gap": float(np.std(gaps)),
        "max_gap": float(np.max(gaps)),
        "min_gap": float(np.min(gaps)),
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "feasibility_rate": float(np.mean([r["is_feasible"] for r in results])),
        "throughput": len(results) / np.sum(times),
        "results": results,
    }

    print("\nRandom Results:")
    print(f"  Mean Gap: {stats['mean_gap']:.2f}%")
    print(f"  Median Gap: {stats['median_gap']:.2f}%")
    print(f"  Mean Time: {stats['mean_time'] * 1000:.2f} ms")

    return stats


def get_ortools_stats(test_dataset) -> dict:
    """Get OR-Tools statistics (already solved)"""
    print("\n" + "=" * 70)
    print("OR-TOOLS STATISTICS (Exact Solver)")
    print("=" * 70)

    times = [inst.solve_time for inst in test_dataset.instances if inst.solve_time is not None]

    if not times:
        print("  Warning: No solve times available")
        return None

    stats = {
        "method": "OR-Tools (Exact)",
        "mean_gap": 0.0,  # Always optimal
        "median_gap": 0.0,
        "std_gap": 0.0,
        "max_gap": 0.0,
        "min_gap": 0.0,
        "mean_time": float(np.mean(times)),
        "median_time": float(np.median(times)),
        "feasibility_rate": 1.0,
        "throughput": len(times) / np.sum(times),
    }

    print("\nOR-Tools Results:")
    print(f"  Mean Gap: {stats['mean_gap']:.2f}% (always optimal)")
    print(f"  Mean Time: {stats['mean_time'] * 1000:.2f} ms")
    print(f"  Throughput: {stats['throughput']:.1f} inst/s")

    return stats


def create_comparison_table(all_stats: list[dict]) -> None:
    """Print comparison table"""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON TABLE")
    print("=" * 70)
    print()

    # Header
    print(f"{'Method':<25} | {'Mean Gap':<10} | {'Time (ms)':<12} | {'Speedup':<10} | Throughput")
    print("-" * 75)

    # Get OR-Tools time for speedup calculation
    ortools_time = next(
        (s["mean_time"] for s in all_stats if s["method"] == "OR-Tools (Exact)"), None
    )

    # Print rows
    for stats in all_stats:
        method = stats["method"]
        gap = f"{stats['mean_gap']:.2f}%"
        time_ms = f"{stats['mean_time'] * 1000:.4f}"
        throughput = f"{stats['throughput']:.1f} inst/s"

        if ortools_time and stats["mean_time"] > 0:
            speedup = f"{ortools_time / stats['mean_time']:.1f}x"
        else:
            speedup = "N/A"

        print(f"{method:<25} | {gap:<10} | {time_ms:<12} | {speedup:<10} | {throughput}")

    print()


def plot_comparison(all_stats: list[dict], output_dir: str):
    """Create comparison visualizations"""
    print("\nGenerating comparison plots...")

    # Filter out OR-Tools from some plots (it's the reference)
    methods_to_plot = [s for s in all_stats if "OR-Tools" not in s["method"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Optimality Gap Comparison
    ax = axes[0, 0]
    methods = [s["method"] for s in methods_to_plot]
    gaps = [s["mean_gap"] for s in methods_to_plot]
    colors = ["#2E86AB", "#A23B72", "#F18F01"][: len(methods)]

    ax.bar(methods, gaps, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Optimality Gap (%)", fontsize=12)
    ax.set_title("Solution Quality Comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (_, g) in enumerate(zip(methods, gaps, strict=False)):
        ax.text(i, g + 0.5, f"{g:.2f}%", ha="center", fontsize=10)

    # 2. Solve Time Comparison
    ax = axes[0, 1]
    methods_all = [s["method"] for s in all_stats]
    times = [s["mean_time"] * 1000 for s in all_stats]  # Convert to ms
    colors_all = ["#2E86AB", "#A23B72", "#F18F01", "#6A994E"][: len(methods_all)]

    ax.bar(methods_all, times, color=colors_all, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Time (ms)", fontsize=12)
    ax.set_title("Inference Time Comparison", fontsize=14, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=15)

    # 3. Gap Distribution (Box Plot)
    ax = axes[1, 0]
    gap_distributions = []
    method_labels = []

    for stats in methods_to_plot:
        if "results" in stats:
            gaps_list = [
                r["optimality_gap"] for r in stats["results"] if r.get("optimality_gap") is not None
            ]
            if gaps_list:
                gap_distributions.append(gaps_list)
                method_labels.append(stats["method"])

    if gap_distributions:
        bp = ax.boxplot(gap_distributions, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(gap_distributions)], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Optimality Gap (%)", fontsize=12)
        ax.set_title("Gap Distribution", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # 4. Speed vs Accuracy Trade-off
    ax = axes[1, 1]
    gaps = [s["mean_gap"] for s in methods_to_plot]
    times = [s["mean_time"] * 1000 for s in methods_to_plot]
    methods = [s["method"] for s in methods_to_plot]

    for _, (g, t, m, c) in enumerate(
        zip(gaps, times, methods, colors[: len(methods)], strict=False)
    ):
        ax.scatter(t, g, s=300, alpha=0.7, color=c, edgecolor="black", linewidth=2)
        ax.annotate(
            m,
            (t, g),
            fontsize=10,
            ha="center",
            va="bottom",
            xytext=(0, 10),
            textcoords="offset points",
        )

    ax.set_xlabel("Mean Time (ms)", fontsize=12)
    ax.set_ylabel("Mean Gap (%)", fontsize=12)
    ax.set_title("Speed vs Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Optimal")
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, "baseline_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {save_path}")
    plt.close()


def main():
    """Main comparison pipeline"""
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate test data
    test_dataset = load_or_generate_test_data(args)

    # ===== Evaluate All Methods =====
    all_stats = []

    # 1. OR-Tools (already solved)
    ortools_stats = get_ortools_stats(test_dataset)
    if ortools_stats:
        all_stats.append(ortools_stats)

    # 2. GNN
    print("\nLoading GNN model...")
    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = create_model(train_graph_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    test_graph_dataset = KnapsackGraphDataset(test_dataset, normalize_features=True)
    gnn_stats = evaluate_gnn(model, test_graph_dataset, args)
    all_stats.append(gnn_stats)

    # 3. Greedy
    greedy_stats = evaluate_greedy(test_dataset)
    all_stats.append(greedy_stats)

    # 4. Random
    random_stats = evaluate_random(test_dataset, max_attempts=args.random_attempts, seed=args.seed)
    all_stats.append(random_stats)

    # ===== Comparison =====
    create_comparison_table(all_stats)

    # Save results
    comparison_results = {
        method["method"]: {k: v for k, v in method.items() if k != "results"}
        for method in all_stats
    }

    results_path = os.path.join(args.output_dir, "comparison_results.json")
    save_results_to_json(comparison_results, results_path)
    print(f"\nResults saved to: {results_path}")

    # Visualize
    if args.visualize:
        plot_comparison(all_stats, args.output_dir)

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Findings:")

    # Best accuracy
    best_accuracy = min(
        all_stats, key=lambda x: x["mean_gap"] if "OR-Tools" not in x["method"] else float("inf")
    )
    print(f"  🏆 Best Accuracy: {best_accuracy['method']} ({best_accuracy['mean_gap']:.2f}% gap)")

    # Fastest method
    fastest = min(all_stats, key=lambda x: x["mean_time"])
    print(f"  ⚡ Fastest Method: {fastest['method']} ({fastest['mean_time'] * 1000:.4f} ms)")

    # GNN speedup vs OR-Tools
    if ortools_stats and gnn_stats:
        speedup = ortools_stats["mean_time"] / gnn_stats["mean_time"]
        print(f"  🚀 GNN Speedup vs OR-Tools: {speedup:.1f}x faster")
        print(f"  📊 GNN Accuracy: {gnn_stats['mean_gap']:.2f}% gap from optimal")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
