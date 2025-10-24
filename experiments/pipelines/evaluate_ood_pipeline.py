"""
Out-of-Distribution (OOD) Evaluation Script
Tests model generalization on larger problem instances than seen during training
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from knapsack_gnn.data.generator import KnapsackDataset, KnapsackGenerator, KnapsackSolver
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.eval.reporting import print_evaluation_summary, save_results_to_json
from knapsack_gnn.models.pna import create_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="OOD Evaluation for Knapsack GNN")

    # Model parameters
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory containing trained model"
    )
    parser.add_argument(
        "--checkpoint_name", type=str, default="best_model.pt", help="Checkpoint filename"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/datasets",
        help="Directory containing training dataset (for degree histogram)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 150, 200],
        help="OOD problem sizes to test (default: 100 150 200)",
    )
    parser.add_argument(
        "--n_instances_per_size",
        type=int,
        default=50,
        help="Number of instances per size (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=999, help="Random seed for OOD dataset generation"
    )

    # Inference parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive", "lagrangian"],
        help="Inference strategy",
    )
    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of samples for sampling strategy"
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        "--sampling_schedule",
        type=str,
        default="32,64,128",
        help="Comma-separated sampling batch sizes (default: 32,64,128)",
    )
    parser.add_argument(
        "--sampling_tolerance",
        type=float,
        default=1e-3,
        help="Early-stopping tolerance for sampling (default: 1e-3)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum sampling budget (default: n_samples)"
    )
    parser.add_argument(
        "--lagrangian_iters", type=int, default=30, help="Max iterations for Lagrangian decoder"
    )
    parser.add_argument(
        "--lagrangian_tol", type=float, default=1e-4, help="Tolerance for Lagrangian decoder"
    )
    parser.add_argument(
        "--lagrangian_bias",
        type=float,
        default=0.0,
        help="Probability bias for Lagrangian decoding",
    )
    parser.add_argument(
        "--fix_threshold",
        type=float,
        default=0.9,
        help="Probability threshold to fix variables in warm-start ILP",
    )
    parser.add_argument(
        "--ilp_time_limit", type=float, default=1.0, help="Time limit (seconds) for warm-start ILP"
    )
    parser.add_argument(
        "--max_hint_items",
        type=int,
        default=None,
        help="Maximum number of hint variables for ILP (default: all)",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/evaluation/ood)",
    )
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Set torch.set_num_threads for latency runs"
    )
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile")
    parser.add_argument(
        "--quantize", action="store_true", help="Apply dynamic quantization (CPU only)"
    )
    parser.add_argument(
        "--ilp_threads",
        type=int,
        default=None,
        help="Number of threads for warm-start ILP solver (default: 1)",
    )

    return parser.parse_args()


def parse_schedule(schedule_str: str | None) -> tuple[int, ...] | None:
    """Parse comma-separated sampling batches."""
    if schedule_str is None:
        return None
    values: list[int] = []
    for token in schedule_str.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as err:
            raise argparse.ArgumentTypeError(f"Invalid schedule value: {token}") from err
    return tuple(values) if values else None


def resolve_device(device: str) -> str:
    """Resolve device string with CPU fallback."""
    requested = device.lower()
    if requested in ("auto", "best"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        return "cpu"
    return device


def generate_ood_datasets(
    sizes: list[int], n_instances: int, seed: int
) -> dict[int, KnapsackDataset]:
    """
    Generate OOD test datasets for different problem sizes

    Args:
        sizes: List of problem sizes (number of items)
        n_instances: Number of instances per size
        seed: Random seed

    Returns:
        Dictionary mapping size to KnapsackDataset
    """
    print("=" * 70)
    print("GENERATING OOD TEST DATASETS")
    print("=" * 70)

    generator = KnapsackGenerator(seed=seed)
    datasets = {}

    for size in sizes:
        print(f"\nGenerating {n_instances} instances with {size} items...")
        instances = generator.generate_dataset(
            n_instances=n_instances,
            n_items_range=(size, size),  # Fixed size
        )

        print("Solving instances with OR-Tools...")
        instances = KnapsackSolver.solve_batch(instances, verbose=False)

        dataset = KnapsackDataset(instances)
        datasets[size] = dataset

        # Print statistics
        solve_times = [inst.solve_time for inst in instances if inst.solve_time is not None]
        if solve_times:
            print(
                f"  OR-Tools solve time: {np.mean(solve_times) * 1000:.2f} ± {np.std(solve_times) * 1000:.2f} ms"
            )

    return datasets


def evaluate_ood_size(model, dataset: KnapsackDataset, size: int, args) -> dict:
    """
    Evaluate model on a specific OOD size

    Args:
        model: Trained model
        dataset: KnapsackDataset for this size
        size: Problem size (number of items)
        args: Command-line arguments

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATING SIZE: {size} items")
    print(f"{'=' * 70}")

    # Build graph dataset
    graph_dataset = KnapsackGraphDataset(dataset, normalize_features=True)

    # Evaluate
    strategy_kwargs = {}
    if args.strategy == "sampling":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": args.sampling_schedule,
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    elif args.strategy == "threshold":
        strategy_kwargs = {"threshold": 0.5}
    elif args.strategy == "adaptive":
        strategy_kwargs = {"n_trials": args.n_samples}
    elif args.strategy == "lagrangian":
        strategy_kwargs = {
            "lagrangian_iters": args.lagrangian_iters,
            "lagrangian_tol": args.lagrangian_tol,
            "lagrangian_bias": args.lagrangian_bias,
        }
    elif args.strategy == "warm_start":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": args.sampling_schedule,
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
            "fix_threshold": args.fix_threshold,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_threads": args.ilp_threads,
            "max_hint_items": args.max_hint_items,
        }

    results = evaluate_model(
        model=model,
        dataset=graph_dataset,
        strategy=args.strategy,
        device=args.device,
        sampler_kwargs=args.sampler_kwargs,
        **strategy_kwargs,
    )

    # Add size information
    results["problem_size"] = size
    results["n_instances"] = len(dataset)

    # Print summary
    print_evaluation_summary(results)

    return results


def plot_ood_results(all_results: list[dict], training_size_range: tuple, output_dir: str):
    """
    Create visualization of OOD generalization

    Args:
        all_results: List of result dictionaries for each size
        training_size_range: (min, max) training instance sizes
        output_dir: Output directory for plots
    """
    print("\n" + "=" * 70)
    print("GENERATING OOD VISUALIZATION")
    print("=" * 70)

    sizes = [r["problem_size"] for r in all_results]
    mean_gaps = [r["mean_gap"] for r in all_results]
    median_gaps = [r["median_gap"] for r in all_results]
    std_gaps = [r["std_gap"] for r in all_results]
    mean_times = [r.get("mean_inference_time", 0) * 1000 for r in all_results]  # ms

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Optimality Gap vs Problem Size
    ax = axes[0, 0]
    ax.plot(sizes, mean_gaps, "o-", linewidth=2, markersize=8, color="#2E86AB", label="Mean Gap")
    ax.plot(
        sizes, median_gaps, "s--", linewidth=2, markersize=8, color="#A23B72", label="Median Gap"
    )
    ax.fill_between(
        sizes,
        np.array(mean_gaps) - np.array(std_gaps),
        np.array(mean_gaps) + np.array(std_gaps),
        alpha=0.3,
        color="#2E86AB",
    )

    # Shade training region
    ax.axvspan(
        training_size_range[0],
        training_size_range[1],
        alpha=0.2,
        color="green",
        label="Training Range",
    )

    ax.set_xlabel("Problem Size (# items)", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("OOD Generalization: Gap vs Problem Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Inference Time vs Problem Size
    ax = axes[0, 1]
    ax.plot(sizes, mean_times, "o-", linewidth=2, markersize=8, color="#F18F01")
    ax.axvspan(
        training_size_range[0],
        training_size_range[1],
        alpha=0.2,
        color="green",
        label="Training Range",
    )
    ax.set_xlabel("Problem Size (# items)", fontsize=12)
    ax.set_ylabel("Mean Inference Time (ms)", fontsize=12)
    ax.set_title("Inference Time Scaling", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Gap Distribution for Each Size (Box Plot)
    ax = axes[1, 0]
    gap_distributions = []
    labels = []

    for result in all_results:
        if "gaps" in result:
            gap_distributions.append(result["gaps"])
            labels.append(f"n={result['problem_size']}")

    if gap_distributions:
        bp = ax.boxplot(gap_distributions, labels=labels, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(gap_distributions)))
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xlabel("Problem Size", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("Gap Distribution by Size", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # 4. Feasibility Rate vs Size
    ax = axes[1, 1]
    feasibility_rates = [r["feasibility_rate"] * 100 for r in all_results]
    ax.plot(sizes, feasibility_rates, "o-", linewidth=2, markersize=8, color="#6A994E")
    ax.axhline(y=100, color="green", linestyle="--", linewidth=1, alpha=0.5, label="Perfect")
    ax.axvspan(
        training_size_range[0],
        training_size_range[1],
        alpha=0.2,
        color="green",
        label="Training Range",
    )
    ax.set_xlabel("Problem Size (# items)", fontsize=12)
    ax.set_ylabel("Feasibility Rate (%)", fontsize=12)
    ax.set_title("Solution Feasibility", fontsize=14, fontweight="bold")
    ax.set_ylim([95, 101])
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "ood_generalization.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {save_path}")
    plt.close()


def main():
    """Main OOD evaluation pipeline"""
    args = parse_args()
    args.device = resolve_device(args.device)
    args.sampling_schedule = parse_schedule(args.sampling_schedule)
    if args.max_samples is None:
        args.max_samples = args.n_samples
    args.fix_threshold = min(max(args.fix_threshold, 0.0), 1.0)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    args.sampler_kwargs = {
        "num_threads": args.threads,
        "compile_model": args.compile,
        "quantize": args.quantize,
    }

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("OUT-OF-DISTRIBUTION (OOD) EVALUATION")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "evaluation", "ood")
    os.makedirs(args.output_dir, exist_ok=True)

    # ===== STEP 1: Load Model =====
    print("\n" + "=" * 70)
    print("STEP 1: Loading Trained Model")
    print("=" * 70)

    # Load training dataset for degree histogram
    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)

    # Get training size range
    training_sizes = [inst.n_items for inst in train_dataset.instances]
    training_size_range = (min(training_sizes), max(training_sizes))
    print(f"\nTraining size range: {training_size_range[0]}-{training_size_range[1]} items")

    # Load model
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = create_model(train_graph_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print(f"\nModel loaded from: {checkpoint_path}")
    if "epochs_trained" in checkpoint:
        print(f"  Epochs trained: {checkpoint['epochs_trained']}")

    # ===== STEP 2: Generate OOD Datasets =====
    ood_datasets = generate_ood_datasets(args.sizes, args.n_instances_per_size, args.seed)

    # ===== STEP 3: Evaluate on Each OOD Size =====
    all_results = []

    for size in sorted(args.sizes):
        dataset = ood_datasets[size]
        results = evaluate_ood_size(model, dataset, size, args)
        all_results.append(results)

        # Save individual results
        results_path = os.path.join(args.output_dir, f"ood_results_n{size}.json")
        save_results_to_json(results, results_path)

    # ===== STEP 4: Summary Table =====
    print("\n" + "=" * 70)
    print("OOD EVALUATION SUMMARY")
    print("=" * 70)
    print()
    print(
        f"{'Size':<8} | {'Mean Gap':<10} | {'Median Gap':<12} | {'Std':<8} | {'Feasible':<10} | Time (ms)"
    )
    print("-" * 75)

    for result in all_results:
        size = result["problem_size"]
        gap = f"{result['mean_gap']:.2f}%"
        median_gap = f"{result['median_gap']:.2f}%"
        std = f"{result['std_gap']:.2f}%"
        feas = f"{result['feasibility_rate'] * 100:.1f}%"
        time_ms = f"{result.get('mean_inference_time', 0) * 1000:.2f}"

        print(f"{size:<8} | {gap:<10} | {median_gap:<12} | {std:<8} | {feas:<10} | {time_ms}")

    print()

    # Compute degradation
    print("\nGeneralization Analysis:")
    in_dist_gap = all_results[0]["mean_gap"] if args.sizes[0] <= training_size_range[1] else None
    largest_gap = all_results[-1]["mean_gap"]

    if in_dist_gap is not None:
        degradation = largest_gap - in_dist_gap
        print(f"  Gap increase from smallest to largest: {degradation:.2f}%")

    # Check if still generalizing well
    if largest_gap < 5.0:
        print(f"  ✅ Excellent generalization! Gap < 5% even at {args.sizes[-1]} items")
    elif largest_gap < 10.0:
        print(f"  ✓ Good generalization. Gap < 10% at {args.sizes[-1]} items")
    else:
        print(f"  ⚠ Significant degradation. Gap = {largest_gap:.2f}% at {args.sizes[-1]} items")

    # Save summary
    summary = {
        "training_size_range": training_size_range,
        "ood_sizes": args.sizes,
        "results_by_size": {
            r["problem_size"]: {
                "mean_gap": r["mean_gap"],
                "median_gap": r["median_gap"],
                "std_gap": r["std_gap"],
                "feasibility_rate": r["feasibility_rate"],
                "mean_time": r.get("mean_inference_time", 0),
            }
            for r in all_results
        },
    }

    summary_path = os.path.join(args.output_dir, "ood_summary.json")
    save_results_to_json(summary, summary_path)

    # ===== STEP 5: Visualizations =====
    if args.visualize:
        plot_ood_results(all_results, training_size_range, args.output_dir)

    # ===== Final Message =====
    print("\n" + "=" * 70)
    print("OOD EVALUATION COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
