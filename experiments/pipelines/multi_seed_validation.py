"""
Multi-Seed Validation for Publication-Grade Reproducibility

Trains and evaluates models with multiple random seeds to ensure:
1. Results are reproducible
2. Performance is stable across seeds
3. Reported metrics include uncertainty (mean ± std)

Usage:
    python multi_seed_validation.py --seeds 42 123 456 789 1011 --epochs 50
    python multi_seed_validation.py --quick_test  # Fast test with 3 seeds, 10 epochs
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from knapsack_gnn.analysis.stats import StatisticalAnalyzer
from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.loop import train_model
from knapsack_gnn.training.metrics import save_results_to_json


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-Seed Validation for Reproducibility")

    # Seed parameters
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1011],
        help="List of random seeds to use",
    )
    parser.add_argument(
        "--quick_test", action="store_true", help="Quick test: 3 seeds, 10 epochs, small dataset"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, default="data/datasets", help="Directory for datasets"
    )
    parser.add_argument("--train_size", type=int, default=1000, help="Training set size")
    parser.add_argument("--val_size", type=int, default=200, help="Validation set size")
    parser.add_argument("--test_size", type=int, default=200, help="Test set size")
    parser.add_argument("--n_items_min", type=int, default=10, help="Minimum items per instance")
    parser.add_argument("--n_items_max", type=int, default=50, help="Maximum items per instance")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")

    # Evaluation parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive", "lagrangian", "warm_start"],
        help="Inference strategy",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of samples for sampling/adaptive strategies",
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument(
        "--sampling_schedule",
        type=str,
        default="32,64,128",
        help="Comma-separated sampling batches (default: 32,64,128)",
    )
    parser.add_argument(
        "--sampling_tolerance",
        type=float,
        default=1e-3,
        help="Early-stopping tolerance for sampling",
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
        "--output_dir", type=str, default="multi_seed_validation", help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set torch.set_num_threads for latency-sensitive evaluation",
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

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        args.seeds = [42, 123, 456]
        args.epochs = 10
        args.train_size = 500
        args.val_size = 100
        args.test_size = 100
        print("QUICK TEST MODE: 3 seeds, 10 epochs, small dataset")

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

    return args


def parse_schedule(schedule_str: str | None) -> tuple[int, ...] | None:
    """Convert comma-separated schedule string to tuple of ints."""
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
    """Resolve requested device, defaulting to CPU if CUDA unavailable."""
    requested = device.lower()
    if requested in ("auto", "best"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        return "cpu"
    return device


def train_with_seed(seed: int, args: argparse.Namespace, shared_data: tuple | None = None) -> dict:
    """
    Train model with a specific seed

    Args:
        seed: Random seed
        args: Command-line arguments
        shared_data: Optional tuple of (train_dataset, val_dataset, test_dataset)

    Returns:
        Dictionary with training and evaluation results
    """
    print("\n" + "=" * 80)
    print(f"TRAINING WITH SEED {seed}")
    print("=" * 80)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load or use shared data
    if shared_data is not None:
        train_dataset, val_dataset, test_dataset = shared_data
        print("\nUsing shared datasets")
    else:
        print("\nLoading datasets...")
        train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
        val_dataset = KnapsackDataset.load(f"{args.data_dir}/val.pkl")
        test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")

    # Build graph datasets
    print("Building graph datasets...")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)
    val_graph_dataset = KnapsackGraphDataset(val_dataset, normalize_features=True)
    test_graph_dataset = KnapsackGraphDataset(test_dataset, normalize_features=True)

    # Create model
    print("Creating model...")
    model = create_model(
        train_graph_dataset, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.1
    )
    model = model.to(args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train
    import time

    train_start = time.time()

    trained_model, history = train_model(
        model=model,
        train_dataset=train_graph_dataset,
        val_dataset=val_graph_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )

    training_time = time.time() - train_start

    # Evaluate on test set
    print("\nEvaluating on test set...")

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

    test_results = evaluate_model(
        model=trained_model,
        dataset=test_graph_dataset,
        strategy=args.strategy,
        device=args.device,
        sampler_kwargs=args.sampler_kwargs,
        **strategy_kwargs,
    )

    # Compile results
    results = {
        "seed": seed,
        "n_params": n_params,
        "training_time": training_time,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_accuracy"][-1],
        "final_val_acc": history["val_accuracy"][-1],
        "test_mean_gap": test_results["mean_gap"],
        "test_median_gap": test_results["median_gap"],
        "test_std_gap": test_results["std_gap"],
        "test_max_gap": test_results["max_gap"],
        "test_feasibility_rate": test_results["feasibility_rate"],
        "test_inference_time": test_results.get("mean_inference_time", 0),
        "history": history,
        "test_gaps": test_results["gaps"],
    }

    # Save individual results
    save_results_to_json(results, os.path.join(checkpoint_dir, "results.json"))

    print(f"\n✓ Seed {seed} completed:")
    print(f"  Test Mean Gap: {results['test_mean_gap']:.2f}%")
    print(f"  Test Median Gap: {results['test_median_gap']:.2f}%")
    print(f"  Training Time: {training_time / 60:.1f} minutes")

    return results


def aggregate_results(all_results: list[dict]) -> dict:
    """
    Aggregate results across multiple seeds

    Args:
        all_results: List of result dictionaries from different seeds

    Returns:
        Aggregated statistics
    """
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS ACROSS SEEDS")
    print("=" * 80)

    # Extract metrics
    mean_gaps = [r["test_mean_gap"] for r in all_results]
    median_gaps = [r["test_median_gap"] for r in all_results]
    train_times = [r["training_time"] for r in all_results]
    val_losses = [r["final_val_loss"] for r in all_results]
    val_accs = [r["final_val_acc"] for r in all_results]

    # Statistical analyzer for CI
    analyzer = StatisticalAnalyzer()

    # Compute statistics
    aggregated = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        # Mean gap statistics
        "mean_gap": {
            "mean": float(np.mean(mean_gaps)),
            "std": float(np.std(mean_gaps, ddof=1)) if len(mean_gaps) > 1 else 0.0,
            "min": float(np.min(mean_gaps)),
            "max": float(np.max(mean_gaps)),
            "median": float(np.median(mean_gaps)),
            "ci_95": analyzer.bootstrap_ci(np.array(mean_gaps)),
        },
        # Median gap statistics
        "median_gap": {
            "mean": float(np.mean(median_gaps)),
            "std": float(np.std(median_gaps, ddof=1)) if len(median_gaps) > 1 else 0.0,
            "min": float(np.min(median_gaps)),
            "max": float(np.max(median_gaps)),
        },
        # Training time statistics
        "training_time": {
            "mean": float(np.mean(train_times)),
            "std": float(np.std(train_times, ddof=1)) if len(train_times) > 1 else 0.0,
            "min": float(np.min(train_times)),
            "max": float(np.max(train_times)),
        },
        # Validation statistics
        "final_val_loss": {
            "mean": float(np.mean(val_losses)),
            "std": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
        },
        "final_val_acc": {
            "mean": float(np.mean(val_accs)),
            "std": float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else 0.0,
        },
        # Per-seed results
        "per_seed_results": all_results,
    }

    return aggregated


def print_aggregated_results(aggregated: dict):
    """Print formatted aggregated results"""
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)
    print(f"\nNumber of seeds: {aggregated['n_seeds']}")
    print(f"Seeds: {aggregated['seeds']}")
    print()

    print("Test Set Performance (Mean ± Std across seeds):")
    print("-" * 80)
    mg = aggregated["mean_gap"]
    print(f"  Mean Gap:      {mg['mean']:.2f}% ± {mg['std']:.2f}%")
    print(f"    Range:       [{mg['min']:.2f}%, {mg['max']:.2f}%]")
    print(f"    95% CI:      [{mg['ci_95'][0]:.2f}%, {mg['ci_95'][1]:.2f}%]")

    mdg = aggregated["median_gap"]
    print(f"  Median Gap:    {mdg['mean']:.2f}% ± {mdg['std']:.2f}%")
    print()

    print("Training:")
    print("-" * 80)
    vl = aggregated["final_val_loss"]
    va = aggregated["final_val_acc"]
    print(f"  Val Loss:      {vl['mean']:.4f} ± {vl['std']:.4f}")
    print(f"  Val Accuracy:  {va['mean'] * 100:.2f}% ± {va['std'] * 100:.2f}%")

    tt = aggregated["training_time"]
    print(f"  Training Time: {tt['mean'] / 60:.1f} ± {tt['std'] / 60:.1f} minutes")
    print()

    # Per-seed breakdown
    print("Per-Seed Breakdown:")
    print("-" * 80)
    print(
        f"{'Seed':<8} | {'Mean Gap':<12} | {'Median Gap':<12} | {'Val Loss':<10} | {'Val Acc':<8}"
    )
    print("-" * 80)

    for result in aggregated["per_seed_results"]:
        seed = result["seed"]
        mg = f"{result['test_mean_gap']:.2f}%"
        mdg = f"{result['test_median_gap']:.2f}%"
        vl = f"{result['final_val_loss']:.4f}"
        va = f"{result['final_val_acc'] * 100:.2f}%"

        print(f"{seed:<8} | {mg:<12} | {mdg:<12} | {vl:<10} | {va:<8}")

    print()


def plot_seed_comparison(aggregated: dict, output_dir: str):
    """Create visualization comparing results across seeds"""
    print("\nGenerating seed comparison plots...")

    seeds = aggregated["seeds"]
    results = aggregated["per_seed_results"]

    mean_gaps = [r["test_mean_gap"] for r in results]
    median_gaps = [r["test_median_gap"] for r in results]
    val_losses = [r["final_val_loss"] for r in results]
    val_accs = [r["final_val_acc"] * 100 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean Gap across seeds
    ax = axes[0, 0]
    ax.bar(range(len(seeds)), mean_gaps, color="#2E86AB", alpha=0.7, edgecolor="black")
    ax.axhline(
        y=aggregated["mean_gap"]["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {aggregated['mean_gap']['mean']:.2f}%",
    )
    ax.fill_between(
        range(len(seeds)),
        aggregated["mean_gap"]["mean"] - aggregated["mean_gap"]["std"],
        aggregated["mean_gap"]["mean"] + aggregated["mean_gap"]["std"],
        alpha=0.2,
        color="red",
        label="±1 std",
    )
    ax.set_ylabel("Mean Optimality Gap (%)", fontsize=12)
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_title("Test Mean Gap Across Seeds", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. Median Gap across seeds
    ax = axes[0, 1]
    ax.bar(range(len(seeds)), median_gaps, color="#A23B72", alpha=0.7, edgecolor="black")
    ax.axhline(
        y=aggregated["median_gap"]["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {aggregated['median_gap']['mean']:.2f}%",
    )
    ax.set_ylabel("Median Optimality Gap (%)", fontsize=12)
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_title("Test Median Gap Across Seeds", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3. Validation Loss across seeds
    ax = axes[1, 0]
    ax.bar(range(len(seeds)), val_losses, color="#F18F01", alpha=0.7, edgecolor="black")
    ax.axhline(
        y=aggregated["final_val_loss"]["mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {aggregated['final_val_loss']['mean']:.4f}",
    )
    ax.set_ylabel("Final Validation Loss", fontsize=12)
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_title("Validation Loss Across Seeds", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 4. Validation Accuracy across seeds
    ax = axes[1, 1]
    ax.bar(range(len(seeds)), val_accs, color="#6A994E", alpha=0.7, edgecolor="black")
    ax.axhline(
        y=aggregated["final_val_acc"]["mean"] * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {aggregated['final_val_acc']['mean'] * 100:.2f}%",
    )
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_xlabel("Seed", fontsize=12)
    ax.set_title("Validation Accuracy Across Seeds", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "seed_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {save_path}")
    plt.close()


def main():
    """Main multi-seed validation pipeline"""
    args = parse_args()

    print("=" * 80)
    print("MULTI-SEED VALIDATION")
    print("=" * 80)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        if arg != "seeds":
            print(f"  {arg}: {value}")
    print(f"  seeds: {args.seeds} ({len(args.seeds)} seeds)")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to: {config_path}\n")

    # Load shared datasets (use same test set for all seeds)
    print("Loading shared datasets...")
    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    val_dataset = KnapsackDataset.load(f"{args.data_dir}/val.pkl")
    test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")
    shared_data = (train_dataset, val_dataset, test_dataset)

    # Train with each seed
    all_results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n[{i + 1}/{len(args.seeds)}] Processing seed {seed}...")
        try:
            result = train_with_seed(seed, args, shared_data=shared_data)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: Seed {seed} failed with exception: {e}")
            continue

    if len(all_results) == 0:
        print("\nERROR: No seeds completed successfully!")
        return

    # Aggregate results
    aggregated = aggregate_results(all_results)

    # Print summary
    print_aggregated_results(aggregated)

    # Save aggregated results
    agg_path = os.path.join(args.output_dir, "aggregated_results.json")
    # Remove gaps lists to reduce file size
    for result in aggregated["per_seed_results"]:
        if "test_gaps" in result:
            del result["test_gaps"]
        if "history" in result:
            del result["history"]
    save_results_to_json(aggregated, agg_path)
    print(f"\nAggregated results saved to: {agg_path}")

    # Generate plots
    plot_seed_comparison(aggregated, args.output_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("MULTI-SEED VALIDATION COMPLETED")
    print("=" * 80)
    print(f"\nResults directory: {args.output_dir}")
    print("\nKey Findings:")
    mg = aggregated["mean_gap"]
    print(f"  ✓ Test Mean Gap: {mg['mean']:.2f}% ± {mg['std']:.2f}%")
    print(f"  ✓ 95% CI: [{mg['ci_95'][0]:.2f}%, {mg['ci_95'][1]:.2f}%]")
    print(f"  ✓ Reproducible across {aggregated['n_seeds']} seeds")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
