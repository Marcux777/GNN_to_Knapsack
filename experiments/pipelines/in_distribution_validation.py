# mypy: ignore-errors
"""
In-Distribution Validation Pipeline

Generates and evaluates model on in-distribution test sets with specific size bins.
This ensures rigorous validation with adequate sample sizes per bin.

Usage:
    python experiments/pipelines/in_distribution_validation.py \
        --checkpoint-dir checkpoints/run_20251020_104533 \
        --sizes 10 25 50 75 100 150 200 \
        --n-instances-per-size 100 \
        --strategies sampling warm_start
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analysis.distribution_analysis import analyze_distribution  # noqa: E402
from knapsack_gnn.data.generator import (  # noqa: E402
    KnapsackDataset,
    KnapsackGenerator,
    KnapsackSolver,
)
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset  # noqa: E402
from knapsack_gnn.decoding.sampling import evaluate_model  # noqa: E402
from knapsack_gnn.eval.reporting import (  # noqa: E402
    print_evaluation_summary,
    save_results_to_json,
)
from knapsack_gnn.models.pna import create_model  # noqa: E402


def generate_size_specific_datasets(
    sizes: list[int],
    n_instances_per_size: int,
    seed: int = 999,
) -> dict[int, KnapsackDataset]:
    """
    Generate test datasets for specific problem sizes.

    Args:
        sizes: List of problem sizes (number of items)
        n_instances_per_size: Number of instances per size
        seed: Random seed

    Returns:
        Dictionary mapping size -> KnapsackDataset
    """
    print("=" * 80)
    print("GENERATING SIZE-SPECIFIC DATASETS")
    print("=" * 80)
    print(f"Sizes: {sizes}")
    print(f"Instances per size: {n_instances_per_size}")
    print(f"Total instances: {len(sizes) * n_instances_per_size}")
    print()

    generator = KnapsackGenerator(seed=seed)
    datasets = {}

    for size in sizes:
        print(f"Generating {n_instances_per_size} instances with {size} items...")

        # Generate instances for this specific size
        instances = generator.generate_dataset(
            n_instances=n_instances_per_size,
            n_items_range=(size, size),  # Fixed size
        )

        # Solve with OR-Tools
        print("  Solving with OR-Tools...")
        instances = KnapsackSolver.solve_batch(instances, verbose=False)

        # Create dataset
        dataset = KnapsackDataset(instances)
        datasets[size] = dataset

        # Print statistics
        solve_times = [inst.solve_time for inst in instances if inst.solve_time is not None]
        if solve_times:
            print(
                f"  Solve time: {np.mean(solve_times) * 1000:.2f} ± {np.std(solve_times) * 1000:.2f} ms"
            )

        opt_values = [inst.optimal_value for inst in instances if inst.optimal_value is not None]
        if opt_values:
            print(f"  Optimal value: {np.mean(opt_values):.1f} ± {np.std(opt_values):.1f}")
        print()

    print("=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print()

    return datasets


def load_model(
    checkpoint_dir: Path,
    device: str,
    sample_dataset: KnapsackGraphDataset,
    hidden_dim: int = 64,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> torch.nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_dir}...")

    checkpoint_path = checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create model
    model = create_model(
        dataset=sample_dataset,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    # Load weights
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    if "epochs_trained" in state:
        print(f"  Epochs trained: {state['epochs_trained']}")

    return model


def evaluate_size_specific(
    model,
    dataset: KnapsackDataset,
    size: int,
    strategy: str,
    args,
) -> dict:
    """Evaluate model on a specific size dataset."""
    print(f"\n{'=' * 80}")
    print(f"EVALUATING SIZE: {size} items (strategy: {strategy})")
    print(f"{'=' * 80}")

    # Build graph dataset
    graph_dataset = KnapsackGraphDataset(dataset, normalize_features=True)

    # Prepare strategy kwargs
    strategy_kwargs = {}
    if strategy == "sampling":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": tuple(args.sampling_schedule),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    elif strategy == "threshold":
        strategy_kwargs = {"threshold": 0.5}
    elif strategy == "adaptive":
        strategy_kwargs = {"n_trials": args.n_samples}
    elif strategy == "lagrangian":
        strategy_kwargs = {
            "lagrangian_iters": args.lagrangian_iters,
            "lagrangian_tol": args.lagrangian_tol,
            "lagrangian_bias": args.lagrangian_bias,
        }
    elif strategy == "warm_start":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": tuple(args.sampling_schedule),
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
            "fix_threshold": args.fix_threshold,
            "ilp_time_limit": args.ilp_time_limit,
            "max_hint_items": args.max_hint_items,
            "ilp_threads": args.ilp_threads,
        }

    # Build sampler kwargs
    sampler_kwargs = {
        "num_threads": args.threads,
        "compile_model": args.compile,
        "quantize": args.quantize,
    }

    # Evaluate
    results = evaluate_model(
        model=model,
        dataset=graph_dataset,
        strategy=strategy,
        device=args.device,
        sampler_kwargs=sampler_kwargs,
        **strategy_kwargs,
    )

    # Add size information
    results["problem_size"] = size
    results["n_instances"] = len(dataset)

    # Add per-instance sizes for later analysis
    results["sizes"] = [size] * len(dataset)

    # Print summary
    print_evaluation_summary(results)

    return results


def combine_results(results_by_size: dict[int, dict]) -> dict:
    """
    Combine results from different sizes into a single result dict.

    Args:
        results_by_size: Dict mapping size -> results

    Returns:
        Combined results dictionary
    """
    combined: dict[str, list] = {
        "gaps": [],
        "sizes": [],
        "values": [],
        "optimal_values": [],
        "inference_times": [],
        "samples_used": [],
        "ilp_wall_times": [],
    }

    for size, results in results_by_size.items():
        n = results["n_instances"]
        combined["gaps"].extend(results.get("gaps", []))
        combined["sizes"].extend([size] * n)
        combined["values"].extend(results.get("values", []))
        combined["optimal_values"].extend(results.get("optimal_values", []))
        combined["inference_times"].extend(results.get("inference_times", []))
        combined["samples_used"].extend(results.get("samples_used", []))
        combined["ilp_wall_times"].extend(results.get("ilp_wall_times", []))

    # Compute aggregate statistics
    if combined["gaps"]:
        combined["mean_gap"] = float(np.mean(combined["gaps"]))
        combined["median_gap"] = float(np.median(combined["gaps"]))
        combined["std_gap"] = float(np.std(combined["gaps"]))
        combined["max_gap"] = float(np.max(combined["gaps"]))

    combined["total_instances"] = len(combined["gaps"])
    combined["n_sizes"] = len(results_by_size)

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-distribution validation pipeline")

    # Model/checkpoint
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Dataset generation
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 25, 50, 75, 100, 150, 200],
        help="Problem sizes to evaluate",
    )
    parser.add_argument(
        "--n-instances-per-size",
        type=int,
        default=100,
        help="Number of instances per size",
    )
    parser.add_argument("--seed", type=int, default=999, help="Random seed for dataset generation")

    # Evaluation
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["sampling", "warm_start"],
        help="Strategies to evaluate",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--quantize", action="store_true")

    # Strategy parameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-schedule", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--sampling-tolerance", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--lagrangian-iters", type=int, default=30)
    parser.add_argument("--lagrangian-tol", type=float, default=1e-4)
    parser.add_argument("--lagrangian-bias", type=float, default=0.0)
    parser.add_argument("--fix-threshold", type=float, default=0.9)
    parser.add_argument("--ilp-time-limit", type=float, default=1.0)
    parser.add_argument("--max-hint-items", type=int, default=None)
    parser.add_argument("--ilp-threads", type=int, default=None)

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/evaluation/in_dist)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    if args.output_dir is None:
        output_dir = checkpoint_dir / "evaluation" / "in_dist"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    print("=" * 80)
    print("IN-DISTRIBUTION VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Sizes: {args.sizes}")
    print(f"Instances per size: {args.n_instances_per_size}")
    print(f"Strategies: {args.strategies}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()

    # Generate datasets
    datasets_by_size = generate_size_specific_datasets(
        sizes=args.sizes,
        n_instances_per_size=args.n_instances_per_size,
        seed=args.seed,
    )

    # Load model (use first dataset as sample for architecture)
    sample_size = args.sizes[0]
    sample_graph_dataset = KnapsackGraphDataset(
        datasets_by_size[sample_size],
        normalize_features=True,
    )

    model = load_model(
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        sample_dataset=sample_graph_dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Evaluate each strategy
    for strategy in args.strategies:
        print(f"\n{'=' * 80}")
        print(f"STRATEGY: {strategy}")
        print(f"{'=' * 80}\n")

        results_by_size = {}

        for size in args.sizes:
            dataset = datasets_by_size[size]
            results = evaluate_size_specific(
                model=model,
                dataset=dataset,
                size=size,
                strategy=strategy,
                args=args,
            )
            results_by_size[size] = results

            # Save individual size results
            size_output = output_dir / strategy
            size_output.mkdir(parents=True, exist_ok=True)
            result_path = size_output / f"results_n{size}.json"
            save_results_to_json(results, str(result_path))

        # Combine results
        combined_results = combine_results(results_by_size)

        # Save combined results
        combined_path = output_dir / strategy / "results_combined.json"
        save_results_to_json(combined_results, str(combined_path))

        # Run distribution analysis
        print(f"\n{'=' * 80}")
        print(f"RUNNING DISTRIBUTION ANALYSIS FOR {strategy}")
        print(f"{'=' * 80}\n")

        analysis_output = output_dir / strategy / "analysis"
        analyze_distribution(
            gaps=combined_results["gaps"],
            sizes=combined_results["sizes"],
            output_dir=analysis_output,
            strategy_name=strategy,
            size_bins=args.sizes,
        )

    print("\n" + "=" * 80)
    print("IN-DISTRIBUTION VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
