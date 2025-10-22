"""
Evaluation Script for Knapsack GNN
Evaluates trained model on test set with multiple inference strategies
"""

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import project modules
from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.decoding.sampling import evaluate_model, KnapsackSampler
from knapsack_gnn.training.metrics import (
    plot_optimality_gaps,
    plot_performance_vs_size,
    plot_solution_comparison,
    print_evaluation_summary,
    save_results_to_json,
    benchmark_time,
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate Knapsack GNN")

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="Directory containing trained model"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename (default: best_model.pt)",
    )

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, default="data/datasets", help="Directory containing datasets"
    )
    parser.add_argument(
        "--test_only", action="store_true", help="Evaluate only on test set (skip val set)"
    )

    # Inference parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive", "lagrangian", "warm_start"],
        help="Inference strategy (default: sampling)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples for sampling strategy (default: 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for threshold strategy (default: 0.5)",
    )
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
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples (default: n_samples)",
    )
    parser.add_argument(
        "--lagrangian_iters",
        type=int,
        default=30,
        help="Max iterations for Lagrangian decoder (default: 30)",
    )
    parser.add_argument(
        "--lagrangian_tol",
        type=float,
        default=1e-4,
        help="Tolerance for Lagrangian decoder (default: 1e-4)",
    )
    parser.add_argument(
        "--lagrangian_bias",
        type=float,
        default=0.0,
        help="Bias weight for probabilities in Lagrangian decoding (default: 0.0)",
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

    # Visualization parameters
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    parser.add_argument(
        "--n_visualize", type=int, default=5, help="Number of instances to visualize (default: 5)"
    )

    # Benchmark parameters
    parser.add_argument("--benchmark", action="store_true", help="Run timing benchmark")

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint_dir/evaluation)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set torch.set_num_threads for latency-sensitive runs",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile model with torch.compile for inference"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization to linear layers (CPU only)",
    )
    parser.add_argument(
        "--ilp_threads",
        type=int,
        default=None,
        help="Number of threads for ILP warm-start solver (default: 1)",
    )

    return parser.parse_args()


def parse_schedule(schedule_str: Optional[str]) -> Optional[tuple[int, ...]]:
    """Parse comma-separated sampling schedule."""
    if schedule_str is None:
        return None
    values = []
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
    """Resolve device string with graceful CPU fallback."""
    requested = device.lower()
    if requested in ("auto", "best"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA requested but not available; falling back to CPU.")
            return "cpu"
    return device


def load_model(checkpoint_path, train_dataset, device):
    """Load trained model from checkpoint"""
    # Create model with same architecture
    model = create_model(train_dataset, hidden_dim=64, num_layers=3, dropout=0.1)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    if "epochs_trained" in checkpoint:
        print(f"  Epochs trained: {checkpoint['epochs_trained']}")
    if "best_val_loss" in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")

    return model


def main():
    """Main evaluation pipeline"""
    args = parse_args()
    args.device = resolve_device(args.device)
    args.sampling_schedule = parse_schedule(args.sampling_schedule)
    if args.max_samples is None:
        args.max_samples = args.n_samples
    args.fix_threshold = min(max(args.fix_threshold, 0.0), 1.0)
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "evaluation")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("KNAPSACK GNN EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # ===== STEP 1: Load Data =====
    print("=" * 70)
    print("STEP 1: Loading Data")
    print("=" * 70)

    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    val_dataset = KnapsackDataset.load(f"{args.data_dir}/val.pkl")
    test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")

    print("\nBuilding graph datasets...")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)
    val_graph_dataset = KnapsackGraphDataset(val_dataset, normalize_features=True)
    test_graph_dataset = KnapsackGraphDataset(test_dataset, normalize_features=True)

    # ===== STEP 2: Load Model =====
    print("\n" + "=" * 70)
    print("STEP 2: Loading Model")
    print("=" * 70)

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = load_model(checkpoint_path, train_graph_dataset, args.device)

    # ===== STEP 3: Evaluate on Test Set =====
    print("\n" + "=" * 70)
    print("STEP 3: Evaluating on Test Set")
    print("=" * 70)

    strategy_kwargs = {}
    if args.strategy == "sampling":
        strategy_kwargs = {
            "temperature": args.temperature,
            "sampling_schedule": args.sampling_schedule,
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    elif args.strategy == "threshold":
        strategy_kwargs = {"threshold": args.threshold}
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

    sampler_kwargs = {
        "num_threads": args.threads,
        "compile_model": args.compile,
        "quantize": args.quantize,
    }

    test_results = evaluate_model(
        model=model,
        dataset=test_graph_dataset,
        strategy=args.strategy,
        device=args.device,
        sampler_kwargs=sampler_kwargs,
        **strategy_kwargs,
    )

    print_evaluation_summary(test_results)

    # Save results
    results_path = os.path.join(args.output_dir, f"test_results_{args.strategy}.json")
    save_results_to_json(test_results, results_path)

    # ===== STEP 4: Evaluate on Validation Set (optional) =====
    if not args.test_only:
        print("\n" + "=" * 70)
        print("STEP 4: Evaluating on Validation Set")
        print("=" * 70)

        val_results = evaluate_model(
            model=model,
            dataset=val_graph_dataset,
            strategy=args.strategy,
            device=args.device,
            sampler_kwargs=sampler_kwargs,
            **strategy_kwargs,
        )

        print_evaluation_summary(val_results)

        val_results_path = os.path.join(args.output_dir, f"val_results_{args.strategy}.json")
        save_results_to_json(val_results, val_results_path)

    # ===== STEP 5: Benchmark Timing (optional) =====
    if args.benchmark:
        print("\n" + "=" * 70)
        print("STEP 5: Benchmarking Timing")
        print("=" * 70)

        timing_results = benchmark_time(model, test_graph_dataset, device=args.device)
        print("\nTiming Results:")
        print(f"  Mean time: {timing_results['mean_time'] * 1000:.2f} ms")
        print(f"  Median time: {timing_results['median_time'] * 1000:.2f} ms")
        print(f"  Throughput: {timing_results['throughput']:.2f} instances/sec")

        timing_path = os.path.join(args.output_dir, "timing_results.json")
        save_results_to_json(timing_results, timing_path)

    # ===== STEP 6: Visualizations (optional) =====
    if args.visualize:
        print("\n" + "=" * 70)
        print("STEP 6: Generating Visualizations")
        print("=" * 70)

        # Plot optimality gaps
        if test_results["gaps"]:
            print("\nPlotting optimality gap distribution...")
            plot_optimality_gaps(
                test_results["gaps"],
                title=f"Test Set Optimality Gaps ({args.strategy} strategy)",
                save_path=os.path.join(args.output_dir, f"gaps_{args.strategy}.png"),
            )

        # Plot performance vs size
        if test_results["gaps"]:
            print("Plotting performance vs problem size...")
            instance_sizes = [data.n_items for data in test_graph_dataset]
            plot_performance_vs_size(
                instance_sizes,
                test_results["gaps"],
                metric_name="Optimality Gap (%)",
                title=f"Performance vs Problem Size ({args.strategy})",
                save_path=os.path.join(args.output_dir, f"performance_vs_size_{args.strategy}.png"),
            )

        # Visualize individual solutions
        print(f"Visualizing {args.n_visualize} example solutions...")
        sampler = KnapsackSampler(model, device=args.device)

        for i in range(min(args.n_visualize, len(test_graph_dataset))):
            data = test_graph_dataset[i]
            result = sampler.solve(data, strategy=args.strategy, **strategy_kwargs)

            plot_solution_comparison(
                instance_data=data,
                predicted_solution=result["solution"],
                probabilities=result["probabilities"],
                title=f"Instance {i} - Gap: {result.get('optimality_gap', 0):.2f}%",
                save_path=os.path.join(args.output_dir, f"solution_{i}.png"),
            )

        print(f"\nVisualizations saved to {args.output_dir}")

    # ===== STEP 7: Compare Strategies (bonus) =====
    if args.strategy == "sampling":
        print("\n" + "=" * 70)
        print("BONUS: Comparing All Strategies")
        print("=" * 70)

        strategies = {
            "threshold": {"threshold": 0.5},
            "sampling": {"n_samples": 100, "temperature": 1.0},
            "adaptive": {"n_trials": 20},
        }

        comparison_results = {}
        for strategy_name, strategy_kwargs in strategies.items():
            print(f"\nEvaluating with {strategy_name} strategy...")
            results = evaluate_model(
                model=model,
                dataset=test_graph_dataset[:50],  # Use subset for speed
                strategy=strategy_name,
                device=args.device,
                **strategy_kwargs,
            )
            comparison_results[strategy_name] = {
                "mean_gap": results["mean_gap"],
                "median_gap": results["median_gap"],
                "std_gap": results["std_gap"],
            }

        print("\n=== Strategy Comparison ===")
        for strategy_name, metrics in comparison_results.items():
            print(f"\n{strategy_name.upper()}:")
            print(f"  Mean gap: {metrics['mean_gap']:.2f}%")
            print(f"  Median gap: {metrics['median_gap']:.2f}%")
            print(f"  Std gap: {metrics['std_gap']:.2f}%")

        # Save comparison
        comparison_path = os.path.join(args.output_dir, "strategy_comparison.json")
        save_results_to_json(comparison_results, comparison_path)

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"\nKey Metrics (Test Set):")
    print(f"  Mean Optimality Gap: {test_results['mean_gap']:.2f}%")
    print(f"  Median Optimality Gap: {test_results['median_gap']:.2f}%")
    print(f"  Feasibility Rate: {test_results['feasibility_rate'] * 100:.2f}%")
    if test_results.get("gap_mean_ci_95"):
        ci_low, ci_high = test_results["gap_mean_ci_95"]
        print(f"  95% CI (Mean Gap): [{ci_low:.2f}%, {ci_high:.2f}%]")
    if test_results.get("mean_approx_ratio") is not None:
        print(f"  Mean Approximation Ratio: {test_results['mean_approx_ratio']:.4f}")
    if test_results.get("mean_inference_time") is not None:
        print(f"  Mean Inference Time: {test_results['mean_inference_time'] * 1000:.2f} ms")
    if test_results.get("mean_speedup") is not None:
        print(f"  Mean Speedup vs Exact Solver: {test_results['mean_speedup']:.2f}x")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
