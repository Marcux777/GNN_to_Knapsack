# mypy: ignore-errors
"""
Ablation Study Script
Tests importance of architectural choices and feature engineering
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from knapsack_gnn.data.generator import KnapsackDataset, create_datasets
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.eval.reporting import save_results_to_json
from knapsack_gnn.models.gat import create_gat_model
from knapsack_gnn.models.gcn import create_gcn_model
from knapsack_gnn.models.pna import create_model as create_pna_model
from knapsack_gnn.training.loop import train_model


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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Ablation Study for Knapsack GNN")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["architecture", "features", "both"],
        help="Type of ablation study",
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
    parser.add_argument("--generate_data", action="store_true", help="Generate new datasets")

    # Training parameters (for architecture ablation)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")

    # Feature ablation parameters
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory (for feature ablation)",
    )

    # Inference parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive", "lagrangian", "warm_start"],
        help="Inference strategy for evaluation",
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
    parser.add_argument("--output_dir", type=str, default="ablation_study", help="Output directory")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--threads", type=int, default=None, help="Set torch.set_num_threads for evaluation"
    )
    parser.add_argument(
        "--compile", action="store_true", help="Compile evaluation model with torch.compile"
    )
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


def build_strategy_kwargs(args) -> dict:
    """Build strategy-specific kwargs for inference."""
    if args.strategy == "sampling":
        return {
            "temperature": args.temperature,
            "sampling_schedule": args.sampling_schedule,
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
        }
    if args.strategy == "threshold":
        return {"threshold": 0.5}
    if args.strategy == "adaptive":
        return {"n_trials": args.n_samples}
    if args.strategy == "lagrangian":
        return {
            "lagrangian_iters": args.lagrangian_iters,
            "lagrangian_tol": args.lagrangian_tol,
            "lagrangian_bias": args.lagrangian_bias,
        }
    if args.strategy == "warm_start":
        return {
            "temperature": args.temperature,
            "sampling_schedule": args.sampling_schedule,
            "sampling_tolerance": args.sampling_tolerance,
            "max_samples": args.max_samples,
            "fix_threshold": args.fix_threshold,
            "ilp_time_limit": args.ilp_time_limit,
            "ilp_threads": args.ilp_threads,
            "max_hint_items": args.max_hint_items,
        }
    return {}


def architecture_ablation(args) -> dict:
    """
    Compare different GNN architectures: PNA, GCN, GAT

    Returns:
        Dictionary with results for each architecture
    """
    print("\n" + "=" * 70)
    print("ARCHITECTURE ABLATION STUDY")
    print("=" * 70)

    # Load or generate data
    if args.generate_data or not os.path.exists(f"{args.data_dir}/train.pkl"):
        print("\nGenerating datasets...")
        train_dataset, val_dataset, test_dataset = create_datasets(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            n_items_range=(args.n_items_min, args.n_items_max),
            output_dir=args.data_dir,
        )
    else:
        print("\nLoading existing datasets...")
        train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
        val_dataset = KnapsackDataset.load(f"{args.data_dir}/val.pkl")
        test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")

    # Build graph datasets
    print("Building graph datasets...")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)
    val_graph_dataset = KnapsackGraphDataset(val_dataset, normalize_features=True)
    test_graph_dataset = KnapsackGraphDataset(test_dataset, normalize_features=True)

    strategy_kwargs = build_strategy_kwargs(args)

    # Architectures to compare
    architectures = {
        "PNA": lambda: create_pna_model(
            train_graph_dataset, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.1
        ),
        "GCN": lambda: create_gcn_model(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.1
        ),
        "GAT": lambda: create_gat_model(
            hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.1, num_heads=4
        ),
    }

    results = {}

    for arch_name, model_creator in architectures.items():
        print("\n" + "=" * 70)
        print(f"TRAINING {arch_name} MODEL")
        print("=" * 70)

        # Create model
        model = model_creator()
        model = model.to(args.device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters: {n_params:,}")

        # Train
        training_start = time.time()
        trained_model, history = train_model(
            model=model,
            train_dataset=train_graph_dataset,
            val_dataset=val_graph_dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=os.path.join(args.output_dir, f"{arch_name}_checkpoint"),
            device=args.device,
        )
        training_time = time.time() - training_start

        # Evaluate on test set
        print(f"\nEvaluating {arch_name} on test set...")
        test_results = evaluate_model(
            model=trained_model,
            dataset=test_graph_dataset,
            strategy=args.strategy,
            device=args.device,
            sampler_kwargs=args.sampler_kwargs,
            **strategy_kwargs,
        )

        # Store results
        results[arch_name] = {
            "n_params": n_params,
            "training_time": training_time,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_train_acc": history["train_accuracy"][-1],
            "final_val_acc": history["val_accuracy"][-1],
            "test_mean_gap": test_results["mean_gap"],
            "test_median_gap": test_results["median_gap"],
            "test_std_gap": test_results["std_gap"],
            "test_feasibility_rate": test_results["feasibility_rate"],
            "test_inference_time": test_results.get("mean_inference_time", 0),
            "history": history,
        }

        print(f"\n{arch_name} Results:")
        print(f"  Training time: {training_time / 60:.1f} minutes")
        print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"  Test mean gap: {test_results['mean_gap']:.2f}%")
        print(f"  Test median gap: {test_results['median_gap']:.2f}%")

    return results


def feature_ablation(args) -> dict:
    """
    Test importance of different features

    Feature ablations:
    1. Baseline (all features)
    2. No weights (only values)
    3. No values (only weights)
    4. No capacity normalization
    5. Random features

    Returns:
        Dictionary with results for each ablation
    """
    print("\n" + "=" * 70)
    print("FEATURE ABLATION STUDY")
    print("=" * 70)

    if args.checkpoint_dir is None:
        print("Error: --checkpoint_dir required for feature ablation")
        sys.exit(1)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    KnapsackDataset.load(f"{args.data_dir}/val.pkl")
    test_dataset = KnapsackDataset.load(f"{args.data_dir}/test.pkl")

    # Load trained model
    print(f"\nLoading trained model from {args.checkpoint_dir}...")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)

    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    model = create_pna_model(train_graph_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    # Feature ablations
    ablations = {
        "Baseline (All features)": lambda inst: inst,
        "No weights": lambda inst: remove_weights(inst),
        "No values": lambda inst: remove_values(inst),
        "Random features": lambda inst: randomize_features(inst),
    }

    results = {}

    for ablation_name, transform_fn in ablations.items():
        print(f"\n{'=' * 70}")
        print(f"TESTING: {ablation_name}")
        print(f"{'=' * 70}")

        # Create modified test dataset
        modified_instances = [transform_fn(inst) for inst in test_dataset.instances]
        modified_dataset = KnapsackDataset(modified_instances)
        modified_graph_dataset = KnapsackGraphDataset(modified_dataset, normalize_features=True)

        # Evaluate
        test_results = evaluate_model(
            model=model,
            dataset=modified_graph_dataset,
            strategy=args.strategy,
            device=args.device,
            sampler_kwargs=args.sampler_kwargs,
            **build_strategy_kwargs(args),
        )

        results[ablation_name] = {
            "test_mean_gap": test_results["mean_gap"],
            "test_median_gap": test_results["median_gap"],
            "test_std_gap": test_results["std_gap"],
            "test_feasibility_rate": test_results["feasibility_rate"],
        }

        print(f"\n{ablation_name} Results:")
        print(f"  Mean gap: {test_results['mean_gap']:.2f}%")
        print(f"  Median gap: {test_results['median_gap']:.2f}%")
        print(f"  Feasibility rate: {test_results['feasibility_rate'] * 100:.1f}%")

    return results


def remove_weights(instance):
    """Set all weights to 1 (removes weight information)"""
    from knapsack_gnn.data.generator import KnapsackInstance

    modified = KnapsackInstance(
        weights=np.ones_like(instance.weights),
        values=instance.values.copy(),
        capacity=instance.capacity,
    )
    modified.solution = instance.solution
    modified.optimal_value = instance.optimal_value
    modified.solve_time = instance.solve_time
    return modified


def remove_values(instance):
    """Set all values to 1 (removes value information)"""
    from knapsack_gnn.data.generator import KnapsackInstance

    modified = KnapsackInstance(
        weights=instance.weights.copy(),
        values=np.ones_like(instance.values),
        capacity=instance.capacity,
    )
    modified.solution = instance.solution
    modified.optimal_value = instance.optimal_value
    modified.solve_time = instance.solve_time
    return modified


def randomize_features(instance):
    """Replace features with random values"""
    from knapsack_gnn.data.generator import KnapsackInstance

    rng = np.random.RandomState(42)
    modified = KnapsackInstance(
        weights=rng.randint(1, 100, size=len(instance.weights)),
        values=rng.randint(1, 100, size=len(instance.values)),
        capacity=instance.capacity,
    )
    modified.solution = instance.solution
    modified.optimal_value = instance.optimal_value
    modified.solve_time = instance.solve_time
    return modified


def plot_architecture_results(results: dict, output_dir: str):
    """Create visualization for architecture ablation"""
    print("\nGenerating architecture comparison plots...")

    arch_names = list(results.keys())
    n_params = [results[a]["n_params"] for a in arch_names]
    train_times = [results[a]["training_time"] / 60 for a in arch_names]  # minutes
    test_gaps = [results[a]["test_mean_gap"] for a in arch_names]
    val_accs = [results[a]["final_val_acc"] * 100 for a in arch_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    # 1. Model Parameters
    ax = axes[0, 0]
    ax.bar(arch_names, n_params, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Number of Parameters", fontsize=12)
    ax.set_title("Model Complexity", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (_name, param) in enumerate(zip(arch_names, n_params, strict=False)):
        ax.text(i, param + max(n_params) * 0.02, f"{param:,}", ha="center", fontsize=10)

    # 2. Test Optimality Gap
    ax = axes[0, 1]
    ax.bar(arch_names, test_gaps, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Mean Optimality Gap (%)", fontsize=12)
    ax.set_title("Test Set Performance", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (_name, gap) in enumerate(zip(arch_names, test_gaps, strict=False)):
        ax.text(i, gap + 0.1, f"{gap:.2f}%", ha="center", fontsize=10)

    # 3. Training Time
    ax = axes[1, 0]
    ax.bar(arch_names, train_times, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Training Time (minutes)", fontsize=12)
    ax.set_title("Training Efficiency", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, (_name, time_val) in enumerate(zip(arch_names, train_times, strict=False)):
        ax.text(i, time_val + max(train_times) * 0.02, f"{time_val:.1f}m", ha="center", fontsize=10)

    # 4. Validation Accuracy
    ax = axes[1, 1]
    ax.bar(arch_names, val_accs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Training Performance", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([min(val_accs) - 5, 100])
    for i, (_name, acc) in enumerate(zip(arch_names, val_accs, strict=False)):
        ax.text(i, acc + 1, f"{acc:.1f}%", ha="center", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "architecture_ablation.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {save_path}")
    plt.close()

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax = axes[0]
    for _i, (arch_name, color) in enumerate(zip(arch_names, colors, strict=False)):
        history = results[arch_name]["history"]
        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(
            epochs,
            history["train_loss"],
            label=f"{arch_name} (train)",
            color=color,
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            epochs,
            history["val_loss"],
            label=f"{arch_name} (val)",
            color=color,
            linestyle="--",
            linewidth=2,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Curves - Loss", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy curves
    ax = axes[1]
    for _i, (arch_name, color) in enumerate(zip(arch_names, colors, strict=False)):
        history = results[arch_name]["history"]
        epochs = range(1, len(history["train_accuracy"]) + 1)
        ax.plot(
            epochs,
            np.array(history["train_accuracy"]) * 100,
            label=f"{arch_name} (train)",
            color=color,
            linestyle="-",
            linewidth=2,
        )
        ax.plot(
            epochs,
            np.array(history["val_accuracy"]) * 100,
            label=f"{arch_name} (val)",
            color=color,
            linestyle="--",
            linewidth=2,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Training Curves - Accuracy", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {save_path}")
    plt.close()


def plot_feature_results(results: dict, output_dir: str):
    """Create visualization for feature ablation"""
    print("\nGenerating feature ablation plots...")

    ablation_names = list(results.keys())
    gaps = [results[a]["test_mean_gap"] for a in ablation_names]
    feasibility = [results[a]["test_feasibility_rate"] * 100 for a in ablation_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#6A994E", "#BC4B51", "#F4A259", "#8B8C89"]

    # 1. Optimality Gap
    ax = axes[0]
    bars = ax.barh(
        ablation_names, gaps, color=colors[: len(ablation_names)], alpha=0.7, edgecolor="black"
    )
    ax.set_xlabel("Mean Optimality Gap (%)", fontsize=12)
    ax.set_title("Feature Importance (Gap)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for _i, (bar, gap) in enumerate(zip(bars, gaps, strict=False)):
        ax.text(
            gap + max(gaps) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{gap:.2f}%",
            va="center",
            fontsize=10,
        )

    # 2. Feasibility Rate
    ax = axes[1]
    bars = ax.barh(
        ablation_names,
        feasibility,
        color=colors[: len(ablation_names)],
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Feasibility Rate (%)", fontsize=12)
    ax.set_title("Feature Impact on Feasibility", fontsize=14, fontweight="bold")
    ax.set_xlim([90, 101])
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for _i, (bar, feas) in enumerate(zip(bars, feasibility, strict=False)):
        ax.text(
            feas + 0.5, bar.get_y() + bar.get_height() / 2, f"{feas:.1f}%", va="center", fontsize=10
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "feature_ablation.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {save_path}")
    plt.close()


def main():
    """Main ablation study pipeline"""
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}

    # Run ablation studies
    if args.mode in ["architecture", "both"]:
        arch_results = architecture_ablation(args)
        all_results["architecture"] = arch_results

        # Save results
        save_results_to_json(
            arch_results, os.path.join(args.output_dir, "architecture_ablation.json")
        )

        # Plot
        plot_architecture_results(arch_results, args.output_dir)

    if args.mode in ["features", "both"]:
        feat_results = feature_ablation(args)
        all_results["features"] = feat_results

        # Save results
        save_results_to_json(feat_results, os.path.join(args.output_dir, "feature_ablation.json"))

        # Plot
        plot_feature_results(feat_results, args.output_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")

    if "architecture" in all_results:
        print("\nArchitecture Results:")
        for arch, res in all_results["architecture"].items():
            print(f"  {arch}: {res['test_mean_gap']:.2f}% gap, {res['n_params']:,} params")

    if "features" in all_results:
        print("\nFeature Results:")
        baseline_gap = all_results["features"]["Baseline (All features)"]["test_mean_gap"]
        print(f"  Baseline: {baseline_gap:.2f}% gap")
        for feat, res in all_results["features"].items():
            if feat != "Baseline (All features)":
                degradation = res["test_mean_gap"] - baseline_gap
                print(f"  {feat}: {res['test_mean_gap']:.2f}% gap (+{degradation:.2f}% worse)")

    print()


if __name__ == "__main__":
    main()
