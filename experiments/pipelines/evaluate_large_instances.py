"""
Comprehensive Large Instance Evaluation
Tests model on n=500, n=1000, n=2000 with sufficient samples for statistical significance
"""

import argparse
import json
import os

import numpy as np
import torch

from knapsack_gnn.data.generator import KnapsackDataset, KnapsackGenerator, KnapsackSolver
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.eval.reporting import print_evaluation_summary, save_results_to_json
from knapsack_gnn.models.pna import create_model


def main():
    parser = argparse.ArgumentParser(description="Large Instance Evaluation")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/run_20251020_104533",
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--checkpoint_name", type=str, default="best_model.pt", help="Checkpoint filename"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/datasets",
        help="Directory containing training dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoint_dir/evaluation/large_instances)",
    )
    parser.add_argument(
        "--n_instances", type=int, default=30, help="Number of instances per size (default: 30)"
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument("--strategy", type=str, default="sampling", help="Decoding strategy")
    parser.add_argument(
        "--n_samples", type=int, default=500, help="Number of samples for large instances"
    )
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "evaluation", "large_instances")
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 80)
    print("LARGE INSTANCE COMPREHENSIVE EVALUATION")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint_dir}/{args.checkpoint_name}")
    print(f"  Output: {args.output_dir}")
    print(f"  Instances per size: {args.n_instances}")
    print(f"  Samples per instance: {args.n_samples}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    train_dataset = KnapsackDataset.load(f"{args.data_dir}/train.pkl")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)

    training_sizes = [inst.n_items for inst in train_dataset.instances]
    training_size_range = (min(training_sizes), max(training_sizes))
    print(f"Training size range: {training_size_range[0]}-{training_size_range[1]} items")

    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = create_model(train_graph_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    print("Model loaded successfully")
    if "epochs_trained" in checkpoint:
        print(f"Epochs trained: {checkpoint['epochs_trained']}")

    # Test sizes
    test_sizes = [500, 1000, 2000]

    all_results = {}
    generator = KnapsackGenerator(seed=args.seed)

    for size in test_sizes:
        print("\n" + "=" * 80)
        print(f"EVALUATING SIZE: n={size}")
        print("=" * 80)

        # Generate instances
        print(f"\n1. Generating {args.n_instances} instances...")
        instances = generator.generate_dataset(
            n_instances=args.n_instances,
            n_items_range=(size, size),
        )

        # Solve with OR-Tools
        print("2. Solving with OR-Tools (this may take a while for large instances)...")
        instances = KnapsackSolver.solve_batch(instances, verbose=True)

        # Check solve status
        solved = [inst for inst in instances if inst.optimal_value is not None]
        print(f"   Successfully solved: {len(solved)}/{len(instances)}")

        if len(solved) == 0:
            print("   ⚠️  WARNING: No instances were solved by OR-Tools!")
            continue

        # Use only solved instances
        dataset = KnapsackDataset(solved)

        # Save dataset for future use
        dataset_path = os.path.join(args.output_dir, f"dataset_n{size}.pkl")
        dataset.save(dataset_path)
        print(f"   Dataset saved to: {dataset_path}")

        # Build graph dataset
        print("3. Building graph dataset...")
        graph_dataset = KnapsackGraphDataset(dataset, normalize_features=True)

        # Evaluate
        print("4. Running GNN evaluation...")
        print(f"   Strategy: {args.strategy}")
        print(f"   Samples: {args.n_samples}")
        print(f"   Temperature: {args.temperature}")

        # Use adaptive sampling schedule for large instances
        sampling_schedule = (32, 64, 128, 256) if size >= 1000 else (32, 64, 128)

        results = evaluate_model(
            model=model,
            dataset=graph_dataset,
            strategy=args.strategy,
            device=args.device,
            temperature=args.temperature,
            sampling_schedule=sampling_schedule,
            sampling_tolerance=1e-3,
            max_samples=args.n_samples,
            sampler_kwargs={
                "num_threads": None,
                "compile_model": False,
                "quantize": False,
            },
        )

        # Add metadata
        results["problem_size"] = size
        results["n_instances"] = len(solved)
        results["n_requested"] = args.n_instances
        results["training_size_range"] = training_size_range

        # Print summary
        print(f"\n5. Results for n={size}:")
        print_evaluation_summary(results)

        # Save individual results
        results_path = os.path.join(args.output_dir, f"results_n{size}.json")
        save_results_to_json(results, results_path)
        print(f"   Results saved to: {results_path}")

        all_results[size] = results

    # Generate comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)

    print("\n" + "-" * 80)
    print(
        f"{'Size':>6} | {'N':>4} | {'Mean Gap':>10} | {'Median Gap':>12} | "
        f"{'Std Gap':>10} | {'Feasible':>9} | {'Time (ms)':>10}"
    )
    print("-" * 80)

    for size in test_sizes:
        if size not in all_results:
            continue
        r = all_results[size]
        print(
            f"{size:>6} | {r['n_instances']:>4} | "
            f"{r['mean_gap']:>9.2f}% | {r['median_gap']:>11.2f}% | "
            f"{r['std_gap']:>9.2f}% | {r['feasibility_rate'] * 100:>8.1f}% | "
            f"{r.get('mean_inference_time', 0) * 1000:>10.2f}"
        )

    print("-" * 80)

    # Analysis
    print("\nGeneralization Analysis:")
    print(f"  Training range: {training_size_range[0]}-{training_size_range[1]} items")

    for size in test_sizes:
        if size not in all_results:
            continue
        r = all_results[size]
        extrapolation_factor = size / training_size_range[1]
        gap = r["mean_gap"]

        print(f"\n  Size n={size} (extrapolation {extrapolation_factor:.1f}x):")
        print(f"    Mean gap: {gap:.2f}%")
        print(f"    Median gap: {r['median_gap']:.2f}%")
        print(f"    Std gap: {r['std_gap']:.2f}%")
        print(f"    Feasibility: {r['feasibility_rate'] * 100:.1f}%")

        if gap < 2.0:
            print("    ✅ Excellent! Gap < 2%")
        elif gap < 5.0:
            print("    ✓ Good. Gap < 5%")
        elif gap < 10.0:
            print("    ⚠ Moderate. Gap < 10%")
        else:
            print("    ❌ Poor. Gap >= 10%")

    # Save comprehensive summary
    summary = {
        "training_size_range": training_size_range,
        "test_sizes": test_sizes,
        "n_instances_per_size": args.n_instances,
        "strategy": args.strategy,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "results": {
            str(size): {
                "n_instances": r["n_instances"],
                "mean_gap": r["mean_gap"],
                "median_gap": r["median_gap"],
                "std_gap": r["std_gap"],
                "feasibility_rate": r["feasibility_rate"],
                "mean_inference_time": r.get("mean_inference_time", 0),
                "extrapolation_factor": size / training_size_range[1],
            }
            for size, r in all_results.items()
        },
    }

    summary_path = os.path.join(args.output_dir, "comprehensive_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComprehensive summary saved to: {summary_path}")

    # Generate statistical comparison
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 80)

    if len(all_results) >= 2:
        sizes_sorted = sorted(all_results.keys())
        for i in range(len(sizes_sorted) - 1):
            size1, size2 = sizes_sorted[i], sizes_sorted[i + 1]
            r1, r2 = all_results[size1], all_results[size2]

            if "gaps" in r1 and "gaps" in r2:
                gaps1 = np.array(r1["gaps"])
                gaps2 = np.array(r2["gaps"])

                # Perform t-test
                from scipy import stats

                t_stat, p_value = stats.ttest_ind(gaps1, gaps2)

                print(f"\nn={size1} vs n={size2}:")
                print(f"  Mean gap difference: {r2['mean_gap'] - r1['mean_gap']:.2f}%")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.4f}")

                if p_value < 0.05:
                    if r2["mean_gap"] > r1["mean_gap"]:
                        print("  → Significant degradation (p < 0.05)")
                    else:
                        print("  → Significant improvement (p < 0.05)")
                else:
                    print("  → No significant difference (p >= 0.05)")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(f"\nAll results saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
