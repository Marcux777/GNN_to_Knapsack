# mypy: ignore-errors
"""
Complete Publication-Grade Validation Pipeline

Runs all validation experiments and generates publication-ready outputs:
1. Baseline comparisons with statistical tests
2. Cross-validation for generalization
3. Assumption checking
4. Power analysis
5. LaTeX tables and figures

Usage:
    python publication_validation.py --checkpoint checkpoints/run_xxx --output validation_report
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knapsack_gnn.analysis.reporting import AcademicReporter
from knapsack_gnn.analysis.stats import StatisticalAnalyzer
from knapsack_gnn.analysis.validation import PublicationValidator
from knapsack_gnn.data.generator import KnapsackDataset
from knapsack_gnn.data.graph_builder import KnapsackGraphDataset
from knapsack_gnn.decoding.sampling import evaluate_model
from knapsack_gnn.models.pna import create_model
from knapsack_gnn.training.loop import train_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Publication-Grade Validation Pipeline")

    # Model parameters
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/datasets", help="Directory containing datasets"
    )

    # Validation parameters
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["greedy", "random"],
        help="Baseline methods to compare against",
    )
    parser.add_argument(
        "--run_cv", action="store_true", help="Run cross-validation (requires training)"
    )
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--stratify_cv", action="store_true", help="Stratify CV by problem size")

    # Statistical parameters
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Bootstrap samples for CI")
    parser.add_argument("--check_power", action="store_true", help="Run statistical power analysis")

    # Inference parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="sampling",
        choices=["threshold", "sampling", "adaptive", "lagrangian", "warm_start"],
        help="GNN inference strategy",
    )
    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of samples for sampling strategy"
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="validation_report",
        help="Output directory for validation results",
    )
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--figures", action="store_true", help="Generate publication figures")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def load_model_and_data(checkpoint_path: str, data_dir: str, device: str) -> tuple:
    """Load trained model and datasets"""
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = KnapsackDataset.load(f"{data_dir}/train.pkl")
    val_dataset = KnapsackDataset.load(f"{data_dir}/val.pkl")
    test_dataset = KnapsackDataset.load(f"{data_dir}/test.pkl")

    print(f"  Train: {len(train_dataset)} instances")
    print(f"  Val: {len(val_dataset)} instances")
    print(f"  Test: {len(test_dataset)} instances")

    # Build graph datasets
    print("\nBuilding graph datasets...")
    train_graph_dataset = KnapsackGraphDataset(train_dataset, normalize_features=True)
    val_graph_dataset = KnapsackGraphDataset(val_dataset, normalize_features=True)
    test_graph_dataset = KnapsackGraphDataset(test_dataset, normalize_features=True)

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = create_model(train_graph_dataset, hidden_dim=64, num_layers=3, dropout=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully")

    return (
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        train_graph_dataset,
        val_graph_dataset,
        test_graph_dataset,
    )


def evaluate_gnn(
    model, test_graph_dataset, strategy: str, n_samples: int, device: str
) -> np.ndarray:
    """Evaluate GNN model"""
    print("\n" + "=" * 70)
    print("EVALUATING GNN MODEL")
    print("=" * 70)

    strategy_kwargs = {}
    if strategy == "sampling":
        strategy_kwargs = {"n_samples": n_samples, "temperature": 1.0}

    results = evaluate_model(
        model=model, dataset=test_graph_dataset, strategy=strategy, device=device, **strategy_kwargs
    )

    print("\nGNN Results:")
    print(f"  Mean Gap: {results['mean_gap']:.4f}%")
    print(f"  Median Gap: {results['median_gap']:.4f}%")
    print(f"  Std Gap: {results['std_gap']:.4f}%")
    print(f"  Feasibility: {results['feasibility_rate'] * 100:.2f}%")

    return np.array(results["gaps"])


def main() -> None:
    """Main validation pipeline"""
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("PUBLICATION-GRADE VALIDATION PIPELINE")
    print("=" * 70)
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize validator
    validator = PublicationValidator(
        output_dir=args.output_dir,
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap,
        random_state=args.seed,
    )

    # Load model and data
    (
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        train_graph_dataset,
        val_graph_dataset,
        test_graph_dataset,
    ) = load_model_and_data(args.checkpoint, args.data_dir, args.device)

    # ===== STEP 1: Evaluate GNN =====
    gnn_gaps = evaluate_gnn(model, test_graph_dataset, args.strategy, args.n_samples, args.device)

    # ===== STEP 2: Baseline Comparison =====
    print("\n" + "=" * 70)
    print("STEP 2: BASELINE COMPARISON WITH STATISTICAL VALIDATION")
    print("=" * 70)

    baseline_comparisons = validator.compare_with_baselines(
        gnn_gaps=gnn_gaps,
        dataset=test_dataset,
        baselines=args.baselines,
        gnn_name="GNN-PNA",
        verbose=True,
    )

    # ===== STEP 3: Cross-Validation (Optional) =====
    if args.run_cv:
        print("\n" + "=" * 70)
        print("STEP 3: CROSS-VALIDATION")
        print("=" * 70)

        # Define training and evaluation functions
        def train_fn(train_ds, val_ds, config, device):
            train_graph = KnapsackGraphDataset(train_ds, normalize_features=True)
            val_graph = KnapsackGraphDataset(val_ds, normalize_features=True)

            model = create_model(train_graph, hidden_dim=64, num_layers=3, dropout=0.1)
            model = model.to(device)

            trained_model, history = train_model(
                model=model,
                train_dataset=train_graph,
                val_dataset=val_graph,
                num_epochs=config.get("epochs", 30),
                batch_size=config.get("batch_size", 32),
                learning_rate=config.get("lr", 0.002),
                device=device,
                checkpoint_dir=None,  # Don't save intermediate checkpoints
            )

            return trained_model, history

        def eval_fn(model, test_ds, device):
            test_graph = KnapsackGraphDataset(test_ds, normalize_features=True)
            results = evaluate_model(
                model=model,
                dataset=test_graph,
                strategy=args.strategy,
                device=device,
                n_samples=args.n_samples,
            )
            return results

        config = {"epochs": 30, "batch_size": 32, "lr": 0.002}

        validator.run_cross_validation(
            train_fn=train_fn,
            evaluate_fn=eval_fn,
            dataset=train_dataset,  # Use training data for CV
            config=config,
            n_folds=args.cv_folds,
            stratify=args.stratify_cv,
            device=args.device,
            verbose=True,
        )

    # ===== STEP 4: Power Analysis =====
    if args.check_power and baseline_comparisons:
        print("\n" + "=" * 70)
        print("STEP 4: STATISTICAL POWER ANALYSIS")
        print("=" * 70)

        # Use first baseline for power analysis
        first_baseline = list(baseline_comparisons.keys())[0]
        comparison = baseline_comparisons[first_baseline]

        effect_size = comparison["cohens_d"]["value"]
        sample_size = comparison["n_samples"]

        validator.run_power_analysis(
            observed_effect_size=abs(effect_size),
            current_sample_size=sample_size,
            desired_power=0.8,
            verbose=True,
        )

    # ===== STEP 5: Multiple Methods Comparison =====
    if len(args.baselines) >= 2:
        print("\n" + "=" * 70)
        print("STEP 5: MULTIPLE METHODS COMPARISON")
        print("=" * 70)

        # Collect all method results
        method_results = {"GNN-PNA": gnn_gaps}

        # Add baseline results
        from knapsack_gnn.baselines.greedy import GreedySolver, RandomSolver

        for baseline in args.baselines:
            if baseline.lower() == "greedy":
                solver = GreedySolver()
                results = solver.solve_batch(test_dataset.instances, verbose=False)
                gaps = np.array(
                    [r["optimality_gap"] for r in results if r["optimality_gap"] is not None]
                )
                method_results["Greedy"] = gaps
            elif baseline.lower() == "random":
                solver = RandomSolver(seed=args.seed)
                results = []
                for inst in test_dataset.instances:
                    result = solver.solve(inst, max_attempts=100)
                    results.append(result)
                gaps = np.array(
                    [r["optimality_gap"] for r in results if r["optimality_gap"] is not None]
                )
                method_results["Random"] = gaps

        if len(method_results) >= 3:
            validator.compare_multiple_methods(method_results=method_results, verbose=True)

    # ===== STEP 6: Generate Report =====
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING VALIDATION REPORT")
    print("=" * 70)

    validator.generate_validation_report(include_latex=args.latex, include_figures=args.figures)

    # ===== STEP 7: Generate Figures =====
    if args.figures:
        print("\n" + "=" * 70)
        print("STEP 7: GENERATING PUBLICATION FIGURES")
        print("=" * 70)

        reporter = AcademicReporter()

        # Box plot comparison
        if "Greedy" in method_results and "Random" in method_results:
            reporter.create_boxplot_comparison(
                data=method_results,
                ylabel="Optimality Gap (%)",
                title="Method Comparison",
                save_path=str(output_dir / "method_comparison"),
            )
            print(f"Saved: {output_dir / 'method_comparison.pdf'}")

        # Confidence interval plot
        summary_results = {}
        for method, gaps in method_results.items():
            analyzer = StatisticalAnalyzer()
            ci = analyzer.bootstrap_ci(gaps)
            summary_results[method] = {"mean": np.mean(gaps), "ci_95": ci}

        reporter.create_confidence_interval_plot(
            results=summary_results,
            ylabel="Optimality Gap (%)",
            title="Mean Gap with 95% Confidence Intervals",
            save_path=str(output_dir / "confidence_intervals"),
        )
        print(f"Saved: {output_dir / 'confidence_intervals.pdf'}")

    # ===== Final Summary =====
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - validation_results.json")
    print("  - validation_report.txt")
    if args.latex:
        print("  - baseline_comparison_table.tex")
        print("  - statistical_tests_table.tex")
    if args.figures:
        print("  - method_comparison.pdf")
        print("  - confidence_intervals.pdf")

    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)

    # Print key findings
    if baseline_comparisons:
        print("\n✓ Baseline Comparisons:")
        for baseline, comp in baseline_comparisons.items():
            sig = "✓ SIGNIFICANT" if comp["t_test"]["significant"] else "✗ NOT SIGNIFICANT"
            print(f"  - GNN vs {baseline}: {sig} (p={comp['t_test']['p_value']:.6f})")
            print(
                f"    Effect size: d={comp['cohens_d']['value']:.3f} ({comp['cohens_d']['interpretation']})"
            )

    if args.run_cv and "cross_validation" in validator.results:
        cv = validator.results["cross_validation"]
        print(f"\n✓ Cross-Validation ({cv['n_folds']}-fold):")
        print(f"  Mean Gap: {cv['mean_gap']:.4f}% ± {cv['std_gap']:.4f}%")
        print(f"  95% CI: [{cv['ci_95'][0]:.4f}%, {cv['ci_95'][1]:.4f}%]")

    if args.check_power and "power_analysis" in validator.results:
        pa = validator.results["power_analysis"]
        print("\n✓ Statistical Power:")
        if pa["achieved_power"]:
            print(f"  Achieved power: {pa['achieved_power']:.3f}")
        if pa["sample_size_adequate"] is not None:
            status = "ADEQUATE" if pa["sample_size_adequate"] else "INSUFFICIENT"
            print(f"  Sample size: {status}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
