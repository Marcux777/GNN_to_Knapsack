# mypy: ignore-errors
"""
Publication-Grade Validation Orchestrator

Comprehensive validation framework for academic publications:
- Automated statistical testing
- Cross-validation experiments
- Baseline comparisons with rigorous statistics
- Publication-ready output generation

Usage:
    from knapsack_gnn.analysis.validation import PublicationValidator

    validator = PublicationValidator(output_dir='validation_report')
    validator.run_full_validation(
        gnn_model=model,
        dataset=test_dataset,
        baselines=['greedy', 'random'],
        config=config
    )
"""

import json
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from knapsack_gnn.analysis.cross_validation import KFoldValidator
from knapsack_gnn.analysis.reporting import AcademicReporter
from knapsack_gnn.analysis.stats import StatisticalAnalyzer
from knapsack_gnn.baselines.greedy import GreedySolver, RandomSolver
from knapsack_gnn.data.generator import KnapsackDataset


class PublicationValidator:
    """
    Orchestrates complete validation for academic publications
    """

    def __init__(
        self,
        output_dir: str = "validation_report",
        alpha: float = 0.05,
        n_bootstrap: int = 10000,
        random_state: int = 42,
    ):
        """
        Args:
            output_dir: Directory for validation outputs
            alpha: Significance level for statistical tests
            n_bootstrap: Bootstrap samples for CI
            random_state: Random seed
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

        # Initialize components
        self.stats_analyzer = StatisticalAnalyzer(alpha=alpha, n_bootstrap=n_bootstrap)
        self.reporter = AcademicReporter()

        # Results storage
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "config": {"alpha": alpha, "n_bootstrap": n_bootstrap, "random_state": random_state},
            "baseline_comparisons": {},
            "cross_validation": {},
            "statistical_tests": {},
            "assumptions_checks": {},
            "power_analysis": {},
        }

    def compare_with_baselines(
        self,
        gnn_gaps: np.ndarray,
        dataset: KnapsackDataset,
        baselines: list[str] = None,
        gnn_name: str = "GNN",
        verbose: bool = True,
    ) -> dict:
        """
        Compare GNN against baseline methods with rigorous statistics

        Args:
            gnn_gaps: Optimality gaps from GNN
            dataset: Test dataset
            baselines: List of baseline names
            gnn_name: Name for GNN method
            verbose: Print results

        Returns:
            Dictionary with comparison results
        """
        if baselines is None:
            baselines = ["greedy", "random"]
        if verbose:
            print("\n" + "=" * 70)
            print("BASELINE COMPARISON WITH STATISTICAL VALIDATION")
            print("=" * 70)

        comparison_results = {}
        baseline_gaps = {}

        # Collect baseline results
        for baseline_name in baselines:
            if verbose:
                print(f"\nRunning {baseline_name} solver...")

            if baseline_name.lower() == "greedy":
                solver = GreedySolver()
                results = solver.solve_batch(dataset.instances, verbose=False)
                gaps = np.array(
                    [r["optimality_gap"] for r in results if r["optimality_gap"] is not None]
                )

            elif baseline_name.lower() == "random":
                solver = RandomSolver(seed=self.random_state)
                results = []
                for inst in dataset.instances:
                    result = solver.solve(inst, max_attempts=100)
                    results.append(result)
                gaps = np.array(
                    [r["optimality_gap"] for r in results if r["optimality_gap"] is not None]
                )

            else:
                warnings.warn(f"Unknown baseline: {baseline_name}", stacklevel=2)
                continue

            baseline_gaps[baseline_name] = gaps

            if verbose:
                print(f"  {baseline_name} mean gap: {np.mean(gaps):.4f}%")

        # Perform statistical comparisons
        for baseline_name, baseline_gap in baseline_gaps.items():
            if verbose:
                print(f"\n{'-' * 70}")
                print(f"Statistical Comparison: {gnn_name} vs {baseline_name}")
                print(f"{'-' * 70}")

            # Check assumptions
            assumptions = self.check_test_assumptions(
                gnn_gaps, baseline_gap, method_a_name=gnn_name, method_b_name=baseline_name
            )

            # Paired comparison (same instances)
            comparison = self.stats_analyzer.paired_comparison(
                gnn_gaps, baseline_gap, method_a_name=gnn_name, method_b_name=baseline_name
            )

            # Additional effect sizes
            cliffs_delta = self.stats_analyzer.cliffs_delta(gnn_gaps, baseline_gap)
            vd_a = self.stats_analyzer.vargha_delaney_a(gnn_gaps, baseline_gap)

            comparison["cliffs_delta"] = {
                "value": cliffs_delta,
                "interpretation": self.stats_analyzer._interpret_cliffs_delta(cliffs_delta),
            }
            comparison["vargha_delaney_a"] = {
                "value": vd_a,
                "interpretation": self.stats_analyzer._interpret_vargha_delaney_a(vd_a),
            }
            comparison["assumptions"] = assumptions

            comparison_results[baseline_name] = comparison

            if verbose:
                print(self.reporter.format_statistical_summary(comparison, include_latex=False))

        # Save results
        self.results["baseline_comparisons"] = comparison_results
        self._save_results()

        # Generate LaTeX tables
        self._generate_baseline_tables(gnn_name, gnn_gaps, baseline_gaps, comparison_results)

        return comparison_results

    def check_test_assumptions(
        self,
        method_a: np.ndarray,
        method_b: np.ndarray,
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
    ) -> dict:
        """
        Check assumptions for parametric tests

        Args:
            method_a: Data from method A
            method_b: Data from method B
            method_a_name: Name of method A
            method_b_name: Name of method B

        Returns:
            Dictionary with assumption test results
        """
        assumptions = {}

        # Normality tests
        assumptions["normality_a"] = self.stats_analyzer.shapiro_wilk_test(method_a)
        assumptions["normality_b"] = self.stats_analyzer.shapiro_wilk_test(method_b)

        # Test normality of differences (for paired tests)
        differences = method_a - method_b
        assumptions["normality_differences"] = self.stats_analyzer.shapiro_wilk_test(differences)

        # Homogeneity of variances
        assumptions["equal_variances"] = self.stats_analyzer.levene_test(method_a, method_b)

        # Recommendation
        normality_ok = assumptions["normality_differences"]["is_normal"] or (
            assumptions["normality_a"]["is_normal"] and assumptions["normality_b"]["is_normal"]
        )
        variances_ok = assumptions["equal_variances"]["equal_variances"]

        if normality_ok and variances_ok:
            assumptions["recommendation"] = "parametric"  # Use t-test
        else:
            assumptions["recommendation"] = "non-parametric"  # Use Wilcoxon

        assumptions["summary"] = (
            f"Normality: {'OK' if normality_ok else 'VIOLATED'}, "
            + f"Equal variances: {'OK' if variances_ok else 'VIOLATED'}"
        )

        return assumptions

    def run_cross_validation(
        self,
        train_fn: Callable,
        evaluate_fn: Callable,
        dataset: KnapsackDataset,
        config: dict,
        n_folds: int = 5,
        stratify: bool = True,
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict:
        """
        Run k-fold cross-validation

        Args:
            train_fn: Training function
            evaluate_fn: Evaluation function
            dataset: Full dataset
            config: Configuration
            n_folds: Number of folds
            stratify: Stratify by problem size
            device: Device
            verbose: Verbose output

        Returns:
            Cross-validation results
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"K-FOLD CROSS-VALIDATION (k={n_folds})")
            print("=" * 70)

        validator = KFoldValidator(
            n_splits=n_folds, random_state=self.random_state, stratify=stratify
        )

        cv_results = validator.validate(
            train_fn=train_fn,
            evaluate_fn=evaluate_fn,
            dataset=dataset,
            config=config,
            device=device,
            verbose=verbose,
        )

        # Store results
        self.results["cross_validation"] = {
            "n_folds": n_folds,
            "stratify": stratify,
            "mean_gap": cv_results.mean_gap,
            "std_gap": cv_results.std_gap,
            "ci_95": cv_results.ci_95,
            "fold_results": cv_results.fold_results,
        }
        self._save_results()

        return cv_results

    def run_power_analysis(
        self,
        observed_effect_size: float,
        current_sample_size: int,
        desired_power: float = 0.8,
        verbose: bool = True,
    ) -> dict:
        """
        Perform statistical power analysis

        Args:
            observed_effect_size: Observed Cohen's d
            current_sample_size: Current n
            desired_power: Target power
            verbose: Print results

        Returns:
            Power analysis results
        """
        if verbose:
            print("\n" + "=" * 70)
            print("STATISTICAL POWER ANALYSIS")
            print("=" * 70)

        # Compute achieved power with current sample size
        achieved_power = self.stats_analyzer.statistical_power(
            effect_size=observed_effect_size, n_samples=current_sample_size, alpha=self.alpha
        )

        # Compute required sample size for desired power
        required_n = self.stats_analyzer.required_sample_size(
            effect_size=observed_effect_size, power=desired_power, alpha=self.alpha
        )

        results = {
            "observed_effect_size": observed_effect_size,
            "current_sample_size": current_sample_size,
            "achieved_power": achieved_power,
            "desired_power": desired_power,
            "required_sample_size": required_n,
            "sample_size_adequate": current_sample_size >= required_n if required_n else None,
        }

        self.results["power_analysis"] = results
        self._save_results()

        if verbose:
            print(f"\nObserved effect size (Cohen's d): {observed_effect_size:.3f}")
            print(f"Current sample size: {current_sample_size}")
            print(
                f"Achieved power: {achieved_power:.3f}"
                if achieved_power
                else "Power: N/A (install statsmodels)"
            )
            print(
                f"Required n for power={desired_power}: {required_n}"
                if required_n
                else "Required n: N/A"
            )
            if results["sample_size_adequate"] is not None:
                if results["sample_size_adequate"]:
                    print("✓ Sample size is adequate for desired power")
                else:
                    print(
                        f"✗ Sample size insufficient. Need {required_n - current_sample_size} more samples."
                    )

        return results

    def compare_multiple_methods(
        self, method_results: dict[str, np.ndarray], verbose: bool = True
    ) -> dict:
        """
        Compare multiple methods using Friedman test

        Args:
            method_results: Dict mapping method names to gap arrays
            verbose: Print results

        Returns:
            Comparison results
        """
        if len(method_results) < 3:
            warnings.warn("Friedman test requires at least 3 methods", stacklevel=2)
            return {}

        if verbose:
            print("\n" + "=" * 70)
            print("MULTIPLE METHODS COMPARISON (Friedman Test)")
            print("=" * 70)

        methods = list(method_results.keys())
        gaps = [method_results[m] for m in methods]

        # Friedman test
        friedman_result = self.stats_analyzer.friedman_test(*gaps)

        # Pairwise comparisons with Holm correction
        pairwise_comparisons = []
        p_values = []

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                comparison = self.stats_analyzer.wilcoxon_test(gaps[i], gaps[j])
                comparison["method_a"] = methods[i]
                comparison["method_b"] = methods[j]
                pairwise_comparisons.append(comparison)
                p_values.append(comparison["p_value"])

        # Multiple testing correction
        holm_correction = self.stats_analyzer.holm_correction(p_values, alpha=self.alpha)

        # Update comparisons with corrected significance
        for comp, significant in zip(
            pairwise_comparisons, holm_correction["significant"], strict=False
        ):
            comp["significant_corrected"] = significant

        results = {
            "friedman_test": friedman_result,
            "pairwise_comparisons": pairwise_comparisons,
            "multiple_testing_correction": holm_correction,
        }

        self.results["multiple_methods_comparison"] = results
        self._save_results()

        if verbose:
            print("\nFriedman Test:")
            print(f"  Statistic: {friedman_result['statistic']:.3f}")
            print(f"  p-value: {friedman_result['p_value']:.6f}")
            print(f"  Significant: {'YES' if friedman_result['significant'] else 'NO'}")
            print("\nMean Ranks:")
            for method, rank in zip(methods, friedman_result["mean_ranks"], strict=False):
                print(f"  {method}: {rank:.2f}")

            print("\nPairwise Comparisons (Holm-corrected):")
            for comp in pairwise_comparisons:
                sig = "✓" if comp["significant_corrected"] else "✗"
                print(f"  {comp['method_a']} vs {comp['method_b']}: p={comp['p_value']:.4f} {sig}")

        return results

    def generate_validation_report(
        self, include_latex: bool = True, include_figures: bool = True
    ) -> str:
        """
        Generate comprehensive validation report

        Args:
            include_latex: Include LaTeX tables
            include_figures: Generate figures

        Returns:
            Path to report file
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("PUBLICATION-GRADE VALIDATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"\nGenerated: {self.results['timestamp']}")
        report_lines.append(f"Significance level: α = {self.alpha}")
        report_lines.append(f"Bootstrap samples: {self.n_bootstrap}")

        # Baseline comparisons
        if self.results["baseline_comparisons"]:
            report_lines.append("\n" + "=" * 70)
            report_lines.append("BASELINE COMPARISONS")
            report_lines.append("=" * 70)

            for baseline, comp in self.results["baseline_comparisons"].items():
                report_lines.append(f"\n{comp['method_a']} vs {baseline}:")
                report_lines.append(f"  Paired t-test: p = {comp['t_test']['p_value']:.6f}")
                report_lines.append(f"  Wilcoxon: p = {comp['wilcoxon']['p_value']:.6f}")
                report_lines.append(
                    f"  Cohen's d = {comp['cohens_d']['value']:.3f} ({comp['cohens_d']['interpretation']})"
                )
                report_lines.append(f"  Assumptions: {comp['assumptions']['summary']}")

        # Cross-validation
        if self.results["cross_validation"]:
            cv = self.results["cross_validation"]
            report_lines.append("\n" + "=" * 70)
            report_lines.append("CROSS-VALIDATION")
            report_lines.append("=" * 70)
            report_lines.append(f"\n{cv['n_folds']}-Fold CV Results:")
            report_lines.append(f"  Mean Gap: {cv['mean_gap']:.4f}% ± {cv['std_gap']:.4f}%")
            report_lines.append(f"  95% CI: [{cv['ci_95'][0]:.4f}%, {cv['ci_95'][1]:.4f}%]")

        # Power analysis
        if self.results["power_analysis"]:
            pa = self.results["power_analysis"]
            report_lines.append("\n" + "=" * 70)
            report_lines.append("POWER ANALYSIS")
            report_lines.append("=" * 70)
            report_lines.append(f"\nEffect size: d = {pa['observed_effect_size']:.3f}")
            report_lines.append(f"Sample size: n = {pa['current_sample_size']}")
            if pa["achieved_power"]:
                report_lines.append(f"Achieved power: {pa['achieved_power']:.3f}")
            if pa["required_sample_size"]:
                report_lines.append(f"Required n for 80% power: {pa['required_sample_size']}")

        report_lines.append("\n" + "=" * 70)

        report_text = "\n".join(report_lines)

        # Save report
        report_path = self.output_dir / "validation_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        print(f"\nValidation report saved to: {report_path}")

        return str(report_path)

    def _generate_baseline_tables(
        self,
        gnn_name: str,
        gnn_gaps: np.ndarray,
        baseline_gaps: dict[str, np.ndarray],
        comparison_results: dict,
    ):
        """Generate LaTeX tables for baseline comparison"""
        # Comparison table
        results_dict = {
            gnn_name: {
                "mean_gap": np.mean(gnn_gaps),
                "median_gap": np.median(gnn_gaps),
                "std_gap": np.std(gnn_gaps, ddof=1),
            }
        }

        for baseline, gaps in baseline_gaps.items():
            results_dict[baseline] = {
                "mean_gap": np.mean(gaps),
                "median_gap": np.median(gaps),
                "std_gap": np.std(gaps, ddof=1),
            }

        latex_table = self.reporter.generate_comparison_table(
            results_dict,
            metrics=["mean_gap", "median_gap", "std_gap"],
            caption="Performance comparison across methods",
            label="tab:baseline_comparison",
        )

        latex_path = self.output_dir / "baseline_comparison_table.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)

        print(f"LaTeX table saved to: {latex_path}")

        # Statistical tests table
        test_results = []
        for baseline, comp in comparison_results.items():
            test_results.append(
                {
                    "method_a": gnn_name,
                    "method_b": baseline,
                    "test": "Paired t-test",
                    "t_statistic": comp["t_test"]["t_statistic"],
                    "p_value": comp["t_test"]["p_value"],
                    "significant": comp["t_test"]["significant"],
                }
            )

        latex_tests = self.reporter.generate_statistical_test_table(
            test_results, caption="Statistical significance tests", label="tab:statistical_tests"
        )

        tests_path = self.output_dir / "statistical_tests_table.tex"
        with open(tests_path, "w") as f:
            f.write(latex_tests)

        print(f"Statistical tests table saved to: {tests_path}")

    def _save_results(self):
        """Save results to JSON"""
        results_path = self.output_dir / "validation_results.json"

        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer | np.floating):
                return float(obj)
            return obj

        serializable = json.loads(json.dumps(self.results, default=convert))

        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    print("Publication Validator Module")
    print("=" * 70)
    print("\nComprehensive validation framework for academic publications.")
    print("\nExample usage:")
    print("""
    from knapsack_gnn.analysis.validation import PublicationValidator

    validator = PublicationValidator(output_dir='my_validation')

    # Compare with baselines
    validator.compare_with_baselines(
        gnn_gaps=gnn_results,
        dataset=test_dataset,
        baselines=['greedy', 'random']
    )

    # Run cross-validation
    validator.run_cross_validation(
        train_fn=my_train_fn,
        evaluate_fn=my_eval_fn,
        dataset=full_dataset,
        config=config
    )

    # Generate report
    validator.generate_validation_report()
    """)
