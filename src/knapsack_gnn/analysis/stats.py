"""
Statistical Analysis Module for Publication-Grade Validation

Provides rigorous statistical tests for comparing optimization methods:
- Paired statistical tests (t-test, Wilcoxon, Sign test)
- Independent tests (t-test, Mann-Whitney U)
- Multiple methods comparison (Friedman test)
- Effect size metrics (Cohen's d, Cliff's Delta, Vargha-Delaney A)
- Bootstrap confidence intervals
- Multiple testing correction (Bonferroni, Holm, Benjamini-Hochberg)
- Assumption testing (normality, homoscedasticity)
- Statistical power analysis

Usage:
    from knapsack_gnn.analysis.stats import compare_methods, StatisticalAnalyzer

    analyzer = StatisticalAnalyzer()
    results = analyzer.paired_comparison(method_a_gaps, method_b_gaps)
"""

import warnings

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for optimization algorithm comparison
    """

    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        """
        Args:
            alpha: Significance level (default: 0.05 for 95% CI)
            n_bootstrap: Number of bootstrap samples (default: 10000)
        """
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap

    def paired_t_test(self, method_a: np.ndarray, method_b: np.ndarray) -> dict:
        """
        Paired t-test for comparing two methods on the same instances

        Args:
            method_a: Results from method A (e.g., optimality gaps)
            method_b: Results from method B

        Returns:
            Dictionary with test statistics
        """
        method_a = np.asarray(method_a)
        method_b = np.asarray(method_b)

        if len(method_a) != len(method_b):
            raise ValueError("Methods must have same number of samples for paired test")

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(method_a, method_b)

        # Mean difference and CI
        differences = method_a - method_b
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(len(differences))

        # 95% CI using t-distribution
        df = len(differences) - 1
        t_crit = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        return {
            "test": "Paired t-test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "ci_95": (float(ci_lower), float(ci_upper)),
            "significant": p_value < self.alpha,
            "winner": "method_b" if mean_diff > 0 else "method_a" if mean_diff < 0 else "tie",
        }

    def wilcoxon_test(self, method_a: np.ndarray, method_b: np.ndarray) -> dict:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)

        Recommended when data is not normally distributed

        Args:
            method_a: Results from method A
            method_b: Results from method B

        Returns:
            Dictionary with test statistics
        """
        method_a = np.asarray(method_a)
        method_b = np.asarray(method_b)

        if len(method_a) != len(method_b):
            raise ValueError("Methods must have same number of samples")

        # Wilcoxon signed-rank test
        stat, p_value = stats.wilcoxon(method_a, method_b, alternative="two-sided")

        differences = method_a - method_b
        median_diff = np.median(differences)

        return {
            "test": "Wilcoxon signed-rank",
            "statistic": float(stat),
            "p_value": float(p_value),
            "median_difference": float(median_diff),
            "significant": p_value < self.alpha,
            "winner": "method_b" if median_diff > 0 else "method_a" if median_diff < 0 else "tie",
        }

    def cohens_d(self, method_a: np.ndarray, method_b: np.ndarray) -> float:
        """
        Cohen's d effect size for paired samples

        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

        Args:
            method_a: Results from method A
            method_b: Results from method B

        Returns:
            Cohen's d effect size
        """
        differences = np.asarray(method_a) - np.asarray(method_b)
        return float(np.mean(differences) / np.std(differences, ddof=1))

    def bootstrap_ci(
        self, data: np.ndarray, statistic_fn: callable = np.mean, confidence: float = 0.95
    ) -> tuple[float, float]:
        """
        Bootstrap confidence interval for any statistic

        Args:
            data: Data array
            statistic_fn: Function to compute statistic (default: mean)
            confidence: Confidence level (default: 0.95)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        data = np.asarray(data)
        n = len(data)

        # Generate bootstrap samples
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Compute percentile CI
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def bootstrap_paired_difference(
        self, method_a: np.ndarray, method_b: np.ndarray, confidence: float = 0.95
    ) -> dict:
        """
        Bootstrap confidence interval for paired difference

        Args:
            method_a: Results from method A
            method_b: Results from method B
            confidence: Confidence level

        Returns:
            Dictionary with bootstrap statistics
        """
        differences = np.asarray(method_a) - np.asarray(method_b)
        n = len(differences)

        # Bootstrap
        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(differences, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # Compute CI
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        # Bootstrap p-value (proportion of bootstrap samples with opposite sign)
        observed_mean = np.mean(differences)
        if observed_mean > 0:
            p_value = np.mean(bootstrap_means <= 0) * 2  # Two-tailed
        elif observed_mean < 0:
            p_value = np.mean(bootstrap_means >= 0) * 2
        else:
            p_value = 1.0

        p_value = min(p_value, 1.0)  # Clamp to [0, 1]

        return {
            "test": "Bootstrap (paired difference)",
            "mean_difference": float(np.mean(differences)),
            "ci_95": (float(ci_lower), float(ci_upper)),
            "bootstrap_std": float(np.std(bootstrap_means)),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }

    def bonferroni_correction(self, p_values: list[float], alpha: float | None = None) -> dict:
        """
        Bonferroni correction for multiple testing

        Args:
            p_values: List of p-values from multiple tests
            alpha: Significance level (uses self.alpha if None)

        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha

        p_values = np.array(p_values)
        n_tests = len(p_values)

        # Bonferroni correction
        corrected_alpha = alpha / n_tests
        corrected_p_values = np.minimum(p_values * n_tests, 1.0)

        return {
            "method": "Bonferroni",
            "n_tests": n_tests,
            "original_alpha": alpha,
            "corrected_alpha": corrected_alpha,
            "original_p_values": p_values.tolist(),
            "corrected_p_values": corrected_p_values.tolist(),
            "significant": (corrected_p_values < alpha).tolist(),
        }

    def holm_correction(self, p_values: list[float], alpha: float | None = None) -> dict:
        """
        Holm-Bonferroni correction (less conservative than Bonferroni)

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha

        p_values = np.array(p_values)
        n_tests = len(p_values)

        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        # Holm correction
        reject = np.zeros(n_tests, dtype=bool)
        for i in range(n_tests):
            adjusted_alpha = alpha / (n_tests - i)
            if sorted_p_values[i] <= adjusted_alpha:
                reject[i] = True
            else:
                break  # Stop at first non-rejected

        # Restore original order
        original_reject = np.zeros(n_tests, dtype=bool)
        original_reject[sorted_indices] = reject

        return {
            "method": "Holm-Bonferroni",
            "n_tests": n_tests,
            "original_alpha": alpha,
            "p_values": p_values.tolist(),
            "significant": original_reject.tolist(),
        }

    def paired_comparison(
        self,
        method_a: np.ndarray,
        method_b: np.ndarray,
        method_a_name: str = "Method A",
        method_b_name: str = "Method B",
    ) -> dict:
        """
        Comprehensive paired comparison with all tests

        Args:
            method_a: Results from method A
            method_b: Results from method B
            method_a_name: Name of method A
            method_b_name: Name of method B

        Returns:
            Dictionary with all test results
        """
        method_a = np.asarray(method_a)
        method_b = np.asarray(method_b)

        results = {"method_a": method_a_name, "method_b": method_b_name, "n_samples": len(method_a)}

        # Descriptive statistics
        results["descriptive"] = {
            "method_a_mean": float(np.mean(method_a)),
            "method_a_std": float(np.std(method_a, ddof=1)),
            "method_a_median": float(np.median(method_a)),
            "method_b_mean": float(np.mean(method_b)),
            "method_b_std": float(np.std(method_b, ddof=1)),
            "method_b_median": float(np.median(method_b)),
        }

        # Parametric test
        try:
            results["t_test"] = self.paired_t_test(method_a, method_b)
        except Exception as e:
            warnings.warn(f"Paired t-test failed: {e}", stacklevel=2)
            results["t_test"] = None

        # Non-parametric test
        try:
            results["wilcoxon"] = self.wilcoxon_test(method_a, method_b)
        except Exception as e:
            warnings.warn(f"Wilcoxon test failed: {e}", stacklevel=2)
            results["wilcoxon"] = None

        # Effect size
        try:
            d = self.cohens_d(method_a, method_b)
            results["cohens_d"] = {"value": d, "interpretation": self._interpret_cohens_d(d)}
        except Exception as e:
            warnings.warn(f"Cohen's d failed: {e}", stacklevel=2)
            results["cohens_d"] = None

        # Bootstrap
        try:
            results["bootstrap"] = self.bootstrap_paired_difference(method_a, method_b)
        except Exception as e:
            warnings.warn(f"Bootstrap failed: {e}", stacklevel=2)
            results["bootstrap"] = None

        # Summary
        results["summary"] = self._create_summary(results)

        return results

    def mann_whitney_u_test(self, group_a: np.ndarray, group_b: np.ndarray) -> dict:
        """
        Mann-Whitney U test (independent samples, non-parametric)

        Use when comparing two independent groups (not paired).
        Non-parametric alternative to independent t-test.

        Args:
            group_a: Results from group A
            group_b: Results from group B

        Returns:
            Dictionary with test statistics
        """
        group_a = np.asarray(group_a)
        group_b = np.asarray(group_b)

        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

        # Medians
        median_a = np.median(group_a)
        median_b = np.median(group_b)

        return {
            "test": "Mann-Whitney U",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "median_a": float(median_a),
            "median_b": float(median_b),
            "significant": p_value < self.alpha,
            "winner": "group_a"
            if median_a < median_b
            else "group_b"
            if median_a > median_b
            else "tie",
        }

    def sign_test(self, method_a: np.ndarray, method_b: np.ndarray) -> dict:
        """
        Sign test (non-parametric paired test)

        Simple test based on the sign of differences.
        More robust than Wilcoxon but less powerful.

        Args:
            method_a: Results from method A
            method_b: Results from method B

        Returns:
            Dictionary with test statistics
        """
        method_a = np.asarray(method_a)
        method_b = np.asarray(method_b)

        if len(method_a) != len(method_b):
            raise ValueError("Methods must have same number of samples")

        differences = method_a - method_b
        # Remove ties
        differences = differences[differences != 0]

        if len(differences) == 0:
            return {
                "test": "Sign test",
                "n_positive": 0,
                "n_negative": 0,
                "p_value": 1.0,
                "significant": False,
            }

        # Count signs
        n_positive = np.sum(differences > 0)
        n_negative = np.sum(differences < 0)
        n_total = len(differences)

        # Binomial test (two-tailed)
        p_value = stats.binom_test(n_positive, n_total, p=0.5, alternative="two-sided")

        return {
            "test": "Sign test",
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
            "n_total": int(n_total),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }

    def friedman_test(self, *methods: np.ndarray) -> dict:
        """
        Friedman test (non-parametric test for multiple related samples)

        Use when comparing 3 or more methods on the same instances.
        Non-parametric alternative to repeated measures ANOVA.

        Args:
            *methods: Variable number of method result arrays

        Returns:
            Dictionary with test statistics
        """
        if len(methods) < 3:
            raise ValueError("Friedman test requires at least 3 methods")

        # Check all same length
        lengths = [len(m) for m in methods]
        if len(set(lengths)) > 1:
            raise ValueError("All methods must have same number of samples")

        # Stack methods
        methods_array = np.array(methods)

        # Friedman test
        statistic, p_value = stats.friedmanchisquare(*methods)

        # Compute rankings
        ranks = stats.rankdata(methods_array, axis=0, method="average")
        mean_ranks = np.mean(ranks, axis=1)

        return {
            "test": "Friedman",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "n_methods": len(methods),
            "n_samples": lengths[0],
            "mean_ranks": mean_ranks.tolist(),
        }

    def cliffs_delta(self, method_a: np.ndarray, method_b: np.ndarray) -> float:
        """
        Cliff's Delta effect size (non-parametric)

        Measures how often values in one distribution are larger than
        values in another distribution.

        Interpretation:
        - |delta| < 0.147: negligible
        - 0.147 <= |delta| < 0.33: small
        - 0.33 <= |delta| < 0.474: medium
        - |delta| >= 0.474: large

        Args:
            method_a: Results from method A
            method_b: Results from method B

        Returns:
            Cliff's Delta value in [-1, 1]
        """
        method_a = np.asarray(method_a)
        method_b = np.asarray(method_b)

        n_a = len(method_a)
        n_b = len(method_b)

        # Count dominance
        dominance = 0
        for a in method_a:
            dominance += np.sum(a > method_b) - np.sum(a < method_b)

        delta = dominance / (n_a * n_b)
        return float(delta)

    def vargha_delaney_a(self, method_a: np.ndarray, method_b: np.ndarray) -> float:
        """
        Vargha-Delaney A effect size

        Probability that a random value from method_a is greater than
        a random value from method_b.

        Interpretation:
        - A = 0.5: No difference
        - A > 0.5: method_a tends to be larger
        - A < 0.5: method_b tends to be larger
        - |A - 0.5| < 0.06: negligible
        - 0.06 <= |A - 0.5| < 0.14: small
        - 0.14 <= |A - 0.5| < 0.21: medium
        - |A - 0.5| >= 0.21: large

        Args:
            method_a: Results from method A
            method_b: Results from method B

        Returns:
            A value in [0, 1]
        """
        # A = (cliff's delta + 1) / 2
        delta = self.cliffs_delta(method_a, method_b)
        return float((delta + 1) / 2)

    def shapiro_wilk_test(self, data: np.ndarray) -> dict:
        """
        Shapiro-Wilk normality test

        Tests the null hypothesis that data comes from a normal distribution.

        Args:
            data: Data array

        Returns:
            Dictionary with test results
        """
        data = np.asarray(data)

        if len(data) < 3:
            return {
                "test": "Shapiro-Wilk",
                "statistic": None,
                "p_value": None,
                "is_normal": None,
                "warning": "Sample size too small (n < 3)",
            }

        statistic, p_value = stats.shapiro(data)

        return {
            "test": "Shapiro-Wilk",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value >= self.alpha,
            "interpretation": "normal" if p_value >= self.alpha else "not normal",
        }

    def anderson_darling_test(self, data: np.ndarray) -> dict:
        """
        Anderson-Darling normality test

        More sensitive than Shapiro-Wilk for detecting deviations in tails.

        Args:
            data: Data array

        Returns:
            Dictionary with test results
        """
        data = np.asarray(data)

        result = stats.anderson(data, dist="norm")

        # Find significance level closest to self.alpha
        sig_levels = [0.15, 0.10, 0.05, 0.025, 0.01]
        if self.alpha in sig_levels:
            idx = sig_levels.index(self.alpha)
        else:
            # Find closest
            idx = min(range(len(sig_levels)), key=lambda i: abs(sig_levels[i] - self.alpha))

        critical_value = result.critical_values[idx]
        is_normal = result.statistic < critical_value

        return {
            "test": "Anderson-Darling",
            "statistic": float(result.statistic),
            "critical_value": float(critical_value),
            "significance_level": sig_levels[idx],
            "is_normal": is_normal,
            "interpretation": "normal" if is_normal else "not normal",
        }

    def levene_test(self, *groups: np.ndarray) -> dict:
        """
        Levene's test for homogeneity of variances

        Tests whether k samples have equal variances.

        Args:
            *groups: Variable number of data arrays

        Returns:
            Dictionary with test results
        """
        if len(groups) < 2:
            raise ValueError("Levene's test requires at least 2 groups")

        statistic, p_value = stats.levene(*groups)

        return {
            "test": "Levene",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "equal_variances": p_value >= self.alpha,
            "interpretation": "equal variances" if p_value >= self.alpha else "unequal variances",
        }

    def benjamini_hochberg_correction(
        self, p_values: list[float], alpha: float | None = None
    ) -> dict:
        """
        Benjamini-Hochberg FDR correction (less conservative than Bonferroni)

        Controls False Discovery Rate instead of Family-Wise Error Rate.

        Args:
            p_values: List of p-values
            alpha: Significance level

        Returns:
            Dictionary with corrected results
        """
        if alpha is None:
            alpha = self.alpha

        p_values = np.array(p_values)
        n_tests = len(p_values)

        # Sort p-values and track indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        # BH critical values
        critical_values = (np.arange(1, n_tests + 1) / n_tests) * alpha

        # Find largest i where p[i] <= (i/m) * alpha
        reject = sorted_p_values <= critical_values

        if np.any(reject):
            max_idx = np.where(reject)[0][-1]
            reject[: max_idx + 1] = True

        # Restore original order
        original_reject = np.zeros(n_tests, dtype=bool)
        original_reject[sorted_indices] = reject

        # Adjusted p-values
        adjusted_p = np.minimum.accumulate(
            (n_tests / np.arange(n_tests, 0, -1)) * sorted_p_values[::-1]
        )[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)

        original_adjusted_p = np.zeros(n_tests)
        original_adjusted_p[sorted_indices] = adjusted_p

        return {
            "method": "Benjamini-Hochberg",
            "n_tests": n_tests,
            "original_alpha": alpha,
            "p_values": p_values.tolist(),
            "adjusted_p_values": original_adjusted_p.tolist(),
            "significant": original_reject.tolist(),
        }

    def statistical_power(
        self,
        effect_size: float,
        n_samples: int,
        alpha: float | None = None,
        test_type: str = "paired-t",
    ) -> float | None:
        """
        Compute statistical power for a test

        Power is the probability of detecting an effect when it exists.

        Args:
            effect_size: Expected effect size (Cohen's d)
            n_samples: Sample size
            alpha: Significance level
            test_type: Type of test ('paired-t', 'independent-t')

        Returns:
            Statistical power (0 to 1)
        """
        if alpha is None:
            alpha = self.alpha

        try:
            from statsmodels.stats.power import ttest_power

            if test_type == "paired-t":
                power = ttest_power(effect_size, n_samples, alpha, alternative="two-sided")
            elif test_type == "independent-t":
                # For equal sample sizes
                power = ttest_power(effect_size, n_samples, alpha, alternative="two-sided")
            else:
                raise ValueError(f"Unknown test type: {test_type}")

            return float(power)
        except ImportError:
            warnings.warn(
                "statsmodels not installed. Install with: pip install statsmodels", stacklevel=2
            )
            return None

    def required_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float | None = None,
        test_type: str = "paired-t",
    ) -> int | None:
        """
        Compute required sample size for desired power

        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power (default: 0.8)
            alpha: Significance level
            test_type: Type of test

        Returns:
            Required sample size
        """
        if alpha is None:
            alpha = self.alpha

        try:
            from statsmodels.stats.power import tt_solve_power

            n_samples = tt_solve_power(
                effect_size=effect_size, power=power, alpha=alpha, alternative="two-sided"
            )

            return int(np.ceil(n_samples))
        except ImportError:
            warnings.warn(
                "statsmodels not installed. Install with: pip install statsmodels", stacklevel=2
            )
            return None

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's Delta effect size"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"

    def _interpret_vargha_delaney_a(self, a: float) -> str:
        """Interpret Vargha-Delaney A effect size"""
        diff = abs(a - 0.5)
        if diff < 0.06:
            return "negligible"
        elif diff < 0.14:
            return "small"
        elif diff < 0.21:
            return "medium"
        else:
            return "large"

    def _create_summary(self, results: dict) -> str:
        """Create human-readable summary"""
        summary_lines = []

        method_a = results["method_a"]
        method_b = results["method_b"]

        # Descriptive
        desc = results["descriptive"]
        mean_diff = desc["method_a_mean"] - desc["method_b_mean"]
        summary_lines.append(
            f"{method_a} (μ={desc['method_a_mean']:.2f}±{desc['method_a_std']:.2f}) vs "
            f"{method_b} (μ={desc['method_b_mean']:.2f}±{desc['method_b_std']:.2f})"
        )
        summary_lines.append(f"Mean difference: {mean_diff:.2f}")

        # Statistical significance
        t_test = results.get("t_test")
        if t_test:
            sig = "IS" if t_test["significant"] else "is NOT"
            summary_lines.append(
                f"Paired t-test: {sig} statistically significant (p={t_test['p_value']:.4f})"
            )

        wilcoxon = results.get("wilcoxon")
        if wilcoxon:
            sig = "IS" if wilcoxon["significant"] else "is NOT"
            summary_lines.append(
                f"Wilcoxon test: {sig} statistically significant (p={wilcoxon['p_value']:.4f})"
            )

        # Effect size
        cohens_d = results.get("cohens_d")
        if cohens_d:
            summary_lines.append(
                f"Effect size: Cohen's d = {cohens_d['value']:.3f} ({cohens_d['interpretation']})"
            )

        return "\n".join(summary_lines)


def compare_methods(
    method_a_results: np.ndarray,
    method_b_results: np.ndarray,
    method_a_name: str = "Method A",
    method_b_name: str = "Method B",
    alpha: float = 0.05,
) -> dict:
    """
    Convenience function for paired method comparison

    Args:
        method_a_results: Results (e.g., gaps) from method A
        method_b_results: Results from method B
        method_a_name: Name of method A
        method_b_name: Name of method B
        alpha: Significance level

    Returns:
        Dictionary with all comparison results
    """
    analyzer = StatisticalAnalyzer(alpha=alpha)
    return analyzer.paired_comparison(
        method_a_results, method_b_results, method_a_name, method_b_name
    )


def print_comparison_report(results: dict):
    """
    Print formatted comparison report

    Args:
        results: Results from paired_comparison()
    """
    print("=" * 80)
    print(f"STATISTICAL COMPARISON: {results['method_a']} vs {results['method_b']}")
    print("=" * 80)
    print()

    # Descriptive statistics
    print("Descriptive Statistics:")
    print("-" * 80)
    desc = results["descriptive"]
    print(
        f"  {results['method_a']}: μ={desc['method_a_mean']:.4f}, "
        f"σ={desc['method_a_std']:.4f}, median={desc['method_a_median']:.4f}"
    )
    print(
        f"  {results['method_b']}: μ={desc['method_b_mean']:.4f}, "
        f"σ={desc['method_b_std']:.4f}, median={desc['method_b_median']:.4f}"
    )
    print()

    # Paired t-test
    if results.get("t_test"):
        print("Paired t-test:")
        print("-" * 80)
        t = results["t_test"]
        print(f"  t-statistic: {t['t_statistic']:.4f}")
        print(f"  p-value: {t['p_value']:.6f}")
        print(f"  Mean difference: {t['mean_difference']:.4f}")
        print(f"  95% CI: [{t['ci_95'][0]:.4f}, {t['ci_95'][1]:.4f}]")
        print(f"  Significant: {'YES ✓' if t['significant'] else 'NO'}")
        print(f"  Winner: {t['winner']}")
        print()

    # Wilcoxon test
    if results.get("wilcoxon"):
        print("Wilcoxon Signed-Rank Test:")
        print("-" * 80)
        w = results["wilcoxon"]
        print(f"  Statistic: {w['statistic']:.4f}")
        print(f"  p-value: {w['p_value']:.6f}")
        print(f"  Median difference: {w['median_difference']:.4f}")
        print(f"  Significant: {'YES ✓' if w['significant'] else 'NO'}")
        print()

    # Effect size
    if results.get("cohens_d"):
        print("Effect Size:")
        print("-" * 80)
        d = results["cohens_d"]
        print(f"  Cohen's d: {d['value']:.4f}")
        print(f"  Interpretation: {d['interpretation']}")
        print()

    # Bootstrap
    if results.get("bootstrap"):
        print("Bootstrap Analysis:")
        print("-" * 80)
        b = results["bootstrap"]
        print(f"  Mean difference: {b['mean_difference']:.4f}")
        print(f"  95% CI: [{b['ci_95'][0]:.4f}, {b['ci_95'][1]:.4f}]")
        print(f"  Bootstrap p-value: {b['p_value']:.6f}")
        print(f"  Significant: {'YES ✓' if b['significant'] else 'NO'}")
        print()

    # Summary
    print("Summary:")
    print("-" * 80)
    print(results["summary"])
    print()
    print("=" * 80)


def compute_percentiles(
    data: np.ndarray, percentiles: list[float] | None = None
) -> dict[str, float]:
    """
    Compute percentiles for gap distribution.

    Args:
        data: Array of gap values
        percentiles: List of percentiles to compute (default: [50, 90, 95, 99])

    Returns:
        Dictionary mapping percentile names to values
    """
    if percentiles is None:
        percentiles = [50, 90, 95, 99]

    data = np.asarray(data)
    if data.size == 0:
        return {f"p{int(p)}": None for p in percentiles}

    result = {}
    for p in percentiles:
        result[f"p{int(p)}"] = float(np.percentile(data, p))

    return result


def compute_gap_statistics_by_size(
    gaps: list[float], sizes: list[int], size_bins: list[int] | None = None
) -> dict[int, dict]:
    """
    Compute gap statistics grouped by problem size.

    Args:
        gaps: List of optimality gap percentages
        sizes: List of problem sizes (number of items) corresponding to gaps
        size_bins: Optional list of size bins to group by (default: unique sizes)

    Returns:
        Dictionary mapping size -> statistics dict containing:
            - mean, median, std, min, max
            - p50, p90, p95, p99
            - count (number of instances)
            - ci_95 (bootstrap CI for mean)

    Example:
        >>> gaps = [0.05, 0.12, 0.03, 0.08, 0.15, 0.02, 0.10]
        >>> sizes = [10, 10, 10, 25, 25, 50, 50]
        >>> stats = compute_gap_statistics_by_size(gaps, sizes)
        >>> print(stats[10])
        {'mean': 0.0667, 'p95': 0.12, ...}
    """
    gaps = np.asarray(gaps)
    sizes = np.asarray(sizes)

    if len(gaps) != len(sizes):
        raise ValueError("gaps and sizes must have the same length")

    if size_bins is None:
        size_bins = sorted(set(sizes))

    results = {}
    analyzer = StatisticalAnalyzer()

    for size in size_bins:
        mask = sizes == size
        size_gaps = gaps[mask]

        if len(size_gaps) == 0:
            results[size] = {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                **{f"p{p}": None for p in [50, 90, 95, 99]},
                "ci_95": (None, None),
            }
            continue

        # Basic statistics
        stats_dict = {
            "count": int(len(size_gaps)),
            "mean": float(np.mean(size_gaps)),
            "median": float(np.median(size_gaps)),
            "std": float(np.std(size_gaps, ddof=1)) if len(size_gaps) > 1 else 0.0,
            "min": float(np.min(size_gaps)),
            "max": float(np.max(size_gaps)),
        }

        # Percentiles
        percentiles = compute_percentiles(size_gaps, [50, 90, 95, 99])
        stats_dict.update(percentiles)

        # Bootstrap CI for mean (if sample size >= 10)
        if len(size_gaps) >= 10:
            try:
                ci_lower, ci_upper = analyzer.bootstrap_ci(size_gaps, statistic_fn=np.mean)
                stats_dict["ci_95"] = (float(ci_lower), float(ci_upper))
            except Exception:
                stats_dict["ci_95"] = (None, None)
        else:
            stats_dict["ci_95"] = (None, None)

        results[size] = stats_dict

    return results


def compute_cdf(
    data: np.ndarray, x_values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution function.

    Args:
        data: Data array
        x_values: Optional array of x-values at which to evaluate CDF
                  (default: sorted unique values from data)

    Returns:
        Tuple of (x_values, cdf_values)
            x_values: Array of x-coordinates
            cdf_values: Array of CDF values in [0, 1]

    Example:
        >>> data = np.array([1, 2, 2, 3, 5])
        >>> x, cdf = compute_cdf(data)
        >>> # cdf[i] = proportion of data <= x[i]
    """
    data = np.asarray(data)
    if data.size == 0:
        return np.array([]), np.array([])

    if x_values is None:
        x_values = np.sort(np.unique(data))
    else:
        x_values = np.asarray(x_values)

    cdf_values = np.array([np.mean(data <= x) for x in x_values])

    return x_values, cdf_values


def compute_cdf_by_size(
    gaps: list[float], sizes: list[int], size_bins: list[int] | None = None
) -> dict[int, dict[str, np.ndarray]]:
    """
    Compute CDF of gaps grouped by problem size.

    Args:
        gaps: List of optimality gap percentages
        sizes: List of problem sizes corresponding to gaps
        size_bins: Optional list of size bins (default: unique sizes)

    Returns:
        Dictionary mapping size -> {'x': x_values, 'cdf': cdf_values}

    Example:
        >>> gaps = [0.05, 0.12, 0.03, 0.08, 0.15]
        >>> sizes = [10, 10, 10, 25, 25]
        >>> cdfs = compute_cdf_by_size(gaps, sizes)
        >>> x, cdf = cdfs[10]['x'], cdfs[10]['cdf']
    """
    gaps = np.asarray(gaps)
    sizes = np.asarray(sizes)

    if len(gaps) != len(sizes):
        raise ValueError("gaps and sizes must have the same length")

    if size_bins is None:
        size_bins = sorted(set(sizes))

    results = {}

    for size in size_bins:
        mask = sizes == size
        size_gaps = gaps[mask]

        if len(size_gaps) == 0:
            results[size] = {"x": np.array([]), "cdf": np.array([])}
            continue

        x_values, cdf_values = compute_cdf(size_gaps)
        results[size] = {"x": x_values, "cdf": cdf_values}

    return results


def check_sample_size_adequacy(
    data: np.ndarray, target_error: float = 0.5, confidence: float = 0.95
) -> dict:
    """
    Check if sample size is adequate for estimating mean with desired precision.

    Uses formula: n ≈ (z * σ / ε)²
    where z is the critical value, σ is std, ε is target error.

    Args:
        data: Data array
        target_error: Target margin of error for mean estimate (default: 0.5%)
        confidence: Confidence level (default: 0.95)

    Returns:
        Dictionary with:
            - current_n: Current sample size
            - current_std: Current standard deviation
            - required_n: Required sample size for target error
            - adequate: Boolean indicating if current n is sufficient
            - margin_of_error: Current margin of error

    Example:
        >>> gaps = np.random.gamma(2, 0.5, 50)
        >>> adequacy = check_sample_size_adequacy(gaps, target_error=0.5)
        >>> print(adequacy['adequate'])
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        return {
            "current_n": n,
            "current_std": None,
            "required_n": None,
            "adequate": False,
            "margin_of_error": None,
            "warning": "Sample size too small (n < 2)",
        }

    std = np.std(data, ddof=1)

    # Critical value (z-score for normal, t-score for small samples)
    if n >= 30:
        from scipy.stats import norm

        z = norm.ppf((1 + confidence) / 2)
    else:
        from scipy.stats import t

        z = t.ppf((1 + confidence) / 2, df=n - 1)

    # Required sample size
    required_n = int(np.ceil((z * std / target_error) ** 2))

    # Current margin of error
    current_margin = z * std / np.sqrt(n)

    return {
        "current_n": int(n),
        "current_std": float(std),
        "required_n": int(required_n),
        "adequate": n >= required_n,
        "margin_of_error": float(current_margin),
        "target_error": float(target_error),
        "confidence": float(confidence),
    }


if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Module")
    print("=" * 80)
    print()
    print("Example: Comparing two methods")
    print()

    # Simulate some results
    np.random.seed(42)
    gnn_gaps = np.random.gamma(2, 0.5, 100)  # GNN gaps: mean ~1%
    greedy_gaps = np.random.gamma(2, 2, 100)  # Greedy gaps: mean ~4%

    # Compare
    results = compare_methods(gnn_gaps, greedy_gaps, method_a_name="GNN", method_b_name="Greedy")

    # Print report
    print_comparison_report(results)

    # Example: Gap statistics by size
    print("\n" + "=" * 80)
    print("Example: Gap statistics by problem size")
    print("=" * 80)

    # Simulate gaps by size
    np.random.seed(42)
    gaps_all = []
    sizes_all = []
    for size in [10, 25, 50, 100]:
        n_instances = 50
        # Larger sizes have slightly higher gaps
        gaps = np.random.gamma(2, 0.3 + size / 500, n_instances)
        gaps_all.extend(gaps)
        sizes_all.extend([size] * n_instances)

    stats_by_size = compute_gap_statistics_by_size(gaps_all, sizes_all)

    print("\nStatistics by Size:")
    print(f"{'Size':<8} {'Count':<8} {'Mean':<8} {'Median':<8} {'p95':<8} {'p99':<8} {'CI 95%'}")
    print("-" * 80)
    for size in sorted(stats_by_size.keys()):
        s = stats_by_size[size]
        ci_str = (
            f"[{s['ci_95'][0]:.2f}, {s['ci_95'][1]:.2f}]" if s["ci_95"][0] is not None else "N/A"
        )
        print(
            f"{size:<8} {s['count']:<8} {s['mean']:<8.2f} {s['median']:<8.2f} {s['p95']:<8.2f} {s['p99']:<8.2f} {ci_str}"
        )

    # Sample size adequacy
    print("\nSample Size Adequacy Check:")
    print("-" * 80)
    for size in sorted(stats_by_size.keys()):
        gaps_for_size = [g for g, sz in zip(gaps_all, sizes_all, strict=False) if sz == size]
        adequacy = check_sample_size_adequacy(np.array(gaps_for_size), target_error=0.5)
        status = "✓" if adequacy["adequate"] else "✗"
        print(
            f"Size {size}: n={adequacy['current_n']}, required_n={adequacy['required_n']}, "
            f"margin=±{adequacy['margin_of_error']:.2f}% {status}"
        )
