"""
Academic Reporting Utilities

Generate publication-ready outputs:
- LaTeX tables for papers
- High-quality matplotlib figures
- Statistical test summaries
- Formatted result reports

Usage:
    from knapsack_gnn.analysis.reporting import AcademicReporter

    reporter = AcademicReporter()
    latex_table = reporter.generate_comparison_table(results)
    reporter.save_publication_figure(fig, "comparison.pdf")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class AcademicReporter:
    """
    Generate publication-ready academic outputs
    """

    def __init__(
        self, style: str = "ieee", dpi: int = 300, font_family: str = "serif", font_size: int = 10
    ):
        """
        Args:
            style: Plot style ('ieee', 'neurips', 'acm')
            dpi: Figure DPI for saving
            font_size: Base font size
        """
        self.style = style
        self.dpi = dpi
        self.font_family = font_family
        self.font_size = font_size

        # Configure matplotlib for publication quality
        self._configure_matplotlib()

    def _configure_matplotlib(self) -> None:
        """Configure matplotlib for publication-quality figures"""
        # Use LaTeX if available
        try:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": self.font_family,
                    "font.size": self.font_size,
                    "axes.labelsize": self.font_size,
                    "xtick.labelsize": self.font_size - 1,
                    "ytick.labelsize": self.font_size - 1,
                    "legend.fontsize": self.font_size - 1,
                    "figure.titlesize": self.font_size + 2,
                    "figure.dpi": 100,
                    "savefig.dpi": self.dpi,
                    "savefig.bbox": "tight",
                    "savefig.pad_inches": 0.05,
                    "axes.grid": True,
                    "grid.alpha": 0.3,
                    "lines.linewidth": 1.5,
                    "lines.markersize": 6,
                }
            )
        except Exception:
            # Fallback without LaTeX
            plt.rcParams.update(
                {
                    "font.family": self.font_family,
                    "font.size": self.font_size,
                    "figure.dpi": 100,
                    "savefig.dpi": self.dpi,
                    "savefig.bbox": "tight",
                }
            )

    def generate_comparison_table(
        self,
        results: dict[str, dict],
        metrics: list[str] | None = None,
        caption: str = "Performance comparison",
        label: str = "tab:comparison",
        highlight_best: bool = True,
    ) -> str:
        """
        Generate LaTeX table comparing multiple methods

        Args:
            results: Dict mapping method names to result dicts
            metrics: List of metric names to include
            caption: Table caption
            label: LaTeX label
            highlight_best: Bold the best result per metric

        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ["mean_gap", "median_gap", "std_gap", "feasibility_rate", "mean_time"]

        # Header
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")

        # Column specification
        len(metrics) + 1
        col_spec = "l" + "c" * len(metrics)
        latex.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex.append(r"\toprule")

        # Header row
        metric_names = {
            "mean_gap": "Mean Gap (\\%)",
            "median_gap": "Median Gap (\\%)",
            "std_gap": "Std Gap (\\%)",
            "max_gap": "Max Gap (\\%)",
            "feasibility_rate": "Feasibility",
            "mean_time": "Time (ms)",
            "throughput": "Throughput",
        }

        header = "Method"
        for metric in metrics:
            header += " & " + metric_names.get(metric, metric)
        header += r" \\"
        latex.append(header)
        latex.append(r"\midrule")

        # Find best values for each metric
        best_values = {}
        if highlight_best:
            for metric in metrics:
                values = []
                for _, result in results.items():
                    if metric in result:
                        values.append(result[metric])
                if values:
                    # Lower is better for gaps and time
                    if "gap" in metric or "time" in metric:
                        best_values[metric] = min(values)
                    else:
                        best_values[metric] = max(values)

        # Data rows
        for method, result in results.items():
            row = method.replace("_", "\\_")

            for metric in metrics:
                if metric in result:
                    value = result[metric]

                    # Format value
                    if "gap" in metric or "rate" in metric:
                        if "rate" in metric:
                            formatted = f"{value * 100:.2f}\\%"
                        else:
                            formatted = f"{value:.2f}"
                    elif "time" in metric:
                        if value < 1:
                            formatted = f"{value * 1000:.2f}"  # ms
                        else:
                            formatted = f"{value:.2f}"
                    else:
                        formatted = f"{value:.2f}"

                    # Bold if best
                    if highlight_best and metric in best_values:
                        if abs(value - best_values[metric]) < 1e-6:
                            formatted = f"\\textbf{{{formatted}}}"

                    row += f" & {formatted}"
                else:
                    row += " & --"

            row += r" \\"
            latex.append(row)

        # Footer
        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def generate_statistical_test_table(
        self,
        test_results: list[dict],
        caption: str = "Statistical test results",
        label: str = "tab:tests",
    ) -> str:
        """
        Generate LaTeX table for statistical test results

        Args:
            test_results: List of test result dicts
            caption: Table caption
            label: LaTeX label

        Returns:
            LaTeX table string
        """
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append(r"\begin{tabular}{lcccc}")
        latex.append(r"\toprule")
        latex.append(r"Comparison & Test & Statistic & $p$-value & Significant \\")
        latex.append(r"\midrule")

        for result in test_results:
            comparison = f"{result.get('method_a', 'A')} vs {result.get('method_b', 'B')}"
            comparison = comparison.replace("_", "\\_")

            test_name = result.get("test", "Unknown")
            statistic = result.get("t_statistic") or result.get("statistic", 0)
            p_value = result.get("p_value", 1.0)
            significant = result.get("significant", False)

            sig_str = r"$\checkmark$" if significant else ""

            # Format p-value
            if p_value < 0.001:
                p_str = "$< 0.001$"
            else:
                p_str = f"${p_value:.4f}$"

            row = f"{comparison} & {test_name} & {statistic:.3f} & {p_str} & {sig_str} \\\\"
            latex.append(row)

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def generate_effect_size_table(
        self,
        comparisons: list[dict],
        caption: str = "Effect sizes for method comparisons",
        label: str = "tab:effect_sizes",
    ) -> str:
        """
        Generate LaTeX table for effect sizes

        Args:
            comparisons: List of comparison results with effect sizes
            caption: Table caption
            label: LaTeX label

        Returns:
            LaTeX table string
        """
        latex = []
        latex.append(r"\begin{table}[t]")
        latex.append(r"\centering")
        latex.append(f"\\caption{{{caption}}}")
        latex.append(f"\\label{{{label}}}")
        latex.append(r"\begin{tabular}{lccc}")
        latex.append(r"\toprule")
        latex.append(r"Comparison & Cohen's $d$ & Cliff's $\delta$ & Interpretation \\")
        latex.append(r"\midrule")

        for comp in comparisons:
            comparison = f"{comp.get('method_a', 'A')} vs {comp.get('method_b', 'B')}"
            comparison = comparison.replace("_", "\\_")

            cohens_d = comp.get("cohens_d", {})
            d_value = cohens_d.get("value", 0.0)
            interpretation = cohens_d.get("interpretation", "unknown")

            # Cliff's delta if available
            cliffs_delta = comp.get("cliffs_delta", 0.0)

            row = f"{comparison} & {d_value:.3f} & {cliffs_delta:.3f} & {interpretation} \\\\"
            latex.append(row)

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")

        return "\n".join(latex)

    def save_publication_figure(
        self, fig: plt.Figure, filepath: str | Path, formats: list[str] | None = None
    ) -> None:
        """
        Save figure in publication-ready formats

        Args:
            fig: Matplotlib figure
            filepath: Base filepath (without extension)
            formats: List of formats to save
        """
        if formats is None:
            formats = ["pdf", "png"]
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            save_path = path.with_suffix(f".{fmt}")
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight", format=fmt)
            print(f"Saved figure: {save_path}")

    def create_comparison_plot(
        self,
        results: dict[str, dict],
        metric: str = "mean_gap",
        ylabel: str | None = None,
        title: str | None = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create bar plot comparing methods

        Args:
            results: Dict mapping method names to result dicts
            metric: Metric to plot
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        methods = list(results.keys())
        values = [results[m].get(metric, 0) for m in methods]
        errors = [results[m].get(f"std_{metric}", 0) for m in methods]

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(methods))
        bars = ax.bar(x, values, yerr=errors, capsize=5, alpha=0.7, edgecolor="black")

        # Color bars
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for bar, color in zip(bars, colors, strict=False):
            bar.set_color(color)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")

        if ylabel is None:
            ylabel = metric.replace("_", " ").title()
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            self.save_publication_figure(fig, save_path)

        return fig

    def create_boxplot_comparison(
        self,
        data: dict[str, np.ndarray],
        ylabel: str = "Optimality Gap (%)",
        title: str = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create boxplot comparing distributions

        Args:
            data: Dict mapping method names to data arrays
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        methods = list(data.keys())
        values = [data[m] for m in methods]

        bp = ax.boxplot(values, labels=methods, patch_artist=True)

        # Color boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        if save_path:
            self.save_publication_figure(fig, save_path)

        return fig

    def create_confidence_interval_plot(
        self,
        results: dict[str, dict],
        metric: str = "mean_gap",
        ylabel: str = None,
        title: str = None,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Create plot with confidence intervals

        Args:
            results: Dict with 'mean', 'ci_95' for each method
            metric: Metric to plot
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = list(results.keys())
        means = []
        ci_lows = []
        ci_highs = []

        for method in methods:
            result = results[method]
            if isinstance(result, dict):
                mean = result.get(metric, result.get("mean", 0))
                ci = result.get(f"{metric}_ci_95", result.get("ci_95", (mean, mean)))
            else:
                mean = result
                ci = (mean, mean)

            means.append(mean)
            ci_lows.append(ci[0])
            ci_highs.append(ci[1])

        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        # Error bars
        errors_low = means - ci_lows
        errors_high = ci_highs - means

        x = np.arange(len(methods))
        ax.errorbar(
            x,
            means,
            yerr=[errors_low, errors_high],
            fmt="o",
            markersize=8,
            capsize=5,
            capthick=2,
            linewidth=2,
            label="95% CI",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")

        if ylabel is None:
            ylabel = metric.replace("_", " ").title()
        ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title)

        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            self.save_publication_figure(fig, save_path)

        return fig

    def format_statistical_summary(
        self, comparison_result: dict, include_latex: bool = True
    ) -> str:
        """
        Format statistical test results as text

        Args:
            comparison_result: Result from StatisticalAnalyzer.paired_comparison()
            include_latex: Include LaTeX formatting

        Returns:
            Formatted string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("STATISTICAL COMPARISON")
        lines.append("=" * 70)

        method_a = comparison_result.get("method_a", "Method A")
        method_b = comparison_result.get("method_b", "Method B")

        lines.append(f"\nComparing: {method_a} vs {method_b}")
        lines.append(f"Sample size: n = {comparison_result.get('n_samples', 0)}")

        # Descriptive statistics
        desc = comparison_result.get("descriptive", {})
        lines.append("\nDescriptive Statistics:")
        lines.append(
            f"  {method_a}: μ = {desc.get('method_a_mean', 0):.4f} ± {desc.get('method_a_std', 0):.4f}"
        )
        lines.append(
            f"  {method_b}: μ = {desc.get('method_b_mean', 0):.4f} ± {desc.get('method_b_std', 0):.4f}"
        )

        # T-test
        t_test = comparison_result.get("t_test", {})
        if t_test:
            lines.append("\nPaired t-test:")
            lines.append(
                f"  t = {t_test.get('t_statistic', 0):.4f}, p = {t_test.get('p_value', 1):.6f}"
            )
            lines.append(
                f"  95% CI: [{t_test.get('ci_95', (0, 0))[0]:.4f}, {t_test.get('ci_95', (0, 0))[1]:.4f}]"
            )
            lines.append(f"  Significant: {'YES' if t_test.get('significant') else 'NO'}")

        # Wilcoxon
        wilcoxon = comparison_result.get("wilcoxon", {})
        if wilcoxon:
            lines.append("\nWilcoxon signed-rank test:")
            lines.append(
                f"  W = {wilcoxon.get('statistic', 0):.4f}, p = {wilcoxon.get('p_value', 1):.6f}"
            )
            lines.append(f"  Significant: {'YES' if wilcoxon.get('significant') else 'NO'}")

        # Effect size
        cohens_d = comparison_result.get("cohens_d", {})
        if cohens_d:
            lines.append("\nEffect Size:")
            lines.append(
                f"  Cohen's d = {cohens_d.get('value', 0):.4f} ({cohens_d.get('interpretation', 'unknown')})"
            )

        lines.append("=" * 70)

        text = "\n".join(lines)

        # Add LaTeX version
        if include_latex:
            text += "\n\n" + "=" * 70
            text += "\nLaTeX Format:"
            text += "\n" + "=" * 70
            text += f"\n{method_a} achieved $\\mu = {desc.get('method_a_mean', 0):.2f}\\%$ "
            text += f"optimality gap, compared to {method_b}'s $\\mu = {desc.get('method_b_mean', 0):.2f}\\%$. "
            text += "A paired $t$-test showed this difference "
            if t_test.get("significant"):
                text += f"was statistically significant ($t = {t_test.get('t_statistic', 0):.2f}$, "
                text += f"$p < {0.001 if t_test.get('p_value', 1) < 0.001 else t_test.get('p_value', 1):.3f}$) "
            else:
                text += f"was not statistically significant ($p = {t_test.get('p_value', 1):.3f}$) "
            text += f"with a {cohens_d.get('interpretation', 'unknown')} effect size "
            text += f"(Cohen's $d = {cohens_d.get('value', 0):.2f}$)."

        return text


if __name__ == "__main__":
    print("Academic Reporting Module")
    print("=" * 70)
    print("\nThis module generates publication-ready outputs.")
    print("\nExample usage:")
    print("""
    from knapsack_gnn.analysis.reporting import AcademicReporter

    reporter = AcademicReporter()

    # Generate LaTeX table
    results = {
        'GNN': {'mean_gap': 0.07, 'std_gap': 0.34, 'feasibility_rate': 1.0},
        'Greedy': {'mean_gap': 0.49, 'std_gap': 1.23, 'feasibility_rate': 1.0}
    }
    latex_table = reporter.generate_comparison_table(results)
    print(latex_table)

    # Create publication figure
    fig = reporter.create_comparison_plot(results, save_path='comparison')
    """)
