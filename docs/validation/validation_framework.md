# Publication-Grade Validation Framework

## Overview

This validation framework provides comprehensive statistical validation for your GNN-based Knapsack solver, meeting the rigorous standards required for academic publication. It automates statistical testing, cross-validation, baseline comparisons, and generates publication-ready outputs.

## Key Features

### ✅ Statistical Rigor
- **Parametric Tests**: Paired t-test, independent t-test
- **Non-parametric Tests**: Wilcoxon signed-rank, Mann-Whitney U, Sign test, Friedman test
- **Effect Sizes**: Cohen's d, Cliff's Delta, Vargha-Delaney A
- **Assumption Checking**: Normality tests (Shapiro-Wilk, Anderson-Darling), variance homogeneity (Levene's test)
- **Multiple Testing Correction**: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg
- **Confidence Intervals**: Bootstrap CI with 10,000+ samples
- **Statistical Power Analysis**: Sample size justification

### ✅ Cross-Validation
- **K-Fold CV**: Standard k-fold with configurable folds
- **Stratified CV**: Stratify by problem size for balanced folds
- **Leave-One-Size-Out**: Extreme OOD test
- **Nested CV**: Unbiased hyperparameter selection

### ✅ Publication-Ready Outputs
- **LaTeX Tables**: Copy-paste ready for papers
- **High-Quality Figures**: 300 DPI PDF/PNG for publications
- **Statistical Reports**: Formatted summaries with interpretations
- **JSON Results**: Machine-readable validation data

## Quick Start

### Basic Validation

```bash
# Run validation with default settings
knapsack-gnn validate --checkpoint checkpoints/run_20251020_104533

# This runs:
# 1. Baseline comparisons (Greedy, Random)
# 2. Statistical significance tests
# 3. Effect size calculations
# 4. LaTeX table generation
# 5. Publication figures
```

### Full Validation (Recommended for Publications)

```bash
# Complete validation with all features
knapsack-gnn validate \
  --checkpoint checkpoints/run_20251020_104533 \
  --run-cv \
  --cv-folds 5 \
  --stratify-cv \
  --check-power \
  --latex \
  --figures \
  --output-dir validation_full
```

### Custom Configuration

```bash
# Use configuration file for reproducibility
knapsack-gnn validate \
  --checkpoint checkpoints/run_20251020_104533 \
  --config experiments/configs/validation_config.yaml
```

## Output Structure

After running validation, you'll get:

```
validation_report/
├── validation_results.json              # Complete results (machine-readable)
├── validation_report.txt                # Human-readable summary
├── baseline_comparison_table.tex        # LaTeX table for paper
├── statistical_tests_table.tex          # Statistical test results
├── method_comparison.pdf                # Box plot comparison
├── confidence_intervals.pdf             # CI plot
└── figures/                             # Additional figures
    ├── method_comparison.png
    └── confidence_intervals.png
```

## Statistical Tests Explained

### 1. Paired Tests (Same Instances)

**When to use**: Comparing two methods on the same test instances.

```python
# Automatically performed for baseline comparisons
validator.compare_with_baselines(
    gnn_gaps=gnn_results,
    dataset=test_dataset,
    baselines=['greedy', 'random']
)
```

**Tests performed**:
- **Paired t-test**: Parametric test assuming normality
- **Wilcoxon signed-rank**: Non-parametric alternative
- **Sign test**: Simple, robust alternative
- **Cohen's d**: Standardized effect size
- **Bootstrap CI**: 95% confidence interval

**Interpretation**:
```
Paired t-test: p < 0.001 ⭐ SIGNIFICANT
Cohen's d = 1.23 (large effect)
95% CI: [0.42%, 0.68%]

→ GNN significantly outperforms Greedy
```

### 2. Independent Tests

**When to use**: Comparing independent groups (e.g., different training runs).

```python
# Mann-Whitney U test for independent samples
result = stats_analyzer.mann_whitney_u_test(
    group_a=run1_gaps,
    group_b=run2_gaps
)
```

### 3. Multiple Methods Comparison

**When to use**: Comparing 3+ methods simultaneously.

```python
# Friedman test for multiple related samples
validator.compare_multiple_methods({
    'GNN-PNA': gnn_gaps,
    'Greedy': greedy_gaps,
    'Random': random_gaps
})
```

**Output**:
```
Friedman Test: χ² = 45.2, p < 0.001 ⭐
Pairwise comparisons (Holm-corrected):
  GNN vs Greedy: p < 0.001 ✓
  GNN vs Random: p < 0.001 ✓
  Greedy vs Random: p < 0.001 ✓
```

### 4. Assumption Checking

Before using parametric tests, check assumptions:

```python
assumptions = validator.check_test_assumptions(
    method_a=gnn_gaps,
    method_b=greedy_gaps
)

# Output:
# Normality: OK (Shapiro-Wilk p=0.23)
# Equal variances: OK (Levene p=0.45)
# Recommendation: Use parametric tests
```

### 5. Power Analysis

Justify your sample size:

```python
validator.run_power_analysis(
    observed_effect_size=1.2,  # Cohen's d
    current_sample_size=200,
    desired_power=0.8
)

# Output:
# Achieved power: 0.95 ✓
# Required n for 80% power: 120
# ✓ Sample size is adequate
```

## Cross-Validation

### Standard K-Fold

```python
from knapsack_gnn.analysis.cross_validation import KFoldValidator

validator = KFoldValidator(n_splits=5, stratify=True)
cv_results = validator.validate(
    train_fn=my_train_function,
    evaluate_fn=my_eval_function,
    dataset=full_dataset,
    config=config
)

print(cv_results.summary())
# Output:
# Mean Gap: 0.072% ± 0.015%
# 95% CI: [0.045%, 0.098%]
```

### Leave-One-Size-Out (Extreme OOD)

```python
from knapsack_gnn.analysis.cross_validation import LeaveOneSizeOutValidator

loso_validator = LeaveOneSizeOutValidator()
results = loso_validator.validate(
    train_fn=my_train_function,
    evaluate_fn=my_eval_function,
    dataset=dataset
)
```

## LaTeX Table Generation

### Baseline Comparison Table

```python
from knapsack_gnn.analysis.reporting import AcademicReporter

reporter = AcademicReporter()

results = {
    'GNN-PNA': {'mean_gap': 0.07, 'std_gap': 0.34, 'feasibility_rate': 1.0},
    'Greedy': {'mean_gap': 0.49, 'std_gap': 1.23, 'feasibility_rate': 1.0},
    'Random': {'mean_gap': 11.47, 'std_gap': 8.32, 'feasibility_rate': 1.0}
}

latex = reporter.generate_comparison_table(
    results,
    caption="Performance comparison on 200 test instances",
    label="tab:baseline_comparison"
)

print(latex)
```

**Output** (copy-paste ready):
```latex
\begin{table}[t]
\centering
\caption{Performance comparison on 200 test instances}
\label{tab:baseline_comparison}
\begin{tabular}{lccc}
\toprule
Method & Mean Gap (\%) & Median Gap (\%) & Feasibility \\
\midrule
GNN-PNA & \textbf{0.07} & \textbf{0.00} & \textbf{100.00\%} \\
Greedy & 0.49 & 0.13 & 100.00\% \\
Random & 11.47 & 12.67 & 100.00\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Statistical Tests Table

```latex
\begin{table}[t]
\caption{Statistical significance tests}
\begin{tabular}{lcccc}
\toprule
Comparison & Test & Statistic & $p$-value & Significant \\
\midrule
GNN vs Greedy & Paired t-test & 8.45 & $< 0.001$ & $\checkmark$ \\
GNN vs Random & Paired t-test & 15.2 & $< 0.001$ & $\checkmark$ \\
\bottomrule
\end{tabular}
\end{table}
```

## Publication Figures

### Box Plot Comparison

```python
reporter = AcademicReporter()

fig = reporter.create_boxplot_comparison(
    data={
        'GNN': gnn_gaps,
        'Greedy': greedy_gaps,
        'Random': random_gaps
    },
    ylabel="Optimality Gap (%)",
    title="Method Comparison",
    save_path="comparison"
)
# Saves: comparison.pdf and comparison.png (300 DPI)
```

### Confidence Interval Plot

```python
fig = reporter.create_confidence_interval_plot(
    results={
        'GNN': {'mean': 0.07, 'ci_95': (0.02, 0.12)},
        'Greedy': {'mean': 0.49, 'ci_95': (0.32, 0.66)}
    },
    ylabel="Optimality Gap (%)",
    save_path="confidence_intervals"
)
```

## Writing Results for Papers

### Describing Statistical Results

**Template**:
```
The GNN-PNA model achieved a mean optimality gap of X% ± Y% (M ± SD), 
significantly outperforming the Greedy baseline (Z% ± W%, t = A.BC, 
p < 0.001, d = D.EF [large effect]). A Wilcoxon signed-rank test 
confirmed this difference (p < 0.001), with 95% CI [L%, U%].
```

**Example**:
```
The GNN-PNA model achieved a mean optimality gap of 0.07% ± 0.34%, 
significantly outperforming the Greedy baseline (0.49% ± 1.23%, 
t(199) = 8.45, p < 0.001, d = 1.23 [large effect]). A Wilcoxon 
signed-rank test confirmed this difference (W = 1850, p < 0.001), 
with 95% CI [0.02%, 0.12%].
```

### Reporting Cross-Validation

```
Five-fold cross-validation yielded a mean optimality gap of 0.072% 
± 0.015% (95% CI: [0.045%, 0.098%]), demonstrating robust 
generalization performance.
```

### Sample Size Justification

```
Power analysis revealed that with n = 200 test instances and an 
observed effect size of d = 1.23, our study achieved a statistical 
power of 0.95, exceeding the conventional threshold of 0.80.
```

## Best Practices for Academic Validation

### ✅ Do's

1. **Always report**:
   - p-values AND confidence intervals
   - Effect sizes (not just significance)
   - Sample sizes
   - Statistical power (when possible)

2. **Check assumptions**:
   - Run normality tests
   - Check variance homogeneity
   - Use non-parametric tests if assumptions violated

3. **Correct for multiple comparisons**:
   - Use Holm or Benjamini-Hochberg correction
   - Report both raw and corrected p-values

4. **Be transparent**:
   - Report all comparisons (not just significant ones)
   - Include negative results
   - Specify all hyperparameters

5. **Use cross-validation**:
   - For generalization estimates
   - Especially if test set is small

### ❌ Don'ts

1. **Don't p-hack**:
   - Don't run multiple tests until you get p < 0.05
   - Pre-register your analysis plan if possible

2. **Don't rely solely on p-values**:
   - Always report effect sizes
   - p < 0.05 ≠ practically meaningful

3. **Don't ignore assumptions**:
   - Don't use t-tests on non-normal data
   - Check and report assumption violations

4. **Don't cherry-pick results**:
   - Report all planned comparisons
   - Use multiple testing correction

5. **Don't over-claim**:
   - Correlation ≠ causation
   - Statistical significance ≠ practical importance

## Configuration Reference

### validation_config.yaml

```yaml
# Statistical Parameters
statistical:
  alpha: 0.05              # Significance level
  n_bootstrap: 10000       # Bootstrap samples
  desired_power: 0.8       # Target power

# Baselines
baselines:
  - greedy
  - random

# Cross-Validation
cross_validation:
  enabled: true
  n_folds: 5
  stratify: true

# Experiments
experiments:
  baseline_comparison: true
  cross_validation: false  # Set true for full validation
  power_analysis: true
  assumption_checking: true

# Output
output:
  generate_latex: true
  generate_figures: true
  figure_dpi: 300
```

## API Reference

### StatisticalAnalyzer

```python
from knapsack_gnn.analysis.stats import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(alpha=0.05, n_bootstrap=10000)

# Paired tests
t_test = analyzer.paired_t_test(method_a, method_b)
wilcoxon = analyzer.wilcoxon_test(method_a, method_b)
sign = analyzer.sign_test(method_a, method_b)

# Independent tests
mann_whitney = analyzer.mann_whitney_u_test(group_a, group_b)

# Multiple methods
friedman = analyzer.friedman_test(method1, method2, method3)

# Effect sizes
cohens_d = analyzer.cohens_d(method_a, method_b)
cliffs_delta = analyzer.cliffs_delta(method_a, method_b)
vd_a = analyzer.vargha_delaney_a(method_a, method_b)

# Assumptions
normality = analyzer.shapiro_wilk_test(data)
variances = analyzer.levene_test(group_a, group_b)

# Power analysis
power = analyzer.statistical_power(effect_size=0.5, n_samples=200)
required_n = analyzer.required_sample_size(effect_size=0.5, power=0.8)
```

### PublicationValidator

```python
from knapsack_gnn.analysis.validation import PublicationValidator

validator = PublicationValidator(output_dir='validation_report')

# Compare with baselines
validator.compare_with_baselines(
    gnn_gaps=gnn_results,
    dataset=test_dataset,
    baselines=['greedy', 'random']
)

# Cross-validation
validator.run_cross_validation(
    train_fn=my_train_fn,
    evaluate_fn=my_eval_fn,
    dataset=full_dataset,
    config=config
)

# Power analysis
validator.run_power_analysis(
    observed_effect_size=1.2,
    current_sample_size=200
)

# Generate report
validator.generate_validation_report()
```

## Troubleshooting

### "Sample size too small for power analysis"

```python
# Solution: Increase test set size or accept lower power
# For small-scale experiments, report descriptive statistics
```

### "Normality assumption violated"

```python
# Solution: Use non-parametric tests
# Wilcoxon instead of t-test
wilcoxon_result = analyzer.wilcoxon_test(method_a, method_b)
```

### "statsmodels not installed"

```bash
# Solution: Install optional dependency
pip install statsmodels
```

## References

1. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. JMLR.
2. Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
3. Wasserstein, R. L., & Lazar, N. A. (2016). The ASA's statement on p-values. The American Statistician.
4. Cumming, G. (2014). The new statistics: Why and how. Psychological Science.

## Support

For issues or questions about the validation framework:
1. Check this documentation
2. Review examples in `experiments/pipelines/publication_validation.py`
3. Open an issue on GitHub

---

**Last Updated**: 2025-01-20  
**Framework Version**: 1.0.0
