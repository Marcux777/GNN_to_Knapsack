# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional governance files (CONTRIBUTING.md, CODE_OF_CONDUCT.md, CITATION.cff, CHANGELOG.md)
- Comprehensive documentation structure in `docs/` with organized reports, guides, and architecture docs
- Detailed experimental results report (`docs/reports/experimental_results.md`)

### Changed
- Refactored README.md from 572 to 327 lines for better readability
- Moved detailed experimental results to separate documentation file

## [1.0.0] - 2025-10-22

### Added
- **Scientific validation framework** (~3,200 lines of code):
  - Bootstrap confidence intervals (B=10,000 iterations)
  - Probability calibration (ECE, Brier score, Temperature/Platt scaling)
  - Solution repair mechanisms (greedy repair + local search)
  - Ablation study framework (PNA vs GCN vs GAT comparisons)
- **New modules**:
  - `src/knapsack_gnn/analysis/calibration.py` - Probability calibration tools
  - `src/knapsack_gnn/decoding/repair.py` - Solution repair strategies
- **Validation pipelines**:
  - `experiments/pipelines/in_distribution_validation.py` - Structured validation by size
  - `experiments/pipelines/ablation_study_models.py` - Architecture comparisons
  - `experiments/pipelines/create_publication_figure.py` - Publication-ready figures
  - `experiments/analysis/calibration_study.py` - Calibration analysis
  - `experiments/analysis/normalization_check.py` - Size invariance verification
  - `experiments/analysis/distribution_analysis.py` - Statistical analysis
- **Publication-ready visualizations**:
  - 4-panel figures (300 DPI) with gap analysis, CDF, violin plots, and reliability diagrams
  - LaTeX table generation
- **Hybrid configuration structure**:
  - Top-level `configs/` for canonical defaults
  - `experiments/configs/` for experiment-specific overrides
- **Project reorganization**:
  - Organized documentation in `docs/` (guides/, reports/, architecture/, validation/)
  - Created `results/` for publication snapshots
  - Refactored `.gitignore` to be directory-specific
  - Added Git LFS configuration for binary artifacts

### Changed
- Reorganized project structure for publication-grade clarity
- Improved `.gitignore` from global wildcards to specific directory ignores
- Enhanced statistical analysis in `src/knapsack_gnn/analysis/stats.py`
- Extended sampling strategies with repair variants

### Performance
- **Main results** (run_20251020_104533):
  - 99.93% of optimal value (0.068% gap) with sampling strategy
  - 100% feasibility across all test instances
  - 14.5 ms mean inference latency (CPU)
  - 7× more accurate than greedy baseline (0.07% vs 0.49% gap)

## [0.9.0] - 2025-10-20

### Added
- PNA-based GNN architecture for 0-1 Knapsack Problem
- Multiple decoding strategies:
  - Threshold-based decoding
  - Vectorized sampling with adaptive schedule
  - Warm-start ILP refinement
- OR-Tools integration for exact solutions and training labels
- Comprehensive evaluation pipeline with metrics:
  - Optimality gap analysis
  - Timing benchmarks
  - Feasibility checking
- CLI interface (`knapsack-gnn`) with subcommands:
  - `train` - Model training
  - `eval` - Evaluation
  - `ood` - Out-of-distribution testing
  - `pipeline` - Full workflow automation
  - `ablation` - Ablation studies
- Full test suite with unit and integration tests
- CI/CD pipeline with GitHub Actions
- Documentation structure

### Features
- **Bipartite graph representation**: Item nodes connected to capacity node
- **PNA architecture**: Principal Neighborhood Aggregation for expressive message passing
- **Exact solver integration**: OR-Tools generates optimal labels
- **Rich visualizations**: Gap histograms, learning curves, comparison plots

### Validation
- Out-of-distribution generalization tests (100-200 items, up to 4× training size)
- Baseline comparisons (Greedy, Random heuristics)
- Feature ablation studies
- Architecture ablation studies (PNA, GCN, GAT)

## [0.1.0] - 2025-10-15

### Added
- Initial prototype implementation
- Basic GNN training pipeline
- Dataset generation utilities for Knapsack instances
- Simple evaluation metrics
- OR-Tools solver integration

---

## Version Guidelines

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

### Unreleased Section

The `[Unreleased]` section tracks upcoming changes. When releasing a new version:
1. Create a new version section with release date
2. Move items from `[Unreleased]` to the new version
3. Clear the `[Unreleased]` section

### Categories

Changes are grouped by category:
- **Added** - New features
- **Changed** - Changes to existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security patches
- **Performance** - Performance improvements
