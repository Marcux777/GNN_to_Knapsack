# Results - Publication Snapshots

**Last updated:** 2025-10-22

This directory contains **curated, publication-ready results** from experiments.

## 📁 Purpose

Unlike `checkpoints/` (which contains raw outputs from every run), `results/` contains:
- ✅ **Canonical snapshots** that are referenced in papers/docs
- ✅ **Baseline comparisons** for README figures
- ✅ **Publication-quality plots** (300 DPI PNG)
- ✅ **Summary tables** (CSV, JSON) for reference

## 🎯 What Belongs Here

**DO put here:**
- Figures referenced in README.md or documentation
- Baseline comparison results (e.g., GNN vs Greedy vs Random)
- Final validation metrics from published experiments
- Summary tables for paper submissions
- Any result you want to cite/reference later

**DON'T put here:**
- Raw outputs from individual training runs (→ `checkpoints/<run_id>/`)
- Intermediate debugging results
- Temporary experiment outputs
- Large binary files (→ Git LFS if truly needed)

## 📂 Current Structure

```
results/
├── ablations/
│   ├── architecture/                       # Architecture comparisons (PNA/GCN/GAT)
│   │   ├── architecture_ablation.json      # Aggregate metrics table
│   │   ├── architecture_ablation.png       # Bar chart used in docs
│   │   ├── learning_curves.png             # Overlayed learning curves
│   │   └── {gat,gcn,pna}/                  # Per-arch training curves + histories
│   └── features/
│       ├── feature_ablation.json           # Summary of feature drop tests
│       └── feature_ablation.png            # Visualization referenced in reports
├── baselines/                              # Baseline algorithm comparisons
│   ├── baseline_comparison.png             # GNN vs Greedy vs Random
│   └── comparison_results.json             # Numerical results
└── README.md                               # This file
```

Ablation checkpoints that accompany these figures are stored separately under
`checkpoints/ablations/architecture/<arch>/` to keep large binaries out of the
`results/` tree.

## 🔄 Workflow

1. **Run experiment** → outputs go to `checkpoints/<run_id>/evaluation/`
2. **Identify canonical results** → copy specific files here
3. **Reference in docs** → use paths like `results/baselines/figure.png`
4. **Version control** → commit these files (they're small and important)

## 📊 Versioning Policy

- ✅ **DO version:** PNG plots, small JSON/CSV files (<1 MB)
- ❌ **DON'T version:** Large datasets, model weights (use Git LFS or cloud storage)
- 📝 **Document origin:** Add comments or metadata about which run produced each file

## 🔗 See Also

- [Documentation Index](../docs/index.md) - All project documentation
- [Validation Report](../docs/reports/validation_report_2025-10-20.md) - Detailed experimental results
- [Execution Guide](../docs/guides/execution_guide.md) - How to reproduce results

---

**Philosophy:** This is the "source of truth" for publishable results. If it's in `results/`, it should be stable and reproducible.
