# Configuration Files

**Last updated:** 2025-10-22

This directory contains **canonical default configurations** for the GNN to Knapsack project.

## 📁 Hybrid Configuration Structure

We use a hybrid approach to avoid "configuration treasure hunts":

```
configs/                          # ← Canonical defaults (you are here)
├── train_default.yaml            # Default training configuration
├── validation_config.yaml        # Scientific validation settings
└── README.md                     # This file

experiments/configs/              # ← Experiment-specific overrides
├── ablations/                    # Ablation study configs
├── eval_sampling.yaml            # Sampling evaluation settings
├── eval_warm_start.yaml          # Warm-start evaluation settings
└── pipeline_full.yaml            # Full pipeline configuration
```

## 🎯 When to Use Which

### Use `configs/` (this directory) for:
- **Default parameters** that work for most use cases
- **Training from scratch** with standard settings
- **Quick starts** without special requirements
- **Documentation examples** in README/guides

### Use `experiments/configs/` for:
- **Paper-specific experiments** that need special tuning
- **Ablation studies** with modified architectures
- **Evaluation-specific settings** (sampling strategies, time limits)
- **Research pipelines** that combine multiple steps

## 📝 Available Canonical Configs

### `train_default.yaml`
Default training configuration:
- PNA architecture (3 layers, 64 hidden dims)
- 50 epochs, batch size 32
- Adam optimizer, lr=0.001
- Training on instances 10-50 items

**Usage:**
```bash
knapsack-gnn train --config configs/train_default.yaml
```

### `validation_config.yaml`
Scientific validation framework settings:
- Bootstrap CI (B=10,000)
- Statistical targets (p95 ≤ 1%, ECE < 0.1)
- Repair strategies enabled
- Comprehensive metrics collection

**Usage:**
```bash
python experiments/pipelines/in_distribution_validation.py \
    --config configs/validation_config.yaml
```

## 🔄 Override Pattern

You can override specific parameters without editing files:

```bash
# Use default config but change epochs
knapsack-gnn train --config configs/train_default.yaml --epochs 100

# Use default but run on GPU
knapsack-gnn train --config configs/train_default.yaml --device cuda
```

Or create experiment-specific YAML in `experiments/configs/` that inherits from defaults.

## 🚫 What NOT to Put Here

- ❌ One-off experimental configs (use `experiments/configs/`)
- ❌ Results or outputs (those go in `checkpoints/` or `results/`)
- ❌ Hardcoded paths specific to your machine
- ❌ Credentials or secrets (use environment variables)

## 📚 See Also

- [Main README](../README.md) - Project overview and quick start
- [Execution Guide](../docs/guides/execution_guide.md) - How to run experiments
- [Validation Report](../docs/reports/validation_report_2025-10-20.md) - Scientific validation details

---

**Philosophy:** Defaults should "just work" for the common case. Special cases get special configs in `experiments/`.
