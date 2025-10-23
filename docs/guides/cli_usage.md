# CLI Usage Guide

The `knapsack-gnn` CLI provides a unified interface for all operations.

## Available Commands

### Training

```bash
# Train with default configuration
knapsack-gnn train --config experiments/configs/train_default.yaml

# Train with custom parameters
knapsack-gnn train --seed 42 --device cuda --epochs 100
```

### Evaluation

```bash
# Evaluate with sampling strategy
knapsack-gnn eval --checkpoint checkpoints/run_001 --strategy sampling

# Evaluate with warm-start ILP
knapsack-gnn eval --checkpoint checkpoints/run_001 --strategy warm_start
```

### Pipeline

```bash
# Run full pipeline (train + eval)
knapsack-gnn pipeline --strategies sampling,warm_start --seed 1337
```

### Ablation Studies

```bash
# Feature ablation
knapsack-gnn ablation --mode features

# Architecture comparison
knapsack-gnn ablation --mode architecture
```

### Baseline Comparison

```bash
knapsack-gnn compare --checkpoint checkpoints/run_001 --baseline greedy
```

## Global Options

All commands support:
- `--seed` - Random seed for reproducibility
- `--device` - Device to use (cpu/cuda)
- `--verbose` - Increase output verbosity

See `knapsack-gnn --help` for full details.
