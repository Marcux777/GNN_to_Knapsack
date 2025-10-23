# Quick Start

This guide will get you up and running with knapsack-gnn in minutes.

## Installation

```bash
pip install -e .
```

## Basic Usage

### 1. Train a Model

```bash
knapsack-gnn train --config experiments/configs/train_default.yaml
```

Or using the Makefile:

```bash
make train EPOCHS=50 DEVICE=cpu
```

### 2. Evaluate

```bash
knapsack-gnn eval --checkpoint checkpoints/run_<timestamp> --strategy sampling
```

### 3. Run Full Pipeline

```bash
knapsack-gnn pipeline --strategies sampling,warm_start --seed 1337
```

## Next Steps

- Read the [Execution Guide](execution_guide.md) for detailed instructions
- Check the [CLI Usage Guide](cli_usage.md) for all available commands
- Explore the [API Reference](../api/index.md) for programmatic usage
