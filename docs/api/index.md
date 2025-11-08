# API Reference

This section provides detailed API documentation for the `knapsack-gnn` library.

## Overview

The library is organized into the following modules:

### Core Modules

- **[Data](data.md)** - Problem generation and graph construction
  - Dataset classes for knapsack problems
  - Graph builder for bipartite graph representation
  - Data loading utilities

- **[Models](models.md)** - GNN architectures
  - PNA (Principal Neighborhood Aggregation)
  - GCN (Graph Convolutional Network)
  - GAT (Graph Attention Network)

- **[Training](training.md)** - Training loops and utilities
  - Trainer class with early stopping
  - Loss functions
  - Metrics and logging

- **[Decoding](decoding.md)** - Solution decoding strategies
  - Threshold decoding
  - Sampling strategies (vectorized, adaptive)
  - Warm-start ILP refinement
  - Lagrangian projection

- **[Evaluation](eval.md)** - Evaluation and reporting
  - Performance metrics
  - Result aggregation
  - Visualization utilities

- **[Analysis](analysis.md)** - Statistical analysis and validation
  - Distribution analysis
  - Probability calibration
  - Solution repair
  - Ablation studies

## Quick Example

```python
from knapsack_gnn.data import KnapsackDataset
from knapsack_gnn.models import PNAModel
from knapsack_gnn.training import Trainer

# Load dataset
dataset = KnapsackDataset(
    data_dir="data/datasets",
    n_items_min=10,
    n_items_max=50,
    split="train"
)

# Create model
model = PNAModel(
    hidden_dim=64,
    num_layers=3,
    dropout=0.1
)

# Train
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    learning_rate=0.002
)
trainer.train(num_epochs=50)
```

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## CLI Interface

The library provides a unified CLI interface:

```bash
# Train a model
knapsack-gnn train --config experiments/configs/train_default.yaml

# Evaluate a model
knapsack-gnn eval --checkpoint checkpoints/run_001 --strategy sampling

# Run full pipeline
knapsack-gnn pipeline --strategies sampling,warm_start
```

See the [CLI Usage Guide](../guides/cli_usage.md) for more details.

## Type Definitions

Common types used throughout the library are defined in `knapsack_gnn.types`.

## Further Reading

- [User Guide](../guides/execution_guide.md) - Installation and usage instructions
- [Development Guide](../development.md) - Contributing and development setup
