# Data Directory

**Last updated:** 2025-10-22

This directory stores **generated datasets** for training, validation, and testing.

## 🚨 Important Note

**This directory is NOT versioned in Git** (see `.gitignore`). Datasets are generated programmatically.

## 📦 Generating Datasets

Datasets are generated automatically during training or can be pre-generated:

### Automatic Generation (During Training)
```bash
knapsack-gnn train --config configs/train_default.yaml
# Datasets will be generated in data/datasets/
```

### Manual Pre-generation
```bash
python -c "
from knapsack_gnn.data import KnapsackDatasetGenerator
gen = KnapsackDatasetGenerator(
    num_instances=1000,
    min_items=10,
    max_items=50,
    seed=1337
)
gen.save('data/datasets/train.pkl')
"
```

## 📂 Expected Structure

After generation, you should see:
```
data/
├── datasets/
│   ├── train.pkl      # Training dataset
│   ├── val.pkl        # Validation dataset
│   └── test.pkl       # Test dataset
└── README.md          # This file
```

## 🔧 Dataset Properties

**Default training dataset (10-50 items):**
- Number of instances: 1,000
- Items per instance: 10-50 (uniform distribution)
- Weight/value generation: Uniform integers [1, 100]
- Capacity: ~50% of total weight
- Seed: 1337 (for reproducibility)

**Test dataset:**
- Number of instances: 200
- Same distribution as training
- Different seed for independence

## 🔄 Reproducibility

To reproduce exact datasets from papers/reports:
1. Check the seed in the configuration file
2. Use the same `KnapsackDatasetGenerator` parameters
3. Verify with checksums if provided in documentation

## 📝 Data Format

Datasets are stored as pickled Python objects containing:
- `instances`: List of knapsack problem instances
- `metadata`: Generation parameters, seeds, timestamps
- `optimal_solutions`: Precomputed optimal solutions (from OR-Tools)

## 🎓 For Research

If you're reproducing published results:
1. Check [Validation Report](../docs/reports/validation_report_2025-10-20.md) for exact parameters
2. Use the same seed (default: 1337)
3. Verify instance counts match documentation
4. Compare summary statistics (mean capacity, mean optimal value)

## 🔗 See Also

- [Execution Guide](../docs/guides/execution_guide.md) - How to run experiments
- [Data Generator Code](../src/knapsack_gnn/data/generator.py) - Implementation details
- [Main README](../README.md) - Project overview

---

**Note:** If you need the exact datasets used in a published paper, contact the authors or check supplementary materials.
