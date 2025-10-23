## Reproducibility Guide

This guide ensures you can reproduce experiments exactly and understand how our system maintains reproducibility.

## Quick Start

### 1. Download Pre-trained Checkpoints

```bash
# From GitHub Releases (< 100MB files)
python scripts/download_artifacts.py --checkpoint run_20251020_104533 --source github

# From Zenodo (complete datasets)
python scripts/download_artifacts.py --source zenodo --doi 10.5281/zenodo.XXXXX
```

### 2. Reproduce Main Results

```bash
# Reproduce sampling strategy results
make pipeline SKIP_TRAIN=1 CHECKPOINT_DIR=checkpoints/run_20251020_104533 PIPELINE_STRATEGIES="sampling"

# Full pipeline with warm-start
make pipeline SKIP_TRAIN=1 CHECKPOINT_DIR=checkpoints/run_20251020_104533 PIPELINE_STRATEGIES="sampling warm_start"
```

### 3. Verify Reproducibility

```bash
# Re-run training with same seed and verify results match
python scripts/verify_reproducibility.py --checkpoint checkpoints/run_20251020_104533 --tolerance 1e-6
```

## Seed Management

### How Seeds Work

Our system uses a centralized `set_seed()` function that sets seeds for:
- Python's `random` module
- NumPy
- PyTorch (CPU and CUDA)
- Python's hash seed (PYTHONHASHSEED)

**Example:**

```python
from knapsack_gnn.training.utils import set_seed

# Full determinism (slower but reproducible)
set_seed(42, deterministic=True)

# Faster but may not be fully deterministic on GPU
set_seed(42, deterministic=False)
```

### Seed Specification

Seeds can be set via:

**1. Configuration files:**

```yaml
seed: 1337

reproducibility:
  deterministic: true
  benchmark: false  # False for reproducibility, True for speed
```

**2. CLI arguments:**

```bash
knapsack-gnn train --seed 42 --epochs 50
```

**3. Environment variable:**

```bash
export KNAPSACK_SEED=42
python experiments/pipelines/train_pipeline.py
```

### Determinism vs Performance

| Mode | Speed | Reproducibility | Use Case |
|------|-------|-----------------|----------|
| `deterministic=True, benchmark=False` | Slowest | ✅ Exact | Publication, verification |
| `deterministic=False, benchmark=True` | Fastest | ⚠️ GPU-dependent | Development, hyperparameter search |

## Configuration Management

### Schema Validation

All configs are validated using Pydantic schemas:

```bash
# Validate all configs
make validate-configs

# Check single config
python -c "from knapsack_gnn.config import load_config; load_config('configs/train_default.yaml')"
```

### Config Snapshots

Every checkpoint automatically saves:

```
checkpoints/run_YYYYMMDD_HHMMSS/
├── config.yaml             # Exact config used
├── environment.txt         # pip freeze output
├── git_info.json          # Git commit, branch, diff
├── reproducibility.json   # Seed, hardware, determinism flags
├── metadata.json          # Timestamp, paths
├── model.pt               # Model weights
└── git_diff.patch        # Uncommitted changes (if any)
```

### Config Versioning

Track configuration changes in `CHANGELOG_CONFIGS.md`:

```bash
# View config history
cat CHANGELOG_CONFIGS.md

# When modifying schemas, document changes there
```

## Hardware Considerations

### CPU Reproducibility

✅ **Fully deterministic** with `deterministic=True`

```bash
# Results should match bit-for-bit
python experiments/pipelines/train_pipeline.py --seed 42 --device cpu
```

### GPU Reproducibility

⚠️ **Mostly deterministic** but some operations may vary across hardware

**Requirements for full determinism:**

```bash
# Set environment variables BEFORE running
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # OR :16:8

# Then run with deterministic mode
python experiments/pipelines/train_pipeline.py --seed 42 --device cuda
```

**Known Issues:**
- Some CUDA operations don't have deterministic implementations
- Results may differ slightly across GPU architectures
- For publication, use same GPU model or compare CPU results

### Checking Hardware Info

```python
from knapsack_gnn.utils.checkpoint import load_checkpoint_metadata

metadata = load_checkpoint_metadata("checkpoints/run_20251020_104533")
print(metadata["reproducibility"]["hardware"])
```

## Regenerating Datasets

Datasets are generated with fixed seeds and can be reproduced exactly:

```bash
# Generate datasets with specific seed
python scripts/generate_datasets.py --config configs/train_default.yaml --seed 1337

# Verify hash matches original
python scripts/generate_datasets.py --config configs/train_default.yaml --verify-hash
```

### Dataset Registry

`configs/datasets/registry.yaml` tracks all generated datasets:

```yaml
train_default_seed1337:
  seed: 1337
  n_items_min: 10
  n_items_max: 50
  train_size: 1000
  val_size: 200
  test_size: 200
  sha256: abc123...
  created: 2025-10-20T10:45:33
```

## Troubleshooting

### Results Don't Match

**1. Check exact seed and config:**

```bash
# Compare configs
diff checkpoints/run_A/config.yaml checkpoints/run_B/config.yaml

# Compare seeds
jq '.seed' checkpoints/run_A/reproducibility.json
jq '.seed' checkpoints/run_B/reproducibility.json
```

**2. Check determinism settings:**

```bash
jq '.deterministic' checkpoints/run_*/reproducibility.json
```

**3. Check environment differences:**

```bash
diff checkpoints/run_A/environment.txt checkpoints/run_B/environment.txt
```

**4. Check hardware:**

```bash
jq '.hardware' checkpoints/run_*/reproducibility.json
```

### CUDA Non-determinism

If results vary on GPU:

1. **Ensure CUBLAS config is set:**
   ```bash
   echo $CUBLAS_WORKSPACE_CONFIG  # Should be :4096:8 or :16:8
   ```

2. **Use CPU for exact reproducibility:**
   ```bash
   make pipeline DEVICE=cpu
   ```

3. **Accept small numerical differences:**
   - Use tolerance-based comparison (e.g., `atol=1e-6`)
   - Statistical tests instead of exact match

### Different PyTorch Versions

Minor differences expected across PyTorch versions. For exact reproduction:

```bash
# Install exact version from checkpoint
grep "^torch==" checkpoints/run_20251020_104533/environment.txt
pip install torch==X.Y.Z
```

## Best Practices

### For Authors (Running Experiments)

1. ✅ **Always specify seed** in config or CLI
2. ✅ **Use deterministic mode** for final results
3. ✅ **Document hardware** (automatically saved in checkpoint)
4. ✅ **Commit code** before experiments (git hash saved)
5. ✅ **Save config snapshot** (automatic with our system)
6. ✅ **Test reproducibility** before publication:
   ```bash
   python scripts/verify_reproducibility.py --checkpoint <dir>
   ```

### For Reviewers (Verifying Results)

1. **Download checkpoint:**
   ```bash
   python scripts/download_artifacts.py --checkpoint run_20251020_104533
   ```

2. **Check metadata:**
   ```bash
   cat checkpoints/run_20251020_104533/reproducibility.json
   ```

3. **Verify config is valid:**
   ```bash
   python -c "from knapsack_gnn.config import load_config; load_config('checkpoints/run_20251020_104533/config.yaml')"
   ```

4. **Re-run evaluation** (faster than full training):
   ```bash
   make evaluate CHECKPOINT_DIR=checkpoints/run_20251020_104533 TEST_ONLY=1
   ```

5. **(Optional) Full retraining:**
   ```bash
   python scripts/verify_reproducibility.py --checkpoint checkpoints/run_20251020_104533 --retrain
   ```

## Checklist

See [Reproducibility Checklist](../checklists/reproducibility_checklist.md) for complete verification steps.

## Additional Resources

- [Configuration Changelog](../../CHANGELOG_CONFIGS.md)
- [Artifact Management](artifacts.md)
- [Contributing Guide](contributing.md)

---

**Last Updated:** 2025-10-23
