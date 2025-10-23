# Zenodo Artifacts Registry

This document tracks all datasets and checkpoints published on Zenodo for permanent archival and citation.

## Published Artifacts

### Checkpoints

| Run ID | DOI | Description | Size | Upload Date | Status |
|--------|-----|-------------|------|-------------|--------|
| run_20251020_104533 | [DOI pending] | Main results checkpoint (PNA, 99.93% optimal) | ~50 MB | - | ⏳ Pending upload |

### Datasets

| Dataset | DOI | Description | Size | Upload Date | Status |
|---------|-----|-------------|------|-------------|--------|
| train_default_seed1337 | [DOI pending] | Training dataset (n=10-50, seed=1337) | ~100 MB | - | ⏳ Pending upload |

---

## How to Upload to Zenodo

### 1. Prepare Checkpoint for Upload

```bash
# Create archive with metadata
cd checkpoints
tar -czf run_20251020_104533.tar.gz run_20251020_104533/

# Compute SHA256 hash
shasum -a 256 run_20251020_104533.tar.gz
```

### 2. Upload to Zenodo

1. Go to https://zenodo.org/
2. Click "Upload" → "New upload"
3. Upload `run_20251020_104533.tar.gz`
4. Fill metadata:

```yaml
Upload type: Dataset
Title: "Knapsack GNN Checkpoint: run_20251020_104533"
Authors: Marcus Vinicius
Description: |
  Pre-trained PNA-based Graph Neural Network checkpoint for solving 0-1 Knapsack Problem.

  Achieves 99.93% of optimal value (0.068% gap) on instances with 10-50 items.

  Contents:
  - model.pt: Trained model weights
  - config.yaml: Exact configuration used
  - environment.txt: Package versions
  - git_info.json: Code version (commit hash)
  - reproducibility.json: Seeds, hardware info

  See https://github.com/Marcux777/GNN_to_Knapsack for code and reproduction instructions.

Keywords: graph-neural-networks, combinatorial-optimization, knapsack-problem, deep-learning
License: MIT
Related identifiers:
  - GitHub repository: https://github.com/Marcux777/GNN_to_Knapsack
```

5. Click "Publish"
6. Copy DOI and update this file

### 3. Update Documentation

After getting DOI, update:

1. **This file (ZENODO.md)**:
   ```markdown
   | run_20251020_104533 | https://doi.org/10.5281/zenodo.XXXXX | ... | ... | 2025-10-23 | ✅ Published |
   ```

2. **README.md**:
   Add download instructions with Zenodo DOI

3. **Reproducibility Guide**:
   Update download commands

---

## Download from Zenodo

### Using Our Script

```bash
# Download checkpoint
python scripts/download_artifacts.py \
  --source zenodo \
  --doi 10.5281/zenodo.XXXXX \
  --filename run_20251020_104533.tar.gz
```

### Manual Download

```bash
# Download directly
wget https://zenodo.org/record/XXXXX/files/run_20251020_104533.tar.gz

# Verify hash
shasum -a 256 run_20251020_104533.tar.gz

# Extract
tar -xzf run_20251020_104533.tar.gz -C checkpoints/
```

---

## Metadata Template for New Uploads

When uploading new artifacts, use this template:

```yaml
# Checkpoint Metadata Template
Upload type: Dataset
Title: "Knapsack GNN Checkpoint: <RUN_ID>"
Authors: <YOUR_NAME>
Description: |
  Pre-trained checkpoint for Knapsack GNN.

  Model: <MODEL_TYPE>
  Performance: <MEAN_GAP>% gap, <ACCURACY>% accuracy
  Training: <TRAIN_SIZE> instances, <EPOCHS> epochs, seed <SEED>

  Contents:
  - model.pt: Model weights
  - config.yaml: Configuration
  - environment.txt: Dependencies
  - git_info.json: Code version
  - reproducibility.json: Metadata

  Repository: https://github.com/Marcux777/GNN_to_Knapsack
  Commit: <GIT_HASH>

Keywords: graph-neural-networks, combinatorial-optimization, knapsack-problem
License: MIT
Version: v1.0.0
Related identifiers:
  - Repository: https://github.com/Marcux777/GNN_to_Knapsack (isSupplementTo)
  - Paper DOI: <IF_APPLICABLE> (cites)
```

---

## Versioning Strategy

- **Checkpoints**: Use git commit hash + run ID
- **Datasets**: Use schema version + seed (e.g., v1.0_seed1337)
- **Zenodo versions**: Create new version for significant changes, otherwise new record

---

## Citation

If you use these artifacts, please cite:

```bibtex
@dataset{knapsack_gnn_checkpoint_2025,
  author       = {Marcus Vinicius},
  title        = {Knapsack GNN Checkpoint: run_20251020_104533},
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXX}
}
```

---

**Last Updated:** 2025-10-23
