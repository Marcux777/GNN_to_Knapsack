# Reproducibility Checklist

Use this checklist to ensure your experiments are reproducible.

## For Authors

### Before Running Experiments

- [ ] Code is committed to git (no uncommitted changes)
- [ ] Configuration file is validated (`make validate-configs`)
- [ ] Seed is specified in config or CLI
- [ ] Device is specified (cpu/cuda)
- [ ] Deterministic mode is enabled for final results

### After Running Experiments

- [ ] Checkpoint contains all metadata files:
  - [ ] `config.yaml` - Exact configuration used
  - [ ] `environment.txt` - Package versions
  - [ ] `git_info.json` - Code version
  - [ ] `reproducibility.json` - Seeds and hardware
- [ ] Results are documented with:
  - [ ] Mean and standard deviation
  - [ ] Confidence intervals
  - [ ] Hardware used (CPU/GPU model)
  - [ ] Runtime information
- [ ] Experiment can be reproduced:
  - [ ] Re-run produces same results (within tolerance)
  - [ ] Documented any expected variations (e.g., GPU non-determinism)

### Before Publication/Sharing

- [ ] Checkpoint uploaded to GitHub Releases or Zenodo
- [ ] SHA256 hashes documented
- [ ] README updated with reproduction instructions
- [ ] Results table includes seed and hardware info
- [ ] Configuration changes documented in `CHANGELOG_CONFIGS.md`

---

## For Reviewers

### Initial Verification

- [ ] Checkpoint/code is publicly accessible
- [ ] Configuration file is included
- [ ] Seed is specified
- [ ] Hardware specs are documented

### Metadata Check

- [ ] `reproducibility.json` contains:
  - [ ] Seed value
  - [ ] Deterministic flag
  - [ ] Hardware info
  - [ ] PyTorch version
- [ ] `git_info.json` shows clean state (no uncommitted changes)
- [ ] `environment.txt` lists all dependencies

### Reproduction Attempt

**Level 1: Checkpoint Evaluation (Fastest)**

- [ ] Download checkpoint
- [ ] Run evaluation on test set
- [ ] Compare metrics with reported results

**Level 2: Full Re-training (Complete Verification)**

- [ ] Install exact environment from `environment.txt`
- [ ] Checkout exact git commit
- [ ] Run training with same seed and config
- [ ] Compare final model performance

**Level 3: Statistical Verification**

- [ ] Run multiple seeds (e.g., 3-5 different seeds)
- [ ] Verify reported results fall within confidence intervals
- [ ] Check for cherry-picking (results should be typical, not best)

### Red Flags

- ⚠️ No seed specified
- ⚠️ Incomplete metadata (missing environment.txt, git info, etc.)
- ⚠️ Uncommitted code changes (`git_info.json` shows `is_dirty: true`)
- ⚠️ Results vary significantly across runs (> 5% difference)
- ⚠️ Cannot reproduce with same seed
- ⚠️ Hardware requirements not documented
- ⚠️ Only "best" run reported (no confidence intervals)

---

## Commands Reference

```bash
# Validate configuration
make validate-configs

# Download checkpoint
python scripts/download_artifacts.py --checkpoint run_20251020_104533 --source github

# Verify reproducibility
python scripts/verify_reproducibility.py --checkpoint checkpoints/run_20251020_104533

# Re-run evaluation only
make evaluate CHECKPOINT_DIR=checkpoints/run_20251020_104533 TEST_ONLY=1

# Full re-training
knapsack-gnn train --config checkpoints/run_20251020_104533/config.yaml
```

---

**Standard:** For publication-grade reproducibility, all items should be checked ✅
