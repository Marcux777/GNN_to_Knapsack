---
title: Profit-Loss Weight Calibration
---

# Profit-Loss Weight × Dropout Study

This document tracks the calibration experiment requested in `todo.md` for the
hybrid BCE + profit gap loss.

## How to Run

Use the new pipeline script:

```bash
python experiments/pipelines/profit_loss_study.py \
  --weights 0.1 0.25 0.5 \
  --dropouts 0.0 0.1 0.2 \
  --seeds 11 23 37 47 59 \
  --epochs 40 \
  --train_size 1000 --val_size 200 --test_size 200 \
  --data_dir data/datasets \
  --output_dir results/profit_loss_study
```

Outputs:

- `raw_results.json`: one record per (weight, dropout, seed) run with training time and gap stats.
- `summary.json`: aggregated mean/std gaps and average training time for each combination.

## Notes

- The script shares datasets across runs to keep comparisons consistent.
- Each run trains the same PNA architecture (`hidden_dim=64`, `num_layers=3`) and evaluates using the sampling decoder.
- Adjust `--weights`, `--dropouts`, `--seeds`, and other flags as needed for extended sweeps.
