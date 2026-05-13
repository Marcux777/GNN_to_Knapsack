# GNN_to_Knapsack

Learning-to-Optimize (L2O) toolkit that trains Graph Neural Networks (PNA, GCN, GAT) to solve the 0/1 knapsack problem with OR-Tools supervision, rich inference strategies, and publication-grade evaluation.

## Highlights
- **Graph-first pipeline** – item/capacity graphs, feature builders, dataset generators, and solver labels under `data/`.
- **Model zoo** – interchangeable PNA/GCN/GAT architectures plus greedy/repair/sampling decoders in `models/` and `inference/`.
- **Evaluation-ready** – CLI scripts for training, benchmarking, OOD tests, ablations, and interpretability (see `experiments/` + `results/`).
- **Guardrails** – `.codex/` playbooks, `make ci-local`, and a focused CI job (`codex-and-tests`) keep quality and reproducibility baked in.

## Quickstart
```bash
# 1) Install deps
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install && pre-commit install --hook-type pre-push  # optional, recommended

# 2) Generate a small dataset + train
python train.py --generate_data --train_size 1000 --val_size 200 --test_size 200

# 3) Evaluate or run OOD experiments
python evaluate.py --checkpoint_dir checkpoints/run_<stamp> --strategy sampling
python evaluate_ood.py --checkpoint_dir checkpoints/run_<stamp> --sizes 100 150 200

# 4) Local CI before any PR
make ci-local
```

### Graph feature toggles
- All dataset builders (`train_pipeline.py`, `evaluate_pipeline.py`, `experiments/pipelines/main.py`, `multi_seed_validation.py`, etc.) accept `--graph_features` plus `--graph_feature_buckets`.
- Tokens: `density` (capacity ÷ total weight), `quadratic` (`value²/weight`), `bucket` (value quantile buckets). Combine via commas (e.g., `density,quadratic`) or use `all`/`none`.
- Example:
  ```bash
  PYTHONPATH=src python experiments/pipelines/train_pipeline.py \
    --data_dir data/datasets/feature_ablation_small \
    --graph_features density,quadratic,bucket \
    --graph_feature_buckets 6
  ```
- Evaluation scripts accept `--graph_features auto` to reuse the checkpoint config. Headless runs can also set `KNAPSACK_GNN_GRAPH_FEATURES=all` (and `KNAPSACK_GNN_GRAPH_FEATURE_BUCKETS=4`) to turn features on everywhere by default.

### Visualizing the bipartite graphs
```bash
PYTHONPATH=src python scripts/plot_bipartite_graph.py \
  --dataset data/datasets/test.pkl \
  --indices 0 5 10 \
  --graph_features density,quadratic \
  --output_dir results/bipartite_graphs
```

## Repository map
| Path | Purpose |
| --- | --- |
| `.codex/` | Operating manual (system/project/style/tasks/runbook/tools/risks/eval). Read before editing. |
| `data/` | Dataset builders, graph conversion utilities, and storage (`data/datasets/` is local-only). |
| `models/`, `training/`, `inference/` | Core GNN modules, trainer loops, and decoding/repair strategies. |
| `experiments/` | Scripts for ablations, OOD runs, interpretability, and BC ranker experiments. |
| `results/` | Versioned artifacts (JSON, PNG, reports) ready for publication. |
| `scripts/` | Utilities (`bc_ranker_inspect.py`, `plot_bipartite_graph.py`, `verify_codex.py`, etc.). |
| `tests/` | Unit/integration tests (run via `make ci-local` or `make test`). |
| `checkpoints/` | Local checkpoints (ignored except for `ablations/**`); see `checkpoints/README.md`. |
| `artifacts/` | Personal scratch space for large or temporary outputs (ignored by git). |
| `docs/` | All documentation (guides, reports, validation, architecture). Use `docs/index.md` as entry point. |

## Workflow & quality gates
- **Kick off every task with `.codex/tasks.md`**: copy the BUGFIX/FEATURE/REFACTOR/EXPERIMENTO template into the issue/PR and fill the blanks.
- **`make ci-local` is mandatory**: runs `ruff format`, `ruff check`, `mypy`, and a quick pytest pass. The CI job `codex-and-tests` reruns this bundle and verifies `.codex/*` exists.
- **Use `scripts/verify_codex.py` + pre-commit**: prevents accidental removal of governance files; install hooks with `pre-commit install`.
- **Store noisy outputs under `artifacts/` or throwaway checkpoints**: everything else should be reproducible and checked in.

## Documentation & further reading
- [Documentation index](docs/index.md) – portal for execution guides, architecture notes, and language-specific reports.
- [Execution guide](docs/guides/execution_guide.md) – full pipelines, CLI arguments, and reproducibility tips.
- [Validation framework](docs/validation/validation_framework.md) – scientific protocol, metrics, and reporting cadence.
- [Experimental results](docs/reports/experimental_results_summary.md) – benchmarks, ablations, and diagrams ready for publication.
- [Executive summary (PT-BR)](docs/reports/sumario_executivo_pt-br.md) – high-level findings for stakeholders.
- [Profit-loss calibration](docs/reports/profit_loss_study.md) – how to rerun the hybrid-loss sweep added in this iteration.

### Profit-loss weight × dropout study (quick run)
To unblock the TODO request, we ran a lightweight sweep over the 3×3 grid using a small dataset (80/20/20 instances, seeds = {11}) so it fits within the current execution budget. The automation lives in `experiments/pipelines/profit_loss_study.py` and drops artefacts under `results/profit_loss_study_quick/` (see `raw_results.json` + `summary.json`).

Command executed (plots disabled via the new `plot_curves` flag on `train_model`):

```bash
PYTHONPATH=src python experiments/pipelines/profit_loss_study.py \
  --weights 0.1 0.25 0.5 \
  --dropouts 0.0 0.1 0.2 \
  --seeds 11 \
  --epochs 2 \
  --train_size 80 --val_size 20 --test_size 20 \
  --n_items_min 6 --n_items_max 9 \
  --output_dir results/profit_loss_study_quick \
  --data_dir data/datasets
```

| weight | dropout | mean gap (%) | max gap (%) | train time (s) |
| --- | --- | --- | --- | --- |
| 0.10 | 0.0 | 1.50 | 8.37 | 0.18 |
| 0.10 | 0.1 | 1.25 | 8.37 | 0.14 |
| 0.10 | 0.2 | **0.69** | 8.37 | 0.10 |
| 0.25 | 0.0 | 2.67 | 19.35 | 0.09 |
| 0.25 | 0.1 | 1.94 | 11.54 | 0.11 |
| 0.25 | 0.2 | 2.61 | 18.73 | 0.10 |
| 0.50 | 0.0 | 1.69 | 11.16 | 0.17 |
| 0.50 | 0.1 | 1.78 | 12.93 | 0.13 |
| 0.50 | 0.2 | 1.36 | **21.12** | 0.43 |

> **Note:** This was a “quick” run (single seed, tiny dataset) purely to collect empirical evidence for the calibration task without exhausting the allocated runtime. The pipeline supports larger seeds/epochs/datasets—simply bump the flags to reproduce the full-blown study described in `docs/reports/profit_loss_study.md`.

### Feature toggle ablation (quick run)
To validate the new optional graph inputs we trained four lightweight models on the same dataset (`train/val/test = 400/80/80`, items ∈ [8, 25], 6 epochs, seed 1337) and evaluated them with the sampling decoder (48 draws, schedule 16→32). Training command (spec placeholder):

```bash
PYTHONPATH=src python experiments/pipelines/train_pipeline.py \
  --data_dir data/datasets/feature_ablation_small \
  --checkpoint_dir checkpoints/feature_ablation \
  --train_size 400 --val_size 80 --test_size 80 \
  --n_items_min 8 --n_items_max 25 \
  --num_epochs 6 --batch_size 32 --learning_rate 0.0015 \
  --device cpu --graph_features <FEATURE_SPEC> --graph_feature_buckets 4
```

Evaluation reused `load_model_from_checkpoint` + `evaluate_model` with identical graph toggles (raw JSON was removed during repo cleanup; numbers below reflect the final run). Summary:

| Spec | Extra signals | Mean gap (%) | Median gap (%) | Max gap (%) | Feasibility | Mean latency (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | base features only | 2.40 | 0.00 | 55.92 | 100% | 1.52 |
| density | + capacity density | 3.07 | 0.00 | 39.80 | 100% | 1.59 |
| density+quadratic | + density & `value²/weight` | **1.83** | 0.00 | 36.85 | 100% | 1.58 |
| all | density + quadratic + bucket ranks | 2.10 | 0.00 | 36.85 | 100% | 1.58 |

`density+quadratic` delivered the tightest mean gap without hurting feasibility or latency, so it is the recommended knob for future sweeps. Temporary checkpoints/datasets used for this sweep were deleted during the cleanup—rerun the command above to regenerate them if needed.

## Contributing & support
- Follow the [Contributing guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).
- Keep commits in `type(scope): summary` format and run `make ci-local` before every push.
- Cite the project via [CITATION.cff](CITATION.cff). Questions or ideas? Open an issue referencing the filled template from `.codex/tasks.md`.
