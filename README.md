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

### Visualizing the bipartite graphs
```bash
PYTHONPATH=src python scripts/plot_bipartite_graph.py \
  --dataset data/datasets/test.pkl \
  --indices 0 5 10 \
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
- [Experimental results](docs/reports/experimental_results.md) – benchmarks, ablations, and diagrams ready for publication.
- [Executive summary (PT-BR)](docs/reports/sumario_executivo_pt-br.md) – high-level findings for stakeholders.

## Contributing & support
- Follow the [Contributing guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).
- Keep commits in `type(scope): summary` format and run `make ci-local` before every push.
- Cite the project via [CITATION.cff](CITATION.cff). Questions or ideas? Open an issue referencing the filled template from `.codex/tasks.md`.
