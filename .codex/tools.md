Comandos padrão:
- `make format`
- `make lint`
- `make mypy`
- `make test-quick`
- `make test`
- `make pipeline`
- `make clean`

Sem Make? use:
- `ruff format src/ experiments/ tests/`
- `ruff check src/ experiments/ tests/`
- `mypy src/knapsack_gnn/ experiments/ --ignore-missing-imports`
- `pytest tests/ -v -m "not slow" --maxfail=1`
- `python Knapsack_GNN.py --demo quick`

Scripts úteis:
- `python verify_install.py`
- `python scripts/download_artifacts.py --checkpoint <run>`
- `python scripts/debug_repair.py`
- `python ablation_study.py --mode {features|architecture|both}`
- `python baselines/compare_baselines.py --checkpoint_dir <run>`
