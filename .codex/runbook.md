Pré-requisitos:
- Python 3.10+ (pyenv/conda). CUDA opcional.
- Pip/uv para instalar dependências. Docker não é obrigatório, mas pode ser usado para reprodutibilidade.

Setup rápido:
- `pip install -r requirements.txt`
- `pip install -r requirements-dev.txt`
- `pre-commit install && pre-commit install --hook-type pre-push` (opcional, recomendado)

Execução local:
- `python Knapsack_GNN.py --demo quick` (demo reduzida)
- `python train.py --generate_data --train_size 1000 --val_size 200 --test_size 200`
- `python evaluate.py --checkpoint_dir checkpoints/run_<stamp> --strategy sampling`
- `python evaluate_ood.py --checkpoint_dir <run> --sizes 100 150 200 --n_instances_per_size 50`
- `python baselines/compare_baselines.py --checkpoint_dir <run> --test_size 100`

Testes:
- `make ci-local` para rodar format + lint + mypy + testes rápidos de uma vez.
- `make test-quick` para feedback rápido.
- `make test` para suíte completa com cobertura.
- `make lint`, `make mypy` e `make format` antes do PR (caso precise rodar individualmente).

Dados/demos:
- Datasets gerados em `data/datasets/` (não versionados). Use `--data_dir` para customizar.
- Checkpoints padrão em `checkpoints/run_<timestamp>`; remova ou arquive antes do PR.
- Resultados oficiais vivem em `results/` com README descrevendo origem.

Variáveis e flags:
- `DEVICE` ou `--device` para escolher cpu/cuda.
- `--seed` em scripts para reprodutibilidade (default 42).
- `SAMPLING_SCHEDULE`, `--n_samples`, `--max_samples` controlam decoding.
- Ajuste `TRAIN_SIZE`, `EPOCHS`, `N_ITEMS_MAX` para rodadas rápidas.

Troubleshooting:
- Execute `python verify_install.py` se Torch/PyG/OR-Tools falharem.
- Reduza `--batch_size` e `hidden_dim` ao encontrar OOM.
- Use `CHECKPOINT_DIR=<path> make pipeline` para pipelines completos já treinados.
