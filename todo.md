
# TODO — GNN_to_Knapsack

Formato: checklist objetivo, linguagem direta. Cada item aponta onde mexer e o que validar.

## 🔥 Prioridade Máxima (0–2 semanas)

- [x] **Trocar `print()` por `logging` (debug seguro)**
  - **Onde**: `src/knapsack_gnn/decoding/sampling.py` (ex.: `anytime_sampling` ~linhas 482–487 tem prints de debug), demais módulos com prints.
  - **Ação**: criar `knapsack_gnn/utils/logging.py` com `get_logger(__name__)`; substituir prints; níveis: INFO para fluxo, DEBUG para métricas internas.
  - **Aceite**: `pytest -q` sem capturar stdout; `ruff check` sem avisos de print.

- [x] **Testes unitários mínimos (caminho feliz + borda)**
  - **Onde**: criar `tests/`
  - **Casos**:
    - `test_generator_solver.py`: gera 20 instâncias; `KnapsackSolver.solve` retorna solução viável e `optimal_value` crescente se adicionar item de valor alto.
    - `test_graph_builder.py`: `Data` com `n_items+1` nós; arestas bidirecionais; `node_types` marca restrição; sem `NaN` após normalização.
    - `test_sampling.py`: `threshold/sampling/lagrangian` produzem solução viável; `sampling` respeita `max_samples`.
    - `test_cp_sat.py`: `solve_knapsack_warm_start` respeita `fixed_variables` e `time_limit`.
    - `test_cli.py`: `knapsack-gnn --help` e subcomandos básicos executam.
  - **Aceite**: cobertura mínima 70% nos módulos centrais.

- [x] **Determinismo e reprodutibilidade reforçados**
  - **Onde**: `experiments/pipelines/*`, `src/knapsack_gnn/training/loop.py`
  - **Ação**: `torch.use_deterministic_algorithms(True)` (proteção), `torch.backends.cudnn.deterministic=True`, `benchmark=False`; centralizar `set_seed`. Persistir seeds e hash do dataset no checkpoint.
  - **Aceite**: duas execuções com mesma seed produzem métricas idênticas (tolerância 1e-6).

- [x] **Pipeline de CI ampliado**
  - **Onde**: `.github/workflows/ci.yml`
  - **Ação**: matriz Python 3.10/3.11; cache de pip; jobs: `ruff format && ruff check`, `mypy`, `pytest -q`, build docs.
  - **Aceite**: CI verde no PR; artefatos de `pytest` e cobertura publicados.

- [x] **Verificações de normalização/robustez**
  - **Onde**: `graph_builder.py` e testes correspondentes
  - **Ação**: testar instâncias degeneradas (pesos/valores iguais, capacidade 0, item de peso 0, vetor vazio).
  - **Aceite**: sem exceções; `x` sem `NaN/Inf`.

## ✅ Qualidade/Manutenibilidade (2–6 semanas)

- [x] **Loss híbrido já existente: calibrar `profit_loss_weight`**
  - **Onde**: `src/knapsack_gnn/training/loop.py` (`_profit_gap_loss`)
  - **Ação**: estudo 3×3 para `profit_loss_weight ∈ {0.1, 0.25, 0.5}` × `dropout ∈ {0.0, 0.1, 0.2}` com 5 seeds; registrar `mean/median/max gap` e tempo.
  - **Aceite**: configuração com melhor p95-gap ≤ 1% e sem queda de viabilidade.

- [x] **Baselines adicionais para comparação**
  - **Onde**: `src/knapsack_gnn/baselines/`
  - **Ação**: implementar FPTAS e *meet-in-the-middle*; expor no CLI `knapsack-gnn compare`.
  - **Aceite**: relatórios exibem tempo vs. gap dessas variantes.

- [x] **Exportação e benchmarking de inferência**
  - **Onde**: novo comando `knapsack-gnn export --onnx` + `bench`
  - **Ação**: exportar ONNX; script de *benchmark* CPU-only (threads 1/N). Suporte a `torch.compile` e quantização dinâmica já presentes no sampler.
  - **Aceite**: planilha `results/bench_*.csv` com throughput e latência p50/p90/p99.

- [x] **Avaliação OOD mais ampla**
  - **Onde**: `experiments/pipelines/evaluate_ood_pipeline.py`
  - **Ação**: adicionar tamanhos 250/300/400; cenários com distribuição de valores/pesos não uniformes (ex.: lei de potência).
  - **Aceite**: figura “OOD generalization” atualizada; gap < 10% até 300 itens.

- [x] **Recursos de entrada extras (ablação de features)**
  - **Onde**: `graph_builder.py`
  - **Ação**: experimentar features: densidade normalizada, valor^2/weight, *bucketized ranks*; ativar via YAML.
  - **Aceite**: tabela de ablação em `docs/reports/experimental_results_summary.md`.
  - CLI expõe `--graph_features/--graph_feature_buckets` (e fallback via `KNAPSACK_GNN_GRAPH_FEATURES`), README/Execution docs descrevem os tokens.
  - Execução rápida (`train_size=400/80/80`, 6 épocas) documentada nas tabelas do README e em `docs/reports/experimental_results_summary.md` (artefatos removidos na limpeza).

- [ ] **Melhorias no *decoder***
  - **Onde**: `sampling.py`
  - **Ação**: `warm_start_repair` com 2-opt opcional; heurística de *biased sampling* por logits (top-k aquecidos).
  - **Aceite**: redução do gap máx. sem custo significativo de tempo.

## 🧱 Organização & DevEx

- [ ] **Configs com Hydra**
  - **Onde**: `experiments/configs/`
  - **Ação**: migrar para Hydra, centralizando seeds/paths/estratégias; `python -m experiments.pipelines.main full +model=pna +decoding=sampling`.
  - **Aceite**: execução parametrizável por linha de comando sem editar YAMLs.

- [ ] **Docs com MkDocs + GitHub Pages**
  - **Onde**: `docs/` → `mkdocs.yml`
  - **Ação**: gerar site a partir de `docs/guides/*`, `docs/api/*`, `docs/validation/*`.
  - **Aceite**: *workflow* publica site em cada tag.

- [ ] **Empacotamento e CLI**
  - **Onde**: `pyproject.toml`
  - **Ação**: garantir `console_scripts = ["knapsack-gnn=knapsack_gnn.cli:main"]`; *extras* `[cpu] [cuda] [dev]`.
  - **Aceite**: `pip install -e .[dev]` registra comando `knapsack-gnn`.

- [ ] **Higiene de repositório**
  - **Onde**: raiz do repo
  - **Ação**: adicionar `CODEOWNERS`, *issue/PR templates*, `SECURITY.md`, *Conventional Commits* e *release please*.
  - **Aceite**: primeira *release* v1.x com artefatos (checkpoints + JSONs).

- [ ] **Publicação de artefatos (Zenodo)**
  - **Onde**: `ZENODO.md`
  - **Ação**: *workflow* que anexa checkpoints e relatórios; atualiza DOI automaticamente.
  - **Aceite**: badge DOI no README apontando para o *record* da release.

## 🔬 Pesquisa & Roadmap

- [ ] **Arquiteturas alternativas**
  - **Ação**: `TransformerGNN` e `GAT` (já há infra de ablação); comparar contra PNA em 10–50 e 100–300 itens.
  - **Aceite**: quadro comparativo com *effect sizes* (Cohen’s d).

- [ ] **Incerteza & calibração**
  - **Ação**: MC Dropout, *temperature scaling* e Platt; usar ECE já existente como métrica; *budget-aware decoding* (parar cedo quando gap estimado < ε).
  - **Aceite**: ECE < 0,1 e melhora de *early stop* em ≥ 10% do tempo.

- [ ] **Planejamento de experimentos**
  - **Ação**: registrar protocolo (dados, seeds, métricas, *embargo*), CSVs em `checkpoints/*/evaluation`.
  - **Aceite**: reexecução bit‑a‑bit com `make verify-reproducibility`.

---

### Comandos de referência

```bash
# Testes e qualidade
pytest -q
ruff format . && ruff check .
mypy src/knapsack_gnn experiments

# Pipelines
PYTHONPATH=src python experiments/pipelines/main.py full --device cpu --skip-train --checkpoint-dir checkpoints/run_20251020_104533
python experiments/pipelines/evaluate_pipeline.py --checkpoint_dir checkpoints/run_20251020_104533 --strategy sampling --n_samples 128

# OOD
python experiments/pipelines/evaluate_ood_pipeline.py --checkpoint_dir checkpoints/run_20251020_104533 --sizes 100 150 200 300
```
