Nome do projeto: GNN_to_Knapsack (Learning to Optimize 0-1 Knapsack)
Stack: Python 3.10+, PyTorch 2.x, PyTorch Geometric, OR-Tools, PyTest, Ruff, Mypy.

Arquitetura (resumo):
- train.py / evaluate.py / Knapsack_GNN.py  # entrypoints de treino, avaliação e demo
- src/knapsack_gnn/data/                    # geradores de instâncias e grafos
- src/knapsack_gnn/models/                  # PNA, GCN, GAT e helpers
- src/knapsack_gnn/decoding/                # estratégias de decodificação + reparo
- src/knapsack_gnn/training/                # loops de treino e métricas
- src/knapsack_gnn/eval/ & utils/           # relatórios, helpers e métricas
- experiments/                              # pipelines e estudos (ablações, OOD)
- training/                                 # scripts legados de treino/checkpoints
- baselines/                                # heurísticas e comparadores
- inference/                                # samplers e lógica de inferência
- results/                                  # snapshots versionados (PNG/JSON)
- docs/                                     # documentação em Markdown

Fluxos críticos:
1) Geração de datasets → `KnapsackDataset` → transformação para grafos → treino em `train.py`.
2) Avaliação → `evaluate.py`/`evaluate_ood.py` → executa decoders (sampling, warm-start, etc.) → outputs JSON/plots.
3) Ablações → `ablation_study.py` → escreve resultados em `results/ablations` e checkpoints em `checkpoints/ablations`.
4) Demo/Inference → `Knapsack_GNN.py` ou `inference/sampler.py` → pipeline ponta a ponta.

Variáveis de ambiente relevantes:
- DEVICE (cpu/cuda) ou `--device` nos scripts.
- OMP_NUM_THREADS / TORCH_NUM_THREADS para limitar CPU.
- Seeds (`--seed`, `PYTHONHASHSEED`) para reprodutibilidade.

Mapa do repo (1–2 linhas por diretório):
- src/knapsack_gnn: núcleo da biblioteca (data, models, decoding, training, utils).
- experiments: pipelines de pesquisa, scripts CLI longos.
- training: versão antiga do loop de treino em Python puro.
- baselines: heurísticas greedy/random e comparações com OR-Tools.
- inference: componentes de amostragem para uso em produção.
- docs: guias, relatórios e sumários (EN/PT-BR).
- results: artefatos publicados (PNG, JSON, README).
- scripts: utilidades como `download_artifacts.py`, `debug_repair.py`.
- configs: YAMLs com configs de treino/datasets.
- tests: testes automatizados (unit/integration).

Pontos de extensão:
- Novas arquiteturas → adicionar em `src/knapsack_gnn/models/` e registrar nos scripts.
- Novos decoders/heurísticas → `src/knapsack_gnn/decoding/` + CLI.
- Novos experimentos → `experiments/` ou `docs/reports/` com instruções.
- Novos resultados oficiais → `results/` com README indicando origem.
