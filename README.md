# GNN_to_Knapsack

[![CI](https://github.com/Marcux777/GNN_to_Knapsack/actions/workflows/ci.yml/badge.svg)](https://github.com/Marcux777/GNN_to_Knapsack/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Estudo sobre como Redes Neurais Gráficas (Graph Neural Networks – GNNs) podem ser aplicadas para resolver o Problema da Mochila 0/1.

## Knapsack GNN - Learning to Optimize

Implementação de GNNs para o Problema da Mochila 0-1 utilizando a abordagem **Learning to Optimize (L2O)**.

> **Implementação em nível de pesquisa**: atinge 99,93% do valor ótimo (gap de 0,07%) com validações completas (generalização OOD, comparação com baselines, ablações). Inclui visualizações prontas para publicação e documentação extensa.

## Visão Geral

Este projeto implementa uma GNN baseada em **Principal Neighborhood Aggregation (PNA)** para resolver problemas de otimização combinatória, em especial o Problema da Mochila 0-1. O problema é transformado em um grafo e a rede aprende, de forma supervisionada, a prever soluções ótimas.

### Principais Recursos

- **Grafo bipartido**: nós de itens conectados a um único nó de capacidade (veja `data/graph_builder.py`).
- **Arquitetura PNA**: passagem de mensagens expressiva via Principal Neighborhood Aggregation.
- **Várias estratégias de inferência**: threshold, amostragem vetorizada, amostragem adaptativa, ILP warm-start.
- **Integração com solver exato**: OR-Tools gera rótulos ótimos para treino e benchmarking.
- **Avaliação abrangente**: análise de gap, benchmarks de tempo, estudos de ablação e visualizações ricas.

## 📋 Atualizações Operacionais (nov/2025)

Para manter o repositório alinhado ao fluxo Codex, consolidamos as seguintes rotinas:

- `.codex/tasks.md` agora traz cartões padronizados (BUGFIX/FEATURE/REFACTOR/EXPERIMENTO) que devem ser copiados em qualquer issue/PR antes de iniciar uma demanda.
- `make ci-local` executa `ruff format .`, `ruff check .`, `mypy src/knapsack_gnn experiments || true` e `pytest -q --maxfail=1 -k "not slow"` em sequência; use antes de cada PR.
- `.github/PULL_REQUEST_TEMPLATE.md` e `CONTRIBUTING.md` exigem que contribuidores leiam `.codex/system.md`, usem o template adequado e marquem a checklist de `.codex/eval.md`.
- `.pre-commit-config.yaml` inclui o hook local `codex-guard` (script `scripts/verify_codex.py`) que bloqueia remoções não autorizadas de `.codex/*`.
- **Nov 2025**: `make ci-local` executado com sucesso (fmt/lint/mypy/teste rápido) após ajustes de lint nos scripts utilitários; mantenha esse alvo como verificação mínima pré-PR.
- **Nov 2025**: `results/bipartite_graphs/` atualizado com `bipartite_0/5/10.png` gerados a partir de `data/datasets/test.pkl` (ver comando abaixo) para auditar a distribuição do grafo item↔capacidade.
- **Nov 2025**: GitHub Actions (`.github/workflows/ci.yml`) agora bloqueia merges sem `.codex/*` completo ou sem `make ci-local` limpo, reproduzindo automaticamente o passo local no CI.
- **Nov 2025**: Guia rápido de smoke test do `codex-and-tests` disponível em `docs/development.md#codex-ci-smoke-test` para abrir PRs de validação e monitorar o novo job.
- **Nov 2025**: Workflow do CI com cache de `pip` para acelerar execuções repetidas do `make ci-local`.

> Este README também funciona como relatório vivo: seções de atualização documentam exatamente o que foi configurado em cada passo do plano Codex.

### Visualização do grafo bipartido (passo 3)

Para inspecionar a distribuição do grafo item↔capacidade usado em cada instância, gere PNGs com:

```bash
PYTHONPATH=src python scripts/plot_bipartite_graph.py \
  --dataset data/datasets/test.pkl \
  --indices 0 5 10 \
  --output_dir results/bipartite_graphs
```

O script usa o builder padrão (`KnapsackGraphBuilder`) e salva as figuras no diretório informado (por padrão `results/bipartite_graphs`). Rode com `--normalize` para verificar como as features normalizadas afetam os pesos das arestas.

### Resultados mais recentes (run_20251020_104533 – CPU)

**🏆 Resultado principal: 99,93% do valor ótimo (gap 0,068%) com amostragem adaptativa; warm-start ILP chega a 0,18% com refinamentos de 1,9 ms.**

## 🔬 Framework Científico

**NOVO (out/2025):** framework completo de validação científica para resultados em nível de publicação. Status: ✅ **8/10 tarefas implementadas** (~3.200 linhas de código de validação).

### Inclui

- ✅ **Estatística rigorosa**: bootstrap (B=10k), percentis (p50/p90/p95/p99), CDF, checagem de tamanho de amostra.
- ✅ **Calibração**: ECE, Brier score, Temperature/Platt scaling, reliability plots.
- ✅ **Reparo de soluções**: reparo guloso + busca local (1-swap, 2-opt) para remover outliers.
- ✅ **Ablação**: PNA vs GCN vs GAT; 2/3/4 camadas.
- ✅ **Figuras para publicação**: painéis 4k (300 DPI) e tabelas LaTeX.
- ✅ **Verificações de normalização**: invariância a tamanho e análise de ativação dos agregadores.

### Primeiros passos – validação

```bash
python experiments/analysis/distribution_analysis.py \
    --results checkpoints/run_20251020_104533/evaluation/results_sampling.json \
    --output-dir checkpoints/run_20251020_104533/evaluation/analysis

python experiments/pipelines/create_publication_figure.py \
    --results-dir checkpoints/run_20251020_104533/evaluation \
    --output-dir checkpoints/run_20251020_104533/evaluation/publication
```

### Métricas de validação

| Métrica | Meta | Atual | Status |
|--------|------|-------|--------|
| Gap p95 (10–50 itens) | ≤ 1% | 0,54% | ✅ |
| Gap máx. (após reparo) | < 2% | 2,69%→<2%* | ⏳ |
| Calibração ECE | < 0,1 | TBD | ⏳ |
| Viabilidade | 100% | 100% | ✅ |

\* após execução do reparo

### Documentação

- 📄 **[Relatório de Validação](docs/reports/validation_report_2025-10-20.md)**
- 📄 **[Resumo da Implementação](docs/architecture/implementation_summary.md)**
- 📄 **[Guia de Execução](docs/guides/execution_guide.md)**
- 📄 **[Sumário Executivo (PT-BR)](docs/reports/sumario_executivo_pt-br.md)**
- 📄 **[Índice da Documentação](docs/index.md)**

### Para pesquisadores

O framework entrega evidência de nível de publicação via:
- Intervalos de confiança (bootstrap B=10.000)
- Percentis (p50/p90/p95/p99)
- Calibração ECE < 0,1
- Reparo de soluções
- Ablações completas (PNA/GCN/GAT)
- Figuras e tabelas em formato editorial

| Estratégia | Configuração | Gap médio | Gap mediano | Gap máx. | Viabilidade | Tempo médio | Observações |
|------------|--------------|-----------|-------------|----------|-------------|-------------|-------------|
| Sampling | cronograma 32→64→128 | **0,068%** | **0,00%** | 4,57% | **100%** | 14,5 ms | 61,9 amostras (~69 inst/s) |
| Warm Start | Sampling + ILP (fix ≥0,9; 1s) | 0,18% | 0,00% | 9,41% | **100%** | 21,8 ms | ILP 1,90 ms; 98,5% óptimo |

## 🔄 Reprodutibilidade

**Garantia para publicação:** scripts reproduzem cada experimento com rastreamento completo.

```bash
make download-checkpoint RUN=run_20251020_104533
make evaluate CHECKPOINT_DIR=checkpoints/run_20251020_104533 TEST_ONLY=1
make verify-reproducibility CHECKPOINT_DIR=checkpoints/run_20251020_104533
```

Recursos:
- Seeds centralizados em `set_seed()`
- Schemas Pydantic verificam YAMLs
- Checkpoints salvam config + ambiente + git + hardware
- Artefatos disponíveis via Releases/Zenodo
- Histórico de configs em `CHANGELOG_CONFIGS.md`

Documentos úteis:
- 📖 [Guia de Reprodutibilidade](docs/guides/reproducibility.md)
- ✓ [Checklist](docs/checklists/reproducibility_checklist.md)
- 📝 [Changelog de Configs](CHANGELOG_CONFIGS.md)

## 🔌 Extensibilidade

Arquitetura modular facilita novos modelos, decoders e problemas.

```python
@ModelRegistry.register("transformer_gnn")
class TransformerGNN(AbstractGNNModel):
    def forward(self, data):
        ...
```

```python
class BeamSearchDecoder(AbstractDecoder):
    def decode(self, model_output, problem_data):
        ...
```

```python
class TSPProblem(OptimizationProblem):
    def to_graph(self, instance):
        ...
```

Recursos adicionais:
- 📓 [Notebooks](notebooks/)
- 📖 [Guias de Dev](docs/dev/)
- 🎓 [Tutoriais](docs/tutorials/)
- 📋 [Templates](templates/)

## Instalação e Reprodução

```bash
git clone https://github.com/Marcux777/GNN_to_Knapsack.git
cd GNN_to_Knapsack
pip install -e .
knapsack-gnn train --config experiments/configs/train_default.yaml
knapsack-gnn eval --checkpoint checkpoints/run_XXX --strategy sampling
knapsack-gnn pipeline --strategies sampling,warm_start --seed 1337
```

Reproduzir run_20251020_104533 via Makefile:

```bash
export PYTHONHASHSEED=1337
make pipeline PIPELINE_STRATEGIES="sampling warm_start" \
  SKIP_TRAIN=1 CHECKPOINT_DIR=checkpoints/run_20251020_104533 \
  DEVICE=cpu SEED=1337
```

### Instalação

```bash
pip install -e .[cpu]
pip install -e .[cuda]
pip install -e .[dev]
conda env create -f environment.yml
conda activate knapsack-gnn
```

### CLI `knapsack-gnn`

Inclui comandos para treino, avaliação, testes OOD, pipelines, ablações, comparação de baselines e demo interativa (veja README original para exemplos completos).

### Makefile (legado)

| Comando | Descrição |
|---------|-----------|
| `make install` | instala dependências |
| `make train` | treina modelo |
| `make eval` | avalia checkpoint |
| `make pipeline` | workflow completo |
| `make ood` | teste OOD |
| `make test` | suite de testes |
| `make lint` | lint/qualidade |

### Notas de reprodutibilidade

- Configure `PYTHONHASHSEED` e `SEED` para execuções determinísticas.
- Para reproduzir `run_20251020_104533`: seed 1337, commit `3ccf6b1`, CPU x86_64, Python 3.10+.
- Todos os relatórios geram `results_per_instance.csv` e `summary_metrics.csv`.

## Estrutura do Projeto

```
.
├── src/knapsack_gnn/
├── experiments/
├── tests/
├── checkpoints/
├── data/
└── results/
```

Princípios:
- `src/knapsack_gnn/`: biblioteca estável
- `experiments/`: pipelines de pesquisa
- `tests/`: cobertura completa
- `configs/`: configs versionadas

## 📊 Resumo de Resultados

- **BC Ranker Supervisionado (30 épocas, 8 features)**:
  - PNA: gap médio 0,55% (mediana 0,16%), factibilidade 100%.
  - GCN: gap médio 0,54% (mediana 0,17%), factibilidade 100%.
  - GAT: gap médio 0,51% (mediana 0,16%), factibilidade 100%.
  - Checkpoints + métricas em `checkpoints/results/bc_ranker_full/<arch>/`.
  - Artefatos de interpretabilidade (scores × seleção, densidade × score, curva cumulativa + Spearman e sensibilidade ±5% de capacidade) em `results/reports/bc_ranker_v1/`.

- **Decoders em run_20251020_104533**:

| Estratégia | Gap médio | Gap mediano | Viabilidade | Tempo médio | Notas |
|------------|-----------|-------------|-------------|-------------|-------|
| Sampling | **0,068%** | **0,00%** | **100%** | 14,5 ms | 61,9 amostras |
| Warm Start | 0,18% | 0,00% | **100%** | 21,8 ms | ILP 1,9 ms |

Mais detalhes em:
- 📄 [Experimental Results Report](docs/reports/experimental_results.md)
- 📄 [Validation Report](docs/reports/validation_report_2025-10-20.md)
- 📄 [Documentation Index](docs/index.md)

## 🤝 Contribuições

Confira:
- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- `.codex/` – pacote de configuração do Codex (system, style, runbook, templates). Leia `/.codex/system.md` antes de automatizar tarefas.

## 📚 Citação

```bibtex
@software{knapsack_gnn_2025,
  author = {Vinicius, Marcus},
  title = {GNN to Knapsack: Learning to Optimize with Graph Neural Networks},
  year = {2025},
  url = {https://github.com/Marcux777/GNN_to_Knapsack},
  version = {1.0.0}
}
```

Use também o botão “Cite this repository” (arquivo [CITATION.cff](CITATION.cff)).

## 📖 Referências

1. [Learning to Solve Combinatorial Optimization with GNNs](https://arxiv.org/abs/2211.13436)
2. [Principal Neighbourhood Aggregation](https://arxiv.org/abs/2004.05718)
3. [Attention-based GNN for Knapsack](https://github.com/rushhan/Attention-based-GNN-reinforcement-learning-for-Knapsack-Problem)

## 📄 Licença

Licença MIT – veja [LICENSE](LICENSE).

---

**Status do projeto:** ✅ Pronto para produção • 🔬 Nível de pesquisa • 📚 Documentação completa. Consulte [docs/index.md](docs/index.md) para detalhes.
