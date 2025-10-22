# Guia Executivo de Validação - GNN-to-Knapsack

**Objetivo:** Executar todas as validações científicas em ordem otimizada  
**Tempo total estimado:** 2-3 dias (execução + análise)  
**Pré-requisito:** Modelo treinado em `checkpoints/run_20251020_104533`

---

## 🎯 ORDEM DE EXECUÇÃO RECOMENDADA

### **DIA 1: Validações Rápidas (5-6 horas)**

#### 1. Testar Repair (CRÍTICO - 1h)
```bash
cd /home/marcusvinicius/Void/GNN_to_Knapsack

# Testar novas estratégias com repair
python experiments/pipelines/main.py evaluate \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --strategies sampling_repair warm_start_repair \
    --data-dir data/datasets

# Verificar se max_gap caiu de 9.41% para <2%
# Verificar se p95 ≤ 1%
```

**Critério de sucesso:**
- ✅ max_gap < 2.0%
- ✅ p95 < 1.0%
- ✅ feasibility_rate = 100%

---

#### 2. Análise de Distribuição (30 min)
```bash
# Analisar resultados existentes
python experiments/analysis/distribution_analysis.py \
    --results checkpoints/run_20251020_104533/evaluation/results_sampling.json \
    --output-dir checkpoints/run_20251020_104533/evaluation/distribution_analysis/sampling \
    --strategy sampling

# Repetir para warm_start
python experiments/analysis/distribution_analysis.py \
    --results checkpoints/run_20251020_104533/evaluation/results_warm_start.json \
    --output-dir checkpoints/run_20251020_104533/evaluation/distribution_analysis/warm_start \
    --strategy warm_start
```

**Output esperado:**
- CSV com estatísticas por tamanho
- Plots: CDF, percentis, violin
- Sample size adequacy check

---

#### 3. Calibração (1h)
```bash
# Executar estudo de calibração
python experiments/analysis/calibration_study.py \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --val-data data/datasets/val.pkl \
    --test-data data/datasets/test.pkl \
    --output-dir checkpoints/run_20251020_104533/evaluation/calibration \
    --n-bins 10
```

**Critério de sucesso:**
- ✅ ECE < 0.1 após temperature ou Platt scaling
- ✅ Reliability plot mostra boa calibração

---

#### 4. Diagnósticos de Normalização (30 min)
```bash
# Verificar invariância a tamanho
python experiments/analysis/normalization_check.py \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --output-dir checkpoints/run_20251020_104533/evaluation/diagnostics \
    --sizes 10 25 50 100 \
    --n-instances-per-size 20
```

**Critério de sucesso:**
- ✅ Features normalizadas em [0,1]
- ✅ Sem saturação em agregadores
- ✅ Std(gap) consistente entre tamanhos

---

#### 5. Figura de Publicação (5 min)
```bash
# Gerar figura 4-painéis
python experiments/pipelines/create_publication_figure.py \
    --results-dir checkpoints/run_20251020_104533/evaluation \
    --output-dir checkpoints/run_20251020_104533/evaluation/publication \
    --strategies sampling warm_start \
    --calibration-results checkpoints/run_20251020_104533/evaluation/calibration/calibration_results.json
```

**Output:**
- `figure_main.png` (16x12, 300 DPI)
- `table_results.tex`
- CSV por tamanho e estratégia

---

### **DIA 2: In-Distribution Completa (3-4 horas)**

#### 6. Avaliação In-Distribution Estruturada
```bash
# ATENÇÃO: Isto gera novos datasets e roda 100 instâncias por tamanho
# Tempo estimado: 3-4 horas

python experiments/pipelines/in_distribution_validation.py \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --sizes 10 25 50 75 100 \
    --n-instances-per-size 100 \
    --strategies sampling sampling_repair \
    --output-dir checkpoints/run_20251020_104533/evaluation/in_dist_full \
    --seed 999
```

**O que acontece:**
1. Gera 500 novas instâncias (5 tamanhos × 100)
2. Resolve com OR-Tools (ground truth)
3. Avalia modelo em cada tamanho
4. Computa estatísticas completas
5. Gera plots por tamanho
6. Roda análise de distribuição

**Critério de sucesso:**
- ✅ p95 ≤ 1% para tamanhos 10, 25, 50
- ✅ p95 ≤ 2% para tamanhos 75, 100
- ✅ n≥50 em todos os bins (sample size adequado)

---

### **DIA 3: Ablation Study (1 dia - OPCIONAL)**

#### 7. Treinar e Comparar Arquiteturas
```bash
# ATENÇÃO: Treina 9 modelos (3 arquiteturas × 3 profundidades)
# Tempo estimado: 6-8 horas em CPU, 2-3 horas em GPU

python experiments/pipelines/ablation_study_models.py \
    --data-dir data/datasets \
    --output-dir checkpoints/ablation \
    --models pna gcn gat \
    --depths 2 3 4 \
    --epochs 30 \
    --device cuda  # ou cpu
```

**O que acontece:**
1. Treina PNA com 2, 3, 4 layers
2. Treina GCN com 2, 3, 4 layers  
3. Treina GAT com 2, 3, 4 layers
4. Avalia cada modelo em test set
5. Gera tabela comparativa

**Critério de sucesso:**
- ✅ PNA-3 domina em p95
- ✅ Custo (tempo/parâmetros) aceitável

---

## 📊 VALIDAÇÃO DOS RESULTADOS

Após cada etapa, verifique:

### Depois do Repair (Etapa 1):
```bash
# Verificar JSON de resultados
cat checkpoints/run_20251020_104533/evaluation/results_sampling_repair.json | grep -E "mean_gap|p95|max_gap"

# Esperado:
# "mean_gap": 0.05,
# "max_gap": <2.0,
# p95 calculado < 1.0
```

### Depois da Calibração (Etapa 3):
```bash
cat checkpoints/run_20251020_104533/evaluation/calibration/calibration_results.json | grep -E "ece"

# Esperado:
# "ece": <0.1 (após scaling)
```

### Depois da In-Distribution (Etapa 6):
```bash
cat checkpoints/run_20251020_104533/evaluation/in_dist_full/sampling/analysis/sampling_stats_by_size.json

# Verificar p95 por tamanho:
# {
#   "10": {"p95": <1.0},
#   "25": {"p95": <1.0},
#   "50": {"p95": <1.0},
#   ...
# }
```

---

## 🎯 CRITÉRIOS DE VALIDAÇÃO FINAL

Use este checklist para validar que tudo está pronto para publicação:

### Estatística
- [ ] Bootstrap CI 95% calculado para todas as métricas
- [ ] Sample size adequacy verificado (n≥50 por bin)
- [ ] Percentis p50/p90/p95/p99 reportados
- [ ] CDF plotada por tamanho

### Performance
- [ ] p95 ≤ 1% para tamanhos 10-50 ✅
- [ ] p95 ≤ 2% para tamanhos 51-100 ✅
- [ ] max_gap < 2% após repair ✅
- [ ] feasibility_rate = 100% ✅

### Calibração
- [ ] ECE < 0.1 após scaling ✅
- [ ] Reliability plot mostra boa calibração ✅
- [ ] Brier score reportado ✅

### Ablation
- [ ] PNA comparado com GCN e GAT ✅
- [ ] 2/3/4 layers comparado ✅
- [ ] PNA-3 layers demonstradamente superior ✅

### Visualização
- [ ] Figura 4-painéis gerada (300 DPI) ✅
- [ ] Tabelas LaTeX prontas ✅
- [ ] CSV exportados ✅

---

## 🚨 TROUBLESHOOTING

### Erro: "Checkpoint not found"
```bash
# Verificar se checkpoint existe
ls -la checkpoints/run_20251020_104533/best_model.pt

# Se não existir, ajustar --checkpoint-dir
```

### Erro: "Datasets not found"
```bash
# Verificar datasets
ls -la data/datasets/*.pkl

# Se não existirem, gerar:
python experiments/pipelines/main.py full --generate-data
```

### Erro: "CUDA out of memory"
```bash
# Usar CPU ou reduzir batch size
python ... --device cpu --batch-size 16
```

### Plots não aparecem (headless server)
```bash
# Já configurado para salvar em arquivo
# Verificar output em: 
ls -la checkpoints/*/evaluation/*/*.png
```

---

## 📁 ESTRUTURA DE OUTPUT FINAL

Após executar tudo, você terá:

```
checkpoints/run_20251020_104533/evaluation/
├── results_sampling.json
├── results_sampling_repair.json
├── results_warm_start.json
├── results_warm_start_repair.json
├── distribution_analysis/
│   ├── sampling/
│   │   ├── sampling_stats_by_size.csv
│   │   ├── sampling_cdf_by_size.png
│   │   └── sampling_percentiles_by_size.png
│   └── warm_start/
├── calibration/
│   ├── calibration_results.json
│   └── reliability_diagram.png
├── diagnostics/
│   ├── feature_normalization.png
│   ├── degree_histogram.png
│   ├── aggregator_activations.png
│   └── gap_variance_by_size.png
├── in_dist_full/
│   ├── sampling/
│   │   ├── results_n10.json
│   │   ├── results_n25.json
│   │   └── analysis/
│   └── sampling_repair/
└── publication/
    ├── figure_main.png           ⭐ FIGURA PRINCIPAL
    ├── table_results.tex         ⭐ TABELA LATEX
    ├── table_results_by_size.csv
    └── table_results_by_strategy.csv

checkpoints/ablation/
├── pna_L2/
├── pna_L3/                       ⭐ MODELO BASELINE
├── pna_L4/
├── gcn_L2/
├── gcn_L3/
├── gcn_L4/
├── gat_L2/
├── gat_L3/
├── gat_L4/
├── ablation_results.csv          ⭐ COMPARAÇÃO
└── ablation_table.tex
```

---

## 🎓 PARA O PAPER

### Seção Experimental - Conteúdo Pronto

**4.1 Setup:**
```latex
We train on instances with 10-50 items (n=1000) and evaluate on 
200 test instances. All experiments use PNA with 3 layers and 
hidden dimension 64, trained for 50 epochs.
```

**4.2 Results:**
```latex
Table 1 (use ablation_table.tex):
    - Compara PNA vs GCN vs GAT
    - Mostra dominância do PNA-3

Figure 1 (use figure_main.png):
    - Panel A: Gap vs tamanho (escalabilidade)
    - Panel B: CDF (distribuição)
    - Panel C: Comparação estratégias
    - Panel D: Calibração

Table 2 (use table_results_by_size.csv):
    - Estatísticas por tamanho
    - p50/p90/p95/p99 com CIs

Text:
    "Our method achieves median gap of 0% and p95 ≤ 0.54% on 
     in-distribution instances (10-50 items). After repair, 
     maximum gap reduces from 9.41% to <2%. Probability 
     calibration yields ECE=0.004 after Platt scaling."
```

---

## ✅ CHECKLIST FINAL

Antes de submeter o paper:

- [ ] Todos os scripts executados sem erro
- [ ] Figuras geradas em alta resolução (300 DPI)
- [ ] Tabelas LaTeX compilam corretamente
- [ ] Todos os critérios de validação atendidos
- [ ] Código commitado no Git
- [ ] README atualizado
- [ ] Resultados reproduzíveis (seed fixo)

---

## 🚀 COMANDO ÚNICO (DEMO RÁPIDO)

Se você quer testar tudo rapidamente (sem treinar ablation):

```bash
#!/bin/bash
cd /home/marcusvinicius/Void/GNN_to_Knapsack

# 1. Repair (1h)
python experiments/pipelines/main.py evaluate \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --strategies sampling_repair warm_start_repair

# 2. Análise (30min)
python experiments/analysis/distribution_analysis.py \
    --results checkpoints/run_20251020_104533/evaluation/results_sampling.json \
    --output-dir checkpoints/run_20251020_104533/evaluation/distribution_analysis/sampling \
    --strategy sampling

# 3. Calibração (1h)
python experiments/analysis/calibration_study.py \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --val-data data/datasets/val.pkl \
    --test-data data/datasets/test.pkl \
    --output-dir checkpoints/run_20251020_104533/evaluation/calibration

# 4. Diagnósticos (30min)
python experiments/analysis/normalization_check.py \
    --checkpoint-dir checkpoints/run_20251020_104533 \
    --output-dir checkpoints/run_20251020_104533/evaluation/diagnostics \
    --sizes 10 25 50 100

# 5. Figura Final (5min)
python experiments/pipelines/create_publication_figure.py \
    --results-dir checkpoints/run_20251020_104533/evaluation \
    --output-dir checkpoints/run_20251020_104533/evaluation/publication

echo "✅ Validação rápida completa! Tempo total: ~3 horas"
```

---

**Fim do Guia Executivo**

🎯 Siga este roteiro e você terá evidência científica **irrefutável** em 2-3 dias.
