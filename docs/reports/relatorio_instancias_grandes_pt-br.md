# Relatório: Desempenho em Instâncias Grandes

## Sumário Executivo

**Pergunta:** Para instâncias grandes, o modelo está indo bem?

**Resposta curta:** **DEPENDE DO TAMANHO**
- ✅ **n=500 (2.5x extrapolação)**: Maioria excelente, alguns outliers
- ❌ **n=2000 (10x extrapolação)**: Desempenho ruim (~21% gap)

---

## Resultados Detalhados

### Configuração do Experimento

- **Treinamento**: n=50-200 itens
- **Teste**: n=200, n=500, n=2000
- **Estratégia**: Sampling com temperatura 0.8
- **Amostras**: Até 500 por instância

### Performance por Tamanho

| Tamanho | N inst | Extrapolação | Gap Médio | Gap Mediano | Gap Máx | Viabilidade |
|---------|--------|--------------|-----------|-------------|---------|-------------|
| n=200   | 50     | 1.0x (treino)| **0.53%** | **0.08%**   | 21.25%  | 100%        |
| n=500   | 30     | 2.5x         | **5.70%** | **0.08%**   | 25.62%  | 100%        |
| n=2000  | 1      | 10.0x        | **20.99%**| **20.99%**  | 20.99%  | 100%        |

---

## Análise Crítica

### 🎯 n=200 (Dentro da Distribuição)
**Status: ✅ EXCELENTE**

- Gap mediano: 0.08% (praticamente ótimo!)
- 98% das instâncias com gap < 1%
- ⚠️ Alta variância: 1 outlier extremo puxando média para 0.53%

### 🎯 n=500 (Extrapolação Moderada - 2.5x)
**Status: ⚠️ MISTO**

**Pontos Positivos:**
- Gap mediano: 0.08% (excelente!)
- 73.3% das instâncias com gap < 1%
- Todos os casos são viáveis (100%)
- Speedup de ~46x vs OR-Tools

**Pontos Negativos:**
- Gap médio: 5.70% (puxado por outliers)
- Alta variância (std = 9.46%)
- 26.7% das instâncias com gap >= 5%
- Pior caso: 25.62% de gap

**Interpretação:**
- O modelo **generaliza bem** para a maioria dos casos
- Mas existe um **subgrupo de instâncias difíceis** (~27%) onde o desempenho degrada significativamente
- Degradação estatisticamente significativa vs n=200 (p < 0.001, Cohen's d = 0.83)

### 🎯 n=2000 (Extrapolação Extrema - 10x)
**Status: ❌ RUIM**

**Problemas:**
- Gap: 20.99% (muito alto!)
- Apenas 79% do valor ótimo alcançado
- ⚠️ **Amostra insuficiente**: Apenas 1 instância testada!

**Observação:** Não podemos tirar conclusões estatísticas robustas com N=1. Precisamos de mais dados.

---

## Teste Estatístico

### n=200 vs n=500

```
Mean gap difference: +5.17%
t-statistic: -3.539
p-value: 0.0007 (altamente significativo)
Cohen's d: 0.830 (efeito grande)
```

**Conclusão:** Há degradação estatisticamente significativa de n=200 para n=500, mas o efeito é **heterogêneo** (alguns casos excelentes, outros ruins).

---

## Diagnóstico do Problema

### Por que o desempenho degrada?

#### 1. **Outliers em n=500**

Analisando os casos ruins:
- 8 instâncias (26.7%) têm gap > 5%
- Dessas, 6 têm gap > 15%
- Hipóteses:
  - Estrutura de correlação peso-valor diferente
  - Casos onde sampling precisa de mais amostras
  - Possível overfitting em padrões de instâncias pequenas

#### 2. **Colapso em n=2000**

Gap de 21% sugere que o modelo:
- Não captura a estrutura global do problema em larga escala
- Precisa de muito mais amostras (testamos com max=500)
- Pode ter problemas de propagação de informação na GNN (3 camadas podem ser insuficientes)

---

## Visualização

Visualização detalhada salva em:
```
checkpoints/run_20251020_104533/evaluation/large_scale_analysis/large_scale_performance.png
```

Mostra:
1. Comparação média vs mediana por tamanho
2. Tendência de generalização com barras de erro
3. Distribuição de gaps com outliers destacados
4. Performance vs fator de extrapolação

---

## Recomendações

### 📊 Para n=500 (Melhorar casos ruins)

1. **Aumentar budget de sampling**
   - Testar com 1000-2000 amostras
   - Adaptar sampling schedule: (64, 128, 256, 512)

2. **Estratégia de repair**
   - Usar ILP warm-start nos casos difíceis
   - Detectar quando sampling está com dificuldade

3. **Análise de outliers**
   - Identificar padrões nas 8 instâncias ruins
   - Verificar se há correlação peso-valor específica

### 🔬 Para n=2000 (Problema mais grave)

1. **Coleta de mais dados**
   - Testar 20-30 instâncias para ter estatísticas confiáveis
   - Usar solver com timeout para OR-Tools não travar

2. **Arquitetura**
   - Testar com 5-7 camadas GNN (mais alcance)
   - Aumentar hidden_dim (128 ou 256)
   - Considerar Graph Transformer

3. **Training augmentation**
   - Retreinar incluindo instâncias de n=300-500 no training set
   - Curriculum learning: treinar progressivamente em tamanhos maiores

4. **Sampling adaptativo**
   - Budget proporcional ao tamanho: n=2000 → 5000-10000 amostras
   - Early stopping mais sofisticado

---

## Conclusão Final

### Resumo por Caso de Uso

| Tamanho | Recomendação | Justificativa |
|---------|--------------|---------------|
| n ≤ 200 | ✅ **Usar GNN** | Gap < 1% na maioria, 46x mais rápido |
| n = 500 | ⚠️ **Usar com cautela** | 73% excelente, mas 27% problemático. Considerar ensemble com solver exato |
| n ≥ 1000 | ❌ **Não usar ainda** | Degradação severa. Precisa de melhorias arquiteturais |

### Próximos Passos Sugeridos

1. **Curto prazo**: Implementar estratégia híbrida para n=500
   - GNN primeiro, se gap estimado > threshold → warm-start ILP
   
2. **Médio prazo**: Retreinar com instâncias maiores
   - Expandir training set para n=50-500
   - 50 épocas adicionais
   
3. **Longo prazo**: Arquitetura melhorada
   - Graph Transformer ou GNN mais profunda
   - Attention mechanisms para capturar dependências de longo alcance

---

## Arquivos Gerados

```
checkpoints/run_20251020_104533/evaluation/
├── large_instances/
│   ├── dataset_n500.pkl         # Dataset de teste (30 instâncias)
│   └── results_n500.json        # Resultados detalhados
└── large_scale_analysis/
    ├── analysis_summary.json     # Sumário estatístico
    └── large_scale_performance.png  # Visualizações
```

---

**Data da Análise:** 2025-10-21  
**Modelo:** checkpoints/run_20251020_104533 (47 épocas)  
**Autor:** Análise Automática via Claude Code
