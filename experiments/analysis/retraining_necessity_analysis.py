"""
Analysis: Do we need to retrain for large instances?
Compares retraining vs other strategies
"""

import json
from pathlib import Path


def analyze_retraining_necessity():
    """Analyze if retraining is necessary or if other strategies suffice"""

    print("=" * 90)
    print("ANÁLISE: NECESSIDADE DE RETREINAMENTO PARA INSTÂNCIAS GRANDES")
    print("=" * 90)

    # Load current results
    checkpoint_dir = Path("checkpoints/run_20251020_104533")

    results_n500_path = checkpoint_dir / "evaluation" / "large_instances" / "results_n500.json"
    with open(results_n500_path) as f:
        results_n500 = json.load(f)

    print("\n" + "=" * 90)
    print("1. DIAGNÓSTICO DO PROBLEMA")
    print("=" * 90)

    # Analyze n=500 performance
    gaps_n500 = results_n500["gaps"]
    median_gap = results_n500["median_gap"]
    mean_gap = results_n500["mean_gap"]

    # Categorize instances
    excellent = sum(1 for g in gaps_n500 if g < 1.0)
    good = sum(1 for g in gaps_n500 if 1.0 <= g < 5.0)
    bad = sum(1 for g in gaps_n500 if g >= 5.0)

    print(f"\nResultados n=500 (30 instâncias):")
    print(f"  • Excelente (gap < 1%):  {excellent} ({excellent / 30 * 100:.1f}%)")
    print(f"  • Bom (1% ≤ gap < 5%):   {good} ({good / 30 * 100:.1f}%)")
    print(f"  • Ruim (gap ≥ 5%):       {bad} ({bad / 30 * 100:.1f}%)")
    print(f"\n  → Gap mediano: {median_gap:.2f}% (excelente!)")
    print(f"  → Gap médio:   {mean_gap:.2f}% (puxado por {bad} outliers)")

    print("\n" + "-" * 90)
    print("INTERPRETAÇÃO:")
    print("-" * 90)

    if median_gap < 1.0:
        print("✓ O modelo JÁ GENERALIZA BEM para a maioria das instâncias n=500")
        print("✓ A mediana baixa indica que o modelo aprendeu padrões transferíveis")
        print("✗ Problema: SUBGRUPO de instâncias difíceis (~27%)")
        print("\n→ Isso sugere que o problema NÃO É falta de capacidade geral,")
        print("  mas sim CASOS ESPECÍFICOS que o modelo não cobre.")

    print("\n" + "=" * 90)
    print("2. CUSTO-BENEFÍCIO: RETREINAMENTO vs OUTRAS ESTRATÉGIAS")
    print("=" * 90)

    strategies = [
        {
            "name": "1. AUMENTAR SAMPLING BUDGET",
            "cost": "Baixo (apenas inferência)",
            "implementation": "Imediato (mudar parâmetro)",
            "expected_gain": "Moderado (5-10% de melhoria)",
            "pros": [
                "Sem custo de treinamento",
                "Funciona com modelo atual",
                "Fácil de testar incrementalmente",
            ],
            "cons": ["Aumenta tempo de inferência", "Pode não resolver outliers extremos"],
            "recommendation": "✅ TESTAR PRIMEIRO",
        },
        {
            "name": "2. ESTRATÉGIA HÍBRIDA (GNN + ILP)",
            "cost": "Baixo (implementação)",
            "implementation": "1-2 dias",
            "expected_gain": "Alto (resolver 90%+ dos casos ruins)",
            "pros": [
                "Sem retreinamento necessário",
                "Usa melhor dos dois mundos",
                "GNN rápido para casos fáceis, ILP para difíceis",
            ],
            "cons": ["Complexidade adicional no código", "Precisa detectar quando usar ILP"],
            "recommendation": "✅ ALTA PRIORIDADE",
        },
        {
            "name": "3. RETREINAR COM INSTÂNCIAS MAIORES",
            "cost": "Alto (dados + computação + tempo)",
            "implementation": "1-2 semanas",
            "expected_gain": "Incerto (pode não resolver outliers)",
            "pros": ["Pode melhorar generalização geral", "Útil para n>500 também"],
            "cons": [
                "Requer gerar milhares de instâncias grandes",
                "OR-Tools lento para n>500 (horas de CPU)",
                "Treino mais lento (grafos maiores)",
                "Não garante resolver casos específicos ruins",
                "Pode degradar performance em n<200",
            ],
            "recommendation": "⚠️ CONSIDERAR APÓS TESTAR 1 e 2",
        },
        {
            "name": "4. ARQUITETURA MELHORADA (mais camadas/attention)",
            "cost": "Muito Alto (novo modelo + treino)",
            "implementation": "2-4 semanas",
            "expected_gain": "Alto para n>1000, baixo para n=500",
            "pros": ["Melhor para n>>500 (n=2000+)", "Captura dependências de longo alcance"],
            "cons": [
                "Retreinamento completo necessário",
                "Mais lento na inferência",
                "Overkill para n=500",
            ],
            "recommendation": "⚠️ APENAS SE FOCO FOR n≥1000",
        },
    ]

    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'-' * 90}")
        print(f"{strategy['name']}")
        print(f"{'-' * 90}")
        print(f"Custo:            {strategy['cost']}")
        print(f"Implementação:    {strategy['implementation']}")
        print(f"Ganho esperado:   {strategy['expected_gain']}")
        print(f"\nPrós:")
        for pro in strategy["pros"]:
            print(f"  ✓ {pro}")
        print(f"\nContras:")
        for con in strategy["cons"]:
            print(f"  ✗ {con}")
        print(f"\nRecomendação: {strategy['recommendation']}")

    print("\n" + "=" * 90)
    print("3. ANÁLISE ESPECÍFICA: QUANDO RETREINAR FAZ SENTIDO?")
    print("=" * 90)

    print("\nRetreinar FAZ SENTIDO quando:")
    print("  ✓ Gap mediano também é alto (modelo não captura o problema)")
    print("  ✓ Degradação uniforme em todas as instâncias")
    print("  ✓ Problema é falta de capacidade arquitetural")
    print("  ✓ Foco principal é n>1000")

    print("\nRetreinar NÃO FAZ SENTIDO quando:")
    print("  ✗ Gap mediano é baixo, mas média alta (nosso caso!)")
    print("  ✗ Maioria das instâncias vai bem")
    print("  ✗ Problema é subgrupo específico de casos")
    print("  ✗ Outras estratégias mais baratas não foram testadas")

    print("\n" + "=" * 90)
    print("4. RECOMENDAÇÃO FINAL")
    print("=" * 90)

    print("\n🎯 PARA n=500:")
    print("  → NÃO RETREINAR AGORA")
    print("  → Razão: 73% já excelente, retreino pode não resolver outliers específicos")
    print("  → Estratégia:")
    print("     1. Testar sampling com 1000-2000 amostras (custo zero)")
    print("     2. Implementar híbrido GNN→ILP (1-2 dias)")
    print("     3. Avaliar novamente: se ainda ruim, aí considerar retreino")

    print("\n🎯 PARA n=2000:")
    print("  → RETREINAR PROVAVELMENTE NECESSÁRIO")
    print("  → Razão: Extrapolação 10x é muito extrema")
    print("  → Mas ANTES:")
    print("     1. Coletar mais dados (N=1 insuficiente)")
    print("     2. Testar com muito mais samples (5000-10000)")
    print("     3. Se continuar ruim, retreinar com n=300-1000 no training set")

    print("\n" + "=" * 90)
    print("5. PLANO DE AÇÃO SUGERIDO")
    print("=" * 90)

    action_plan = [
        {
            "step": "FASE 1: Otimizar Inferência (0 custo de treino)",
            "tasks": [
                "Testar n=500 com 1000 samples (vs 500 atual)",
                "Testar n=500 com 2000 samples",
                "Medir melhoria nos casos ruins (gap>5%)",
                "Custo: ~30min de experimentos",
            ],
            "go_criteria": "Se gap médio cair para <3%, PARAR AQUI",
        },
        {
            "step": "FASE 2: Estratégia Híbrida (baixo custo)",
            "tasks": [
                "Implementar detector de 'caso difícil'",
                "Se gap estimado >5%, usar ILP warm-start",
                "Testar em n=500",
                "Custo: ~2 dias implementação",
            ],
            "go_criteria": "Se 95%+ casos ficarem gap<2%, PARAR AQUI",
        },
        {
            "step": "FASE 3: Coleta de Dados n=1000-2000 (médio custo)",
            "tasks": [
                "Gerar 20-30 instâncias n=1000",
                "Gerar 20-30 instâncias n=2000",
                "Usar OR-Tools com timeout 300s",
                "Testar estratégias 1+2 nessas instâncias",
                "Custo: ~1 dia",
            ],
            "go_criteria": "Se gap médio n=1000 <5% e n=2000 <10%, PARAR AQUI",
        },
        {
            "step": "FASE 4: Retreinamento (SÓ SE NECESSÁRIO)",
            "tasks": [
                "Gerar 2000-5000 instâncias n=50-500",
                "Resolver com OR-Tools (paralelo, ~2-3 dias CPU)",
                "Retreinar 50 épocas (~1 dia GPU)",
                "Avaliar em todas as escalas",
                "Custo: ~1-2 semanas",
            ],
            "go_criteria": "ÚLTIMA OPÇÃO - só se Fases 1-3 falharem",
        },
    ]

    for i, phase in enumerate(action_plan, 1):
        print(f"\n{'=' * 90}")
        print(f"{phase['step']}")
        print(f"{'=' * 90}")
        for task in phase["tasks"]:
            print(f"  • {task}")
        print(f"\n  Critério de parada: {phase['go_criteria']}")

    print("\n" + "=" * 90)
    print("CONCLUSÃO")
    print("=" * 90)

    print("\n📊 RESPOSTA CURTA:")
    print("  → Para n=500: NÃO precisa retreinar AGORA")
    print("  → Para n=2000: PROVAVELMENTE precisa, mas TESTAR OUTRAS OPÇÕES PRIMEIRO")

    print("\n🎯 RAZÃO:")
    print("  O modelo já generaliza bem (mediana 0.08% em n=500)")
    print("  Problema são casos específicos, não capacidade geral")
    print("  Retreino é caro e pode não resolver outliers")
    print("  Estratégias mais baratas devem ser testadas primeiro")

    print("\n⚡ PRÓXIMO PASSO IMEDIATO:")
    print("  Executar FASE 1: testar n=500 com mais samples")
    print("  Tempo: 30 minutos")
    print("  Se funcionar, economiza 1-2 semanas de trabalho")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    analyze_retraining_necessity()
