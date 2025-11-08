# Fluxo diário com `.codex/tasks.md`

Abra **toda** demanda (issue, comentário, PR) colando um dos blocos abaixo já preenchidos. Seja direto e inclua comandos reais.

## BUGFIX (ex.: DataLoader travando com batch grande)
```
Tarefa: BUGFIX
Contexto:
- Sintoma: DataLoader trava em batches >128 em GPU 0 (PNA no dataset X).
- Reprodutibilidade: CUDA_VISIBLE_DEVICES=0 python -m knapsack_gnn.train --batch_size 256 --epochs 1
- Esperado vs Obtido: treinamento deveria iniciar; fica pendurado após 1º epoch.

Requisitos:
- Adicionar teste que falha sem o fix (timeout curto em smoke de DataLoader).
- Não quebrar APIs públicas de train().
- Tempo de execução do teste <= 5s.

Checklist:
[ ] teste falha antes do fix
[ ] fix mínimo
[ ] ruff/mypy ok
[ ] explicação do root cause
```

## FEATURE (ex.: early stopping por validação estratificada)
```
Tarefa: FEATURE
Problema do usuário: reduzir overfitting nos cenários OOD com instâncias 200-itens.
Critérios de aceitação (G/W/T):
- Given split estratificado por densidade, When treinar PNA, Then parar em paciência=10 sem piorar gap.
Impacto: treinos mais curtos; sem regressão de qualidade.
Plano de teste: unit (callback), integração (loop treino com dummy dataset), e2e (script de treino rápido).
Docs/migração: README + exemplo em configs/exp_earlystop.yaml
```

## REFACTOR (ex.: unificar loop de treino/val)
```
Tarefa: REFACTOR
Cheiro: duplicação em train.py e validate.py, lógica de hooks repetida.
Hipótese: extrair Engine com callbacks (on_batch_end, on_eval_epoch_end).
Riscos: quebra de logs/seed; medir mesmas métricas.
Plano: passos pequenos com cobertura de testes.
Critérios de não-regressão: métricas e logs idênticos em seed fixa.
```

## EXPERIMENTO (ex.: PNA vs GAT com augment de instâncias 150–220)
```
Tarefa: EXPERIMENTO
Hipótese: PNA+augment melhora gap OOD vs GAT puro.
Métrica: optimality gap @ 200-itens; t-test p<0.05; efeito ≥ 0.3.
Setup: seeds=[13,37,73], 3 runs/cfg, dataset_ood_200.pkl
Plano: rodar run_exp.sh; coletar CSV; script de stats.
Reversão: descartar checkpoints/artefatos; não mergear código.
```
