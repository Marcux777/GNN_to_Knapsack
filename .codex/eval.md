Checklist de revisão:
[ ] Reproduzi o bug/cenário com comando exato
[ ] Teste novo falha sem o fix (quando aplicável)
[ ] Fix mínimo e bem localizado
[ ] `make lint` limpo
[ ] `make mypy` limpo (ou justificativa clara)
[ ] `make test-quick`/`make test` passando em tempo razoável
[ ] Impacto em performance/segurança/documentado
[ ] README/docs/resultados atualizados quando necessário
[ ] Plano de rollback simples (git revert / flag)
