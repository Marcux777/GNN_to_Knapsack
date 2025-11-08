Código:
- Siga PEP8; Ruff é obrigatório (ver `pyproject.toml`).
- Docstrings no estilo Google, principalmente em APIs públicas de `src/knapsack_gnn` e CLI.
- Funções curtas, com responsabilidade única. Prefira helpers nomeados a blocos longos.
- Use type hints; `mypy` é tolerante, mas novos trechos devem ser anotados.
- Evite efeitos colaterais em import; inicializações pesadas ficam em `if __name__ == "__main__"` ou funções específicas.

Testes:
- PyTest com dados determinísticos (`np.random.seed`, `torch.manual_seed`).
- `test_<area>_<comportamento>_<condição>` como convenção de nomes.
- Adicione testes unitários para lógica local; use testes de integração apenas quando necessário (fluxos de treino/decoding).

Commits:
- Formato `type(scope): summary` com types {feat, fix, refactor, docs, test, chore}.
  - Ex.: `fix(decoding): guard repair when no feasible items`.
- Commits atômicos descrevendo *por que* mudou; mantenha mensagem curta (
<= 72 col).

PRs:
- Pequenos e focados; linke issues/tarefas quando existirem.
- Siga o checklist de `.codex/eval.md` antes de abrir PR.
- Documente impacto em performance, memória e compatibilidade com checkpoints.
