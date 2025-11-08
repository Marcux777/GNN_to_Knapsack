Deadlocks/Threads:
- Estratégias de decoding podem travar se ILP/Lagrangiano passar do tempo. Sempre respeite `ilp_time_limit` e trate timeouts.
- Evite processos longos em testes; use datasets pequenos e seeds fixos.

I/O e Artefatos Pesados:
- Nunca versione checkpoints (`checkpoints/`) ou datasets (`data/datasets/`). Use README/resultados oficiais.
- `results/` só deve conter artefatos pequenos e documentados.

Concorrência/Performance:
- PNA com `hidden_dim` alto pode estourar memória; documente alterações que aumentem parâmetros ou tempo.
- Samplers com `n_samples` grande podem ficar lentos; mantenha limites na CLI.

Reprodutibilidade:
- Sempre definir `--seed` e documentar datasets gerados. Mudanças em geradores devem explicar impacto em métricas publicadas.

Dependências:
- PyTorch/PyG são pesados; evite atualizar sem necessidade e descreva impacto.
- Novas libs precisam de revisão (licença, compatibilidade com CUDA).

Erros comuns:
- `KnapsackSampler` degradar solução após reparo; rode scripts de debug quando tocar em `src/knapsack_gnn/decoding/`.
- Links quebrados em docs quando mover resultados (atualize `docs/reports/experimental_results.md`).
