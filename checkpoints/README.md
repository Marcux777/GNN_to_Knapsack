# Checkpoints

Esta pasta guarda checkpoints locais de treino. Por padrão, somente os artefatos de
ablação (`checkpoints/ablations/**`) são versionados. Todos os demais subdiretórios
são ignorados via `.gitignore`, então cada desenvolvedor pode salvar execuções locais
sem risco de poluir os commits.

Regras rápidas:
- Use `make train`/`python train.py` para gerar runs em `checkpoints/run_<timestamp>`.
- Remova checkpoints antigos ou compacte-os antes de compartilhar.
- Para compartilhar pesos relevantes, mova-os para `checkpoints/ablations/` ou publique
  em um release/artefato externo.
