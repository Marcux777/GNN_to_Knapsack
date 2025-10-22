# CI Workflow Fixes - GitHub Actions

## 🔧 Problemas Identificados

O workflow CI do GitHub Actions estava falhando porque referenciava **diretórios antigos** que não existem mais após a refatoração do projeto.

### Estrutura Antiga (antes da refatoração):
```
.
├── utils/
├── models/
├── training/
├── data/
└── inference/
```

### Estrutura Atual (pós-refatoração):
```
.
├── src/knapsack_gnn/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   ├── decoding/
│   ├── solvers/
│   ├── baselines/
│   ├── analysis/
│   └── eval/
├── experiments/
└── tests/
```

## ✅ Correções Aplicadas

### 1. **Lint Job** (ruff)
**Antes:**
```yaml
- name: Run ruff
  run: ruff check .
```

**Depois:**
```yaml
- name: Run ruff
  run: ruff check src/ experiments/ tests/
```

**Motivo:** Escaneia apenas os diretórios relevantes, evitando problemas com arquivos temporários ou cache.

---

### 2. **Type Check Job** (mypy)
**Antes:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install mypy
    pip install -r requirements.txt

- name: Run mypy
  run: mypy utils/ models/ training/ --ignore-missing-imports
```

**Depois:**
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install mypy
    pip install -e .

- name: Run mypy
  run: mypy src/knapsack_gnn/ --ignore-missing-imports
```

**Motivos:**
- ✅ Usa `pip install -e .` ao invés de `requirements.txt` (instala o pacote em modo editável)
- ✅ Verifica o caminho correto: `src/knapsack_gnn/`
- ✅ Remove referências a `utils/`, `models/`, `training/` (não existem mais na raiz)

---

### 3. **Test Job** (pytest)
**Antes:**
```yaml
- name: Run tests with coverage
  env:
    PYTHONHASHSEED: 42
  run: |
    pytest tests/ -v --cov=. --cov-report=xml --cov-report=term-missing
```

**Depois:**
```yaml
- name: Run tests with coverage
  env:
    PYTHONHASHSEED: 42
  run: |
    pytest tests/ -v --cov=src/knapsack_gnn --cov-report=xml --cov-report=term-missing
```

**Motivo:** Mede cobertura apenas do código fonte em `src/knapsack_gnn/`, não de todo o repositório (evita incluir scripts, configs, etc).

---

## 📊 Resumo das Mudanças

| Job | Comando Original | Comando Corrigido | Status |
|-----|------------------|-------------------|--------|
| **Lint** | `ruff check .` | `ruff check src/ experiments/ tests/` | ✅ Corrigido |
| **Type Check** | `mypy utils/ models/ training/` | `mypy src/knapsack_gnn/` | ✅ Corrigido |
| **Test Coverage** | `--cov=.` | `--cov=src/knapsack_gnn` | ✅ Corrigido |
| **Dependencies** | `pip install -r requirements.txt` | `pip install -e .` | ✅ Melhorado |

---

## 🧪 Validação Local

Para testar se as mudanças funcionam localmente (requer ferramentas instaladas):

```bash
# 1. Instalar dependências de desenvolvimento
pip install -e .[dev]

# 2. Testar linting
ruff check src/ experiments/ tests/

# 3. Testar type checking
mypy src/knapsack_gnn/ --ignore-missing-imports

# 4. Rodar testes com cobertura
pytest tests/ -v --cov=src/knapsack_gnn --cov-report=term-missing
```

---

## 🚀 Próximos Passos

1. **Commit as mudanças:**
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "fix(ci): Update workflow paths after project refactoring"
   ```

2. **Push para o GitHub:**
   ```bash
   git push origin main
   ```

3. **Verificar GitHub Actions:**
   - Acesse: https://github.com/Marcux777/GNN_to_Knapsack/actions
   - O workflow deve executar com sucesso ✅

---

## 📝 Notas Adicionais

- ✅ Sintaxe YAML validada localmente
- ✅ Compatível com Python 3.10 e 3.11
- ✅ Mantém integração com Codecov
- ✅ Configuração de seeds (PYTHONHASHSEED=42) preservada para reprodutibilidade

---

**Data:** 2025-10-21  
**Responsável:** Claude AI Assistant  
**Motivo:** Atualização pós-refatoração (commit 5bb113b)
