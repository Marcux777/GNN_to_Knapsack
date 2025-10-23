# Development Guide

This guide covers how to set up your development environment and contribute to the `knapsack-gnn` project.

## Development Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone https://github.com/Marcux777/GNN_to_Knapsack.git
cd GNN_to_Knapsack

# Install development dependencies
pip install -e .[dev]
```

### 2. Install Pre-commit Hooks

Pre-commit hooks automatically format and lint your code before each commit:

```bash
# Install pre-commit hooks
pre-commit install

# Install pre-push hooks (for type checking and tests)
pre-commit install --hook-type pre-push

# Run hooks manually on all files
pre-commit run --all-files
```

### 3. Verify Setup

```bash
# Run tests
make test

# Check code formatting
make format

# Run linter
make lint

# Type check
make mypy

# Build documentation
make docs
```

## Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write tests for new functionality
   - Update documentation as needed
   - Follow the code style (enforced by pre-commit hooks)

3. **Run checks locally**
   ```bash
   make test        # Run tests with coverage
   make lint        # Check code quality
   make mypy        # Type check
   make docs        # Build docs locally
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   ```
   Pre-commit hooks will automatically format and lint your code.

5. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Pre-push hooks will run type checking and quick tests.

## Code Style

### Formatting and Linting

We use **Ruff** for both formatting and linting (replacing black, isort, and flake8):

```bash
# Format code
make format

# Lint code
make lint

# Or use ruff directly
ruff format src/ experiments/ tests/
ruff check src/ experiments/ tests/
```

### Type Hints

We use **mypy** for static type checking:

```bash
# Run type checker
make mypy

# Or use mypy directly
mypy src/knapsack_gnn/ experiments/ --ignore-missing-imports
```

Type hints are encouraged but not strictly enforced. Key guidelines:
- Add type hints to public APIs
- Use `typing` module for complex types
- Ignore missing imports with `# type: ignore`

### Docstrings

We use **Google-style docstrings**:

```python
def function_name(param1: int, param2: str) -> bool:
    """Brief description of function.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something goes wrong.

    Example:
        ```python
        result = function_name(42, "test")
        ```
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests with coverage
make test

# Run quick tests (exclude slow tests)
make test-quick

# Run specific test file
pytest tests/unit/test_models.py -v

# Run tests with specific marker
pytest -m "not slow" -v
```

### Writing Tests

- Place tests in `tests/` directory
- Use `pytest` framework
- Name test files `test_*.py`
- Mark slow tests with `@pytest.mark.slow`

Example test:

```python
import pytest
from knapsack_gnn.models import PNAModel

def test_pna_model_forward():
    """Test PNA model forward pass."""
    model = PNAModel(hidden_dim=64, num_layers=3)
    # ... test code ...

@pytest.mark.slow
def test_full_training_pipeline():
    """Test complete training pipeline (slow)."""
    # ... test code ...
```

### Coverage

We maintain a minimum coverage of **70%**:

```bash
# Run tests with coverage report
make test

# View detailed HTML coverage report
open htmlcov/index.html
```

## Dependency Management

### Adding Dependencies

1. **Add to pyproject.toml**
   ```toml
   [project]
   dependencies = [
       "new-package>=1.0.0",
   ]
   ```

2. **Sync dependencies**
   ```bash
   make sync-deps
   ```
   This regenerates `requirements.txt` and `requirements-dev.txt` with hashes.

3. **Commit changes**
   ```bash
   git add pyproject.toml requirements*.txt
   git commit -m "deps: add new-package"
   ```

### Checking Dependency Drift

```bash
# Check if requirements files are in sync with pyproject.toml
make check-deps
```

This is automatically checked in pre-push hooks and CI.

## Documentation

### Building Documentation

```bash
# Build documentation
make docs

# Serve documentation locally
make docs-serve
# Visit http://127.0.0.1:8000
```

### Adding Documentation

- **User guides**: Add to `docs/guides/`
- **API documentation**: Automatically generated from docstrings
- **Reports**: Add to `docs/reports/`

Documentation uses **MkDocs** with **Material theme** and **mkdocstrings** for API docs.

## Makefile Commands

Common development commands:

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make sync-deps` | Regenerate requirements from pyproject.toml |
| `make check-deps` | Verify dependency sync |
| `make format` | Format code with ruff |
| `make lint` | Lint code with ruff |
| `make mypy` | Type check with mypy |
| `make test` | Run tests with coverage |
| `make test-quick` | Run quick tests only |
| `make docs` | Build documentation |
| `make docs-serve` | Serve docs locally |
| `make clean` | Clean build artifacts |

## CI/CD

### GitHub Actions

The CI pipeline runs on every push and PR:

1. **Format Check** - Verifies code is formatted
2. **Lint** - Checks code quality with ruff
3. **Type Check** - Runs mypy type checking
4. **Tests** - Runs test suite with coverage (70% minimum)
5. **Dependency Check** - Verifies requirements are in sync
6. **Docs Build** - Builds documentation with strict mode

See `.github/workflows/ci.yml` for details.

### Pre-commit Hooks

**Pre-commit stage** (fast, <2s):
- Auto-format code (ruff format)
- Auto-fix linting issues (ruff --fix)
- Fix file endings and trailing whitespace
- Check YAML/TOML syntax

**Pre-push stage** (heavier checks):
- Type check with mypy
- Run quick tests
- Check dependency sync

Skip hooks temporarily:
```bash
SKIP=mypy,tests git commit
SKIP=mypy git push
```

## Troubleshooting

### Pre-commit Hooks Failing

```bash
# Update hooks to latest versions
pre-commit autoupdate

# Clear cache and reinstall
pre-commit clean
pre-commit install --install-hooks
```

### Import Errors

```bash
# Reinstall package in editable mode
pip install -e .
```

### Type Checking Errors

```bash
# Install type stubs
mypy --install-types --non-interactive
```

## Project Structure

```
.
├── src/knapsack_gnn/          # Main library code
│   ├── data/                  # Data generation and loading
│   ├── models/                # GNN architectures
│   ├── training/              # Training loops
│   ├── decoding/              # Solution decoders
│   ├── eval/                  # Evaluation
│   ├── analysis/              # Statistical analysis
│   └── cli.py                 # CLI interface
├── experiments/               # Research pipelines
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── docs/                      # Documentation
├── .github/workflows/         # CI/CD pipelines
├── pyproject.toml             # Project metadata and deps
├── Makefile                   # Development commands
└── .pre-commit-config.yaml    # Pre-commit hooks
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Marcux777/GNN_to_Knapsack/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Marcux777/GNN_to_Knapsack/discussions)
- **Documentation**: [Full Documentation](https://marcux777.github.io/GNN_to_Knapsack/)

## Code of Conduct

Please read our [Code of Conduct](../CODE_OF_CONDUCT.md) before contributing.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
