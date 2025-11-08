# Contributing Guide

Thank you for your interest in contributing to Knapsack GNN! This guide will help you understand our development workflow, coding standards, and best practices.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Coding Standards](#coding-standards)
4. [Commit Message Guidelines](#commit-message-guidelines)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA-compatible GPU for training

### Setup Development Environment

1. **Clone the repository**

```bash
git clone https://github.com/Marcux777/GNN_to_Knapsack.git
cd GNN_to_Knapsack
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or using uv (faster)
uv pip install -e ".[dev]"
```

4. **Install pre-commit hooks**

```bash
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

This will automatically run formatters, linters, and checks before commits and pushes.

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following our [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Type check
make mypy

# Run tests
make test-quick  # Fast tests only
make test        # Full test suite with coverage
```

### 4. Commit Changes

Use Conventional Commits format (enforced by commitlint):

```bash
# Interactive commit message helper (recommended)
make commit

# Or manually
git add .
git commit -m "feat(models): add PNA architecture support"
```

### 5. Push and Create Pull Request

```bash
git push origin feat/your-feature-name
```

Then create a pull request on GitHub using the PR template.

## Coding Standards

### Code Style and Formatting

We use **Ruff** for both linting and formatting, configured in `pyproject.toml`.

**Key Style Rules:**
- Line length: 100 characters
- Indentation: 4 spaces (no tabs)
- String quotes: Double quotes preferred
- Import sorting: Automated by Ruff (replaces isort)

**Run formatters:**

```bash
make format
```

This runs:
- `ruff format` - Code formatting (replaces black)
- `ruff check --fix` - Auto-fix linting issues

### Type Hints

We use **mypy** with hybrid strictness levels:

- **Strict** in `src/knapsack_gnn/` (production code)
  - All functions must have type hints
  - No untyped definitions allowed

- **Moderate** in `experiments/` and `tests/`
  - Type hints encouraged but not enforced
  - Warnings for missing types

**Example:**

```python
from pathlib import Path
from typing import List, Optional

def load_checkpoint(
    checkpoint_dir: Path,
    device: str = "cpu"
) -> dict[str, Any]:
    """
    Load model checkpoint from directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load model on

    Returns:
        Dictionary containing model state and config

    Raises:
        FileNotFoundError: If checkpoint file not found
    """
    ...
```

**Run type checker:**

```bash
make mypy
```

### Error Handling

Use custom exceptions from `src/knapsack_gnn/utils/error_handler.py`:

```python
from knapsack_gnn.utils.error_handler import (
    CheckpointError,
    ConfigurationError,
    ValidationError,
    validate_checkpoint_dir,
)

# Use validation helpers
checkpoint_path = validate_checkpoint_dir(args.checkpoint)

# Raise custom exceptions with helpful messages
if not config_file.exists():
    raise ConfigurationError(
        f"Configuration file not found: {config_file}",
        suggestion="Check the path or create a config file (see examples/ for templates)."
    )
```

**Benefits:**
- Informative error messages
- Actionable suggestions for users
- Consistent error handling across CLI

### Logging

Use the centralized logger from `src/knapsack_gnn/utils/logger.py`:

```python
from knapsack_gnn.utils.logger import setup_logger, log_metrics

# Setup logger with file output
logger = setup_logger(
    name="training",
    log_file=checkpoint_dir / "training.log",
    level=logging.INFO
)

# Log experiment configuration
logger.info("=" * 60)
logger.info(f"Training Configuration")
logger.info("=" * 60)
for key, value in config.items():
    logger.info(f"  {key:.<30} {value}")

# Log metrics
log_metrics(
    logger,
    {"loss": 0.123, "accuracy": 0.956, "gap": 0.0007},
    prefix="Epoch 10 |",
    precision=4
)
```

**Best Practices:**
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include context in messages
- Use structured logging for metrics
- Don't log sensitive information

### Documentation

All public functions, classes, and modules must have docstrings in Google style:

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    num_epochs: int = 50,
) -> dict[str, list[float]]:
    """
    Train GNN model on knapsack instances.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer instance
        num_epochs: Number of training epochs

    Returns:
        Dictionary containing training history:
            - 'loss': List of loss values per epoch
            - 'accuracy': List of accuracy values per epoch

    Raises:
        ValueError: If num_epochs <= 0
        RuntimeError: If training fails

    Example:
        >>> model = GCN(hidden_dim=64)
        >>> optimizer = Adam(model.parameters(), lr=0.002)
        >>> history = train_model(model, train_loader, optimizer)
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")
    """
    ...
```

## Commit Message Guidelines

We follow **Conventional Commits** specification, enforced by commitlint.

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code formatting (no logic changes)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system or dependencies
- **ci**: CI/CD configuration
- **chore**: Maintenance tasks
- **revert**: Reverting previous commits

### Scopes

Optional but recommended:

- `models` - GNN architectures (GCN, GAT, PNA)
- `data` - Data generation and loading
- `training` - Training loop and optimization
- `decoder` - Solution decoding strategies
- `eval` - Evaluation and metrics
- `analysis` - Analysis and reporting
- `cli` - Command-line interface
- `config` - Configuration files
- `utils` - Utility functions
- `deps` - Dependencies
- `docs` - Documentation

### Examples

```bash
# Feature addition
feat(models): add PNA architecture support

# Bug fix
fix(decoder): handle empty solution edge case

# Documentation
docs: update installation instructions

# Refactoring
refactor(training): simplify loss calculation

# Performance improvement
perf(data): optimize graph construction with sparse tensors

# Test addition
test(decoder): add edge case tests for sampling strategy
```

### Using Commitizen (Recommended)

For interactive commit message creation:

```bash
make commit
```

This will guide you through creating a properly formatted commit message.

## Testing

### Running Tests

```bash
# Quick tests (exclude slow tests)
make test-quick

# Full test suite with coverage
make test

# Run specific test file
pytest tests/unit/test_data_generator.py -v

# Run tests matching pattern
pytest tests/ -k "decoder" -v
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Use pytest fixtures from `tests/conftest.py`
- Aim for >70% code coverage
- Mark slow tests with `@pytest.mark.slow`

**Example:**

```python
import pytest
from knapsack_gnn.data.generator import generate_knapsack_instance

def test_generate_knapsack_instance():
    """Test basic knapsack instance generation."""
    n_items = 50
    capacity_ratio = 0.5

    instance = generate_knapsack_instance(
        n_items=n_items,
        capacity_ratio=capacity_ratio,
        seed=42
    )

    assert len(instance.weights) == n_items
    assert len(instance.values) == n_items
    assert instance.capacity > 0
    assert all(w > 0 for w in instance.weights)
    assert all(v > 0 for v in instance.values)

@pytest.mark.slow
def test_train_full_pipeline():
    """Test full training pipeline (slow test)."""
    # Long-running integration test
    ...
```

### Coverage Requirements

- Minimum 70% code coverage required (enforced in CI)
- New features should include tests
- Bug fixes should include regression tests

## Documentation

Documentation lives directly inside the `docs/` directory as Markdown. There is no generated site,
so updating or adding documentation is as simple as editing the relevant `.md` file and making sure
it is linked from the README or another guide.

### Documentation Structure

```
docs/
├── api/              # Module notes and API references
├── guides/           # User guides and tutorials
│   ├── quickstart.md
│   ├── cli_usage.md
│   └── contributing.md
└── stylesheets/      # Optional assets for rendered Markdown
```

### Adding Documentation

1. **Guides**: Add Markdown files in `docs/guides/`
2. **API references**: Update the files in `docs/api/`
3. **Reports**: Add new documents in `docs/reports/`
4. **Link it**: Reference the new content from an existing page so readers can find it

## Pull Request Process

### Before Submitting

1. ✅ All tests pass (`make test`)
2. ✅ Code is formatted (`make format`)
3. ✅ Linting passes (`make lint`)
4. ✅ Type checking passes (`make mypy`)
5. ✅ Documentation updated
6. ✅ Commit messages follow Conventional Commits
7. ✅ Branch is up to date with main

### PR Checklist

Use the PR template to ensure:

- [ ] Clear description of changes
- [ ] Type of change specified
- [ ] Related issues linked
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Code quality checks passed

### Review Process

1. Automated checks run (CI/CD)
2. Code review by maintainer
3. Address feedback
4. Approval and merge

### Merge Strategy

- Squash and merge (default)
- Keep commit history clean
- Preserve PR description in merge commit

## Development Tips

### Quick Commands Reference

```bash
# Code quality
make format          # Format code with ruff
make lint           # Run linting
make mypy           # Type checking

# Testing
make test-quick     # Fast tests
make test          # Full test suite

# Documentation
# Edit Markdown in docs/ and link from README (no build step)

# Commits
make commit        # Interactive commit helper

# Cleanup
make clean         # Remove build artifacts and caches
```

### Pre-commit Hooks

Our pre-commit hooks run:

**On commit:**
- Ruff formatting
- Ruff linting (auto-fix)
- File hygiene (trailing whitespace, end-of-file fixer)
- YAML/TOML validation
- Commitlint (message validation)

**On push:**
- Mypy type checking
- Pytest quick tests
- Dependency sync check

### Troubleshooting

**Q: Pre-commit hooks are slow**
```bash
# Skip hooks temporarily (not recommended)
SKIP=mypy,pytest-quick git commit -m "..."

# Or skip specific hook
SKIP=mypy git commit -m "..."
```

**Q: Type checking fails**
```bash
# Run mypy to see errors
make mypy

# Add type hints or use type: ignore comments sparingly
```

**Q: Tests fail locally but pass in CI**
```bash
# Ensure dependencies are synced
make check-deps
make sync-deps

# Clear caches
make clean
```

## Getting Help

- 📚 [Documentation](https://github.com/Marcux777/GNN_to_Knapsack/tree/main/docs)
- 💬 [GitHub Discussions](https://github.com/Marcux777/GNN_to_Knapsack/discussions)
- 🐛 [Issue Tracker](https://github.com/Marcux777/GNN_to_Knapsack/issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

---

Thank you for contributing to Knapsack GNN! 🎉
