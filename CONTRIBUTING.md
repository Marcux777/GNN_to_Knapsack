# Contributing to GNN to Knapsack

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the GNN to Knapsack library.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Code Standards](#code-standards)
- [Development Setup](#development-setup)
- [Branch Flow](#branch-flow)
- [Pull Request Process](#pull-request-process)
- [Commit Convention](#commit-convention)
- [Adding New Components](#adding-new-components)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with the following tools:

**Linting:**
```bash
ruff check src/ experiments/ tests/
```

**Formatting:**
```bash
black src/ experiments/ tests/
```

**Type Checking:**
```bash
mypy src/knapsack_gnn/ --ignore-missing-imports
```

**Run all checks:**
```bash
make lint
```

### Docstrings

- Use **Google style** docstrings for all public APIs
- Include `Args`, `Returns`, `Raises`, and `Examples` sections
- Keep docstrings concise but complete

**Example:**
```python
def solve_instance(
    weights: np.ndarray,
    values: np.ndarray,
    capacity: float
) -> Tuple[np.ndarray, float]:
    """Solve a knapsack instance using the trained GNN model.

    Args:
        weights: Item weights array of shape (n_items,).
        values: Item values array of shape (n_items,).
        capacity: Knapsack capacity constraint.

    Returns:
        Tuple of (solution, objective_value) where solution is a
        binary array indicating selected items.

    Raises:
        ValueError: If weights and values have different lengths.
        ValueError: If capacity is negative.

    Examples:
        >>> weights = np.array([1, 2, 3])
        >>> values = np.array([10, 20, 30])
        >>> solution, value = solve_instance(weights, values, capacity=5.0)
    """
```

### Type Hints

- All public functions **must** have type hints
- Use modern syntax (Python 3.10+)
- Import types from `typing`, `numpy.typing`, etc.

**Example:**
```python
from typing import Optional, List, Dict, Tuple
from numpy.typing import NDArray
import numpy as np

def function(
    data: NDArray[np.float32],
    labels: Optional[NDArray[np.int64]] = None
) -> Dict[str, float]:
    ...
```

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/GNN_to_Knapsack.git
cd GNN_to_Knapsack
```

### 2. Install Development Dependencies

```bash
pip install -e .[dev]
```

This installs the package in editable mode with development tools (pytest, ruff, mypy, black).

### 3. Verify Installation

```bash
make test
make lint
```

---

## Branch Flow

We follow a simplified Git Flow:

- **`main`** - Stable, production-ready code
- **`feature/<name>`** - New features
- **`fix/<name>`** - Bug fixes
- **`docs/<name>`** - Documentation updates
- **`refactor/<name>`** - Code refactoring

### Creating a Branch

```bash
git checkout -b feature/my-awesome-feature
```

### Branch Naming

- Use lowercase and hyphens
- Be descriptive but concise
- Examples:
  - `feature/add-beam-search-decoder`
  - `fix/sampling-probability-nan`
  - `docs/improve-installation-guide`

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation

### 3. Run Quality Checks

```bash
make test      # Run test suite
make lint      # Check code quality
mypy src/      # Type checking
```

### 4. Commit Changes

Follow the [commit convention](#commit-convention):

```bash
git commit -m "feat(decoding): add beam search strategy"
```

### 5. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub with:
- Clear title following commit convention
- Description of changes
- Link to related issues
- Screenshots/results if applicable

### 6. Code Review

- Address reviewer feedback
- Keep commits organized
- Squash if needed before merge

---

## Commit Convention

We use **Conventional Commits** format:

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only
- **style**: Formatting (no code change)
- **refactor**: Code restructuring
- **perf**: Performance improvement
- **test**: Adding tests
- **chore**: Maintenance tasks
- **ci**: CI/CD changes

### Scopes

Common scopes:
- `models` - GNN architectures
- `decoding` - Solution decoding strategies
- `training` - Training loop
- `data` - Dataset generation
- `eval` - Evaluation metrics
- `cli` - Command-line interface
- `docs` - Documentation

### Examples

```bash
feat(decoding): add greedy repair strategy
fix(sampling): handle empty probability vector
docs(readme): update installation instructions
refactor(models): simplify PNA encoder
test(decoding): add tests for warm-start ILP
chore(deps): update PyTorch to 2.0
```

---

## Adding New Components

### New Decoder Strategy

1. **Implement** in `src/knapsack_gnn/decoding/`
2. **Add enum** to `DecodingStrategy` in `decoding/__init__.py`
3. **Write tests** in `tests/unit/test_decoding.py`
4. **Document** in `docs/guides/decoding_strategies.md`
5. **Add example** in `experiments/examples/`

**Example:**
```python
# src/knapsack_gnn/decoding/my_strategy.py

def my_decoding_strategy(
    probs: np.ndarray,
    weights: np.ndarray,
    capacity: float
) -> np.ndarray:
    """My awesome decoding strategy.

    Args:
        probs: Item selection probabilities from GNN.
        weights: Item weights.
        capacity: Knapsack capacity.

    Returns:
        Binary solution array.
    """
    # Your implementation
    ...
```

### New Model Architecture

1. **Implement** in `src/knapsack_gnn/models/`
2. **Follow pattern**: encoder → message passing layers → decoder
3. **Add to registry** in `models/__init__.py`
4. **Write ablation script** in `experiments/`
5. **Document** in `docs/architecture/`

**Example:**
```python
# src/knapsack_gnn/models/my_model.py

class MyGNN(nn.Module):
    """My custom GNN architecture.

    Args:
        hidden_dim: Hidden dimension size.
        num_layers: Number of message passing layers.
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        # Your architecture
        ...

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyTorch Geometric Data object.

        Returns:
            Item selection probabilities.
        """
        ...
```

---

## Testing

### Running Tests

```bash
# All tests
pytest tests/

# Specific file
pytest tests/unit/test_sampling.py

# With coverage
pytest tests/ --cov=src/knapsack_gnn/ --cov-report=html

# Fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Aim for >80% coverage
- Use descriptive test names

**Example:**
```python
# tests/unit/test_decoding.py

def test_sampling_decoder_feasibility():
    """Test that sampling decoder always produces feasible solutions."""
    weights = np.array([1, 2, 3, 4, 5])
    values = np.array([10, 20, 30, 40, 50])
    capacity = 7.0
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    solution = sampling_decode(probs, weights, capacity)

    assert np.sum(solution * weights) <= capacity
    assert solution.shape == weights.shape
    assert np.all((solution == 0) | (solution == 1))
```

---

## Documentation

### Code Documentation

- **Docstrings** for all public functions/classes
- **Type hints** for all function signatures
- **Inline comments** for complex logic only

### User Documentation

- Update `docs/` when adding features
- Add examples to `experiments/examples/`
- Update README if changing CLI or installation

### Documentation Build

```bash
# Generate API docs (if using Sphinx)
cd docs/
make html

# View locally
python -m http.server --directory docs/_build/html
```

---

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Contact maintainers via email (see `pyproject.toml`)

---

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Acknowledged in papers using this code

Thank you for contributing! 🎉
