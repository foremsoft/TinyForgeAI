# Contributing to TinyForgeAI

Thank you for your interest in contributing to TinyForgeAI! This document provides guidelines and information for contributors.

## Table of Contents

1. [Ways to Contribute](#ways-to-contribute)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style Guidelines](#code-style-guidelines)
5. [Making Changes](#making-changes)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)
8. [Community Guidelines](#community-guidelines)

---

## Ways to Contribute

There are many ways to contribute to TinyForgeAI:

| Contribution Type | Description | Difficulty |
|-------------------|-------------|------------|
| Report Bugs | Found a bug? Open an issue! | Easy |
| Documentation | Fix typos, improve explanations | Easy |
| Answer Questions | Help others in discussions | Easy |
| Write Tutorials | Create guides for new users | Medium |
| Fix Bugs | Implement fixes for issues | Medium |
| Add Features | Implement new functionality | Medium-Hard |
| Improve Tests | Add test coverage | Medium |
| Performance | Optimize code | Hard |

---

## Getting Started

### First Time Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/TinyForgeAI.git
   cd TinyForgeAI
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/anthropics/TinyForgeAI.git
   ```
4. **Set up development environment** (see below)

### Finding Issues to Work On

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - We need your help!
- `documentation` - Docs improvements
- `bug` - Known bugs to fix

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) GPU with CUDA for training features

### Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 3. Install in development mode with all dependencies
pip install -e ".[all]"

# 4. Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest

# Check imports
python -c "from backend.training.real_trainer import RealTrainer; print('OK')"
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Python Test Explorer

Settings (`.vscode/settings.json`):
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.formatting.provider": "black"
}
```

#### PyCharm

1. Open project directory
2. Configure interpreter to use venv
3. Mark `tests/` as Test Sources Root
4. Enable pytest as test runner

---

## Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications:

```python
# Good: Clear, documented code
def train_model(
    data_path: str,
    config: TrainingConfig,
    verbose: bool = False
) -> TrainedModel:
    """
    Train a model on the given dataset.

    Args:
        data_path: Path to training data (JSONL format).
        config: Training configuration.
        verbose: If True, print progress.

    Returns:
        Trained model ready for inference.

    Raises:
        FileNotFoundError: If data_path doesn't exist.
        ValueError: If data format is invalid.
    """
    # Implementation...
```

### Naming Conventions

```python
# Classes: PascalCase
class TrainingConfig:
    pass

# Functions/variables: snake_case
def train_model():
    learning_rate = 0.001

# Constants: SCREAMING_SNAKE_CASE
DEFAULT_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 512

# Private members: _leading_underscore
def _internal_helper():
    pass
```

### Import Order

```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party packages
import torch
import numpy as np
from transformers import AutoModel

# 3. Local imports
from backend.config import Settings
from backend.training import RealTrainer
```

### Type Hints

Use type hints for all public functions:

```python
from typing import List, Dict, Optional, Union

def process_data(
    items: List[Dict[str, str]],
    max_length: Optional[int] = None
) -> Union[List[str], None]:
    """Process data items."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief one-line description.

    Longer description if needed. Can span multiple lines
    and provide more context.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param2 is negative.

    Example:
        >>> example_function("test", 5)
        True
    """
    pass
```

---

## Making Changes

### Branch Naming

```
feature/add-lora-support
fix/memory-leak-in-trainer
docs/improve-installation-guide
test/add-indexer-tests
refactor/simplify-config
```

### Commit Messages

Follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

Examples:
```
feat(training): add LoRA support for efficient fine-tuning

Implemented LoRA (Low-Rank Adaptation) using the PEFT library.
This allows fine-tuning large models with minimal memory.

Closes #123
```

```
fix(indexer): prevent memory leak on large documents

Documents larger than 10MB were not being garbage collected
after indexing. Added explicit cleanup in finally block.

Fixes #456
```

### Keep Changes Focused

- One feature/fix per pull request
- Keep PRs small and reviewable (<500 lines ideally)
- Split large changes into multiple PRs

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_trainer.py

# Specific test
pytest tests/test_trainer.py::test_training_config

# With coverage
pytest --cov=backend --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Writing Tests

```python
# tests/test_my_feature.py

import pytest
from backend.my_module import MyClass


class TestMyClass:
    """Tests for MyClass."""

    def test_basic_functionality(self):
        """Test basic usage works correctly."""
        obj = MyClass()
        result = obj.do_something("input")
        assert result == "expected_output"

    def test_edge_case_empty_input(self):
        """Test behavior with empty input."""
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.do_something("")

    @pytest.fixture
    def configured_instance(self):
        """Fixture providing a configured MyClass instance."""
        return MyClass(option="value")

    def test_with_fixture(self, configured_instance):
        """Test using the fixture."""
        result = configured_instance.do_something("input")
        assert result is not None

    @pytest.mark.slow
    def test_long_running_operation(self):
        """Test that takes a long time (marked as slow)."""
        # ...
        pass

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU not available"
    )
    def test_gpu_feature(self):
        """Test that requires GPU."""
        pass
```

### Test Coverage Requirements

- New features should include tests
- Bug fixes should include regression tests
- Aim for >80% coverage on new code
- Critical paths should have 100% coverage

---

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Run tests
   pytest

   # Check code style
   flake8 .

   # Check types (optional)
   mypy backend/
   ```

3. **Update documentation** if needed

### Creating the PR

1. Push your branch:
   ```bash
   git push origin feature/my-feature
   ```

2. Open PR on GitHub

3. Fill out the PR template:
   ```markdown
   ## Description
   Brief description of changes.

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Refactoring
   - [ ] Other (describe)

   ## Testing
   Describe how you tested the changes.

   ## Checklist
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No new warnings
   ```

### Review Process

1. **Automated checks** run on your PR
2. **Maintainers review** the code
3. **Address feedback** by pushing new commits
4. **Maintainer approves** and merges

### After Merge

```bash
# Update your local main
git checkout main
git pull upstream main

# Delete your feature branch
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

---

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. We expect all participants to:

- **Be respectful** and considerate
- **Be collaborative** and helpful
- **Be patient** with newcomers
- **Accept constructive criticism** gracefully
- **Focus on what's best** for the community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions

### Getting Help

If you're stuck:

1. Check the documentation
2. Search existing issues
3. Ask in GitHub Discussions
4. Tag your issue with `help wanted`

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README (for significant contributions)

---

## Appendix: Quick Reference

### Common Commands

```bash
# Setup
pip install -e ".[all]"
pre-commit install

# Development
pytest                    # Run tests
pytest -x                 # Stop on first failure
pytest --cov              # With coverage
flake8 .                  # Check style

# Git workflow
git fetch upstream
git rebase upstream/main
git push origin feature/x
```

### PR Checklist

```
¡ Tests pass locally
¡ Code follows style guidelines
¡ Docstrings for new functions
¡ Type hints for public APIs
¡ Documentation updated
¡ No new warnings
¡ Commit messages are clear
¡ Branch is up to date with main
```

### File Templates

New module template:
```python
"""
Module description.

This module provides...

Example:
    >>> from module import Class
    >>> obj = Class()
"""

from typing import List, Optional

__all__ = ["MainClass", "helper_function"]


class MainClass:
    """Main class description."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize with optional config."""
        self.config = config or {}


def helper_function(data: List[str]) -> int:
    """Helper function description."""
    return len(data)
```

---

## Thank You!

Every contribution, no matter how small, makes TinyForgeAI better. Thank you for being part of our community!

Questions? Open an issue or start a discussion on GitHub.
