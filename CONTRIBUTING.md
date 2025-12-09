# Contributing to TinyForgeAI

Thank you for your interest in contributing to TinyForgeAI! This document provides guidelines and best practices for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) Docker for container testing

### Development Setup

```bash
# Clone the repository
git clone https://github.com/anthropics/TinyForgeAI.git
cd TinyForgeAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest -q
```

## Branching Policy

We use a feature branch workflow:

- `main` - Stable release branch, always deployable
- `feature/*` - New features (e.g., `feature/add-notion-connector`)
- `fix/*` - Bug fixes (e.g., `fix/dataset-loader-unicode`)
- `docs/*` - Documentation updates (e.g., `docs/update-training-guide`)
- `chore/*` - Maintenance tasks (e.g., `chore/update-dependencies`)

### Creating a Branch

```bash
# Sync with main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

## Commit Message Style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `chore:` - Maintenance tasks (deps, configs)
- `ci:` - CI/CD changes
- `test:` - Test additions or fixes
- `refactor:` - Code refactoring (no feature change)
- `style:` - Code style changes (formatting, no logic change)
- `perf:` - Performance improvements

### Examples

```bash
feat(connectors): add Notion API connector
fix(dataset): handle unicode characters in JSONL files
docs(training): add LoRA configuration examples
chore(deps): update fastapi to 0.109.0
test(exporter): add ONNX export edge case tests
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_cli.py

# Run with coverage
pytest --cov=backend --cov=connectors --cov=cli

# Run only fast tests (skip slow/integration)
pytest -m "not slow"
```

## Code Style

We use standard Python formatting tools:

```bash
# Lint code
flake8 backend/ connectors/ cli/ inference_server/

# Format code (if black is installed)
black backend/ connectors/ cli/ inference_server/

# Sort imports (if isort is installed)
isort backend/ connectors/ cli/ inference_server/
```

### Style Guidelines

- Follow PEP 8
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized
- Prefer explicit over implicit

## Submitting Pull Requests

### Before Submitting

1. **Tests pass**: Run `pytest -q` and ensure all tests pass
2. **Lint passes**: Run `flake8` with no errors
3. **Documentation**: Update docs if adding/changing features
4. **Changelog**: Add entry to CHANGELOG.md for user-facing changes

### PR Process

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template completely
4. Request review from maintainers
5. Address review feedback
6. Once approved, maintainer will merge

### PR Title Format

Use the same format as commit messages:

```
feat(scope): description
fix(scope): description
```

## Code Review

### For Authors

- Keep PRs focused and reasonably sized
- Respond to feedback promptly
- Be open to suggestions
- Explain your reasoning when disagreeing

### For Reviewers

- Be constructive and respectful
- Explain the "why" behind suggestions
- Approve when satisfied, don't block on style nitpicks
- Use "Request changes" for blocking issues only

## Adding New Features

### Connectors

1. Create `connectors/your_connector.py` following the pattern of existing connectors
2. Add mock mode support for offline testing
3. Add tests in `tests/test_your_connector.py`
4. Update `docs/connectors.md`

### CLI Commands

1. Add command in `cli/commands/`
2. Register in `cli/__init__.py`
3. Add tests in `tests/test_cli.py`
4. Update README with usage example

### Training Features

1. Implement in `backend/training/`
2. Add tests in `tests/`
3. Update `docs/training.md`

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution (if any)
- Alternatives considered

## Questions?

- Open a [Discussion](https://github.com/anthropics/TinyForgeAI/discussions)
- Check existing [Issues](https://github.com/anthropics/TinyForgeAI/issues)

Thank you for contributing!
