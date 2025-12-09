# TinyForgeAI CI/CD Documentation

This document describes the Continuous Integration (CI) setup for TinyForgeAI.

## Overview

The CI pipeline runs automatically on:
- **Push** to `main` or `master` branches
- **Pull requests** to any branch

The pipeline consists of three jobs:
1. **test** - Runs the pytest test suite
2. **lint** - Runs flake8 linter for code quality
3. **docker-build** - Builds the Docker image (validation only, no push)

## GitHub Actions Workflow

The CI workflow is defined in `.github/workflows/ci.yml`.

### Test Job

Runs the full pytest test suite:
- Uses Python 3.10
- Installs dependencies from `requirements.txt`
- Runs `pytest -q`

### Lint Job

Runs flake8 linting:
- Checks for syntax errors and undefined names (fails on these)
- Reports style issues as warnings (does not fail)
- Ignores long lines (E501) and line break issues (W503)

### Docker Build Job

Builds the Docker image to validate the Dockerfile:
- Uses Docker Buildx for efficient builds
- Does NOT push images to any registry
- Uses GitHub Actions cache for faster builds
- Continues on error if Dockerfile doesn't exist

## Running CI Locally

You can run the same CI checks locally before pushing.

### Using Make (Recommended)

```bash
# Run tests only
make test

# Run tests with verbose output
make test-verbose

# Run linter only
make lint

# Run full CI pipeline (test + lint + docker)
make ci-local
```

### Using pytest directly

```bash
# Quick test run
pytest -q

# Verbose test run
pytest -v

# Run specific test file
pytest tests/test_cli.py -v

# Run with coverage (if installed)
pytest --cov=backend --cov=cli --cov=connectors
```

### Using flake8 directly

```bash
# Check for syntax errors (strict)
flake8 . --select=E9,F63,F7,F82 --show-source

# Full lint check
flake8 . --max-line-length=100 --ignore=E501,W503
```

### Building Docker locally

```bash
# Build the inference server image
docker build -f docker/Dockerfile -t foremsoft/tinyforgeai:local .

# Or using make
make docker-build
```

## Environment Variables

The following environment variables affect CI behavior:

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_VERSION` | Python version for CI | `3.10` |
| `GOOGLE_OAUTH_DISABLED` | Disable Google OAuth (mock mode) | `true` |

## Adding Repository Secrets

For future features that require secrets (e.g., Docker Hub push, deployment):

1. Go to your GitHub repository
2. Navigate to **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Add the secret name and value

Common secrets you might need:
- `DOCKERHUB_USERNAME` - Docker Hub username
- `DOCKERHUB_TOKEN` - Docker Hub access token
- `PYPI_API_TOKEN` - PyPI API token for package publishing

**Note:** Never commit secrets to the repository. Always use GitHub Secrets.

## Troubleshooting

### Tests failing locally but passing in CI

- Ensure you have the same Python version (3.10)
- Install all dependencies: `pip install -r requirements.txt`
- Check for platform-specific issues (Windows vs Linux)

### Lint errors

- Run `make lint` to see all issues
- Most style issues are warnings and won't fail the build
- Syntax errors (E9, F63, F7, F82) will fail the build

### Docker build failing

- Ensure `docker/Dockerfile` exists
- Check that all required files are present
- Review Docker build logs for specific errors

## CI Status Badge

Add this badge to your README to show CI status:

```markdown
![CI](https://github.com/foremsoft/TinyForgeAI/actions/workflows/ci.yml/badge.svg)
```
