# TinyForgeAI Makefile
# Common development tasks

.PHONY: help test lint ci-local docker-build install clean

# Default target
help:
	@echo "TinyForgeAI Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  test        Run pytest test suite"
	@echo "  lint        Run flake8 linter"
	@echo "  ci-local    Run full CI pipeline locally (test + lint + docker)"
	@echo "  docker-build Build Docker image locally"
	@echo "  install     Install dependencies"
	@echo "  clean       Remove cache and build artifacts"
	@echo ""

# Run tests
test:
	pytest -q

# Run tests with verbose output
test-verbose:
	pytest -v

# Run linter
lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics --ignore=E501,W503
	@echo "Linting complete."

# Run full CI pipeline locally
ci-local: test lint
	@echo ""
	@echo "Running Docker build (if Docker is available)..."
	@if command -v docker > /dev/null 2>&1; then \
		$(MAKE) docker-build; \
	else \
		echo "Docker not found, skipping docker-build"; \
	fi
	@echo ""
	@echo "CI local pipeline complete!"

# Build Docker image
docker-build:
	@if [ -f docker/Dockerfile ]; then \
		docker build -f docker/Dockerfile -t foremsoft/tinyforgeai:local .; \
	else \
		echo "docker/Dockerfile not found, skipping build"; \
	fi

# Install dependencies
install:
	pip install -r requirements.txt

# Install in development mode
install-dev:
	pip install -e ".[dev]"

# Clean cache and build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts."
