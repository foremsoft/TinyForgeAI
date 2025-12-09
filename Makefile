# TinyForgeAI Makefile
# Common development tasks

.PHONY: help install install-all install-training install-rag test test-verbose test-cov lint format ci-local docker-build docker-run demo train serve clean

# Default target
help:
	@echo "TinyForgeAI Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install base dependencies"
	@echo "  install-all      Install all dependencies (training + rag)"
	@echo "  install-training Install training dependencies"
	@echo "  install-rag      Install RAG dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run pytest test suite (quick)"
	@echo "  test-verbose     Run tests with verbose output"
	@echo "  test-cov         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run flake8 linter"
	@echo "  format           Format code with black (if installed)"
	@echo "  ci-local         Run full CI pipeline locally"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Running:"
	@echo "  demo             Run the complete demo"
	@echo "  demo-real        Run demo with real training"
	@echo "  train            Run quick start training example"
	@echo "  serve            Start the inference server"
	@echo "  dashboard        Start the dashboard API"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            Remove cache and build artifacts"
	@echo "  check-imports    Verify all imports work"
	@echo ""

# =============================================================================
# Installation
# =============================================================================

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

install-training:
	pip install -e ".[training]"

install-rag:
	pip install -e ".[rag]"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest -q --tb=short

test-verbose:
	pytest -v --tb=short

test-cov:
	pytest --cov=backend --cov=connectors --cov=services --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "Running flake8..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics --ignore=E501,W503
	@echo "Linting complete."

format:
	@if command -v black > /dev/null 2>&1; then \
		black backend connectors services tests; \
	else \
		echo "black not installed. Install with: pip install black"; \
	fi

ci-local: lint test
	@echo ""
	@echo "Running import checks..."
	$(MAKE) check-imports
	@echo ""
	@echo "Running Docker build (if Docker is available)..."
	@if command -v docker > /dev/null 2>&1; then \
		$(MAKE) docker-build; \
	else \
		echo "Docker not found, skipping docker-build"; \
	fi
	@echo ""
	@echo "CI local pipeline complete!"

check-imports:
	python -c "from backend.training.real_trainer import RealTrainer, TrainingConfig; print('real_trainer OK')"
	python -c "from connectors.indexer import DocumentIndexer, IndexerConfig; print('indexer OK')"
	python -c "from services.dashboard_api.main import app; print('dashboard_api OK')"
	@echo "All imports OK!"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t foremsoft/tinyforgeai:local .

docker-build-inference:
	docker build -f docker/Dockerfile -t foremsoft/tinyforgeai-inference:local .

docker-run:
	docker run -p 8000:8000 foremsoft/tinyforgeai:local

# =============================================================================
# Running
# =============================================================================

demo:
	python demo.py

demo-real:
	python demo.py --real

train:
	python examples/training/quick_start.py

train-lora:
	python examples/training/lora_training.py

serve:
	uvicorn inference_server.main:app --reload --port 8000

dashboard:
	uvicorn services.dashboard_api.main:app --reload --port 8001

rag-demo:
	python examples/rag/quick_start_rag.py

# =============================================================================
# Utilities
# =============================================================================

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf build dist *.egg-info
	rm -rf output tmp
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned build artifacts."

# Create a new release tag
release:
	@echo "Current version tags:"
	@git tag -l "v*" | tail -5
	@echo ""
	@echo "To create a release, run:"
	@echo "  git tag -a v0.2.0 -m 'Release v0.2.0'"
	@echo "  git push origin v0.2.0"
