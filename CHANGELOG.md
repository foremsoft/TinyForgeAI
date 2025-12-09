# Changelog

All notable changes to TinyForgeAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and scaffolding (Step A)
- Backend API with FastAPI health endpoints (Step B)
- Training pipeline with dataset loader, dry-run trainer, and PEFT/LoRA adapter (Step C)
- Data connectors: Database, Google Docs (with mock mode), and file ingestion (Step D)
- ONNX export and INT8 quantization stubs (Step E)
- Microservice exporter/builder for packaging inference services (Step F)
- CLI tool (`foremforge`) with init, train, export, and serve commands (Step F.1)
- pytest configuration and GitHub Actions CI workflow (Step G.1)
- Docker configuration with Dockerfile.inference and docker-compose (Step H.1)
- Comprehensive documentation: architecture, training, connectors (Step I.1)
- End-to-end demo scripts (bash and Python) with smoke tests (Step J.1)
- Release preparation: packaging scaffolds, model zoo, governance files (Step K)

### Features
- JSONL dataset loading and validation
- Dry-run training with stub model artifacts
- LoRA adapter simulation for PEFT workflows
- ONNX export with quantization placeholders
- Database connector with SQLite support
- Google Docs connector with mock mode for offline development
- File ingestion for TXT, MD, DOCX, PDF formats
- Inference server template with /health and /predict endpoints
- Docker deployment support with health checks
- Cross-platform e2e demo (bash + Python)

## [0.1.0] - 2025-01-15

### Added
- Initial release of TinyForgeAI
- Complete train -> export -> serve pipeline
- CLI tool for all major operations
- Docker support for inference deployment
- Comprehensive test suite (215+ tests)
- Full documentation

[Unreleased]: https://github.com/anthropics/TinyForgeAI/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anthropics/TinyForgeAI/releases/tag/v0.1.0
