# Changelog

All notable changes to TinyForgeAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - Production-Ready Backend
- **Rate Limiting**: Configurable sliding window rate limiter with in-memory and Redis-based storage
  - Decorator and middleware support for FastAPI
  - Category-based limits (api, auth, inference, hourly)
  - Standard rate limit headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset)
- **Exception Handling**: Structured exception hierarchy with error codes
  - `TinyForgeError` base class with error codes and details
  - Specialized exceptions: `TrainingError`, `ValidationError`, `ConnectorError`, etc.
- **Webhook System**: Event-driven notifications for job lifecycle
  - Support for job started, completed, failed events
  - Configurable webhook endpoints with retry logic
- **Prometheus Metrics**: Integration for observability
  - Request counters, histograms, and gauges
  - Custom metrics registry with namespace support

### Added - REST API Connector
- **APIConnector**: Full-featured connector for consuming REST APIs
  - Multiple authentication types (Bearer, Basic, API Key)
  - Pagination support (offset, page, cursor, link header)
  - Built-in rate limiting and retry logic
  - Response caching with configurable TTL
  - Field mapping for training sample format

### Added - Development Environment
- **Docker Compose Dev Stack**: Complete development environment
  - PostgreSQL 15 database
  - Redis 7 cache for distributed rate limiting
  - pgAdmin for database management
  - Prometheus for metrics collection
  - Grafana for metrics visualization

### Added - A/B Testing Framework
- Model comparison experiments
- Automatic traffic splitting
- Statistical significance testing

### Changed
- Updated CI/CD pipeline with new module import tests
- Expanded test suite to 460+ tests

---

## [0.2.0] - 2025-12-09

### Added - Production Training Features
- **Real Training with HuggingFace Transformers**: Full training pipeline using `transformers` library
- **PEFT/LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation
- **RealTrainer Module**: `backend/training/real_trainer.py` with `TrainingConfig` dataclass
- **RAG Document Indexer**: `connectors/indexer.py` with sentence-transformers embeddings
- **Dashboard API**: `services/dashboard_api/` FastAPI service for monitoring

### Added - Documentation & Tutorials
- **5 Comprehensive Tutorials** in `docs/tutorials/`:
  - 01: Introduction to AI Training (beginner-friendly)
  - 02: Preparing Training Data (JSONL, JSON, CSV formats)
  - 03: Training Your First Model (hands-on walkthrough)
  - 04: Understanding LoRA (efficient fine-tuning)
  - 05: Deploying Your Model (local, Docker, cloud)
- **Architecture Guide**: `docs/ARCHITECTURE.md` with component diagrams
- **Contributing Guide**: `docs/CONTRIBUTING.md` with code style and PR process

### Added - Examples & Demo
- **Complete Demo Script**: `demo.py` - end-to-end demonstration
- **Training Quick Start**: `examples/training/quick_start.py`
- **LoRA Training Example**: `examples/training/lora_training.py`
- **Custom Dataset Example**: `examples/training/custom_dataset.py`
- **Evaluation Example**: `examples/training/evaluation.py`
- **RAG Quick Start**: `examples/rag/quick_start_rag.py`
- **Sample Training Data**: `examples/training/sample_data.jsonl`

### Added - DevOps & CI/CD
- **Docker Build Workflow**: `.github/workflows/docker_build.yml`
- **Root Dockerfile**: Multi-stage build for production
- **Updated CI Workflow**: pyproject.toml installation, import tests
- **Environment Template**: `.env.example` with all configuration options

### Changed
- Updated README.md with tutorials, examples, and new features
- Updated `pyproject.toml` with training and RAG dependencies
- Updated GitHub links to foremsoft organization
- Expanded test suite to 260+ tests

### Fixed
- CI workflow now uses pyproject.toml for installation

---

## [0.1.0] - 2025-01-15

### Added
- Initial release of TinyForgeAI
- Complete train -> export -> serve pipeline
- CLI tool for all major operations
- Docker support for inference deployment
- Comprehensive test suite (215+ tests)
- Full documentation

[Unreleased]: https://github.com/foremsoft/TinyForgeAI/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/foremsoft/TinyForgeAI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/foremsoft/TinyForgeAI/releases/tag/v0.1.0
