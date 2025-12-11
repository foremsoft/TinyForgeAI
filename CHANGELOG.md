# Changelog

All notable changes to TinyForgeAI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-12-11

### Added - Killer Features

- **One-Click URL Training**: Train models directly from URLs without data preparation
  - `foremforge train-url <url>` command
  - Supports Notion pages and databases
  - Supports Google Docs and Google Sheets
  - Supports GitHub files and README
  - Supports generic websites (FAQ pages, documentation)
  - Supports raw JSON/JSONL/CSV files
  - Auto-detection of content type
  - Q&A pair extraction from various formats
  - Preview mode to inspect extracted data

- **Training Data Generator**: Generate synthetic training data from examples
  - `foremforge augment` command
  - Multiple augmentation strategies:
    - Synonym replacement
    - Template-based variations
    - Paraphrasing rules
    - Back-translation simulation
    - LLM-powered augmentation (Claude/OpenAI)
  - Generate 500+ samples from just 5 examples
  - Configurable quality thresholds
  - Deduplication and shuffling

- **Shareable Playground**: Create instant, shareable model demos
  - `foremforge playground` command
  - Beautiful, responsive web UI
  - Three deployment options:
    - Local server (FastAPI-based)
    - Public sharing via ngrok/cloudflare
    - Standalone HTML export (works offline)
  - Real-time inference with latency metrics
  - Example inputs for quick testing
  - Mobile-friendly design

### Added - New Backend Modules
- `backend/augment/` - Data augmentation module
  - `generator.py` - Main data generator
  - `strategies.py` - Augmentation strategies
- `backend/url_trainer/` - URL training module
  - `extractor.py` - Content extraction from URLs
  - `trainer.py` - URL-based training orchestration
- `backend/playground/` - Playground module
  - `server.py` - FastAPI playground server
  - `exporter.py` - Standalone HTML export

### Added - CLI Commands
- `foremforge train-url <url>` - Train from URL
- `foremforge augment` - Generate synthetic data
- `foremforge playground` - Start shareable playground

### Added - Documentation
- New tutorial: `06-killer-features.md` - Guide to new features
- Updated README with killer features section
- Updated Wiki with new feature documentation

---

## [0.4.0] - 2025-12-11

### Added - MCP Integration (Model Context Protocol)
- **MCP Server**: Full Model Context Protocol server for AI assistant integration
  - Works with Claude Desktop, ChatGPT, Gemini, and other MCP-compatible assistants
  - Stdio transport for local execution
  - Comprehensive error handling and logging

- **MCP Tools**: 13 tools for AI-powered control
  - `train_model` - Submit training jobs with configurable parameters
  - `get_training_status` - Monitor job progress and metrics
  - `list_training_jobs` - List all jobs with filtering
  - `cancel_training` - Cancel running jobs
  - `run_inference` - Single model predictions
  - `batch_inference` - Batch predictions on multiple inputs
  - `fetch_data` - Fetch from any connector (DB, API, Notion, Slack, etc.)
  - `ingest_files` - Convert documents to training format
  - `index_documents` - Index for RAG search
  - `search_documents` - Semantic document search
  - `list_models` - Browse Model Zoo
  - `get_model_info` - Get detailed model information
  - `export_model` - Export to ONNX format

- **MCP Resources**: Read-only data access for AI assistants
  - `tinyforge://models/registry` - Complete model registry
  - `tinyforge://jobs/active` - Active training jobs
  - `tinyforge://connectors/available` - Available data connectors
  - `tinyforge://docs/quickstart` - Quick start documentation

### Added - Documentation
- New `docs/mcp.md` - Comprehensive MCP integration guide
- New `mcp/README.md` - MCP server documentation
- New hands-on tutorial: `05-mcp-integration.md`
- Updated Architecture guide with MCP section
- Updated Wiki with MCP quick setup
- Updated Connectors guide with MCP integration info

### Changed
- Updated README with MCP section and badges
- Updated project structure to include `mcp/` directory
- Renumbered ARCHITECTURE.md sections for MCP addition

## [0.3.0] - 2025-12-10

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

### Added - Google Drive Connector
- **GoogleDriveConnector**: Access files and folders from Google Drive
  - List files in folders
  - Download file content (including Google Workspace exports)
  - Stream training samples from JSON/JSONL files
  - Mock mode for offline development (`GOOGLE_DRIVE_MOCK=true`)
  - Support for service account and OAuth authentication

### Added - Notion Connector
- **NotionConnector**: Access pages and databases from Notion workspaces
  - List pages in databases with filtering
  - Get page content with block extraction
  - Stream training samples from database properties
  - Mock mode for offline development (`NOTION_MOCK=true`)
  - Support for all common Notion property types

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

### Added - Confluence Connector
- **ConfluenceConnector**: Access pages and spaces from Atlassian Confluence
  - List spaces with type filtering (global, personal)
  - List and search pages with CQL queries
  - Get page content with HTML to text conversion
  - Stream training samples from wiki pages
  - Mock mode for offline development (`CONFLUENCE_MOCK=true`)
  - Support for Atlassian Cloud API authentication

### Added - Slack Connector
- **SlackConnector**: Access messages and conversations from Slack workspaces
  - List channels (public, private, DMs)
  - Get messages from channels with pagination
  - Get thread replies for conversation context
  - Stream training samples with multiple modes:
    - Thread Q&A: Extract question/answer pairs from threads
    - Consecutive: Create pairs from consecutive messages
    - Reaction filter: Extract messages with specific reactions
  - Message text cleaning (removes mentions, URLs, special commands)
  - Mock mode for offline development (`SLACK_MOCK=true`)
  - User caching for efficient lookups
- **AsyncSlackConnector**: Async version of Slack connector
  - Full async/await support using httpx
  - Concurrent channel message fetching
  - Async context manager for resource management

### Added - Async Connectors
- **AsyncGoogleDriveConnector**: Async version of Google Drive connector
  - Full async/await support using httpx
  - Concurrent file fetching with configurable limits
  - Async context manager for resource management
- **AsyncNotionConnector**: Async version of Notion connector
  - Full async/await support using httpx
  - Concurrent page content fetching
  - Async streaming of training samples

### Added - Model Versioning and Registry
- **ModelRegistry**: Centralized model version management
  - Semantic versioning (SemVer 2.0.0) with major, minor, patch bumping
  - Version lifecycle states: Draft, Active, Staged, Deprecated, Archived
  - Version comparison, rollback, and search capabilities
  - Artifact storage and export functionality
- **ModelMetadata**: Comprehensive training provenance tracking
  - Training configuration (epochs, batch_size, LoRA settings)
  - Training metrics (loss, accuracy, BLEU, ROUGE, F1)
  - Data source tracking and lineage
- **ModelCard**: HuggingFace-style model documentation
  - Markdown export for model cards
  - Intended use, limitations, ethical considerations
- **CLI Interface**: Full command-line management
  - `list`, `versions`, `info`, `register`, `activate`, `deprecate`
  - `compare`, `export`, `rollback`, `search` commands

### Changed
- Updated CI/CD pipeline with new module import tests
- Expanded test suite to 550+ tests

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

[Unreleased]: https://github.com/foremsoft/TinyForgeAI/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/foremsoft/TinyForgeAI/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/foremsoft/TinyForgeAI/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/foremsoft/TinyForgeAI/releases/tag/v0.1.0
