# TinyForgeAI Docker Image
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install package
COPY pyproject.toml setup.cfg MANIFEST.in ./
COPY cli/ ./cli/
COPY backend/ ./backend/
COPY connectors/ ./connectors/
COPY inference_server/ ./inference_server/

RUN pip install --no-cache-dir -e .

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/cli ./cli
COPY --from=builder /app/backend ./backend
COPY --from=builder /app/connectors ./connectors
COPY --from=builder /app/inference_server ./inference_server

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash tinyforge && \
    chown -R tinyforge:tinyforge /app

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output && \
    chown -R tinyforge:tinyforge /app/models /app/data /app/output

USER tinyforge

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose inference server port
EXPOSE 8000

# Default command: run inference server
CMD ["python", "-m", "uvicorn", "inference_server.server:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================================
# Labels
# ============================================
LABEL org.opencontainers.image.title="TinyForgeAI" \
      org.opencontainers.image.description="Fine-tune tiny LLMs and deploy as microservices" \
      org.opencontainers.image.source="https://github.com/foremsoft/TinyForgeAI" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="TinyForgeAI"
