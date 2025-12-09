"""
TinyForgeAI Prometheus Metrics

Provides Prometheus metrics for monitoring the TinyForgeAI platform.

Usage:
    from services.dashboard_api.metrics import (
        METRICS_AVAILABLE,
        MetricsMiddleware,
        get_metrics_router,
    )

    if METRICS_AVAILABLE:
        app.add_middleware(MetricsMiddleware)
        app.include_router(get_metrics_router())
"""

import time
import os
from typing import Callable

from fastapi import Request, Response
from fastapi.routing import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware

# Try to import prometheus_client, make metrics optional
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Check if metrics are enabled via environment
METRICS_ENABLED = os.getenv("TINYFORGE_METRICS_ENABLED", "true").lower() == "true"


if METRICS_AVAILABLE and METRICS_ENABLED:
    # ==========================================================================
    # Application Metrics
    # ==========================================================================

    # Request metrics
    REQUEST_COUNT = Counter(
        "tinyforge_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"]
    )

    REQUEST_LATENCY = Histogram(
        "tinyforge_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )

    REQUEST_IN_PROGRESS = Gauge(
        "tinyforge_http_requests_in_progress",
        "HTTP requests currently in progress",
        ["method", "endpoint"]
    )

    # Training metrics
    TRAINING_JOBS_TOTAL = Counter(
        "tinyforge_training_jobs_total",
        "Total training jobs created",
        ["status"]
    )

    TRAINING_JOBS_ACTIVE = Gauge(
        "tinyforge_training_jobs_active",
        "Currently active training jobs"
    )

    TRAINING_JOB_DURATION = Histogram(
        "tinyforge_training_job_duration_seconds",
        "Training job duration in seconds",
        buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800]
    )

    # Model registry metrics
    MODELS_REGISTERED = Gauge(
        "tinyforge_models_registered_total",
        "Total registered models"
    )

    MODELS_DEPLOYED = Gauge(
        "tinyforge_models_deployed_total",
        "Total deployed models"
    )

    # Inference metrics
    INFERENCE_REQUESTS = Counter(
        "tinyforge_inference_requests_total",
        "Total inference requests",
        ["model", "status"]
    )

    INFERENCE_LATENCY = Histogram(
        "tinyforge_inference_latency_seconds",
        "Inference request latency in seconds",
        ["model"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
    )

    # Service metrics
    SERVICES_TOTAL = Gauge(
        "tinyforge_services_total",
        "Total services",
        ["status"]
    )

    # Application info
    APP_INFO = Info(
        "tinyforge_app",
        "TinyForgeAI application information"
    )
    APP_INFO.info({
        "version": "0.2.0",
        "component": "dashboard-api"
    })


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP request metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not METRICS_AVAILABLE or not METRICS_ENABLED:
            return await call_next(request)

        method = request.method
        # Normalize path to avoid high cardinality
        path = self._normalize_path(request.url.path)

        # Track in-progress requests
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=path).inc()

        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status_code=str(status_code)
            ).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(duration)
            REQUEST_IN_PROGRESS.labels(method=method, endpoint=path).dec()

        return response

    def _normalize_path(self, path: str) -> str:
        """Normalize path to reduce cardinality."""
        # Replace UUIDs and IDs with placeholders
        import re

        # Replace UUID patterns
        path = re.sub(
            r"/[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}",
            "/{id}",
            path
        )
        # Replace short IDs (8 chars)
        path = re.sub(r"/[a-f0-9]{8}(?=/|$)", "/{id}", path)
        # Replace numeric IDs
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)

        return path


def get_metrics_router() -> APIRouter:
    """Create a router for metrics endpoints."""
    router = APIRouter(tags=["Metrics"])

    @router.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        if not METRICS_AVAILABLE:
            return Response(
                content="# Metrics not available - prometheus_client not installed\n",
                media_type="text/plain"
            )

        if not METRICS_ENABLED:
            return Response(
                content="# Metrics disabled\n",
                media_type="text/plain"
            )

        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )

    return router


# ==========================================================================
# Helper Functions for Recording Metrics
# ==========================================================================

def record_training_job_created(status: str = "pending"):
    """Record a new training job."""
    if METRICS_AVAILABLE and METRICS_ENABLED:
        TRAINING_JOBS_TOTAL.labels(status=status).inc()


def record_training_job_status_change(old_status: str, new_status: str):
    """Record a training job status change."""
    if METRICS_AVAILABLE and METRICS_ENABLED:
        TRAINING_JOBS_TOTAL.labels(status=new_status).inc()
        if new_status == "running":
            TRAINING_JOBS_ACTIVE.inc()
        elif old_status == "running" and new_status in ["completed", "failed", "cancelled"]:
            TRAINING_JOBS_ACTIVE.dec()


def record_training_job_duration(duration_seconds: float):
    """Record training job duration."""
    if METRICS_AVAILABLE and METRICS_ENABLED:
        TRAINING_JOB_DURATION.observe(duration_seconds)


def record_inference_request(model: str, success: bool, latency_seconds: float):
    """Record an inference request."""
    if METRICS_AVAILABLE and METRICS_ENABLED:
        status = "success" if success else "error"
        INFERENCE_REQUESTS.labels(model=model, status=status).inc()
        INFERENCE_LATENCY.labels(model=model).observe(latency_seconds)


def update_model_counts(total_models: int, deployed_models: int):
    """Update model registry counts."""
    if METRICS_AVAILABLE and METRICS_ENABLED:
        MODELS_REGISTERED.set(total_models)
        MODELS_DEPLOYED.set(deployed_models)


def update_service_counts(running: int, stopped: int):
    """Update service counts."""
    if METRICS_AVAILABLE and METRICS_ENABLED:
        SERVICES_TOTAL.labels(status="running").set(running)
        SERVICES_TOTAL.labels(status="stopped").set(stopped)
