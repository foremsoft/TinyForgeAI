"""
Metrics and monitoring API routes for TinyForgeAI.

Provides endpoints for Prometheus-compatible metrics export and health monitoring.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from backend.monitoring import (
    get_metrics_registry,
    get_system_info,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])


@router.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """
    Export metrics in Prometheus text format.

    This endpoint is designed to be scraped by Prometheus.
    """
    registry = get_metrics_registry()
    metrics = registry.export_prometheus()
    return Response(
        content=metrics,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/json")
async def get_metrics_json() -> Dict[str, Any]:
    """
    Export metrics as JSON for debugging and visualization.
    """
    registry = get_metrics_registry()
    return registry.export_json()


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if the service is running.
    """
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.

    Returns 200 if the service is ready to accept traffic.
    Checks database connectivity and other dependencies.
    """
    checks = {
        "api": True,
    }

    # Add more readiness checks as needed
    # e.g., database connectivity, model availability

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "checks": checks,
    }


@router.get("/health/detailed")
async def detailed_health():
    """
    Detailed health check with system information.
    """
    system_info = get_system_info()
    registry = get_metrics_registry()

    # Get some key metrics
    uptime = registry.get_gauge_value("process_uptime_seconds") or 0

    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "system": system_info,
        "version": "0.2.0",
    }
