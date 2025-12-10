"""
FastAPI middleware for TinyForgeAI.

Provides request/response middleware for:
- Request timing and metrics collection
- Request ID tracking
- Error handling
- CORS handling (via FastAPI middleware)
"""

import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.monitoring import (
    increment_counter,
    observe_histogram,
    set_gauge,
)

logger = logging.getLogger(__name__)

# Track active connections
_active_connections = 0


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting HTTP request metrics.

    Tracks:
    - Total requests by method, endpoint, and status
    - Request duration histogram
    - Active connections gauge
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        global _active_connections

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Track active connections
        _active_connections += 1
        set_gauge("active_connections", _active_connections)

        # Get normalized path for metrics (avoid high cardinality)
        path = self._normalize_path(request.url.path)
        method = request.method

        start_time = time.time()

        try:
            response = await call_next(request)
            status = str(response.status_code)

            # Record metrics
            increment_counter(
                "http_requests_total",
                labels={"method": method, "endpoint": path, "status": status},
            )

            duration = time.time() - start_time
            observe_histogram(
                "http_request_duration_seconds",
                duration,
                labels={"method": method, "endpoint": path},
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log request (for non-health endpoints)
            if not path.startswith("/health"):
                logger.debug(
                    f"{method} {path} -> {status} ({duration*1000:.1f}ms) "
                    f"[{request_id[:8]}]"
                )

            return response

        except Exception as e:
            # Record error metric
            increment_counter(
                "http_requests_total",
                labels={"method": method, "endpoint": path, "status": "500"},
            )

            duration = time.time() - start_time
            observe_histogram(
                "http_request_duration_seconds",
                duration,
                labels={"method": method, "endpoint": path},
            )

            logger.error(f"Request error: {method} {path} - {e} [{request_id[:8]}]")
            raise

        finally:
            _active_connections -= 1
            set_gauge("active_connections", _active_connections)

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for metrics to avoid high cardinality.

        Replaces dynamic segments (UUIDs, IDs) with placeholders.
        """
        parts = path.split("/")
        normalized = []

        for part in parts:
            if not part:
                continue

            # Check if it looks like a UUID
            if len(part) == 36 and part.count("-") == 4:
                normalized.append("{id}")
            # Check if it's all digits (numeric ID)
            elif part.isdigit():
                normalized.append("{id}")
            # Check if it looks like a version (v1, v2, etc.)
            elif part.startswith("v") and part[1:].isdigit():
                normalized.append(part)
            else:
                normalized.append(part)

        return "/" + "/".join(normalized) if normalized else "/"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request ID to all requests.

    The request ID can be used for distributed tracing and log correlation.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check if client provided a request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response
