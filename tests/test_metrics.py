"""
Tests for TinyForgeAI Prometheus Metrics Module

Tests the metrics collection and endpoint functionality.
"""

import pytest
from fastapi.testclient import TestClient


class TestMetricsModule:
    """Test metrics module functionality."""

    def test_metrics_import(self):
        """Test that metrics module can be imported."""
        from services.dashboard_api.metrics import (
            METRICS_AVAILABLE,
            METRICS_ENABLED,
            MetricsMiddleware,
            get_metrics_router,
        )
        # Module should be importable regardless of prometheus_client
        assert METRICS_AVAILABLE is True or METRICS_AVAILABLE is False
        assert METRICS_ENABLED is True or METRICS_ENABLED is False

    def test_metrics_middleware_class_exists(self):
        """Test that MetricsMiddleware class is defined."""
        from services.dashboard_api.metrics import MetricsMiddleware
        assert MetricsMiddleware is not None

    def test_metrics_router_function_exists(self):
        """Test that get_metrics_router function exists."""
        from services.dashboard_api.metrics import get_metrics_router
        router = get_metrics_router()
        assert router is not None

    def test_helper_functions_exist(self):
        """Test that helper functions are defined."""
        from services.dashboard_api.metrics import (
            record_training_job_created,
            record_training_job_status_change,
            record_training_job_duration,
            record_inference_request,
            update_model_counts,
            update_service_counts,
        )
        # Functions should be callable without error
        assert callable(record_training_job_created)
        assert callable(record_training_job_status_change)
        assert callable(record_training_job_duration)
        assert callable(record_inference_request)
        assert callable(update_model_counts)
        assert callable(update_service_counts)


class TestMetricsEndpoint:
    """Test /metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from services.dashboard_api.main import app
        return TestClient(app)

    def test_metrics_endpoint_returns_200(self, client):
        """Test that /metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type(self, client):
        """Test that /metrics endpoint returns correct content type."""
        response = client.get("/metrics")
        # Should be text/plain or prometheus content type
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "text/plain" in content_type

    def test_metrics_endpoint_contains_tinyforge_metrics(self, client):
        """Test that /metrics endpoint contains TinyForge metrics."""
        from services.dashboard_api.metrics import METRICS_AVAILABLE, METRICS_ENABLED

        response = client.get("/metrics")
        content = response.text

        if METRICS_AVAILABLE and METRICS_ENABLED:
            # Should contain our custom metrics
            assert "tinyforge" in content.lower()
        else:
            # Should indicate metrics unavailable or disabled
            assert "not available" in content.lower() or "disabled" in content.lower()


class TestMetricsHelperFunctions:
    """Test metrics helper functions work without errors."""

    def test_record_training_job_created(self):
        """Test record_training_job_created doesn't raise."""
        from services.dashboard_api.metrics import record_training_job_created
        # Should not raise regardless of metrics availability
        record_training_job_created("pending")
        record_training_job_created("running")

    def test_record_training_job_status_change(self):
        """Test record_training_job_status_change doesn't raise."""
        from services.dashboard_api.metrics import record_training_job_status_change
        record_training_job_status_change("pending", "running")
        record_training_job_status_change("running", "completed")
        record_training_job_status_change("running", "failed")

    def test_record_training_job_duration(self):
        """Test record_training_job_duration doesn't raise."""
        from services.dashboard_api.metrics import record_training_job_duration
        record_training_job_duration(60.0)
        record_training_job_duration(3600.0)

    def test_record_inference_request(self):
        """Test record_inference_request doesn't raise."""
        from services.dashboard_api.metrics import record_inference_request
        record_inference_request("model-1", True, 0.1)
        record_inference_request("model-2", False, 0.5)

    def test_update_model_counts(self):
        """Test update_model_counts doesn't raise."""
        from services.dashboard_api.metrics import update_model_counts
        update_model_counts(10, 5)
        update_model_counts(0, 0)

    def test_update_service_counts(self):
        """Test update_service_counts doesn't raise."""
        from services.dashboard_api.metrics import update_service_counts
        update_service_counts(5, 2)
        update_service_counts(0, 0)


class TestMetricsMiddleware:
    """Test MetricsMiddleware functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from services.dashboard_api.main import app
        return TestClient(app)

    def test_middleware_tracks_requests(self, client):
        """Test that middleware tracks HTTP requests."""
        from services.dashboard_api.metrics import METRICS_AVAILABLE, METRICS_ENABLED

        # Make several requests
        client.get("/health")
        client.get("/api/stats")
        client.get("/api/jobs")

        if METRICS_AVAILABLE and METRICS_ENABLED:
            # Check metrics endpoint shows request counts
            response = client.get("/metrics")
            content = response.text
            assert "tinyforge_http_requests_total" in content

    def test_middleware_normalizes_paths(self, client):
        """Test that middleware normalizes paths with IDs."""
        from services.dashboard_api.metrics import MetricsMiddleware

        middleware = MetricsMiddleware(app=None)

        # Test UUID normalization
        path = middleware._normalize_path("/api/jobs/12345678-1234-1234-1234-123456789012")
        assert "{id}" in path

        # Test numeric ID normalization
        path = middleware._normalize_path("/api/jobs/123")
        assert "{id}" in path

        # Test short ID normalization
        path = middleware._normalize_path("/api/models/abcd1234")
        assert "{id}" in path

    def test_middleware_preserves_normal_paths(self, client):
        """Test that middleware preserves paths without IDs."""
        from services.dashboard_api.metrics import MetricsMiddleware

        middleware = MetricsMiddleware(app=None)

        assert middleware._normalize_path("/health") == "/health"
        assert middleware._normalize_path("/api/jobs") == "/api/jobs"
        assert middleware._normalize_path("/api/stats") == "/api/stats"


class TestHealthCheckIncludesMetrics:
    """Test that health check includes metrics status."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from services.dashboard_api.main import app
        return TestClient(app)

    def test_health_check_includes_metrics_enabled(self, client):
        """Test health check response includes metrics_enabled field."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "metrics_enabled" in data
        assert isinstance(data["metrics_enabled"], bool)
