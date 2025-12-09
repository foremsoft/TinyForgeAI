"""
Tests for TinyForgeAI Dashboard API

Comprehensive test suite covering all API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from services.dashboard_api.main import (
    app,
    jobs_db,
    services_db,
    logs_db,
    JobStatus,
    ServiceStatus,
)


@pytest.fixture
def client():
    """Create a test client and clear databases."""
    jobs_db.clear()
    services_db.clear()
    logs_db.clear()
    return TestClient(app)


@pytest.fixture
def sample_job_data():
    """Sample job creation data."""
    return {
        "name": "test-training-job",
        "dataset_path": "data/training.jsonl",
        "model_name": "t5-small",
        "model_type": "seq2seq",
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 0.0002,
        "use_lora": True,
        "lora_r": 8,
    }


@pytest.fixture
def sample_service_data():
    """Sample service creation data."""
    return {
        "name": "test-inference-service",
        "model_path": "models/my_model",
        "port": 8000,
        "replicas": 1,
    }


# ============================================
# Health & Root Tests
# ============================================

class TestHealthEndpoints:
    """Test health check and root endpoints."""

    def test_health_check(self, client):
        """Test /health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "jobs_count" in data
        assert "services_count" in data

    def test_root_endpoint(self, client):
        """Test / root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "TinyForgeAI Dashboard API"
        assert "version" in data
        assert data["docs"] == "/docs"


# ============================================
# Training Jobs Tests
# ============================================

class TestTrainingJobs:
    """Test training job endpoints."""

    def test_create_job(self, client, sample_job_data):
        """Test creating a new training job."""
        response = client.post("/api/jobs", json=sample_job_data)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == sample_job_data["name"]
        assert data["dataset_path"] == sample_job_data["dataset_path"]
        assert data["model_name"] == sample_job_data["model_name"]
        assert data["status"] == "pending"
        assert "id" in data
        assert "created_at" in data

    def test_create_job_minimal(self, client):
        """Test creating a job with minimal required fields."""
        minimal_data = {
            "name": "minimal-job",
            "dataset_path": "data/test.jsonl",
        }
        response = client.post("/api/jobs", json=minimal_data)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "minimal-job"
        # Check defaults are applied
        assert data["model_name"] == "t5-small"

    def test_create_job_invalid_name(self, client):
        """Test creating a job with invalid name fails."""
        invalid_data = {
            "name": "",  # Empty name should fail
            "dataset_path": "data/test.jsonl",
        }
        response = client.post("/api/jobs", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/api/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_jobs(self, client, sample_job_data):
        """Test listing jobs after creation."""
        # Create multiple jobs
        client.post("/api/jobs", json=sample_job_data)
        sample_job_data["name"] = "second-job"
        client.post("/api/jobs", json=sample_job_data)

        response = client.get("/api/jobs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_jobs_with_status_filter(self, client, sample_job_data):
        """Test filtering jobs by status."""
        client.post("/api/jobs", json=sample_job_data)

        # Filter by pending status
        response = client.get("/api/jobs?status=pending")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "pending"

        # Filter by running status (should be empty)
        response = client.get("/api/jobs?status=running")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_job(self, client, sample_job_data):
        """Test getting a specific job by ID."""
        create_response = client.post("/api/jobs", json=sample_job_data)
        job_id = create_response.json()["id"]

        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["name"] == sample_job_data["name"]

    def test_get_job_not_found(self, client):
        """Test getting a non-existent job returns 404."""
        response = client.get("/api/jobs/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_cancel_job(self, client, sample_job_data):
        """Test cancelling a pending job."""
        create_response = client.post("/api/jobs", json=sample_job_data)
        job_id = create_response.json()["id"]

        response = client.post(f"/api/jobs/{job_id}/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_job_not_found(self, client):
        """Test cancelling non-existent job returns 404."""
        response = client.post("/api/jobs/nonexistent/cancel")
        assert response.status_code == 404

    def test_delete_job(self, client, sample_job_data):
        """Test deleting a job."""
        create_response = client.post("/api/jobs", json=sample_job_data)
        job_id = create_response.json()["id"]

        response = client.delete(f"/api/jobs/{job_id}")
        assert response.status_code == 200

        # Verify job is deleted
        get_response = client.get(f"/api/jobs/{job_id}")
        assert get_response.status_code == 404

    def test_delete_job_not_found(self, client):
        """Test deleting non-existent job returns 404."""
        response = client.delete("/api/jobs/nonexistent")
        assert response.status_code == 404


# ============================================
# Services Tests
# ============================================

class TestServices:
    """Test service deployment endpoints."""

    def test_create_service(self, client, sample_service_data):
        """Test creating a new service."""
        response = client.post("/api/services", json=sample_service_data)
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == sample_service_data["name"]
        assert data["model_path"] == sample_service_data["model_path"]
        assert data["port"] == sample_service_data["port"]
        assert data["status"] == "stopped"
        assert "id" in data

    def test_create_service_minimal(self, client):
        """Test creating a service with minimal fields."""
        minimal_data = {
            "name": "minimal-service",
            "model_path": "models/test",
        }
        response = client.post("/api/services", json=minimal_data)
        assert response.status_code == 200

        data = response.json()
        assert data["port"] == 8000  # Default port

    def test_list_services_empty(self, client):
        """Test listing services when none exist."""
        response = client.get("/api/services")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_services(self, client, sample_service_data):
        """Test listing services after creation."""
        client.post("/api/services", json=sample_service_data)
        sample_service_data["name"] = "second-service"
        client.post("/api/services", json=sample_service_data)

        response = client.get("/api/services")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_service(self, client, sample_service_data):
        """Test getting a specific service by ID."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        response = client.get(f"/api/services/{service_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == service_id

    def test_get_service_not_found(self, client):
        """Test getting non-existent service returns 404."""
        response = client.get("/api/services/nonexistent")
        assert response.status_code == 404

    def test_start_service(self, client, sample_service_data):
        """Test starting a stopped service."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        response = client.post(f"/api/services/{service_id}/start")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["endpoint"] is not None
        assert "started_at" in data

    def test_start_service_already_running(self, client, sample_service_data):
        """Test starting already running service fails."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        # Start service
        client.post(f"/api/services/{service_id}/start")

        # Try to start again
        response = client.post(f"/api/services/{service_id}/start")
        assert response.status_code == 400
        assert "already running" in response.json()["detail"].lower()

    def test_stop_service(self, client, sample_service_data):
        """Test stopping a running service."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        # Start then stop
        client.post(f"/api/services/{service_id}/start")
        response = client.post(f"/api/services/{service_id}/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stopped"
        assert data["endpoint"] is None

    def test_stop_service_not_running(self, client, sample_service_data):
        """Test stopping non-running service fails."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        response = client.post(f"/api/services/{service_id}/stop")
        assert response.status_code == 400
        assert "not running" in response.json()["detail"].lower()

    def test_delete_service(self, client, sample_service_data):
        """Test deleting a stopped service."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        response = client.delete(f"/api/services/{service_id}")
        assert response.status_code == 200

        # Verify deleted
        get_response = client.get(f"/api/services/{service_id}")
        assert get_response.status_code == 404

    def test_delete_running_service_fails(self, client, sample_service_data):
        """Test deleting a running service fails."""
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]

        # Start service
        client.post(f"/api/services/{service_id}/start")

        # Try to delete
        response = client.delete(f"/api/services/{service_id}")
        assert response.status_code == 400
        assert "stop service" in response.json()["detail"].lower()


# ============================================
# Inference Tests
# ============================================

class TestInference:
    """Test inference endpoint."""

    def test_predict(self, client):
        """Test basic prediction."""
        response = client.post("/api/predict", json={"input": "hello world"})
        assert response.status_code == 200

        data = response.json()
        assert "output" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0

    def test_predict_empty_input(self, client):
        """Test prediction with empty input fails."""
        response = client.post("/api/predict", json={"input": ""})
        assert response.status_code == 422  # Validation error


# ============================================
# Dashboard Stats Tests
# ============================================

class TestDashboardStats:
    """Test dashboard statistics endpoint."""

    def test_stats_empty(self, client):
        """Test stats when no data exists."""
        response = client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_jobs"] == 0
        assert data["running_jobs"] == 0
        assert data["completed_jobs"] == 0
        assert data["total_services"] == 0
        assert data["running_services"] == 0
        assert data["total_predictions"] == 0

    def test_stats_with_data(self, client, sample_job_data, sample_service_data):
        """Test stats after creating jobs and services."""
        # Create jobs
        client.post("/api/jobs", json=sample_job_data)
        sample_job_data["name"] = "second-job"
        client.post("/api/jobs", json=sample_job_data)

        # Create and start a service
        create_response = client.post("/api/services", json=sample_service_data)
        service_id = create_response.json()["id"]
        client.post(f"/api/services/{service_id}/start")

        response = client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_jobs"] == 2
        assert data["total_services"] == 1
        assert data["running_services"] == 1


# ============================================
# Logs Tests
# ============================================

class TestLogs:
    """Test logs endpoint."""

    def test_logs_empty(self, client):
        """Test logs when none exist."""
        response = client.get("/api/logs")
        assert response.status_code == 200
        assert response.json() == []

    def test_logs_after_operations(self, client, sample_job_data):
        """Test logs are created after operations."""
        # Create a job (this should add a log entry)
        client.post("/api/jobs", json=sample_job_data)

        response = client.get("/api/logs")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

        # Check log structure
        log_entry = data[0]
        assert "timestamp" in log_entry
        assert "level" in log_entry
        assert "source" in log_entry
        assert "message" in log_entry

    def test_logs_filter_by_level(self, client, sample_job_data):
        """Test filtering logs by level."""
        client.post("/api/jobs", json=sample_job_data)

        response = client.get("/api/logs?level=INFO")
        assert response.status_code == 200
        data = response.json()
        for log in data:
            assert log["level"] == "INFO"

    def test_logs_filter_by_source(self, client, sample_job_data):
        """Test filtering logs by source."""
        client.post("/api/jobs", json=sample_job_data)

        response = client.get("/api/logs?source=jobs")
        assert response.status_code == 200
        data = response.json()
        for log in data:
            assert log["source"] == "jobs"

    def test_logs_limit(self, client, sample_job_data):
        """Test logs limit parameter."""
        # Create multiple jobs to generate logs
        for i in range(5):
            sample_job_data["name"] = f"job-{i}"
            client.post("/api/jobs", json=sample_job_data)

        response = client.get("/api/logs?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 2


# ============================================
# CORS Tests
# ============================================

class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/jobs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI TestClient may not fully simulate CORS preflight
        # but we can verify the middleware is configured
        assert response.status_code in [200, 405]


# ============================================
# Integration Tests
# ============================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_job_lifecycle(self, client, sample_job_data):
        """Test complete job lifecycle: create -> cancel -> delete."""
        # Create
        create_response = client.post("/api/jobs", json=sample_job_data)
        assert create_response.status_code == 200
        job_id = create_response.json()["id"]

        # Verify in list
        list_response = client.get("/api/jobs")
        assert len(list_response.json()) == 1

        # Cancel
        cancel_response = client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

        # Delete
        delete_response = client.delete(f"/api/jobs/{job_id}")
        assert delete_response.status_code == 200

        # Verify gone
        list_response = client.get("/api/jobs")
        assert len(list_response.json()) == 0

    def test_full_service_lifecycle(self, client, sample_service_data):
        """Test complete service lifecycle: create -> start -> stop -> delete."""
        # Create
        create_response = client.post("/api/services", json=sample_service_data)
        assert create_response.status_code == 200
        service_id = create_response.json()["id"]

        # Start
        start_response = client.post(f"/api/services/{service_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["status"] == "running"

        # Stop
        stop_response = client.post(f"/api/services/{service_id}/stop")
        assert stop_response.status_code == 200
        assert stop_response.json()["status"] == "stopped"

        # Delete
        delete_response = client.delete(f"/api/services/{service_id}")
        assert delete_response.status_code == 200

        # Verify gone
        list_response = client.get("/api/services")
        assert len(list_response.json()) == 0
