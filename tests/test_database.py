"""
Tests for TinyForgeAI Database Layer

Tests the SQLite database persistence functionality.
"""

import os
import pytest
from datetime import datetime

# Force database mode for these tests
os.environ["TINYFORGE_USE_DATABASE"] = "true"
os.environ["TINYFORGE_DATABASE_URL"] = "sqlite:///:memory:"

from services.dashboard_api.database import (
    Base,
    reset_engine,
    init_db,
    get_db_context,
    JobRepository,
    ServiceRepository,
    ModelRepository,
    LogRepository,
    JobModel,
    ServiceModel,
    ModelRegistryModel,
    LogModel,
)
from services.dashboard_api import database


@pytest.fixture(autouse=True)
def setup_database():
    """Set up fresh database for each test."""
    reset_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=database.engine)
    yield
    Base.metadata.drop_all(bind=database.engine)


class TestJobRepository:
    """Test JobRepository CRUD operations."""

    def test_create_job(self):
        """Test creating a new job."""
        with get_db_context() as db:
            job_data = {
                "id": "test-001",
                "name": "Test Training Job",
                "dataset_path": "data/test.jsonl",
                "model_name": "t5-small",
                "status": "pending",
                "config": {"epochs": 3, "batch_size": 4},
            }
            job = JobRepository.create(db, job_data)

            assert job.id == "test-001"
            assert job.name == "Test Training Job"
            assert job.status == "pending"
            assert job.progress == 0.0
            assert job.config["epochs"] == 3

    def test_get_job(self):
        """Test retrieving a job by ID."""
        with get_db_context() as db:
            job_data = {
                "id": "test-002",
                "name": "Get Test Job",
                "dataset_path": "data/test.jsonl",
                "model_name": "t5-small",
            }
            JobRepository.create(db, job_data)

            # Retrieve it
            job = JobRepository.get(db, "test-002")
            assert job is not None
            assert job.name == "Get Test Job"

            # Non-existent job
            missing = JobRepository.get(db, "nonexistent")
            assert missing is None

    def test_list_jobs(self):
        """Test listing jobs with filters."""
        with get_db_context() as db:
            # Create multiple jobs
            for i in range(5):
                status = "running" if i < 2 else "pending"
                JobRepository.create(db, {
                    "id": f"job-{i:03d}",
                    "name": f"Job {i}",
                    "dataset_path": "data/test.jsonl",
                    "model_name": "t5-small",
                    "status": status,
                })

            # List all
            all_jobs = JobRepository.list(db)
            assert len(all_jobs) == 5

            # Filter by status
            running = JobRepository.list(db, status="running")
            assert len(running) == 2

            pending = JobRepository.list(db, status="pending")
            assert len(pending) == 3

    def test_update_job(self):
        """Test updating a job."""
        with get_db_context() as db:
            JobRepository.create(db, {
                "id": "update-test",
                "name": "Update Test",
                "dataset_path": "data/test.jsonl",
                "model_name": "t5-small",
                "status": "pending",
            })

            # Update it
            updated = JobRepository.update(db, "update-test", {
                "status": "running",
                "progress": 50.0,
                "started_at": datetime.utcnow(),
            })

            assert updated.status == "running"
            assert updated.progress == 50.0
            assert updated.started_at is not None

    def test_delete_job(self):
        """Test deleting a job."""
        with get_db_context() as db:
            JobRepository.create(db, {
                "id": "delete-test",
                "name": "Delete Test",
                "dataset_path": "data/test.jsonl",
                "model_name": "t5-small",
            })

            # Delete it
            result = JobRepository.delete(db, "delete-test")
            assert result is True

            # Verify it's gone
            job = JobRepository.get(db, "delete-test")
            assert job is None

            # Delete non-existent
            result = JobRepository.delete(db, "nonexistent")
            assert result is False

    def test_count_jobs(self):
        """Test counting jobs."""
        with get_db_context() as db:
            for i in range(3):
                JobRepository.create(db, {
                    "id": f"count-{i}",
                    "name": f"Count Job {i}",
                    "dataset_path": "data/test.jsonl",
                    "model_name": "t5-small",
                    "status": "completed" if i == 0 else "pending",
                })

            total = JobRepository.count(db)
            assert total == 3

            completed = JobRepository.count(db, status="completed")
            assert completed == 1


class TestServiceRepository:
    """Test ServiceRepository CRUD operations."""

    def test_create_service(self):
        """Test creating a new service."""
        with get_db_context() as db:
            service_data = {
                "id": "svc-001",
                "name": "Test Service",
                "model_path": "models/test",
                "status": "stopped",
                "port": 8000,
                "replicas": 1,
            }
            service = ServiceRepository.create(db, service_data)

            assert service.id == "svc-001"
            assert service.name == "Test Service"
            assert service.port == 8000

    def test_service_lifecycle(self):
        """Test service start/stop operations."""
        with get_db_context() as db:
            ServiceRepository.create(db, {
                "id": "lifecycle-test",
                "name": "Lifecycle Service",
                "model_path": "models/test",
                "status": "stopped",
            })

            # Start service
            started = ServiceRepository.update(db, "lifecycle-test", {
                "status": "running",
                "endpoint": "http://localhost:8000",
                "started_at": datetime.utcnow(),
            })
            assert started.status == "running"
            assert started.endpoint == "http://localhost:8000"

            # Stop service
            stopped = ServiceRepository.update(db, "lifecycle-test", {
                "status": "stopped",
                "endpoint": None,
            })
            assert stopped.status == "stopped"
            assert stopped.endpoint is None


class TestLogRepository:
    """Test LogRepository operations."""

    def test_create_log(self):
        """Test creating log entries."""
        with get_db_context() as db:
            log = LogRepository.create(db, "INFO", "test", "Test message")

            assert log.level == "INFO"
            assert log.source == "test"
            assert log.message == "Test message"
            assert log.timestamp is not None

    def test_list_logs_with_filters(self):
        """Test listing logs with filters."""
        with get_db_context() as db:
            LogRepository.create(db, "INFO", "jobs", "Job created")
            LogRepository.create(db, "ERROR", "jobs", "Job failed")
            LogRepository.create(db, "INFO", "services", "Service started")

            # All logs
            all_logs = LogRepository.list(db)
            assert len(all_logs) == 3

            # Filter by level
            errors = LogRepository.list(db, level="ERROR")
            assert len(errors) == 1

            # Filter by source
            job_logs = LogRepository.list(db, source="jobs")
            assert len(job_logs) == 2


class TestModelRepository:
    """Test ModelRepository operations."""

    def test_create_model(self):
        """Test registering a new model."""
        with get_db_context() as db:
            model_data = {
                "id": "model-001",
                "name": "Test Model",
                "description": "A test model",
                "model_type": "seq2seq",
                "base_model": "t5-small",
                "path": "models/test-model",
                "size_bytes": 1024000,
            }
            model = ModelRepository.create(db, model_data)

            assert model.id == "model-001"
            assert model.name == "Test Model"
            assert model.size_bytes == 1024000

    def test_model_to_dict(self):
        """Test model serialization with size formatting."""
        with get_db_context() as db:
            model = ModelRepository.create(db, {
                "id": "dict-test",
                "name": "Dict Test",
                "path": "models/dict-test",
                "size_bytes": 1536000,  # ~1.5 MB
            })

            data = model.to_dict()
            assert data["id"] == "dict-test"
            assert "MB" in data["size"]  # Size should be formatted


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_job_to_dict(self):
        """Test job model serialization."""
        with get_db_context() as db:
            job = JobRepository.create(db, {
                "id": "dict-test",
                "name": "Dict Test Job",
                "dataset_path": "data/test.jsonl",
                "model_name": "t5-small",
                "config": {"epochs": 5},
            })

            data = job.to_dict()
            assert data["id"] == "dict-test"
            assert data["name"] == "Dict Test Job"
            assert data["config"]["epochs"] == 5
            assert "created_at" in data

    def test_concurrent_operations(self):
        """Test multiple operations in sequence."""
        with get_db_context() as db:
            # Create job
            job = JobRepository.create(db, {
                "id": "concurrent-test",
                "name": "Concurrent Test",
                "dataset_path": "data/test.jsonl",
                "model_name": "t5-small",
                "status": "pending",
            })

            # Create service
            service = ServiceRepository.create(db, {
                "id": "concurrent-svc",
                "name": "Concurrent Service",
                "model_path": "models/test",
            })

            # Add logs
            LogRepository.create(db, "INFO", "test", "Job created")
            LogRepository.create(db, "INFO", "test", "Service created")

            # Verify counts
            assert JobRepository.count(db) == 1
            assert ServiceRepository.count(db) == 1
            assert len(LogRepository.list(db)) == 2
