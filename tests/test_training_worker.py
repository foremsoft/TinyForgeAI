"""
Tests for TinyForgeAI Training Worker

Tests the training worker service components.
"""

import os
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from services.training_worker.worker import (
    WorkerConfig,
    ProgressCallback,
    TrainingExecutor,
    TrainingWorker,
)


class TestWorkerConfig:
    """Test WorkerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorkerConfig()
        assert config.api_url == "http://localhost:8001"
        assert config.poll_interval == 5
        assert config.output_base_dir == "./output"
        assert config.max_concurrent_jobs == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkerConfig(
            api_url="http://example.com:9000",
            poll_interval=10,
            output_base_dir="/custom/path",
            max_concurrent_jobs=2,
        )
        assert config.api_url == "http://example.com:9000"
        assert config.poll_interval == 10
        assert config.output_base_dir == "/custom/path"
        assert config.max_concurrent_jobs == 2


class TestProgressCallback:
    """Test ProgressCallback class."""

    def test_progress_callback_creation(self):
        """Test creating a progress callback."""
        callback = ProgressCallback(
            job_id="test-123",
            api_url="http://localhost:8001"
        )
        assert callback.job_id == "test-123"
        assert callback.api_url == "http://localhost:8001"
        callback.close()

    @patch("httpx.Client")
    def test_progress_reporting(self, mock_client_class):
        """Test progress is reported via HTTP."""
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(status_code=200)
        mock_client_class.return_value = mock_client

        callback = ProgressCallback(
            job_id="test-456",
            api_url="http://localhost:8001"
        )

        # Report progress
        callback(50.0, "Halfway there")

        # Verify HTTP call was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "test-456" in call_args[0][0]
        assert call_args[1]["params"]["progress"] == 50.0

        callback.close()

    @patch("httpx.Client")
    def test_failure_reporting(self, mock_client_class):
        """Test failure is reported via HTTP."""
        mock_client = MagicMock()
        mock_client.post.return_value = MagicMock(status_code=200)
        mock_client_class.return_value = mock_client

        callback = ProgressCallback(
            job_id="test-789",
            api_url="http://localhost:8001"
        )

        callback.report_failure("Something went wrong")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "fail" in call_args[0][0]
        assert call_args[1]["params"]["error_message"] == "Something went wrong"

        callback.close()


class TestTrainingExecutor:
    """Test TrainingExecutor class."""

    def test_executor_creation(self, tmp_path):
        """Test creating a training executor."""
        executor = TrainingExecutor(output_base_dir=str(tmp_path))
        assert executor.output_base_dir == tmp_path

    def test_output_directory_created(self, tmp_path):
        """Test output directory is created."""
        output_path = tmp_path / "new_output"
        executor = TrainingExecutor(output_base_dir=str(output_path))
        assert output_path.exists()


class TestTrainingWorker:
    """Test TrainingWorker class."""

    def test_worker_creation(self):
        """Test creating a training worker."""
        config = WorkerConfig(
            api_url="http://localhost:8001",
            poll_interval=10,
        )
        worker = TrainingWorker(config)
        assert worker.config == config
        worker.client.close()

    @patch("httpx.Client")
    def test_get_pending_jobs_success(self, mock_client_class):
        """Test fetching pending jobs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "job-1", "name": "Test Job 1"},
            {"id": "job-2", "name": "Test Job 2"},
        ]
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        worker = TrainingWorker()
        jobs = worker.get_pending_jobs()

        assert len(jobs) == 2
        assert jobs[0]["id"] == "job-1"

    @patch("httpx.Client")
    def test_get_pending_jobs_error(self, mock_client_class):
        """Test handling API errors when fetching jobs."""
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        worker = TrainingWorker()
        jobs = worker.get_pending_jobs()

        assert jobs == []


class TestWorkerIntegration:
    """Integration tests for the training worker."""

    def test_worker_start_stop(self):
        """Test worker can start and stop cleanly."""
        config = WorkerConfig(
            api_url="http://localhost:8001",
            poll_interval=1,
        )
        worker = TrainingWorker(config)

        # Start in background
        worker.start(blocking=False)

        # Give it a moment
        import time
        time.sleep(0.5)

        # Stop
        worker.stop()

        # Verify stopped
        assert worker._stop_event.is_set()

    @patch("httpx.Client")
    def test_worker_processes_job(self, mock_client_class):
        """Test worker processes a job."""
        mock_client = MagicMock()

        # Mock get pending jobs - return one job, then empty
        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            mock_response.status_code = 200
            if call_count[0] == 1:
                mock_response.json.return_value = [{
                    "id": "test-job",
                    "name": "Test Job",
                    "dataset_path": "data/test.jsonl",
                    "model_name": "t5-small",
                    "config": {"epochs": 1},
                }]
            else:
                mock_response.json.return_value = []
            return mock_response

        mock_client.get = mock_get
        mock_client.post.return_value = MagicMock(status_code=200)
        mock_client_class.return_value = mock_client

        # Create worker with mocked executor
        config = WorkerConfig(poll_interval=1)
        worker = TrainingWorker(config)

        # Mock executor to avoid actual training
        mock_executor = MagicMock()
        mock_executor.execute.return_value = {"status": "completed"}
        worker.executor = mock_executor

        # Get and process job
        jobs = worker.get_pending_jobs()
        assert len(jobs) == 1

        # Process would call executor
        worker.process_job(jobs[0])

        # Verify executor was called
        mock_executor.execute.assert_called_once()
