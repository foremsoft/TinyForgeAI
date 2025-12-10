"""Tests for custom exception hierarchy."""

import pytest
from backend.exceptions import (
    TinyForgeError,
    DataValidationError,
    DatasetError,
    EmptyDatasetError,
    TrainingError,
    ModelNotFoundError,
    ModelLoadError,
    TrainingConfigError,
    CheckpointError,
    GPUError,
    InferenceError,
    ModelNotLoadedError,
    InferenceTimeoutError,
    BatchSizeError,
    ConnectorError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    APIError,
    DatabaseError,
    FileConnectorError,
    ExportError,
    ONNXExportError,
    QuantizationError,
    ServiceError,
    TenantNotFoundError,
    TenantQuotaExceededError,
    InvalidAPIKeyError,
    JobNotFoundError,
    JobAlreadyExistsError,
    VersionError,
    VersionNotFoundError,
    VersionConflictError,
)


class TestBaseException:
    """Tests for base TinyForgeError."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        error = TinyForgeError("Test error")
        assert str(error) == "Test error"
        assert error.code == "TINYFORGE_ERROR"
        assert error.details == {}

    def test_with_code(self):
        """Test exception with custom code."""
        error = TinyForgeError("Test", code="CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"

    def test_with_details(self):
        """Test exception with details."""
        details = {"key": "value", "count": 42}
        error = TinyForgeError("Test", details=details)
        assert error.details == details

    def test_to_dict(self):
        """Test conversion to dict."""
        error = TinyForgeError("Test error", code="TEST", details={"x": 1})
        d = error.to_dict()
        assert d["message"] == "Test error"
        assert d["code"] == "TEST"
        assert d["details"] == {"x": 1}

    def test_inheritance(self):
        """Test that all exceptions inherit from TinyForgeError."""
        exceptions = [
            DataValidationError,
            DatasetError,
            TrainingError,
            InferenceError,
            ConnectorError,
            ExportError,
            ServiceError,
            VersionError,
        ]
        for exc_class in exceptions:
            error = exc_class("test")
            assert isinstance(error, TinyForgeError)
            assert isinstance(error, Exception)


class TestDataExceptions:
    """Tests for data-related exceptions."""

    def test_data_validation_error(self):
        """Test DataValidationError."""
        error = DataValidationError("Invalid format")
        assert error.code == "DATA_VALIDATION_ERROR"
        assert "Invalid format" in str(error)

    def test_dataset_error(self):
        """Test DatasetError."""
        error = DatasetError("Failed to load")
        assert error.code == "DATASET_ERROR"

    def test_empty_dataset_error(self):
        """Test EmptyDatasetError."""
        error = EmptyDatasetError("No samples")
        assert error.code == "EMPTY_DATASET"
        assert isinstance(error, DatasetError)


class TestTrainingExceptions:
    """Tests for training-related exceptions."""

    def test_training_error(self):
        """Test TrainingError."""
        error = TrainingError("Training failed")
        assert error.code == "TRAINING_ERROR"

    def test_model_not_found(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError("gpt-99")
        assert error.code == "MODEL_NOT_FOUND"
        assert "gpt-99" in str(error)

    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError("OOM")
        assert error.code == "MODEL_LOAD_ERROR"
        assert isinstance(error, TrainingError)

    def test_training_config_error(self):
        """Test TrainingConfigError."""
        error = TrainingConfigError("Invalid batch size")
        assert error.code == "TRAINING_CONFIG_ERROR"

    def test_checkpoint_error(self):
        """Test CheckpointError."""
        error = CheckpointError("Corrupt checkpoint")
        assert error.code == "CHECKPOINT_ERROR"

    def test_gpu_error(self):
        """Test GPUError."""
        error = GPUError("CUDA OOM")
        assert error.code == "GPU_ERROR"


class TestInferenceExceptions:
    """Tests for inference-related exceptions."""

    def test_inference_error(self):
        """Test InferenceError."""
        error = InferenceError("Prediction failed")
        assert error.code == "INFERENCE_ERROR"

    def test_model_not_loaded(self):
        """Test ModelNotLoadedError."""
        error = ModelNotLoadedError()
        assert error.code == "MODEL_NOT_LOADED"
        assert "not loaded" in str(error).lower()

    def test_inference_timeout(self):
        """Test InferenceTimeoutError."""
        error = InferenceTimeoutError("30s exceeded")
        assert error.code == "INFERENCE_TIMEOUT"

    def test_batch_size_error(self):
        """Test BatchSizeError."""
        error = BatchSizeError("Too large")
        assert error.code == "BATCH_SIZE_ERROR"


class TestConnectorExceptions:
    """Tests for connector-related exceptions."""

    def test_connector_error(self):
        """Test ConnectorError."""
        error = ConnectorError("Connection failed")
        assert error.code == "CONNECTOR_ERROR"

    def test_connection_error(self):
        """Test ConnectionError."""
        error = ConnectionError("Network unreachable")
        assert error.code == "CONNECTION_ERROR"
        assert isinstance(error, ConnectorError)

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid token")
        assert error.code == "AUTHENTICATION_ERROR"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Too many requests")
        assert error.code == "RATE_LIMIT_ERROR"

    def test_api_error(self):
        """Test APIError."""
        error = APIError("500 Server Error")
        assert error.code == "API_ERROR"

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Query failed")
        assert error.code == "DATABASE_ERROR"

    def test_file_connector_error(self):
        """Test FileConnectorError."""
        error = FileConnectorError("File not found")
        assert error.code == "FILE_CONNECTOR_ERROR"


class TestExportExceptions:
    """Tests for export-related exceptions."""

    def test_export_error(self):
        """Test ExportError."""
        error = ExportError("Export failed")
        assert error.code == "EXPORT_ERROR"

    def test_onnx_export_error(self):
        """Test ONNXExportError."""
        error = ONNXExportError("Invalid model graph")
        assert error.code == "ONNX_EXPORT_ERROR"
        assert isinstance(error, ExportError)

    def test_quantization_error(self):
        """Test QuantizationError."""
        error = QuantizationError("Unsupported op")
        assert error.code == "QUANTIZATION_ERROR"
        assert isinstance(error, ExportError)


class TestServiceExceptions:
    """Tests for service-related exceptions."""

    def test_service_error(self):
        """Test ServiceError."""
        error = ServiceError("Service unavailable")
        assert error.code == "SERVICE_ERROR"

    def test_tenant_not_found(self):
        """Test TenantNotFoundError."""
        error = TenantNotFoundError("tenant-123")
        assert error.code == "TENANT_NOT_FOUND"
        assert "tenant-123" in str(error)

    def test_tenant_quota_exceeded(self):
        """Test TenantQuotaExceededError."""
        error = TenantQuotaExceededError("Rate limit")
        assert error.code == "TENANT_QUOTA_EXCEEDED"

    def test_invalid_api_key(self):
        """Test InvalidAPIKeyError."""
        error = InvalidAPIKeyError()
        assert error.code == "INVALID_API_KEY"

    def test_job_not_found(self):
        """Test JobNotFoundError."""
        error = JobNotFoundError("job-456")
        assert error.code == "JOB_NOT_FOUND"

    def test_job_already_exists(self):
        """Test JobAlreadyExistsError."""
        error = JobAlreadyExistsError("job-456")
        assert error.code == "JOB_ALREADY_EXISTS"


class TestVersionExceptions:
    """Tests for version-related exceptions."""

    def test_version_error(self):
        """Test VersionError."""
        error = VersionError("Version mismatch")
        assert error.code == "VERSION_ERROR"

    def test_version_not_found(self):
        """Test VersionNotFoundError."""
        error = VersionNotFoundError("v2.0")
        assert error.code == "VERSION_NOT_FOUND"

    def test_version_conflict(self):
        """Test VersionConflictError."""
        error = VersionConflictError("Concurrent update")
        assert error.code == "VERSION_CONFLICT"


class TestExceptionDetails:
    """Tests for exception details handling."""

    def test_details_preserved(self):
        """Test that details are preserved correctly."""
        details = {
            "field": "email",
            "value": "invalid",
            "expected": "valid email format",
        }
        error = DataValidationError("Validation failed", details=details)
        assert error.details["field"] == "email"
        assert error.details["value"] == "invalid"

    def test_to_dict_complete(self):
        """Test complete to_dict output."""
        error = TrainingError(
            "Training failed at epoch 5",
            code="TRAINING_ERROR",
            details={"epoch": 5, "loss": 2.5},
        )
        d = error.to_dict()
        assert "message" in d
        assert "code" in d
        assert "details" in d
        assert d["details"]["epoch"] == 5

    def test_raise_and_catch(self):
        """Test raising and catching exceptions."""
        with pytest.raises(TinyForgeError) as exc_info:
            raise ModelNotFoundError("test-model")

        assert exc_info.value.code == "MODEL_NOT_FOUND"
        assert "test-model" in str(exc_info.value)

    def test_catch_by_parent(self):
        """Test catching by parent exception type."""
        with pytest.raises(TrainingError):
            raise ModelLoadError("Load failed")

        with pytest.raises(ConnectorError):
            raise DatabaseError("DB error")

        with pytest.raises(ExportError):
            raise ONNXExportError("ONNX error")
