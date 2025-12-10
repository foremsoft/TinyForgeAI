"""
Custom exception hierarchy for TinyForgeAI.

Provides structured error handling with specific exception types for
different failure modes across the application.
"""

from typing import Any, Dict, Optional


class TinyForgeError(Exception):
    """
    Base exception for all TinyForgeAI errors.

    All custom exceptions should inherit from this class to enable
    consistent error handling across the application.

    Attributes:
        message: Human-readable error description.
        code: Machine-readable error code for programmatic handling.
        details: Additional context about the error.
    """

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# =============================================================================
# Data Validation Errors
# =============================================================================


class DataValidationError(TinyForgeError):
    """Raised when input data fails validation."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = repr(value)
        super().__init__(message, "DATA_VALIDATION_ERROR", details)


class DatasetError(TinyForgeError):
    """Raised when dataset loading or processing fails."""

    def __init__(
        self,
        message: str,
        dataset_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if dataset_path:
            details["dataset_path"] = dataset_path
        super().__init__(message, "DATASET_ERROR", details)


class EmptyDatasetError(DatasetError):
    """Raised when a dataset contains no valid samples."""

    def __init__(self, dataset_path: Optional[str] = None):
        super().__init__(
            "Dataset is empty or contains no valid samples",
            dataset_path=dataset_path,
        )


# =============================================================================
# Training Errors
# =============================================================================


class TrainingError(TinyForgeError):
    """Base exception for training-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "TRAINING_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class ModelNotFoundError(TrainingError):
    """Raised when a requested model cannot be found."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model not found: {model_name}",
            "MODEL_NOT_FOUND",
            {"model_name": model_name},
        )


class ModelLoadError(TrainingError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, reason: Optional[str] = None):
        message = f"Failed to load model: {model_name}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message,
            "MODEL_LOAD_ERROR",
            {"model_name": model_name, "reason": reason},
        )


class TrainingConfigError(TrainingError):
    """Raised when training configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, "TRAINING_CONFIG_ERROR", details)


class CheckpointError(TrainingError):
    """Raised when checkpoint save/load fails."""

    def __init__(self, message: str, checkpoint_path: Optional[str] = None):
        details = {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        super().__init__(message, "CHECKPOINT_ERROR", details)


class GPUError(TrainingError):
    """Raised when GPU-related operations fail."""

    def __init__(self, message: str):
        super().__init__(message, "GPU_ERROR")


# =============================================================================
# Inference Errors
# =============================================================================


class InferenceError(TinyForgeError):
    """Base exception for inference-related errors."""

    def __init__(
        self,
        message: str,
        code: str = "INFERENCE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class ModelNotLoadedError(InferenceError):
    """Raised when inference is attempted on an unloaded model."""

    def __init__(self, model_name: Optional[str] = None):
        message = "Model not loaded"
        details = {}
        if model_name:
            message = f"Model not loaded: {model_name}"
            details["model_name"] = model_name
        super().__init__(message, "MODEL_NOT_LOADED", details)


class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds time limit."""

    def __init__(self, timeout_seconds: float):
        super().__init__(
            f"Inference timed out after {timeout_seconds} seconds",
            "INFERENCE_TIMEOUT",
            {"timeout_seconds": timeout_seconds},
        )


class BatchSizeError(InferenceError):
    """Raised when batch size is invalid or exceeds limits."""

    def __init__(self, batch_size: int, max_batch_size: int):
        super().__init__(
            f"Batch size {batch_size} exceeds maximum {max_batch_size}",
            "BATCH_SIZE_ERROR",
            {"batch_size": batch_size, "max_batch_size": max_batch_size},
        )


# =============================================================================
# Connector Errors
# =============================================================================


class ConnectorError(TinyForgeError):
    """Base exception for data connector errors."""

    def __init__(
        self,
        message: str,
        code: str = "CONNECTOR_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class ConnectionError(ConnectorError):
    """Raised when connection to external service fails."""

    def __init__(self, service: str, reason: Optional[str] = None):
        message = f"Failed to connect to {service}"
        if reason:
            message += f": {reason}"
        super().__init__(
            message,
            "CONNECTION_ERROR",
            {"service": service, "reason": reason},
        )


class AuthenticationError(ConnectorError):
    """Raised when authentication fails."""

    def __init__(self, service: str, reason: Optional[str] = None):
        message = f"Authentication failed for {service}"
        if reason:
            message += f": {reason}"
        super().__init__(
            message,
            "AUTHENTICATION_ERROR",
            {"service": service, "reason": reason},
        )


class RateLimitError(ConnectorError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        service: str,
        retry_after: Optional[int] = None,
    ):
        message = f"Rate limit exceeded for {service}"
        details: Dict[str, Any] = {"service": service}
        if retry_after:
            message += f", retry after {retry_after} seconds"
            details["retry_after"] = retry_after
        super().__init__(message, "RATE_LIMIT_ERROR", details)


class APIError(ConnectorError):
    """Raised when an API request fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        details: Dict[str, Any] = {}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]  # Truncate
        super().__init__(message, "API_ERROR", details)


class DatabaseError(ConnectorError):
    """Raised when database operations fail."""

    def __init__(self, message: str, query: Optional[str] = None):
        details = {}
        if query:
            details["query"] = query[:200]  # Truncate for safety
        super().__init__(message, "DATABASE_ERROR", details)


class FileConnectorError(ConnectorError):
    """Raised when file operations fail."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, "FILE_CONNECTOR_ERROR", details)


# =============================================================================
# Export Errors
# =============================================================================


class ExportError(TinyForgeError):
    """Base exception for model export errors."""

    def __init__(
        self,
        message: str,
        code: str = "EXPORT_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class ONNXExportError(ExportError):
    """Raised when ONNX export fails."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        details = {}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message, "ONNX_EXPORT_ERROR", details)


class QuantizationError(ExportError):
    """Raised when model quantization fails."""

    def __init__(self, message: str, quantization_type: Optional[str] = None):
        details = {}
        if quantization_type:
            details["quantization_type"] = quantization_type
        super().__init__(message, "QUANTIZATION_ERROR", details)


# =============================================================================
# API/Service Errors
# =============================================================================


class ServiceError(TinyForgeError):
    """Base exception for service-level errors."""

    def __init__(
        self,
        message: str,
        code: str = "SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class TenantNotFoundError(ServiceError):
    """Raised when a tenant cannot be found."""

    def __init__(self, tenant_id: str):
        super().__init__(
            f"Tenant not found: {tenant_id}",
            "TENANT_NOT_FOUND",
            {"tenant_id": tenant_id},
        )


class TenantQuotaExceededError(ServiceError):
    """Raised when tenant exceeds their quota."""

    def __init__(
        self,
        tenant_id: str,
        quota_type: str,
        limit: int,
        current: int,
    ):
        super().__init__(
            f"Quota exceeded for tenant {tenant_id}: {quota_type}",
            "QUOTA_EXCEEDED",
            {
                "tenant_id": tenant_id,
                "quota_type": quota_type,
                "limit": limit,
                "current": current,
            },
        )


class InvalidAPIKeyError(ServiceError):
    """Raised when API key is invalid or expired."""

    def __init__(self, reason: str = "Invalid or expired API key"):
        super().__init__(reason, "INVALID_API_KEY")


class JobNotFoundError(ServiceError):
    """Raised when a job cannot be found."""

    def __init__(self, job_id: str):
        super().__init__(
            f"Job not found: {job_id}",
            "JOB_NOT_FOUND",
            {"job_id": job_id},
        )


class JobAlreadyExistsError(ServiceError):
    """Raised when attempting to create a duplicate job."""

    def __init__(self, job_id: str):
        super().__init__(
            f"Job already exists: {job_id}",
            "JOB_ALREADY_EXISTS",
            {"job_id": job_id},
        )


# =============================================================================
# Version/Registry Errors
# =============================================================================


class VersionError(TinyForgeError):
    """Base exception for versioning errors."""

    def __init__(
        self,
        message: str,
        code: str = "VERSION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class VersionNotFoundError(VersionError):
    """Raised when a specific version cannot be found."""

    def __init__(self, model_name: str, version: str):
        super().__init__(
            f"Version {version} not found for model {model_name}",
            "VERSION_NOT_FOUND",
            {"model_name": model_name, "version": version},
        )


class VersionConflictError(VersionError):
    """Raised when version conflicts occur."""

    def __init__(self, model_name: str, version: str):
        super().__init__(
            f"Version {version} already exists for model {model_name}",
            "VERSION_CONFLICT",
            {"model_name": model_name, "version": version},
        )
