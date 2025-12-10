"""
Model management API routes for TinyForgeAI.

Provides endpoints for listing, uploading, and managing models.
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.config import settings
from backend.exceptions import ModelNotFoundError, VersionNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    model_type: str = "unknown"
    base_model: Optional[str] = None
    created_at: str
    size_bytes: int
    path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelListResponse(BaseModel):
    """List models response."""
    models: List[ModelInfo]
    total: int


class ModelVersionInfo(BaseModel):
    """Model version information."""
    version: str
    created_at: str
    size_bytes: int
    is_current: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelVersionsResponse(BaseModel):
    """Model versions response."""
    model_name: str
    versions: List[ModelVersionInfo]
    current_version: str


class ModelUploadResponse(BaseModel):
    """Model upload response."""
    name: str
    version: str
    path: str
    message: str


# =============================================================================
# Helper Functions
# =============================================================================


def get_model_registry_path() -> Path:
    """Get the model registry path."""
    path = Path(settings.MODEL_REGISTRY_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model_metadata(model_dir: Path) -> Dict[str, Any]:
    """Load model metadata from directory."""
    metadata_path = model_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def get_model_size(model_dir: Path) -> int:
    """Get total size of model directory in bytes."""
    total = 0
    for dirpath, _, filenames in os.walk(model_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


def get_current_version(model_dir: Path) -> Optional[str]:
    """Get the current version for a model."""
    current_path = model_dir / "current"
    if current_path.exists():
        with open(current_path, "r") as f:
            return f.read().strip()
    # Default to latest version
    versions = list_model_versions(model_dir)
    return versions[-1] if versions else None


def list_model_versions(model_dir: Path) -> List[str]:
    """List all versions of a model."""
    versions = []
    for item in model_dir.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            versions.append(item.name)
    return sorted(versions)


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    registry_path = get_model_registry_path()
    models = []

    for model_dir in registry_path.iterdir():
        if not model_dir.is_dir():
            continue

        current_version = get_current_version(model_dir)
        if not current_version:
            continue

        version_dir = model_dir / current_version
        if not version_dir.exists():
            continue

        metadata = load_model_metadata(version_dir)

        models.append(ModelInfo(
            name=model_dir.name,
            version=current_version,
            model_type=metadata.get("model_type", "unknown"),
            base_model=metadata.get("base_model"),
            created_at=metadata.get("created_at", datetime.now().isoformat()),
            size_bytes=get_model_size(version_dir),
            path=str(version_dir),
            metadata=metadata,
        ))

    return ModelListResponse(models=models, total=len(models))


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str, version: Optional[str] = None):
    """Get information about a specific model."""
    registry_path = get_model_registry_path()
    model_dir = registry_path / model_name

    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}",
        )

    # Use specified version or current
    target_version = version or get_current_version(model_dir)
    if not target_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No versions found for model: {model_name}",
        )

    version_dir = model_dir / target_version
    if not version_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {target_version} not found for model: {model_name}",
        )

    metadata = load_model_metadata(version_dir)

    return ModelInfo(
        name=model_name,
        version=target_version,
        model_type=metadata.get("model_type", "unknown"),
        base_model=metadata.get("base_model"),
        created_at=metadata.get("created_at", datetime.now().isoformat()),
        size_bytes=get_model_size(version_dir),
        path=str(version_dir),
        metadata=metadata,
    )


@router.get("/{model_name}/versions", response_model=ModelVersionsResponse)
async def get_model_versions(model_name: str):
    """Get all versions of a model."""
    registry_path = get_model_registry_path()
    model_dir = registry_path / model_name

    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}",
        )

    current = get_current_version(model_dir)
    versions = []

    for version in list_model_versions(model_dir):
        version_dir = model_dir / version
        metadata = load_model_metadata(version_dir)

        versions.append(ModelVersionInfo(
            version=version,
            created_at=metadata.get("created_at", datetime.now().isoformat()),
            size_bytes=get_model_size(version_dir),
            is_current=(version == current),
            metadata=metadata,
        ))

    return ModelVersionsResponse(
        model_name=model_name,
        versions=versions,
        current_version=current or "",
    )


@router.post("/{model_name}/versions/{version}/activate")
async def activate_model_version(model_name: str, version: str):
    """Set a specific version as the current version."""
    registry_path = get_model_registry_path()
    model_dir = registry_path / model_name

    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}",
        )

    version_dir = model_dir / version
    if not version_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found for model: {model_name}",
        )

    # Update current pointer
    current_path = model_dir / "current"
    with open(current_path, "w") as f:
        f.write(version)

    logger.info(f"Activated version {version} for model {model_name}")

    return {
        "message": f"Version {version} is now active for model {model_name}",
        "model_name": model_name,
        "version": version,
    }


@router.post("", response_model=ModelUploadResponse)
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    version: Optional[str] = Form(None),
    model_type: str = Form("custom"),
    base_model: Optional[str] = Form(None),
):
    """
    Upload a new model or new version of an existing model.

    Accepts a zip file containing model artifacts.
    """
    registry_path = get_model_registry_path()
    model_dir = registry_path / name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine version
    if not version:
        existing_versions = list_model_versions(model_dir)
        if existing_versions:
            # Increment last version
            last = existing_versions[-1]
            num = int(last[1:]) + 1
            version = f"v{num}"
        else:
            version = "v1"

    version_dir = model_dir / version
    if version_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Version {version} already exists for model: {name}",
        )

    version_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded file
        import tempfile
        import zipfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract zip to version directory
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(version_dir)

        os.unlink(tmp_path)

        # Create/update metadata
        metadata = {
            "model_type": model_type,
            "base_model": base_model,
            "created_at": datetime.utcnow().isoformat(),
            "version": version,
            "original_filename": file.filename,
        }

        metadata_path = version_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Set as current if first version
        if not get_current_version(model_dir) or version == "v1":
            current_path = model_dir / "current"
            with open(current_path, "w") as f:
                f.write(version)

        logger.info(f"Uploaded model {name} version {version}")

        return ModelUploadResponse(
            name=name,
            version=version,
            path=str(version_dir),
            message=f"Successfully uploaded model {name} version {version}",
        )

    except zipfile.BadZipFile:
        # Clean up on failure
        shutil.rmtree(version_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid zip file",
        )
    except Exception as e:
        shutil.rmtree(version_dir, ignore_errors=True)
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload model: {str(e)}",
        )


@router.delete("/{model_name}")
async def delete_model(model_name: str, version: Optional[str] = None):
    """
    Delete a model or specific version.

    If version is not specified, deletes the entire model.
    """
    registry_path = get_model_registry_path()
    model_dir = registry_path / model_name

    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}",
        )

    if version:
        # Delete specific version
        version_dir = model_dir / version
        if not version_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Version {version} not found for model: {model_name}",
            )

        shutil.rmtree(version_dir)
        logger.info(f"Deleted version {version} of model {model_name}")

        # Update current if we deleted the current version
        current = get_current_version(model_dir)
        if current == version:
            remaining = list_model_versions(model_dir)
            if remaining:
                current_path = model_dir / "current"
                with open(current_path, "w") as f:
                    f.write(remaining[-1])
            else:
                # No versions left, delete entire model
                shutil.rmtree(model_dir)
                return {"message": f"Deleted model {model_name} (no versions remaining)"}

        return {"message": f"Deleted version {version} of model {model_name}"}
    else:
        # Delete entire model
        shutil.rmtree(model_dir)
        logger.info(f"Deleted model {model_name}")
        return {"message": f"Deleted model {model_name}"}
