"""
Model Registry - Core registry for model versioning and lifecycle management.

Provides comprehensive model storage, versioning, metadata tracking,
and lifecycle management capabilities.
"""

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from backend.model_registry.versioning import SemanticVersion, parse_version

logger = logging.getLogger(__name__)


class VersionStatus(str, Enum):
    """Status of a model version in its lifecycle."""

    DRAFT = "draft"  # Version being prepared, not ready for use
    ACTIVE = "active"  # Current active version
    STAGED = "staged"  # Ready for promotion to active
    DEPRECATED = "deprecated"  # No longer recommended
    ARCHIVED = "archived"  # Removed from active use


@dataclass
class TrainingMetrics:
    """Training metrics for a model version."""

    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    train_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_score: Optional[Dict[str, float]] = None
    perplexity: Optional[float] = None
    training_time_seconds: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingMetrics":
        """Create from dictionary."""
        return cls(
            train_loss=data.get("train_loss"),
            eval_loss=data.get("eval_loss"),
            train_steps=data.get("train_steps"),
            eval_steps=data.get("eval_steps"),
            accuracy=data.get("accuracy"),
            f1_score=data.get("f1_score"),
            bleu_score=data.get("bleu_score"),
            rouge_score=data.get("rouge_score"),
            perplexity=data.get("perplexity"),
            training_time_seconds=data.get("training_time_seconds"),
            custom_metrics=data.get("custom_metrics", {}),
        )


@dataclass
class TrainingConfig:
    """Training configuration used for a model version."""

    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    scheduler: Optional[str] = None
    use_peft: bool = False
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    max_length: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for k, v in asdict(self).items():
            if v is not None and v != {} and v is not False:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            epochs=data.get("epochs"),
            batch_size=data.get("batch_size"),
            learning_rate=data.get("learning_rate"),
            optimizer=data.get("optimizer"),
            scheduler=data.get("scheduler"),
            use_peft=data.get("use_peft", False),
            lora_r=data.get("lora_r"),
            lora_alpha=data.get("lora_alpha"),
            lora_dropout=data.get("lora_dropout"),
            max_length=data.get("max_length"),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps"),
            warmup_steps=data.get("warmup_steps"),
            weight_decay=data.get("weight_decay"),
            custom_config=data.get("custom_config", {}),
        )


@dataclass
class ModelCard:
    """
    Model card following HuggingFace conventions.

    Documents model capabilities, limitations, intended use, and ethical considerations.
    """

    description: str = ""
    intended_use: str = ""
    limitations: str = ""
    ethical_considerations: str = ""
    training_data_description: str = ""
    evaluation_data_description: str = ""
    performance_summary: str = ""
    citation: str = ""
    additional_info: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_markdown(self) -> str:
        """Generate markdown model card."""
        sections = []

        if self.description:
            sections.append(f"## Model Description\n\n{self.description}")

        if self.intended_use:
            sections.append(f"## Intended Use\n\n{self.intended_use}")

        if self.limitations:
            sections.append(f"## Limitations\n\n{self.limitations}")

        if self.ethical_considerations:
            sections.append(f"## Ethical Considerations\n\n{self.ethical_considerations}")

        if self.training_data_description:
            sections.append(f"## Training Data\n\n{self.training_data_description}")

        if self.evaluation_data_description:
            sections.append(f"## Evaluation Data\n\n{self.evaluation_data_description}")

        if self.performance_summary:
            sections.append(f"## Performance\n\n{self.performance_summary}")

        if self.citation:
            sections.append(f"## Citation\n\n```\n{self.citation}\n```")

        return "\n\n".join(sections)


@dataclass
class ModelMetadata:
    """
    Comprehensive metadata for a model version.

    Tracks all relevant information about a trained model including
    provenance, configuration, metrics, and lifecycle status.
    """

    # Identity
    name: str
    version: str
    model_type: str = "custom"

    # Provenance
    base_model: Optional[str] = None
    parent_version: Optional[str] = None  # For derived models
    task_type: Optional[str] = None
    data_path: Optional[str] = None
    data_version: Optional[str] = None
    n_records: int = 0

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: Optional[str] = None
    activated_at: Optional[str] = None
    deprecated_at: Optional[str] = None

    # Status and lifecycle
    status: VersionStatus = VersionStatus.DRAFT
    changelog: str = ""

    # Training details
    training_config: Optional[TrainingConfig] = None
    training_metrics: Optional[TrainingMetrics] = None

    # Documentation
    model_card: Optional[ModelCard] = None
    tags: List[str] = field(default_factory=list)
    license: str = "Apache-2.0"

    # Additional metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "parent_version": self.parent_version,
            "task_type": self.task_type,
            "data_path": self.data_path,
            "data_version": self.data_version,
            "n_records": self.n_records,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "activated_at": self.activated_at,
            "deprecated_at": self.deprecated_at,
            "status": self.status.value if isinstance(self.status, VersionStatus) else self.status,
            "changelog": self.changelog,
            "tags": self.tags,
            "license": self.license,
            "custom_metadata": self.custom_metadata,
        }

        if self.training_config:
            data["training_config"] = self.training_config.to_dict()
        if self.training_metrics:
            data["training_metrics"] = self.training_metrics.to_dict()
        if self.model_card:
            data["model_card"] = self.model_card.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        # Parse nested objects
        training_config = None
        if "training_config" in data and data["training_config"]:
            training_config = TrainingConfig.from_dict(data["training_config"])

        training_metrics = None
        if "training_metrics" in data and data["training_metrics"]:
            training_metrics = TrainingMetrics.from_dict(data["training_metrics"])

        model_card = None
        if "model_card" in data and data["model_card"]:
            model_card = ModelCard.from_dict(data["model_card"])

        # Parse status
        status = data.get("status", "draft")
        if isinstance(status, str):
            status = VersionStatus(status)

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.0.0"),
            model_type=data.get("model_type", "custom"),
            base_model=data.get("base_model"),
            parent_version=data.get("parent_version"),
            task_type=data.get("task_type"),
            data_path=data.get("data_path"),
            data_version=data.get("data_version"),
            n_records=data.get("n_records", 0),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at"),
            activated_at=data.get("activated_at"),
            deprecated_at=data.get("deprecated_at"),
            status=status,
            changelog=data.get("changelog", ""),
            training_config=training_config,
            training_metrics=training_metrics,
            model_card=model_card,
            tags=data.get("tags", []),
            license=data.get("license", "Apache-2.0"),
            custom_metadata=data.get("custom_metadata", {}),
        )


@dataclass
class ModelVersion:
    """
    Represents a specific version of a model in the registry.

    Contains metadata and provides access to model artifacts.
    """

    metadata: ModelMetadata
    path: Path

    @property
    def name(self) -> str:
        """Model name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Version string."""
        return self.metadata.version

    @property
    def semantic_version(self) -> SemanticVersion:
        """Parsed semantic version."""
        return parse_version(self.metadata.version)

    @property
    def status(self) -> VersionStatus:
        """Current status."""
        return self.metadata.status

    @property
    def is_active(self) -> bool:
        """Check if this is the active version."""
        return self.metadata.status == VersionStatus.ACTIVE

    def get_artifact_path(self, artifact_name: str) -> Path:
        """Get path to a specific artifact."""
        return self.path / artifact_name

    def has_artifact(self, artifact_name: str) -> bool:
        """Check if an artifact exists."""
        return (self.path / artifact_name).exists()

    def list_artifacts(self) -> List[str]:
        """List all artifacts in this version."""
        if not self.path.exists():
            return []
        return [f.name for f in self.path.iterdir() if f.is_file()]


class ModelRegistry:
    """
    Central registry for managing model versions.

    Provides functionality for:
    - Registering new model versions
    - Querying and listing models
    - Version lifecycle management (activate, deprecate, archive)
    - Model comparison and rollback
    """

    METADATA_FILE = "model_metadata.json"
    CURRENT_FILE = "current"
    MODEL_CARD_FILE = "MODEL_CARD.md"

    def __init__(self, registry_path: Union[str, Path]):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to the registry directory
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ModelRegistry initialized at {self.registry_path}")

    def _get_model_dir(self, model_name: str) -> Path:
        """Get the directory for a model."""
        return self.registry_path / model_name

    def _get_version_dir(self, model_name: str, version: str) -> Path:
        """Get the directory for a specific version."""
        return self._get_model_dir(model_name) / version

    def _read_metadata(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        """Read metadata for a version."""
        metadata_path = self._get_version_dir(model_name, version) / self.METADATA_FILE
        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ModelMetadata.from_dict(data)

    def _write_metadata(self, model_name: str, version: str, metadata: ModelMetadata) -> None:
        """Write metadata for a version."""
        version_dir = self._get_version_dir(model_name, version)
        version_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = version_dir / self.METADATA_FILE
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _get_current_version(self, model_name: str) -> Optional[str]:
        """Get the current active version for a model."""
        current_file = self._get_model_dir(model_name) / self.CURRENT_FILE
        if not current_file.exists():
            return None

        with open(current_file, "r", encoding="utf-8") as f:
            return f.read().strip()

    def _set_current_version(self, model_name: str, version: str) -> None:
        """Set the current active version for a model."""
        model_dir = self._get_model_dir(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        current_file = model_dir / self.CURRENT_FILE
        with open(current_file, "w", encoding="utf-8") as f:
            f.write(version)

    def list_models(self) -> List[str]:
        """
        List all models in the registry.

        Returns:
            List of model names
        """
        if not self.registry_path.exists():
            return []

        return [
            d.name
            for d in self.registry_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """
        List all versions of a model.

        Args:
            model_name: Name of the model

        Returns:
            List of ModelVersion objects sorted by version
        """
        model_dir = self._get_model_dir(model_name)
        if not model_dir.exists():
            return []

        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir() and not version_dir.name.startswith("."):
                metadata = self._read_metadata(model_name, version_dir.name)
                if metadata:
                    versions.append(ModelVersion(metadata=metadata, path=version_dir))

        # Sort by semantic version
        versions.sort(key=lambda v: v.semantic_version)
        return versions

    def get_version(self, model_name: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get a specific version of a model.

        Args:
            model_name: Name of the model
            version: Version string (if None, returns current active version)

        Returns:
            ModelVersion or None if not found
        """
        if version is None:
            version = self._get_current_version(model_name)
            if version is None:
                # Try to get latest version
                versions = self.list_versions(model_name)
                if not versions:
                    return None
                return versions[-1]

        metadata = self._read_metadata(model_name, version)
        if metadata is None:
            return None

        return ModelVersion(
            metadata=metadata,
            path=self._get_version_dir(model_name, version),
        )

    def get_current_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get the current active version of a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelVersion or None if no active version
        """
        current = self._get_current_version(model_name)
        if current is None:
            return None
        return self.get_version(model_name, current)

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """
        Get the latest version of a model (by semantic version).

        Args:
            model_name: Name of the model

        Returns:
            ModelVersion or None if model doesn't exist
        """
        versions = self.list_versions(model_name)
        if not versions:
            return None
        return versions[-1]

    def register_version(
        self,
        model_name: str,
        version: str,
        metadata: Optional[ModelMetadata] = None,
        artifacts_path: Optional[Path] = None,
        activate: bool = False,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            version: Version string
            metadata: Model metadata (if None, creates minimal metadata)
            artifacts_path: Path to artifacts to copy into registry
            activate: Whether to set this as the active version

        Returns:
            The registered ModelVersion
        """
        version_dir = self._get_version_dir(model_name, version)

        if version_dir.exists():
            raise ValueError(f"Version {version} already exists for model {model_name}")

        version_dir.mkdir(parents=True, exist_ok=True)

        # Create or update metadata
        if metadata is None:
            metadata = ModelMetadata(
                name=model_name,
                version=version,
            )
        else:
            metadata.name = model_name
            metadata.version = version

        if activate:
            metadata.status = VersionStatus.ACTIVE
            metadata.activated_at = datetime.utcnow().isoformat()
        else:
            metadata.status = VersionStatus.DRAFT

        # Copy artifacts if provided
        if artifacts_path and artifacts_path.exists():
            for item in artifacts_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, version_dir / item.name)
                elif item.is_dir():
                    shutil.copytree(item, version_dir / item.name)

        # Save metadata
        self._write_metadata(model_name, version, metadata)

        # Save model card if present
        if metadata.model_card:
            card_path = version_dir / self.MODEL_CARD_FILE
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(metadata.model_card.to_markdown())

        # Set as current if activating
        if activate:
            self._set_current_version(model_name, version)

        logger.info(f"Registered model {model_name} version {version}")
        return ModelVersion(metadata=metadata, path=version_dir)

    def get_next_version(
        self,
        model_name: str,
        bump_type: str = "patch",
    ) -> str:
        """
        Get the next version string for a model.

        Args:
            model_name: Name of the model
            bump_type: Type of version bump ("major", "minor", "patch")

        Returns:
            Next version string
        """
        latest = self.get_latest_version(model_name)
        if latest is None:
            return "1.0.0"

        current = latest.semantic_version
        if bump_type == "major":
            return str(current.bump_major())
        elif bump_type == "minor":
            return str(current.bump_minor())
        else:
            return str(current.bump_patch())

    def activate_version(self, model_name: str, version: str) -> ModelVersion:
        """
        Activate a specific version (make it current).

        Args:
            model_name: Name of the model
            version: Version to activate

        Returns:
            The activated ModelVersion
        """
        # Deactivate current version
        current = self.get_current_version(model_name)
        if current and current.version != version:
            current.metadata.status = VersionStatus.STAGED
            self._write_metadata(model_name, current.version, current.metadata)

        # Activate new version
        model_version = self.get_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Version {version} not found for model {model_name}")

        model_version.metadata.status = VersionStatus.ACTIVE
        model_version.metadata.activated_at = datetime.utcnow().isoformat()
        self._write_metadata(model_name, version, model_version.metadata)
        self._set_current_version(model_name, version)

        logger.info(f"Activated {model_name} version {version}")
        return model_version

    def deprecate_version(
        self,
        model_name: str,
        version: str,
        reason: str = "",
    ) -> ModelVersion:
        """
        Deprecate a model version.

        Args:
            model_name: Name of the model
            version: Version to deprecate
            reason: Reason for deprecation

        Returns:
            The deprecated ModelVersion
        """
        model_version = self.get_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Version {version} not found for model {model_name}")

        model_version.metadata.status = VersionStatus.DEPRECATED
        model_version.metadata.deprecated_at = datetime.utcnow().isoformat()
        if reason:
            model_version.metadata.changelog += f"\n\nDeprecated: {reason}"

        self._write_metadata(model_name, version, model_version.metadata)

        logger.info(f"Deprecated {model_name} version {version}: {reason}")
        return model_version

    def archive_version(self, model_name: str, version: str) -> ModelVersion:
        """
        Archive a model version.

        Args:
            model_name: Name of the model
            version: Version to archive

        Returns:
            The archived ModelVersion
        """
        model_version = self.get_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Version {version} not found for model {model_name}")

        model_version.metadata.status = VersionStatus.ARCHIVED
        self._write_metadata(model_name, version, model_version.metadata)

        logger.info(f"Archived {model_name} version {version}")
        return model_version

    def delete_version(self, model_name: str, version: str, force: bool = False) -> bool:
        """
        Delete a model version.

        Args:
            model_name: Name of the model
            version: Version to delete
            force: Force delete even if active

        Returns:
            True if deleted, False otherwise
        """
        model_version = self.get_version(model_name, version)
        if model_version is None:
            return False

        if model_version.is_active and not force:
            raise ValueError(f"Cannot delete active version {version}. Use force=True or activate another version first.")

        version_dir = self._get_version_dir(model_name, version)
        if version_dir.exists():
            shutil.rmtree(version_dir)

        # Update current pointer if needed
        current = self._get_current_version(model_name)
        if current == version:
            versions = self.list_versions(model_name)
            if versions:
                self._set_current_version(model_name, versions[-1].version)
            else:
                # Remove current file if no versions left
                current_file = self._get_model_dir(model_name) / self.CURRENT_FILE
                if current_file.exists():
                    current_file.unlink()

        logger.info(f"Deleted {model_name} version {version}")
        return True

    def delete_model(self, model_name: str, force: bool = False) -> bool:
        """
        Delete a model and all its versions.

        Args:
            model_name: Name of the model
            force: Force delete without confirmation

        Returns:
            True if deleted, False otherwise
        """
        model_dir = self._get_model_dir(model_name)
        if not model_dir.exists():
            return False

        if not force:
            versions = self.list_versions(model_name)
            if any(v.is_active for v in versions):
                raise ValueError(f"Model {model_name} has active versions. Use force=True to delete.")

        shutil.rmtree(model_dir)
        logger.info(f"Deleted model {model_name}")
        return True

    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str,
    ) -> Dict[str, Any]:
        """
        Compare two versions of a model.

        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version

        Returns:
            Comparison dictionary
        """
        v1 = self.get_version(model_name, version1)
        v2 = self.get_version(model_name, version2)

        if v1 is None or v2 is None:
            raise ValueError("One or both versions not found")

        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {},
            "config_changes": {},
        }

        # Compare metrics
        if v1.metadata.training_metrics and v2.metadata.training_metrics:
            m1 = v1.metadata.training_metrics.to_dict()
            m2 = v2.metadata.training_metrics.to_dict()

            for key in set(m1.keys()) | set(m2.keys()):
                if key in m1 and key in m2:
                    if isinstance(m1[key], (int, float)) and isinstance(m2[key], (int, float)):
                        diff = m2[key] - m1[key]
                        comparison["metrics_comparison"][key] = {
                            "v1": m1[key],
                            "v2": m2[key],
                            "diff": diff,
                            "improved": diff < 0 if "loss" in key else diff > 0,
                        }

        # Compare config
        if v1.metadata.training_config and v2.metadata.training_config:
            c1 = v1.metadata.training_config.to_dict()
            c2 = v2.metadata.training_config.to_dict()

            for key in set(c1.keys()) | set(c2.keys()):
                v1_val = c1.get(key)
                v2_val = c2.get(key)
                if v1_val != v2_val:
                    comparison["config_changes"][key] = {
                        "v1": v1_val,
                        "v2": v2_val,
                    }

        return comparison

    def rollback(self, model_name: str, target_version: str) -> ModelVersion:
        """
        Rollback to a previous version.

        Args:
            model_name: Name of the model
            target_version: Version to rollback to

        Returns:
            The activated (rolled back to) ModelVersion
        """
        target = self.get_version(model_name, target_version)
        if target is None:
            raise ValueError(f"Version {target_version} not found")

        if target.metadata.status == VersionStatus.ARCHIVED:
            raise ValueError(f"Cannot rollback to archived version {target_version}")

        return self.activate_version(model_name, target_version)

    def search_models(
        self,
        task_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[VersionStatus] = None,
    ) -> Iterator[ModelVersion]:
        """
        Search for models matching criteria.

        Args:
            task_type: Filter by task type
            tags: Filter by tags (any match)
            status: Filter by status

        Yields:
            Matching ModelVersion objects
        """
        for model_name in self.list_models():
            for version in self.list_versions(model_name):
                # Apply filters
                if task_type and version.metadata.task_type != task_type:
                    continue
                if tags and not any(t in version.metadata.tags for t in tags):
                    continue
                if status and version.metadata.status != status:
                    continue
                yield version

    def export_model(
        self,
        model_name: str,
        version: str,
        output_path: Path,
    ) -> Path:
        """
        Export a model version to a directory.

        Args:
            model_name: Name of the model
            version: Version to export
            output_path: Path to export to

        Returns:
            Path to exported model
        """
        model_version = self.get_version(model_name, version)
        if model_version is None:
            raise ValueError(f"Version {version} not found for model {model_name}")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy all artifacts
        for item in model_version.path.iterdir():
            if item.is_file():
                shutil.copy2(item, output_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, output_path / item.name)

        logger.info(f"Exported {model_name} v{version} to {output_path}")
        return output_path
