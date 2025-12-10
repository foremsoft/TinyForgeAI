"""
TinyForgeAI Model Registry - Model versioning and lifecycle management.

This module provides comprehensive model versioning, metadata tracking,
and lifecycle management for trained models.
"""

from backend.model_registry.registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetadata,
    ModelCard,
    VersionStatus,
)
from backend.model_registry.versioning import (
    SemanticVersion,
    parse_version,
    compare_versions,
)

__all__ = [
    "ModelRegistry",
    "ModelVersion",
    "ModelMetadata",
    "ModelCard",
    "VersionStatus",
    "SemanticVersion",
    "parse_version",
    "compare_versions",
]
