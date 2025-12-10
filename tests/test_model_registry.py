"""
Tests for the model registry module.

Tests versioning utilities and the core registry functionality.
"""

import json
import pytest
import tempfile
from pathlib import Path

from backend.model_registry.versioning import (
    SemanticVersion,
    parse_version,
    compare_versions,
    get_next_version,
)
from backend.model_registry.registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetadata,
    ModelCard,
    TrainingMetrics,
    TrainingConfig,
    VersionStatus,
)


# ============================================================================
# SemanticVersion Tests
# ============================================================================


class TestSemanticVersion:
    """Tests for SemanticVersion class."""

    def test_create_basic_version(self):
        """Test creating a basic semantic version."""
        v = SemanticVersion(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert str(v) == "1.2.3"

    def test_create_with_prerelease(self):
        """Test creating version with prerelease tag."""
        v = SemanticVersion(1, 0, 0, prerelease="alpha")
        assert v.prerelease == "alpha"
        assert str(v) == "1.0.0-alpha"

    def test_create_with_build(self):
        """Test creating version with build metadata."""
        v = SemanticVersion(1, 0, 0, build="build.123")
        assert v.build == "build.123"
        assert str(v) == "1.0.0+build.123"

    def test_create_with_prerelease_and_build(self):
        """Test creating version with both prerelease and build."""
        v = SemanticVersion(2, 0, 0, prerelease="beta.1", build="20231201")
        assert str(v) == "2.0.0-beta.1+20231201"

    def test_version_validation_negative_major(self):
        """Test that negative major version raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            SemanticVersion(-1, 0, 0)

    def test_version_validation_negative_minor(self):
        """Test that negative minor version raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            SemanticVersion(1, -1, 0)

    def test_version_validation_negative_patch(self):
        """Test that negative patch version raises error."""
        with pytest.raises(ValueError, match="non-negative"):
            SemanticVersion(1, 0, -1)

    def test_version_ordering(self):
        """Test version comparison ordering."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        v3 = SemanticVersion(2, 0, 0)

        assert v1 < v2 < v3
        assert v3 > v2 > v1

    def test_version_equality(self):
        """Test version equality."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        assert v1 == v2

    def test_version_hash(self):
        """Test version is hashable."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 0)
        version_set = {v1, v2}
        assert len(version_set) == 1

    def test_from_string_basic(self):
        """Test parsing basic version string."""
        v = SemanticVersion.from_string("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_from_string_with_v_prefix(self):
        """Test parsing version string with 'v' prefix."""
        v = SemanticVersion.from_string("v1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_from_string_with_prerelease(self):
        """Test parsing version string with prerelease."""
        v = SemanticVersion.from_string("1.0.0-alpha.1")
        assert v.prerelease == "alpha.1"

    def test_from_string_with_build(self):
        """Test parsing version string with build metadata."""
        v = SemanticVersion.from_string("1.0.0+build.123")
        assert v.build == "build.123"

    def test_from_string_simple_version(self):
        """Test parsing simple vN format."""
        v = SemanticVersion.from_string("v2")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 0

    def test_from_string_invalid(self):
        """Test parsing invalid version string."""
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.from_string("not.a.version.string")

    def test_bump_major(self):
        """Test major version bump."""
        v = SemanticVersion(1, 2, 3)
        new_v = v.bump_major()
        assert new_v.major == 2
        assert new_v.minor == 0
        assert new_v.patch == 0

    def test_bump_minor(self):
        """Test minor version bump."""
        v = SemanticVersion(1, 2, 3)
        new_v = v.bump_minor()
        assert new_v.major == 1
        assert new_v.minor == 3
        assert new_v.patch == 0

    def test_bump_patch(self):
        """Test patch version bump."""
        v = SemanticVersion(1, 2, 3)
        new_v = v.bump_patch()
        assert new_v.major == 1
        assert new_v.minor == 2
        assert new_v.patch == 4

    def test_with_prerelease(self):
        """Test adding prerelease tag."""
        v = SemanticVersion(1, 0, 0)
        new_v = v.with_prerelease("rc.1")
        assert new_v.prerelease == "rc.1"
        assert str(new_v) == "1.0.0-rc.1"

    def test_with_build(self):
        """Test adding build metadata."""
        v = SemanticVersion(1, 0, 0)
        new_v = v.with_build("20231201")
        assert new_v.build == "20231201"

    def test_to_tuple(self):
        """Test conversion to tuple."""
        v = SemanticVersion(1, 2, 3)
        assert v.to_tuple() == (1, 2, 3)

    def test_is_prerelease(self):
        """Test prerelease detection."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 0, prerelease="alpha")

        assert not v1.is_prerelease()
        assert v2.is_prerelease()

    def test_is_compatible_major_zero(self):
        """Test compatibility for 0.x versions."""
        v1 = SemanticVersion(0, 1, 0)
        v2 = SemanticVersion(0, 1, 5)
        v3 = SemanticVersion(0, 2, 0)

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)

    def test_is_compatible_major_nonzero(self):
        """Test compatibility for 1.x+ versions."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 5, 0)
        v3 = SemanticVersion(2, 0, 0)

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)


class TestVersioningHelpers:
    """Tests for versioning helper functions."""

    def test_parse_version(self):
        """Test parse_version convenience function."""
        v = parse_version("1.2.3")
        assert isinstance(v, SemanticVersion)
        assert v.major == 1

    def test_compare_versions_less(self):
        """Test compare_versions returns -1 for less than."""
        assert compare_versions("1.0.0", "2.0.0") == -1

    def test_compare_versions_equal(self):
        """Test compare_versions returns 0 for equal."""
        assert compare_versions("1.0.0", "1.0.0") == 0

    def test_compare_versions_greater(self):
        """Test compare_versions returns 1 for greater than."""
        assert compare_versions("2.0.0", "1.0.0") == 1

    def test_compare_versions_with_objects(self):
        """Test compare_versions with SemanticVersion objects."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 1, 0)
        assert compare_versions(v1, v2) == -1

    def test_get_next_version_patch(self):
        """Test get_next_version with patch bump."""
        next_v = get_next_version("1.0.0", "patch")
        assert str(next_v) == "1.0.1"

    def test_get_next_version_minor(self):
        """Test get_next_version with minor bump."""
        next_v = get_next_version("1.0.0", "minor")
        assert str(next_v) == "1.1.0"

    def test_get_next_version_major(self):
        """Test get_next_version with major bump."""
        next_v = get_next_version("1.0.0", "major")
        assert str(next_v) == "2.0.0"

    def test_get_next_version_invalid_bump(self):
        """Test get_next_version with invalid bump type."""
        with pytest.raises(ValueError, match="Invalid bump type"):
            get_next_version("1.0.0", "invalid")


# ============================================================================
# TrainingMetrics Tests
# ============================================================================


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_create_empty_metrics(self):
        """Test creating empty metrics."""
        m = TrainingMetrics()
        assert m.train_loss is None
        assert m.eval_loss is None

    def test_create_with_metrics(self):
        """Test creating metrics with values."""
        m = TrainingMetrics(
            train_loss=0.5,
            eval_loss=0.6,
            accuracy=0.95,
            f1_score=0.92,
        )
        assert m.train_loss == 0.5
        assert m.accuracy == 0.95

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        m = TrainingMetrics(train_loss=0.5)
        d = m.to_dict()
        assert "train_loss" in d
        assert "eval_loss" not in d

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"train_loss": 0.5, "accuracy": 0.95}
        m = TrainingMetrics.from_dict(data)
        assert m.train_loss == 0.5
        assert m.accuracy == 0.95

    def test_custom_metrics(self):
        """Test custom metrics field."""
        m = TrainingMetrics(custom_metrics={"my_metric": 0.99})
        assert m.custom_metrics["my_metric"] == 0.99


# ============================================================================
# TrainingConfig Tests
# ============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_create_empty_config(self):
        """Test creating empty config."""
        c = TrainingConfig()
        assert c.epochs is None
        assert c.use_peft is False

    def test_create_with_config(self):
        """Test creating config with values."""
        c = TrainingConfig(
            epochs=10,
            batch_size=16,
            learning_rate=1e-4,
            use_peft=True,
            lora_r=8,
        )
        assert c.epochs == 10
        assert c.use_peft is True
        assert c.lora_r == 8

    def test_to_dict(self):
        """Test to_dict conversion."""
        c = TrainingConfig(epochs=10, batch_size=16)
        d = c.to_dict()
        assert d["epochs"] == 10
        assert d["batch_size"] == 16

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"epochs": 10, "use_peft": True, "lora_r": 8}
        c = TrainingConfig.from_dict(data)
        assert c.epochs == 10
        assert c.use_peft is True
        assert c.lora_r == 8


# ============================================================================
# ModelCard Tests
# ============================================================================


class TestModelCard:
    """Tests for ModelCard dataclass."""

    def test_create_empty_card(self):
        """Test creating empty model card."""
        card = ModelCard()
        assert card.description == ""

    def test_create_with_content(self):
        """Test creating model card with content."""
        card = ModelCard(
            description="A test model",
            intended_use="For testing purposes",
            limitations="Not for production",
        )
        assert card.description == "A test model"

    def test_to_markdown(self):
        """Test markdown generation."""
        card = ModelCard(
            description="A test model",
            intended_use="For testing purposes",
        )
        md = card.to_markdown()
        assert "## Model Description" in md
        assert "A test model" in md
        assert "## Intended Use" in md

    def test_to_dict(self):
        """Test to_dict conversion."""
        card = ModelCard(description="Test")
        d = card.to_dict()
        assert d["description"] == "Test"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"description": "Test", "intended_use": "Testing"}
        card = ModelCard.from_dict(data)
        assert card.description == "Test"
        assert card.intended_use == "Testing"


# ============================================================================
# ModelMetadata Tests
# ============================================================================


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_create_minimal_metadata(self):
        """Test creating minimal metadata."""
        m = ModelMetadata(name="test-model", version="1.0.0")
        assert m.name == "test-model"
        assert m.version == "1.0.0"
        assert m.status == VersionStatus.DRAFT

    def test_create_full_metadata(self):
        """Test creating full metadata."""
        m = ModelMetadata(
            name="test-model",
            version="1.0.0",
            model_type="transformer",
            base_model="distilbert-base-uncased",
            task_type="classification",
            tags=["test", "example"],
            training_config=TrainingConfig(epochs=10),
            training_metrics=TrainingMetrics(accuracy=0.95),
        )
        assert m.base_model == "distilbert-base-uncased"
        assert "test" in m.tags
        assert m.training_config.epochs == 10

    def test_to_dict(self):
        """Test to_dict conversion."""
        m = ModelMetadata(name="test", version="1.0.0")
        d = m.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "draft"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "name": "test",
            "version": "1.0.0",
            "status": "active",
            "tags": ["a", "b"],
        }
        m = ModelMetadata.from_dict(data)
        assert m.name == "test"
        assert m.status == VersionStatus.ACTIVE
        assert "a" in m.tags


# ============================================================================
# ModelRegistry Tests
# ============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def registry(self, tmp_path):
        """Create a temporary registry for testing."""
        return ModelRegistry(tmp_path / "registry")

    @pytest.fixture
    def populated_registry(self, registry):
        """Create a registry with some models."""
        # Register model v1.0.0
        registry.register_version(
            "test-model",
            "1.0.0",
            metadata=ModelMetadata(
                name="test-model",
                version="1.0.0",
                task_type="classification",
                tags=["test"],
            ),
            activate=True,
        )
        # Register model v1.1.0
        registry.register_version(
            "test-model",
            "1.1.0",
            metadata=ModelMetadata(
                name="test-model",
                version="1.1.0",
                task_type="classification",
                tags=["test", "improved"],
            ),
        )
        return registry

    def test_registry_creation(self, tmp_path):
        """Test registry directory is created."""
        registry_path = tmp_path / "new_registry"
        registry = ModelRegistry(registry_path)
        assert registry_path.exists()

    def test_list_models_empty(self, registry):
        """Test listing models in empty registry."""
        models = registry.list_models()
        assert models == []

    def test_register_version(self, registry):
        """Test registering a new version."""
        version = registry.register_version(
            "my-model",
            "1.0.0",
            metadata=ModelMetadata(name="my-model", version="1.0.0"),
        )
        assert version.name == "my-model"
        assert version.version == "1.0.0"
        assert version.status == VersionStatus.DRAFT

    def test_register_version_activate(self, registry):
        """Test registering and activating a version."""
        version = registry.register_version(
            "my-model",
            "1.0.0",
            activate=True,
        )
        assert version.status == VersionStatus.ACTIVE
        assert version.metadata.activated_at is not None

    def test_register_duplicate_version_fails(self, registry):
        """Test registering duplicate version fails."""
        registry.register_version("my-model", "1.0.0")
        with pytest.raises(ValueError, match="already exists"):
            registry.register_version("my-model", "1.0.0")

    def test_list_models(self, populated_registry):
        """Test listing models."""
        models = populated_registry.list_models()
        assert "test-model" in models

    def test_list_versions(self, populated_registry):
        """Test listing versions of a model."""
        versions = populated_registry.list_versions("test-model")
        assert len(versions) == 2
        assert versions[0].version == "1.0.0"  # Sorted by semver
        assert versions[1].version == "1.1.0"

    def test_list_versions_nonexistent_model(self, registry):
        """Test listing versions of nonexistent model."""
        versions = registry.list_versions("nonexistent")
        assert versions == []

    def test_get_version(self, populated_registry):
        """Test getting a specific version."""
        version = populated_registry.get_version("test-model", "1.0.0")
        assert version is not None
        assert version.version == "1.0.0"

    def test_get_version_nonexistent(self, populated_registry):
        """Test getting nonexistent version."""
        version = populated_registry.get_version("test-model", "9.9.9")
        assert version is None

    def test_get_current_version(self, populated_registry):
        """Test getting current active version."""
        current = populated_registry.get_current_version("test-model")
        assert current is not None
        assert current.version == "1.0.0"
        assert current.is_active

    def test_get_latest_version(self, populated_registry):
        """Test getting latest version."""
        latest = populated_registry.get_latest_version("test-model")
        assert latest is not None
        assert latest.version == "1.1.0"

    def test_get_next_version_from_empty(self, registry):
        """Test getting next version for new model."""
        next_v = registry.get_next_version("new-model")
        assert next_v == "1.0.0"

    def test_get_next_version_patch(self, populated_registry):
        """Test getting next patch version."""
        next_v = populated_registry.get_next_version("test-model", "patch")
        assert next_v == "1.1.1"

    def test_get_next_version_minor(self, populated_registry):
        """Test getting next minor version."""
        next_v = populated_registry.get_next_version("test-model", "minor")
        assert next_v == "1.2.0"

    def test_get_next_version_major(self, populated_registry):
        """Test getting next major version."""
        next_v = populated_registry.get_next_version("test-model", "major")
        assert next_v == "2.0.0"

    def test_activate_version(self, populated_registry):
        """Test activating a version."""
        # Activate v1.1.0 (v1.0.0 is currently active)
        version = populated_registry.activate_version("test-model", "1.1.0")
        assert version.status == VersionStatus.ACTIVE

        # Check v1.0.0 is now staged
        old_version = populated_registry.get_version("test-model", "1.0.0")
        assert old_version.status == VersionStatus.STAGED

        # Check current points to new version
        current = populated_registry.get_current_version("test-model")
        assert current.version == "1.1.0"

    def test_activate_nonexistent_version_fails(self, populated_registry):
        """Test activating nonexistent version fails."""
        with pytest.raises(ValueError, match="not found"):
            populated_registry.activate_version("test-model", "9.9.9")

    def test_deprecate_version(self, populated_registry):
        """Test deprecating a version."""
        version = populated_registry.deprecate_version(
            "test-model", "1.0.0", reason="Outdated"
        )
        assert version.status == VersionStatus.DEPRECATED
        assert version.metadata.deprecated_at is not None
        assert "Outdated" in version.metadata.changelog

    def test_archive_version(self, populated_registry):
        """Test archiving a version."""
        version = populated_registry.archive_version("test-model", "1.1.0")
        assert version.status == VersionStatus.ARCHIVED

    def test_delete_version(self, populated_registry):
        """Test deleting a non-active version."""
        result = populated_registry.delete_version("test-model", "1.1.0")
        assert result is True
        versions = populated_registry.list_versions("test-model")
        assert len(versions) == 1

    def test_delete_active_version_fails(self, populated_registry):
        """Test deleting active version fails without force."""
        with pytest.raises(ValueError, match="Cannot delete active version"):
            populated_registry.delete_version("test-model", "1.0.0")

    def test_delete_active_version_force(self, populated_registry):
        """Test force deleting active version."""
        result = populated_registry.delete_version("test-model", "1.0.0", force=True)
        assert result is True

    def test_delete_model(self, populated_registry):
        """Test deleting entire model."""
        # First deactivate
        populated_registry.archive_version("test-model", "1.0.0")
        result = populated_registry.delete_model("test-model")
        assert result is True
        assert populated_registry.list_models() == []

    def test_delete_model_force(self, populated_registry):
        """Test force deleting model with active versions."""
        result = populated_registry.delete_model("test-model", force=True)
        assert result is True

    def test_compare_versions(self, registry):
        """Test comparing two versions."""
        # Register versions with metrics
        registry.register_version(
            "compare-model",
            "1.0.0",
            metadata=ModelMetadata(
                name="compare-model",
                version="1.0.0",
                training_metrics=TrainingMetrics(
                    train_loss=0.5, accuracy=0.90
                ),
                training_config=TrainingConfig(epochs=5, learning_rate=1e-3),
            ),
        )
        registry.register_version(
            "compare-model",
            "1.1.0",
            metadata=ModelMetadata(
                name="compare-model",
                version="1.1.0",
                training_metrics=TrainingMetrics(
                    train_loss=0.3, accuracy=0.95
                ),
                training_config=TrainingConfig(epochs=10, learning_rate=1e-4),
            ),
        )

        comparison = registry.compare_versions("compare-model", "1.0.0", "1.1.0")

        assert comparison["version1"] == "1.0.0"
        assert comparison["version2"] == "1.1.0"
        assert "train_loss" in comparison["metrics_comparison"]
        assert comparison["metrics_comparison"]["train_loss"]["improved"] is True
        assert "epochs" in comparison["config_changes"]

    def test_rollback(self, populated_registry):
        """Test rollback to previous version."""
        # Activate v1.1.0
        populated_registry.activate_version("test-model", "1.1.0")

        # Rollback to v1.0.0
        version = populated_registry.rollback("test-model", "1.0.0")
        assert version.status == VersionStatus.ACTIVE

        current = populated_registry.get_current_version("test-model")
        assert current.version == "1.0.0"

    def test_rollback_to_archived_fails(self, populated_registry):
        """Test rollback to archived version fails."""
        populated_registry.archive_version("test-model", "1.1.0")
        with pytest.raises(ValueError, match="Cannot rollback to archived"):
            populated_registry.rollback("test-model", "1.1.0")

    def test_search_models_by_task_type(self, populated_registry):
        """Test searching models by task type."""
        results = list(populated_registry.search_models(task_type="classification"))
        assert len(results) == 2

    def test_search_models_by_tags(self, populated_registry):
        """Test searching models by tags."""
        results = list(populated_registry.search_models(tags=["improved"]))
        assert len(results) == 1
        assert results[0].version == "1.1.0"

    def test_search_models_by_status(self, populated_registry):
        """Test searching models by status."""
        results = list(populated_registry.search_models(status=VersionStatus.ACTIVE))
        assert len(results) == 1
        assert results[0].version == "1.0.0"

    def test_export_model(self, populated_registry, tmp_path):
        """Test exporting a model version."""
        export_path = tmp_path / "exported"
        result = populated_registry.export_model("test-model", "1.0.0", export_path)
        assert result.exists()
        assert (result / "model_metadata.json").exists()

    def test_register_with_artifacts(self, registry, tmp_path):
        """Test registering with artifact files."""
        # Create some artifact files
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "model.bin").write_text("model data")
        (artifacts_dir / "config.json").write_text('{"key": "value"}')

        version = registry.register_version(
            "artifact-model",
            "1.0.0",
            artifacts_path=artifacts_dir,
        )

        assert version.has_artifact("model.bin")
        assert version.has_artifact("config.json")
        assert "model.bin" in version.list_artifacts()


class TestModelVersion:
    """Tests for ModelVersion class."""

    @pytest.fixture
    def model_version(self, tmp_path):
        """Create a test model version."""
        version_path = tmp_path / "test-model" / "1.0.0"
        version_path.mkdir(parents=True)
        (version_path / "model.bin").write_text("model data")

        metadata = ModelMetadata(
            name="test-model",
            version="1.0.0",
            status=VersionStatus.ACTIVE,
        )
        return ModelVersion(metadata=metadata, path=version_path)

    def test_name_property(self, model_version):
        """Test name property."""
        assert model_version.name == "test-model"

    def test_version_property(self, model_version):
        """Test version property."""
        assert model_version.version == "1.0.0"

    def test_semantic_version_property(self, model_version):
        """Test semantic_version property."""
        sv = model_version.semantic_version
        assert sv.major == 1
        assert sv.minor == 0

    def test_status_property(self, model_version):
        """Test status property."""
        assert model_version.status == VersionStatus.ACTIVE

    def test_is_active_property(self, model_version):
        """Test is_active property."""
        assert model_version.is_active is True

    def test_get_artifact_path(self, model_version):
        """Test getting artifact path."""
        path = model_version.get_artifact_path("model.bin")
        assert path.name == "model.bin"

    def test_has_artifact(self, model_version):
        """Test checking artifact existence."""
        assert model_version.has_artifact("model.bin") is True
        assert model_version.has_artifact("nonexistent.bin") is False

    def test_list_artifacts(self, model_version):
        """Test listing artifacts."""
        artifacts = model_version.list_artifacts()
        assert "model.bin" in artifacts


# ============================================================================
# Integration Tests
# ============================================================================


class TestModelRegistryIntegration:
    """Integration tests for the model registry."""

    def test_full_lifecycle(self, tmp_path):
        """Test full model lifecycle: register -> activate -> deprecate -> archive."""
        registry = ModelRegistry(tmp_path / "registry")

        # Register initial version
        v1 = registry.register_version(
            "lifecycle-model",
            "1.0.0",
            metadata=ModelMetadata(
                name="lifecycle-model",
                version="1.0.0",
                changelog="Initial release",
            ),
            activate=True,
        )
        assert v1.status == VersionStatus.ACTIVE

        # Register improved version
        v2 = registry.register_version(
            "lifecycle-model",
            "1.1.0",
            metadata=ModelMetadata(
                name="lifecycle-model",
                version="1.1.0",
                parent_version="1.0.0",
                changelog="Bug fixes",
            ),
        )
        assert v2.status == VersionStatus.DRAFT

        # Activate new version
        registry.activate_version("lifecycle-model", "1.1.0")

        # Verify old version is staged
        v1_updated = registry.get_version("lifecycle-model", "1.0.0")
        assert v1_updated.status == VersionStatus.STAGED

        # Deprecate old version
        registry.deprecate_version("lifecycle-model", "1.0.0", "Superseded by 1.1.0")
        v1_deprecated = registry.get_version("lifecycle-model", "1.0.0")
        assert v1_deprecated.status == VersionStatus.DEPRECATED

        # Archive old version
        registry.archive_version("lifecycle-model", "1.0.0")
        v1_archived = registry.get_version("lifecycle-model", "1.0.0")
        assert v1_archived.status == VersionStatus.ARCHIVED

        # Verify current version
        current = registry.get_current_version("lifecycle-model")
        assert current.version == "1.1.0"
        assert current.status == VersionStatus.ACTIVE

    def test_model_with_training_info(self, tmp_path):
        """Test model with full training information."""
        registry = ModelRegistry(tmp_path / "registry")

        # Create comprehensive metadata
        metadata = ModelMetadata(
            name="trained-model",
            version="1.0.0",
            model_type="transformer",
            base_model="distilbert-base-uncased",
            task_type="sentiment",
            data_path="data/sentiment.jsonl",
            n_records=10000,
            training_config=TrainingConfig(
                epochs=10,
                batch_size=16,
                learning_rate=2e-5,
                use_peft=True,
                lora_r=8,
                lora_alpha=16,
            ),
            training_metrics=TrainingMetrics(
                train_loss=0.15,
                eval_loss=0.20,
                accuracy=0.93,
                f1_score=0.91,
            ),
            model_card=ModelCard(
                description="Sentiment analysis model trained on product reviews",
                intended_use="Analyze sentiment of English text",
                limitations="Only works with English text",
            ),
            tags=["sentiment", "english", "production-ready"],
        )

        version = registry.register_version(
            "trained-model",
            "1.0.0",
            metadata=metadata,
            activate=True,
        )

        # Verify all data is stored
        retrieved = registry.get_version("trained-model", "1.0.0")
        assert retrieved.metadata.base_model == "distilbert-base-uncased"
        assert retrieved.metadata.training_config.lora_r == 8
        assert retrieved.metadata.training_metrics.accuracy == 0.93
        assert "sentiment" in retrieved.metadata.tags

        # Check model card was saved
        assert retrieved.has_artifact("MODEL_CARD.md")
