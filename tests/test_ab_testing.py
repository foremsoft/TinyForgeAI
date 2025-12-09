"""
Tests for A/B Testing Module

Comprehensive tests for experiment management, metrics collection,
and statistical analysis.
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from services.ab_testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    TrafficAllocation,
    Variant,
    create_experiment,
)
from services.ab_testing.metrics import (
    ExperimentMetrics,
    MetricsCollector,
    RequestRecord,
    VariantMetrics,
)
from services.ab_testing.analysis import (
    SignificanceResult,
    StatisticalAnalysis,
    analyze_experiment,
    calculate_required_sample_size,
    compare_variants,
)
from services.ab_testing.manager import ABTestManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_ab.db")
        yield db_path


@pytest.fixture
def sample_variants():
    """Create sample variants."""
    return [
        Variant(
            id="control-1",
            name="Control",
            model_id="model-v1",
            description="Control variant",
            is_control=True,
        ),
        Variant(
            id="treatment-1",
            name="Treatment A",
            model_id="model-v2",
            description="Treatment variant",
            is_control=False,
        ),
    ]


@pytest.fixture
def sample_allocations(sample_variants):
    """Create sample traffic allocations."""
    return [
        TrafficAllocation(variant_id=sample_variants[0].id, percentage=50.0),
        TrafficAllocation(variant_id=sample_variants[1].id, percentage=50.0),
    ]


@pytest.fixture
def sample_experiment(sample_variants, sample_allocations):
    """Create a sample experiment."""
    return Experiment(
        id="exp-1",
        name="Test Experiment",
        description="A test experiment",
        variants=sample_variants,
        traffic_allocation=sample_allocations,
        config=ExperimentConfig(),
    )


@pytest.fixture
def manager(temp_db):
    """Create an A/B test manager."""
    return ABTestManager(db_path=temp_db)


# =============================================================================
# Variant Tests
# =============================================================================


class TestVariant:
    """Tests for Variant class."""

    def test_variant_creation(self):
        """Test variant creation."""
        variant = Variant(
            id="v1",
            name="Test Variant",
            model_id="model-1",
            description="Test description",
            is_control=True,
        )
        assert variant.id == "v1"
        assert variant.name == "Test Variant"
        assert variant.model_id == "model-1"
        assert variant.is_control is True

    def test_variant_to_dict(self):
        """Test variant serialization."""
        variant = Variant(
            id="v1",
            name="Test",
            model_id="model-1",
            config={"key": "value"},
            is_control=False,
        )
        data = variant.to_dict()
        assert data["id"] == "v1"
        assert data["config"]["key"] == "value"

    def test_variant_from_dict(self):
        """Test variant deserialization."""
        data = {
            "id": "v1",
            "name": "Test",
            "model_id": "model-1",
            "description": "Desc",
            "config": {"key": "value"},
            "is_control": True,
        }
        variant = Variant.from_dict(data)
        assert variant.id == "v1"
        assert variant.is_control is True


# =============================================================================
# TrafficAllocation Tests
# =============================================================================


class TestTrafficAllocation:
    """Tests for TrafficAllocation class."""

    def test_allocation_creation(self):
        """Test traffic allocation creation."""
        allocation = TrafficAllocation(variant_id="v1", percentage=50.0)
        assert allocation.variant_id == "v1"
        assert allocation.percentage == 50.0

    def test_allocation_serialization(self):
        """Test allocation serialization/deserialization."""
        allocation = TrafficAllocation(variant_id="v1", percentage=75.5)
        data = allocation.to_dict()
        restored = TrafficAllocation.from_dict(data)
        assert restored.variant_id == allocation.variant_id
        assert restored.percentage == allocation.percentage


# =============================================================================
# ExperimentConfig Tests
# =============================================================================


class TestExperimentConfig:
    """Tests for ExperimentConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()
        assert config.min_sample_size == 100
        assert config.confidence_level == 0.95
        assert config.primary_metric == "latency_ms"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExperimentConfig(
            min_sample_size=500,
            confidence_level=0.99,
            primary_metric="success_rate",
            auto_stop_on_significance=True,
        )
        assert config.min_sample_size == 500
        assert config.confidence_level == 0.99
        assert config.auto_stop_on_significance is True

    def test_config_serialization(self):
        """Test config serialization."""
        config = ExperimentConfig(min_sample_size=200)
        data = config.to_dict()
        restored = ExperimentConfig.from_dict(data)
        assert restored.min_sample_size == 200


# =============================================================================
# Experiment Tests
# =============================================================================


class TestExperiment:
    """Tests for Experiment class."""

    def test_experiment_creation(self, sample_variants, sample_allocations):
        """Test experiment creation."""
        exp = Experiment(
            id="exp-1",
            name="Test",
            description="Description",
            variants=sample_variants,
            traffic_allocation=sample_allocations,
            config=ExperimentConfig(),
        )
        assert exp.id == "exp-1"
        assert exp.status == ExperimentStatus.DRAFT
        assert len(exp.variants) == 2

    def test_experiment_validation_no_variants(self):
        """Test validation fails with no variants."""
        with pytest.raises(ValueError, match="at least one variant"):
            Experiment(
                id="exp-1",
                name="Test",
                description="",
                variants=[],
                traffic_allocation=[],
                config=ExperimentConfig(),
            )

    def test_experiment_validation_no_control(self):
        """Test validation fails with no control variant."""
        variants = [
            Variant(id="v1", name="A", model_id="m1", is_control=False),
            Variant(id="v2", name="B", model_id="m2", is_control=False),
        ]
        allocations = [
            TrafficAllocation(variant_id="v1", percentage=50.0),
            TrafficAllocation(variant_id="v2", percentage=50.0),
        ]
        with pytest.raises(ValueError, match="exactly one control"):
            Experiment(
                id="exp-1",
                name="Test",
                description="",
                variants=variants,
                traffic_allocation=allocations,
                config=ExperimentConfig(),
            )

    def test_experiment_validation_traffic_sum(self, sample_variants):
        """Test validation fails if traffic doesn't sum to 100."""
        allocations = [
            TrafficAllocation(variant_id=sample_variants[0].id, percentage=30.0),
            TrafficAllocation(variant_id=sample_variants[1].id, percentage=30.0),
        ]
        with pytest.raises(ValueError, match="sum to 100"):
            Experiment(
                id="exp-1",
                name="Test",
                description="",
                variants=sample_variants,
                traffic_allocation=allocations,
                config=ExperimentConfig(),
            )

    def test_experiment_get_variant(self, sample_experiment):
        """Test getting a variant by ID."""
        variant = sample_experiment.get_variant("control-1")
        assert variant is not None
        assert variant.is_control is True

        missing = sample_experiment.get_variant("nonexistent")
        assert missing is None

    def test_experiment_get_control_variant(self, sample_experiment):
        """Test getting the control variant."""
        control = sample_experiment.get_control_variant()
        assert control.is_control is True
        assert control.id == "control-1"

    def test_experiment_assign_variant_consistent(self, sample_experiment):
        """Test variant assignment is consistent for same user."""
        sample_experiment.start()

        # Same user should always get same variant
        variant1 = sample_experiment.assign_variant("user-123")
        variant2 = sample_experiment.assign_variant("user-123")
        assert variant1.id == variant2.id

    def test_experiment_assign_variant_distribution(self, sample_experiment):
        """Test variant assignment distributes traffic."""
        sample_experiment.start()

        # Assign many users and check distribution
        control_count = 0
        treatment_count = 0
        for i in range(1000):
            variant = sample_experiment.assign_variant(f"user-{i}")
            if variant.is_control:
                control_count += 1
            else:
                treatment_count += 1

        # With 50/50 split, should be roughly equal
        assert 400 < control_count < 600
        assert 400 < treatment_count < 600

    def test_experiment_lifecycle(self, sample_experiment):
        """Test experiment lifecycle transitions."""
        # Start from draft
        assert sample_experiment.can_start()
        sample_experiment.start()
        assert sample_experiment.status == ExperimentStatus.RUNNING
        assert sample_experiment.started_at is not None

        # Pause
        sample_experiment.pause()
        assert sample_experiment.status == ExperimentStatus.PAUSED

        # Resume
        sample_experiment.resume()
        assert sample_experiment.status == ExperimentStatus.RUNNING

        # Complete
        sample_experiment.complete()
        assert sample_experiment.status == ExperimentStatus.COMPLETED
        assert sample_experiment.completed_at is not None

    def test_experiment_cannot_start_twice(self, sample_experiment):
        """Test cannot start an already started experiment."""
        sample_experiment.start()
        with pytest.raises(ValueError):
            sample_experiment.start()

    def test_experiment_serialization(self, sample_experiment):
        """Test experiment serialization/deserialization."""
        data = sample_experiment.to_dict()
        restored = Experiment.from_dict(data)
        assert restored.id == sample_experiment.id
        assert restored.name == sample_experiment.name
        assert len(restored.variants) == len(sample_experiment.variants)


class TestCreateExperiment:
    """Tests for create_experiment helper."""

    def test_create_experiment_basic(self):
        """Test basic experiment creation."""
        exp = create_experiment(
            name="Test Experiment",
            variants=[
                {"name": "Control", "model_id": "model-v1"},
                {"name": "Treatment", "model_id": "model-v2"},
            ],
        )
        assert exp.name == "Test Experiment"
        assert len(exp.variants) == 2
        assert exp.variants[0].is_control is True
        assert exp.variants[1].is_control is False

    def test_create_experiment_with_traffic_split(self):
        """Test experiment creation with custom traffic split."""
        exp = create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
            traffic_split=[80, 20],
        )
        assert exp.traffic_allocation[0].percentage == 80
        assert exp.traffic_allocation[1].percentage == 20


# =============================================================================
# VariantMetrics Tests
# =============================================================================


class TestVariantMetrics:
    """Tests for VariantMetrics class."""

    def test_metrics_initial_values(self):
        """Test initial metric values."""
        metrics = VariantMetrics(variant_id="v1")
        assert metrics.request_count == 0
        assert metrics.success_rate == 0.0
        assert metrics.avg_latency_ms == 0.0

    def test_metrics_record(self):
        """Test recording metrics."""
        metrics = VariantMetrics(variant_id="v1")
        metrics.record(success=True, latency_ms=100.0, tokens_in=10, tokens_out=20)
        metrics.record(success=True, latency_ms=200.0, tokens_in=15, tokens_out=25)
        metrics.record(success=False, latency_ms=50.0, tokens_in=5, tokens_out=0)

        assert metrics.request_count == 3
        assert metrics.success_count == 2
        assert metrics.error_count == 1
        assert metrics.avg_latency_ms == pytest.approx(116.67, rel=0.01)
        assert metrics.success_rate == pytest.approx(0.667, rel=0.01)

    def test_metrics_min_max_latency(self):
        """Test min/max latency tracking."""
        metrics = VariantMetrics(variant_id="v1")
        metrics.record(success=True, latency_ms=50.0)
        metrics.record(success=True, latency_ms=200.0)
        metrics.record(success=True, latency_ms=100.0)

        assert metrics.min_latency_ms == 50.0
        assert metrics.max_latency_ms == 200.0

    def test_metrics_serialization(self):
        """Test metrics serialization."""
        metrics = VariantMetrics(variant_id="v1")
        metrics.record(success=True, latency_ms=100.0)
        data = metrics.to_dict()
        restored = VariantMetrics.from_dict(data)
        assert restored.request_count == metrics.request_count


# =============================================================================
# ExperimentMetrics Tests
# =============================================================================


class TestExperimentMetrics:
    """Tests for ExperimentMetrics class."""

    def test_experiment_metrics_record(self):
        """Test recording experiment metrics."""
        metrics = ExperimentMetrics(experiment_id="exp-1")
        metrics.record(variant_id="v1", success=True, latency_ms=100.0)
        metrics.record(variant_id="v2", success=True, latency_ms=150.0)

        assert metrics.total_requests == 2
        assert "v1" in metrics.variant_metrics
        assert "v2" in metrics.variant_metrics

    def test_experiment_metrics_get_variant(self):
        """Test getting variant metrics."""
        metrics = ExperimentMetrics(experiment_id="exp-1")
        vm = metrics.get_variant_metrics("v1")
        assert vm.variant_id == "v1"
        assert vm.request_count == 0


# =============================================================================
# MetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_collector_in_memory(self):
        """Test in-memory metrics collection."""
        collector = MetricsCollector()
        collector.record_request(
            experiment_id="exp-1",
            variant_id="v1",
            user_id="user-1",
            success=True,
            latency_ms=100.0,
        )
        metrics = collector.get_metrics("exp-1")
        assert metrics.total_requests == 1

    def test_collector_with_db(self, temp_db):
        """Test metrics collection with database persistence."""
        collector = MetricsCollector(db_path=temp_db)
        collector.record_request(
            experiment_id="exp-1",
            variant_id="v1",
            user_id="user-1",
            success=True,
            latency_ms=100.0,
            tokens_in=10,
            tokens_out=20,
        )

        # Create new collector to test persistence
        collector2 = MetricsCollector(db_path=temp_db)
        metrics = collector2.get_metrics("exp-1")
        assert metrics.total_requests == 1

    def test_collector_get_latency_samples(self, temp_db):
        """Test getting latency samples."""
        collector = MetricsCollector(db_path=temp_db)

        for i in range(10):
            collector.record_request(
                experiment_id="exp-1",
                variant_id="v1",
                user_id=f"user-{i}",
                success=True,
                latency_ms=100.0 + i * 10,
            )

        samples = collector.get_latency_samples("exp-1", "v1")
        assert len(samples) == 10

    def test_collector_clear_metrics(self, temp_db):
        """Test clearing metrics."""
        collector = MetricsCollector(db_path=temp_db)
        collector.record_request(
            experiment_id="exp-1",
            variant_id="v1",
            user_id="user-1",
            success=True,
            latency_ms=100.0,
        )

        collector.clear_metrics("exp-1")
        metrics = collector.get_metrics("exp-1")
        assert metrics.total_requests == 0


# =============================================================================
# Statistical Analysis Tests
# =============================================================================


class TestStatisticalAnalysis:
    """Tests for statistical analysis functions."""

    def test_compare_variants_insufficient_data(self):
        """Test comparison with insufficient data."""
        control = VariantMetrics(variant_id="control")
        treatment = VariantMetrics(variant_id="treatment")

        result = compare_variants(control, treatment)
        assert result.is_significant is False
        assert "Insufficient" in result.interpretation

    def test_compare_variants_no_difference(self):
        """Test comparison with no significant difference."""
        control = VariantMetrics(variant_id="control")
        treatment = VariantMetrics(variant_id="treatment")

        # Add similar data
        for i in range(100):
            control.record(success=True, latency_ms=100.0 + i * 0.1)
            treatment.record(success=True, latency_ms=100.0 + i * 0.1)

        result = compare_variants(control, treatment)
        # Should not be significant if data is very similar
        assert result.p_value > 0.05 or abs(result.effect_size) < 0.2

    def test_compare_variants_significant_difference(self):
        """Test comparison with significant difference."""
        control = VariantMetrics(variant_id="control")
        treatment = VariantMetrics(variant_id="treatment")

        # Control: higher latency
        for _ in range(100):
            control.record(success=True, latency_ms=200.0)

        # Treatment: lower latency (better)
        for _ in range(100):
            treatment.record(success=True, latency_ms=100.0)

        result = compare_variants(
            control, treatment,
            control_samples=[200.0] * 100,
            treatment_samples=[100.0] * 100,
        )
        assert result.is_significant is True
        assert result.winner == "treatment"

    def test_analyze_experiment(self, sample_experiment):
        """Test full experiment analysis."""
        metrics = ExperimentMetrics(experiment_id=sample_experiment.id)

        # Add some data
        for i in range(100):
            metrics.record("control-1", success=True, latency_ms=150.0)
            metrics.record("treatment-1", success=True, latency_ms=120.0)

        analysis = analyze_experiment(sample_experiment, metrics)
        assert analysis.experiment_id == sample_experiment.id
        assert analysis.sufficient_sample_size is True
        assert "treatment-1" in analysis.comparisons

    def test_calculate_required_sample_size(self):
        """Test sample size calculation."""
        # For 10% baseline, detecting 5% relative change
        n = calculate_required_sample_size(
            baseline_rate=0.10,
            minimum_detectable_effect=0.05,
        )
        assert n > 0
        assert n < 1000000  # Sanity check


class TestSignificanceResult:
    """Tests for SignificanceResult class."""

    def test_significance_result_to_dict(self):
        """Test result serialization."""
        result = SignificanceResult(
            is_significant=True,
            confidence_level=0.95,
            p_value=0.01,
            effect_size=0.5,
            effect_size_ci_lower=0.3,
            effect_size_ci_upper=0.7,
            winner="v1",
            interpretation="Treatment is better",
        )
        data = result.to_dict()
        assert data["is_significant"] is True
        assert data["winner"] == "v1"


# =============================================================================
# ABTestManager Tests
# =============================================================================


class TestABTestManager:
    """Tests for ABTestManager class."""

    def test_manager_create_experiment(self, manager):
        """Test creating an experiment through manager."""
        exp = manager.create_experiment(
            name="Test Experiment",
            variants=[
                {"name": "Control", "model_id": "model-v1"},
                {"name": "Treatment", "model_id": "model-v2"},
            ],
            description="Test description",
        )
        assert exp.name == "Test Experiment"
        assert len(exp.variants) == 2

    def test_manager_get_experiment(self, manager):
        """Test getting an experiment."""
        created = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        fetched = manager.get_experiment(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    def test_manager_list_experiments(self, manager):
        """Test listing experiments."""
        manager.create_experiment(
            name="Exp 1",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        manager.create_experiment(
            name="Exp 2",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )

        experiments = manager.list_experiments()
        assert len(experiments) == 2

    def test_manager_start_experiment(self, manager):
        """Test starting an experiment."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        started = manager.start_experiment(exp.id)
        assert started.status == ExperimentStatus.RUNNING

    def test_manager_assign_variant(self, manager):
        """Test variant assignment through manager."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        manager.start_experiment(exp.id)

        variant = manager.assign_variant(exp.id, "user-123")
        assert variant is not None

    def test_manager_assign_variant_not_running(self, manager):
        """Test assignment returns None for non-running experiment."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        # Not started
        variant = manager.assign_variant(exp.id, "user-123")
        assert variant is None

    def test_manager_record_and_get_metrics(self, manager):
        """Test recording and retrieving metrics."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        manager.start_experiment(exp.id)

        # Record some requests
        variant = exp.variants[0]
        manager.record_request(
            experiment_id=exp.id,
            variant_id=variant.id,
            user_id="user-1",
            success=True,
            latency_ms=100.0,
        )

        metrics = manager.get_metrics(exp.id)
        assert metrics.total_requests == 1

    def test_manager_delete_experiment(self, manager):
        """Test deleting an experiment."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        assert manager.delete_experiment(exp.id) is True
        assert manager.get_experiment(exp.id) is None

    def test_manager_delete_running_experiment(self, manager):
        """Test cannot delete running experiment without force."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        manager.start_experiment(exp.id)

        with pytest.raises(ValueError):
            manager.delete_experiment(exp.id)

        # Force delete should work
        assert manager.delete_experiment(exp.id, force=True) is True

    def test_manager_update_traffic_allocation(self, manager):
        """Test updating traffic allocation."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )

        # Get variant IDs
        allocations = {
            exp.variants[0].id: 80.0,
            exp.variants[1].id: 20.0,
        }
        updated = manager.update_traffic_allocation(exp.id, allocations)
        assert updated.get_allocation(exp.variants[0].id) == 80.0
        assert updated.get_allocation(exp.variants[1].id) == 20.0

    def test_manager_persistence(self, temp_db):
        """Test manager persists experiments."""
        manager1 = ABTestManager(db_path=temp_db)
        exp = manager1.create_experiment(
            name="Persistent Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )

        # Create new manager to test loading
        manager2 = ABTestManager(db_path=temp_db)
        loaded = manager2.get_experiment(exp.id)
        assert loaded is not None
        assert loaded.name == "Persistent Test"

    def test_manager_experiment_summary(self, manager):
        """Test getting experiment summary."""
        exp = manager.create_experiment(
            name="Test",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
        )
        manager.start_experiment(exp.id)

        # Record some data
        for v in exp.variants:
            for _ in range(10):
                manager.record_request(
                    experiment_id=exp.id,
                    variant_id=v.id,
                    user_id="user",
                    success=True,
                    latency_ms=100.0,
                )

        summary = manager.get_experiment_summary(exp.id)
        assert summary is not None
        assert "experiment" in summary
        assert "metrics" in summary
        assert "analysis" in summary


# =============================================================================
# Integration Tests
# =============================================================================


class TestABTestingIntegration:
    """Integration tests for the full A/B testing workflow."""

    def test_full_experiment_workflow(self, manager):
        """Test a complete A/B testing workflow."""
        # 1. Create experiment
        exp = manager.create_experiment(
            name="Model Comparison",
            description="Compare v1 vs v2 models",
            variants=[
                {"name": "v1 (Control)", "model_id": "model-v1"},
                {"name": "v2 (Treatment)", "model_id": "model-v2"},
            ],
            traffic_split=[50, 50],
            config={
                "min_sample_size": 50,
                "confidence_level": 0.95,
                "primary_metric": "latency_ms",
            },
        )
        assert exp.status == ExperimentStatus.DRAFT

        # 2. Start experiment
        manager.start_experiment(exp.id)
        assert manager.get_experiment(exp.id).status == ExperimentStatus.RUNNING

        # 3. Simulate traffic
        control_id = exp.variants[0].id
        treatment_id = exp.variants[1].id

        for i in range(100):
            user_id = f"user-{i}"
            variant = manager.assign_variant(exp.id, user_id)

            # Simulate different performance
            if variant.id == control_id:
                latency = 150.0 + (i % 20)  # Higher latency
            else:
                latency = 100.0 + (i % 20)  # Lower latency

            manager.record_request(
                experiment_id=exp.id,
                variant_id=variant.id,
                user_id=user_id,
                success=True,
                latency_ms=latency,
            )

        # 4. Check metrics
        metrics = manager.get_metrics(exp.id)
        assert metrics.total_requests == 100

        # 5. Analyze results
        analysis = manager.analyze_experiment(exp.id)
        assert analysis is not None
        assert analysis.sufficient_sample_size is True

        # 6. Complete experiment
        manager.complete_experiment(exp.id)
        assert manager.get_experiment(exp.id).status == ExperimentStatus.COMPLETED

        # 7. Get summary
        summary = manager.get_experiment_summary(exp.id)
        assert summary["experiment"]["status"] == "completed"

    def test_multi_variant_experiment(self, manager):
        """Test experiment with more than 2 variants."""
        exp = manager.create_experiment(
            name="Multi-Variant Test",
            variants=[
                {"name": "Control", "model_id": "model-v1"},
                {"name": "Treatment A", "model_id": "model-v2a"},
                {"name": "Treatment B", "model_id": "model-v2b"},
            ],
            traffic_split=[50, 25, 25],
        )
        assert len(exp.variants) == 3
        assert sum(ta.percentage for ta in exp.traffic_allocation) == 100

    def test_tenant_scoped_experiment(self, manager):
        """Test experiments scoped to tenants."""
        # Create experiments for different tenants
        exp1 = manager.create_experiment(
            name="Tenant 1 Exp",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
            tenant_id="tenant-1",
        )
        exp2 = manager.create_experiment(
            name="Tenant 2 Exp",
            variants=[
                {"name": "Control", "model_id": "m1"},
                {"name": "Treatment", "model_id": "m2"},
            ],
            tenant_id="tenant-2",
        )

        # List by tenant
        tenant1_exps = manager.list_experiments(tenant_id="tenant-1")
        assert len(tenant1_exps) == 1
        assert tenant1_exps[0].id == exp1.id

        tenant2_exps = manager.list_experiments(tenant_id="tenant-2")
        assert len(tenant2_exps) == 1
        assert tenant2_exps[0].id == exp2.id
