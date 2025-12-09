"""
A/B Testing Experiment Models

Defines the core data structures for A/B testing experiments.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ExperimentStatus(str, Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"           # Not yet started
    RUNNING = "running"       # Actively collecting data
    PAUSED = "paused"         # Temporarily stopped
    COMPLETED = "completed"   # Finished, results available
    CANCELLED = "cancelled"   # Stopped without results


@dataclass
class Variant:
    """
    A variant in an A/B test experiment.

    Each variant represents a different model configuration to test.
    """
    id: str
    name: str
    model_id: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    is_control: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "model_id": self.model_id,
            "description": self.description,
            "config": self.config,
            "is_control": self.is_control,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Variant":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            model_id=data["model_id"],
            description=data.get("description", ""),
            config=data.get("config", {}),
            is_control=data.get("is_control", False),
        )


@dataclass
class TrafficAllocation:
    """
    Traffic allocation configuration for variants.

    Defines how traffic is split between variants.
    """
    variant_id: str
    percentage: float  # 0.0 to 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "percentage": self.percentage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrafficAllocation":
        """Create from dictionary."""
        return cls(
            variant_id=data["variant_id"],
            percentage=data["percentage"],
        )


@dataclass
class ExperimentConfig:
    """
    Configuration for an A/B test experiment.
    """
    # Minimum sample size per variant before analysis
    min_sample_size: int = 100
    # Confidence level for statistical significance (e.g., 0.95 for 95%)
    confidence_level: float = 0.95
    # Primary metric to optimize
    primary_metric: str = "latency_ms"
    # Secondary metrics to track
    secondary_metrics: List[str] = field(default_factory=lambda: ["success_rate", "tokens_per_second"])
    # Auto-stop if winner is detected
    auto_stop_on_significance: bool = False
    # Maximum duration in hours (0 = no limit)
    max_duration_hours: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_sample_size": self.min_sample_size,
            "confidence_level": self.confidence_level,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "auto_stop_on_significance": self.auto_stop_on_significance,
            "max_duration_hours": self.max_duration_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            min_sample_size=data.get("min_sample_size", 100),
            confidence_level=data.get("confidence_level", 0.95),
            primary_metric=data.get("primary_metric", "latency_ms"),
            secondary_metrics=data.get("secondary_metrics", ["success_rate", "tokens_per_second"]),
            auto_stop_on_significance=data.get("auto_stop_on_significance", False),
            max_duration_hours=data.get("max_duration_hours", 0),
        )


@dataclass
class Experiment:
    """
    An A/B test experiment for comparing model variants.
    """
    id: str
    name: str
    description: str
    variants: List[Variant]
    traffic_allocation: List[TrafficAllocation]
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT
    tenant_id: Optional[str] = None  # Optional tenant scoping
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experiment configuration."""
        self._validate()

    def _validate(self):
        """Validate the experiment configuration."""
        if not self.variants:
            raise ValueError("Experiment must have at least one variant")

        if not self.traffic_allocation:
            raise ValueError("Experiment must have traffic allocation")

        # Validate traffic allocation sums to 100%
        total_percentage = sum(ta.percentage for ta in self.traffic_allocation)
        if not (99.9 <= total_percentage <= 100.1):
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_percentage}%")

        # Validate all variants have traffic allocation
        variant_ids = {v.id for v in self.variants}
        allocation_ids = {ta.variant_id for ta in self.traffic_allocation}
        if variant_ids != allocation_ids:
            raise ValueError("Traffic allocation must cover all variants")

        # Validate exactly one control variant
        control_variants = [v for v in self.variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Experiment must have exactly one control variant")

    def get_variant(self, variant_id: str) -> Optional[Variant]:
        """Get a variant by ID."""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None

    def get_control_variant(self) -> Variant:
        """Get the control variant."""
        for variant in self.variants:
            if variant.is_control:
                return variant
        raise ValueError("No control variant found")

    def get_allocation(self, variant_id: str) -> float:
        """Get traffic allocation percentage for a variant."""
        for ta in self.traffic_allocation:
            if ta.variant_id == variant_id:
                return ta.percentage
        return 0.0

    def assign_variant(self, user_id: str) -> Variant:
        """
        Assign a variant to a user based on consistent hashing.

        Uses the user_id + experiment_id to ensure consistent assignment
        (same user always gets the same variant for this experiment).
        """
        # Create consistent hash from user_id + experiment_id
        hash_input = f"{user_id}:{self.id}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        hash_value = int.from_bytes(hash_bytes[:4], 'big') % 10000 / 100  # 0.00 to 99.99

        # Assign based on traffic allocation
        cumulative = 0.0
        for ta in self.traffic_allocation:
            cumulative += ta.percentage
            if hash_value < cumulative:
                return self.get_variant(ta.variant_id)

        # Fallback to last variant
        return self.variants[-1]

    def can_start(self) -> bool:
        """Check if experiment can be started."""
        return self.status == ExperimentStatus.DRAFT

    def can_stop(self) -> bool:
        """Check if experiment can be stopped."""
        return self.status in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED)

    def start(self):
        """Start the experiment."""
        if not self.can_start():
            raise ValueError(f"Cannot start experiment in status: {self.status}")
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()

    def pause(self):
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError("Can only pause running experiments")
        self.status = ExperimentStatus.PAUSED
        self.updated_at = datetime.utcnow().isoformat()

    def resume(self):
        """Resume a paused experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError("Can only resume paused experiments")
        self.status = ExperimentStatus.RUNNING
        self.updated_at = datetime.utcnow().isoformat()

    def complete(self):
        """Mark experiment as completed."""
        if not self.can_stop():
            raise ValueError(f"Cannot complete experiment in status: {self.status}")
        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()

    def cancel(self):
        """Cancel the experiment."""
        if self.status == ExperimentStatus.COMPLETED:
            raise ValueError("Cannot cancel a completed experiment")
        self.status = ExperimentStatus.CANCELLED
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": [v.to_dict() for v in self.variants],
            "traffic_allocation": [ta.to_dict() for ta in self.traffic_allocation],
            "config": self.config.to_dict(),
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            variants=[Variant.from_dict(v) for v in data["variants"]],
            traffic_allocation=[TrafficAllocation.from_dict(ta) for ta in data["traffic_allocation"]],
            config=ExperimentConfig.from_dict(data.get("config", {})),
            status=ExperimentStatus(data.get("status", "draft")),
            tenant_id=data.get("tenant_id"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {}),
        )


def create_experiment(
    name: str,
    variants: List[Dict[str, Any]],
    traffic_split: Optional[List[float]] = None,
    description: str = "",
    config: Optional[Dict[str, Any]] = None,
    tenant_id: Optional[str] = None,
) -> Experiment:
    """
    Helper function to create an experiment.

    Args:
        name: Experiment name
        variants: List of variant configs with at least 'name' and 'model_id'
        traffic_split: Optional list of percentages (must sum to 100)
        description: Experiment description
        config: Optional experiment config
        tenant_id: Optional tenant ID for scoping

    Returns:
        Created experiment
    """
    experiment_id = str(uuid.uuid4())

    # Create variants
    variant_objects = []
    for i, v in enumerate(variants):
        variant_objects.append(Variant(
            id=str(uuid.uuid4()),
            name=v.get("name", f"Variant {i}"),
            model_id=v["model_id"],
            description=v.get("description", ""),
            config=v.get("config", {}),
            is_control=(i == 0),  # First variant is control
        ))

    # Create traffic allocation
    if traffic_split is None:
        # Equal split by default
        equal_pct = 100.0 / len(variant_objects)
        traffic_split = [equal_pct] * len(variant_objects)

    traffic_allocation = [
        TrafficAllocation(variant_id=v.id, percentage=pct)
        for v, pct in zip(variant_objects, traffic_split)
    ]

    # Create experiment config
    exp_config = ExperimentConfig.from_dict(config or {})

    return Experiment(
        id=experiment_id,
        name=name,
        description=description,
        variants=variant_objects,
        traffic_allocation=traffic_allocation,
        config=exp_config,
        tenant_id=tenant_id,
    )
