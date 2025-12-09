"""
A/B Testing Manager

Central manager for creating, managing, and running A/B test experiments.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from services.ab_testing.experiment import (
    Experiment,
    ExperimentConfig,
    ExperimentStatus,
    TrafficAllocation,
    Variant,
    create_experiment,
)
from services.ab_testing.metrics import ExperimentMetrics, MetricsCollector
from services.ab_testing.analysis import StatisticalAnalysis, analyze_experiment

logger = logging.getLogger(__name__)


class ABTestManager:
    """
    Manager for A/B testing experiments.

    Handles experiment lifecycle, variant assignment, and metrics collection.
    """

    def __init__(self, db_path: str = "data/ab_testing.db"):
        """
        Initialize A/B test manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.Lock()
        self.metrics_collector = MetricsCollector(db_path=db_path)

        self._init_db()
        self._load_experiments()

    def _init_db(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                experiment_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiment_name
            ON experiments(name)
        """)

        conn.commit()
        conn.close()

    def _load_experiments(self):
        """Load experiments from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, experiment_json FROM experiments")

        for row in cursor.fetchall():
            try:
                data = json.loads(row[1])
                experiment = Experiment.from_dict(data)
                self._experiments[experiment.id] = experiment
            except Exception as e:
                logger.error(f"Failed to load experiment {row[0]}: {e}")

        conn.close()
        logger.info(f"Loaded {len(self._experiments)} experiments")

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO experiments (id, name, experiment_json, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            experiment.id,
            experiment.name,
            json.dumps(experiment.to_dict()),
            datetime.utcnow().isoformat(),
        ))
        conn.commit()
        conn.close()

    def create_experiment(
        self,
        name: str,
        variants: List[Dict[str, Any]],
        traffic_split: Optional[List[float]] = None,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Experiment:
        """
        Create a new A/B test experiment.

        Args:
            name: Experiment name
            variants: List of variant configs with 'name' and 'model_id'
            traffic_split: Optional traffic split percentages
            description: Experiment description
            config: Optional experiment configuration
            tenant_id: Optional tenant ID for scoping

        Returns:
            Created experiment

        Example:
            manager.create_experiment(
                name="Model Comparison",
                variants=[
                    {"name": "Control", "model_id": "distilbert-v1"},
                    {"name": "Treatment", "model_id": "distilbert-v2"},
                ],
                traffic_split=[50, 50],
            )
        """
        experiment = create_experiment(
            name=name,
            variants=variants,
            traffic_split=traffic_split,
            description=description,
            config=config,
            tenant_id=tenant_id,
        )

        with self._lock:
            self._experiments[experiment.id] = experiment
            self._save_experiment(experiment)

        logger.info(f"Created experiment: {experiment.id} - {name}")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        for exp in self._experiments.values():
            if exp.name == name:
                return exp
        return None

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            status: Filter by status
            tenant_id: Filter by tenant ID

        Returns:
            List of experiments
        """
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]

        if tenant_id:
            experiments = [e for e in experiments if e.tenant_id == tenant_id]

        return experiments

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        with self._lock:
            experiment.start()
            self._save_experiment(experiment)

        logger.info(f"Started experiment: {experiment_id}")
        return experiment

    def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        with self._lock:
            experiment.pause()
            self._save_experiment(experiment)

        logger.info(f"Paused experiment: {experiment_id}")
        return experiment

    def resume_experiment(self, experiment_id: str) -> Experiment:
        """Resume a paused experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        with self._lock:
            experiment.resume()
            self._save_experiment(experiment)

        logger.info(f"Resumed experiment: {experiment_id}")
        return experiment

    def complete_experiment(self, experiment_id: str) -> Experiment:
        """Mark an experiment as completed."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        with self._lock:
            experiment.complete()
            self._save_experiment(experiment)

        logger.info(f"Completed experiment: {experiment_id}")
        return experiment

    def cancel_experiment(self, experiment_id: str) -> Experiment:
        """Cancel an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        with self._lock:
            experiment.cancel()
            self._save_experiment(experiment)

        logger.info(f"Cancelled experiment: {experiment_id}")
        return experiment

    def delete_experiment(self, experiment_id: str, force: bool = False) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID
            force: Force delete even if running

        Returns:
            True if deleted
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        if experiment.status == ExperimentStatus.RUNNING and not force:
            raise ValueError("Cannot delete running experiment. Use force=True or stop it first.")

        with self._lock:
            del self._experiments[experiment_id]

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            conn.commit()
            conn.close()

            # Clear metrics
            self.metrics_collector.clear_metrics(experiment_id)

        logger.info(f"Deleted experiment: {experiment_id}")
        return True

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Optional[Variant]:
        """
        Assign a variant to a user for an experiment.

        Args:
            experiment_id: Experiment ID
            user_id: User ID for consistent assignment

        Returns:
            Assigned variant or None if experiment not running
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        if experiment.status != ExperimentStatus.RUNNING:
            return None

        return experiment.assign_variant(user_id)

    def get_active_experiment_for_tenant(
        self,
        tenant_id: str,
    ) -> Optional[Experiment]:
        """Get the active (running) experiment for a tenant."""
        for exp in self._experiments.values():
            if exp.tenant_id == tenant_id and exp.status == ExperimentStatus.RUNNING:
                return exp
        return None

    def record_request(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        success: bool,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a request result for an experiment.

        Args:
            experiment_id: Experiment ID
            variant_id: Variant that was used
            user_id: User ID
            success: Whether request succeeded
            latency_ms: Request latency
            tokens_in: Input tokens
            tokens_out: Output tokens
            metadata: Additional metadata
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return

        self.metrics_collector.record_request(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            success=success,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metadata=metadata,
        )

        # Check for auto-stop if configured
        if experiment.config.auto_stop_on_significance:
            self._check_auto_stop(experiment)

    def _check_auto_stop(self, experiment: Experiment):
        """Check if experiment should auto-stop based on significance."""
        metrics = self.get_metrics(experiment.id)

        # Need minimum samples
        min_required = experiment.config.min_sample_size
        for vm in metrics.variant_metrics.values():
            if vm.request_count < min_required:
                return

        # Analyze and check for significance
        analysis = self.analyze_experiment(experiment.id)
        if analysis and any(r.is_significant for r in analysis.comparisons.values()):
            logger.info(f"Auto-stopping experiment {experiment.id} - significance detected")
            self.complete_experiment(experiment.id)

    def get_metrics(self, experiment_id: str) -> ExperimentMetrics:
        """Get metrics for an experiment."""
        return self.metrics_collector.get_metrics(experiment_id)

    def analyze_experiment(self, experiment_id: str) -> Optional[StatisticalAnalysis]:
        """
        Perform statistical analysis on an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Statistical analysis or None if experiment not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        metrics = self.get_metrics(experiment_id)

        # Get latency samples for more accurate analysis
        latency_samples = {}
        for variant in experiment.variants:
            samples = self.metrics_collector.get_latency_samples(
                experiment_id, variant.id, limit=10000
            )
            if samples:
                latency_samples[variant.id] = samples

        return analyze_experiment(experiment, metrics, latency_samples)

    def get_experiment_summary(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of an experiment including metrics and analysis.

        Args:
            experiment_id: Experiment ID

        Returns:
            Summary dict or None if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        metrics = self.get_metrics(experiment_id)
        analysis = self.analyze_experiment(experiment_id)

        return {
            "experiment": experiment.to_dict(),
            "metrics": metrics.to_dict(),
            "analysis": analysis.to_dict() if analysis else None,
        }

    def update_traffic_allocation(
        self,
        experiment_id: str,
        allocations: Dict[str, float],
    ) -> Experiment:
        """
        Update traffic allocation for an experiment.

        Args:
            experiment_id: Experiment ID
            allocations: Dict of variant_id -> percentage

        Returns:
            Updated experiment
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Validate allocations sum to 100
        total = sum(allocations.values())
        if not (99.9 <= total <= 100.1):
            raise ValueError(f"Allocations must sum to 100%, got {total}%")

        # Create new traffic allocation
        new_allocations = [
            TrafficAllocation(variant_id=vid, percentage=pct)
            for vid, pct in allocations.items()
        ]

        with self._lock:
            experiment.traffic_allocation = new_allocations
            experiment.updated_at = datetime.utcnow().isoformat()
            self._save_experiment(experiment)

        logger.info(f"Updated traffic allocation for experiment: {experiment_id}")
        return experiment
