"""
A/B Testing Metrics Collection

Collects and stores metrics for A/B test experiments.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VariantMetrics:
    """Metrics for a single variant."""
    variant_id: str
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.total_latency_ms == 0:
            return 0.0
        return (self.total_tokens_out * 1000) / self.total_latency_ms

    @property
    def avg_tokens_in(self) -> float:
        """Calculate average input tokens."""
        if self.request_count == 0:
            return 0.0
        return self.total_tokens_in / self.request_count

    @property
    def avg_tokens_out(self) -> float:
        """Calculate average output tokens."""
        if self.request_count == 0:
            return 0.0
        return self.total_tokens_out / self.request_count

    def record(
        self,
        success: bool,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ):
        """Record a request result."""
        self.request_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1

        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant_id": self.variant_id,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0.0,
            "max_latency_ms": self.max_latency_ms,
            "success_rate": round(self.success_rate, 4),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "avg_tokens_in": round(self.avg_tokens_in, 2),
            "avg_tokens_out": round(self.avg_tokens_out, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariantMetrics":
        """Create from dictionary."""
        return cls(
            variant_id=data["variant_id"],
            request_count=data.get("request_count", 0),
            success_count=data.get("success_count", 0),
            error_count=data.get("error_count", 0),
            total_latency_ms=data.get("total_latency_ms", 0.0),
            total_tokens_in=data.get("total_tokens_in", 0),
            total_tokens_out=data.get("total_tokens_out", 0),
            min_latency_ms=data.get("min_latency_ms", float('inf')),
            max_latency_ms=data.get("max_latency_ms", 0.0),
        )


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment."""
    experiment_id: str
    variant_metrics: Dict[str, VariantMetrics] = field(default_factory=dict)
    started_at: Optional[str] = None
    last_updated: Optional[str] = None

    def get_variant_metrics(self, variant_id: str) -> VariantMetrics:
        """Get or create metrics for a variant."""
        if variant_id not in self.variant_metrics:
            self.variant_metrics[variant_id] = VariantMetrics(variant_id=variant_id)
        return self.variant_metrics[variant_id]

    def record(
        self,
        variant_id: str,
        success: bool,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ):
        """Record a request result for a variant."""
        metrics = self.get_variant_metrics(variant_id)
        metrics.record(success, latency_ms, tokens_in, tokens_out)
        self.last_updated = datetime.utcnow().isoformat()

    @property
    def total_requests(self) -> int:
        """Total requests across all variants."""
        return sum(m.request_count for m in self.variant_metrics.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "variant_metrics": {
                vid: vm.to_dict() for vid, vm in self.variant_metrics.items()
            },
            "total_requests": self.total_requests,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentMetrics":
        """Create from dictionary."""
        variant_metrics = {}
        for vid, vm_data in data.get("variant_metrics", {}).items():
            variant_metrics[vid] = VariantMetrics.from_dict(vm_data)

        return cls(
            experiment_id=data["experiment_id"],
            variant_metrics=variant_metrics,
            started_at=data.get("started_at"),
            last_updated=data.get("last_updated"),
        )


@dataclass
class RequestRecord:
    """Record of a single request for detailed analysis."""
    experiment_id: str
    variant_id: str
    user_id: str
    timestamp: str
    success: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and stores metrics for A/B testing experiments.

    Supports both in-memory storage and SQLite persistence.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize metrics collector.

        Args:
            db_path: Optional path to SQLite database for persistence
        """
        self.db_path = db_path
        self._metrics: Dict[str, ExperimentMetrics] = {}
        self._lock = threading.Lock()

        if db_path:
            self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Experiment metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                experiment_id TEXT PRIMARY KEY,
                metrics_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Individual request records for detailed analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS request_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                variant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                tokens_in INTEGER NOT NULL,
                tokens_out INTEGER NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiment_metrics(experiment_id)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_request_experiment
            ON request_records(experiment_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_request_variant
            ON request_records(variant_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_request_timestamp
            ON request_records(timestamp)
        """)

        conn.commit()
        conn.close()

    def _load_metrics(self, experiment_id: str) -> Optional[ExperimentMetrics]:
        """Load metrics from database."""
        if not self.db_path:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT metrics_json FROM experiment_metrics WHERE experiment_id = ?",
            (experiment_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return ExperimentMetrics.from_dict(json.loads(row[0]))
        return None

    def _save_metrics(self, metrics: ExperimentMetrics):
        """Save metrics to database."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO experiment_metrics (experiment_id, metrics_json, updated_at)
            VALUES (?, ?, ?)
        """, (
            metrics.experiment_id,
            json.dumps(metrics.to_dict()),
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()

    def _save_request_record(self, record: RequestRecord):
        """Save individual request record."""
        if not self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO request_records
            (experiment_id, variant_id, user_id, timestamp, success, latency_ms, tokens_in, tokens_out, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.experiment_id,
            record.variant_id,
            record.user_id,
            record.timestamp,
            1 if record.success else 0,
            record.latency_ms,
            record.tokens_in,
            record.tokens_out,
            json.dumps(record.metadata),
        ))
        conn.commit()
        conn.close()

    def get_metrics(self, experiment_id: str) -> ExperimentMetrics:
        """Get metrics for an experiment."""
        with self._lock:
            if experiment_id not in self._metrics:
                # Try loading from database
                loaded = self._load_metrics(experiment_id)
                if loaded:
                    self._metrics[experiment_id] = loaded
                else:
                    self._metrics[experiment_id] = ExperimentMetrics(
                        experiment_id=experiment_id,
                        started_at=datetime.utcnow().isoformat(),
                    )
            return self._metrics[experiment_id]

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
        Record a request result.

        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            user_id: User ID
            success: Whether the request succeeded
            latency_ms: Request latency in milliseconds
            tokens_in: Input tokens
            tokens_out: Output tokens
            metadata: Optional additional metadata
        """
        with self._lock:
            metrics = self.get_metrics(experiment_id)
            metrics.record(variant_id, success, latency_ms, tokens_in, tokens_out)
            self._save_metrics(metrics)

            # Save individual record for detailed analysis
            record = RequestRecord(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                timestamp=datetime.utcnow().isoformat(),
                success=success,
                latency_ms=latency_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                metadata=metadata or {},
            )
            self._save_request_record(record)

    def get_request_records(
        self,
        experiment_id: str,
        variant_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[RequestRecord]:
        """Get individual request records for detailed analysis."""
        if not self.db_path:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if variant_id:
            cursor.execute("""
                SELECT experiment_id, variant_id, user_id, timestamp, success,
                       latency_ms, tokens_in, tokens_out, metadata_json
                FROM request_records
                WHERE experiment_id = ? AND variant_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (experiment_id, variant_id, limit))
        else:
            cursor.execute("""
                SELECT experiment_id, variant_id, user_id, timestamp, success,
                       latency_ms, tokens_in, tokens_out, metadata_json
                FROM request_records
                WHERE experiment_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (experiment_id, limit))

        records = []
        for row in cursor.fetchall():
            records.append(RequestRecord(
                experiment_id=row[0],
                variant_id=row[1],
                user_id=row[2],
                timestamp=row[3],
                success=bool(row[4]),
                latency_ms=row[5],
                tokens_in=row[6],
                tokens_out=row[7],
                metadata=json.loads(row[8]) if row[8] else {},
            ))

        conn.close()
        return records

    def get_latency_samples(
        self,
        experiment_id: str,
        variant_id: str,
        limit: int = 10000,
    ) -> List[float]:
        """Get latency samples for statistical analysis."""
        if not self.db_path:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT latency_ms FROM request_records
            WHERE experiment_id = ? AND variant_id = ? AND success = 1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (experiment_id, variant_id, limit))

        samples = [row[0] for row in cursor.fetchall()]
        conn.close()
        return samples

    def clear_metrics(self, experiment_id: str):
        """Clear metrics for an experiment."""
        with self._lock:
            if experiment_id in self._metrics:
                del self._metrics[experiment_id]

            if self.db_path:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM experiment_metrics WHERE experiment_id = ?",
                    (experiment_id,)
                )
                cursor.execute(
                    "DELETE FROM request_records WHERE experiment_id = ?",
                    (experiment_id,)
                )
                conn.commit()
                conn.close()
