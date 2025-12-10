"""
Monitoring and observability for TinyForgeAI.

Provides metrics collection, health checks, and performance tracking
for the backend API and training jobs.

Supports:
- Request/response metrics
- Training job metrics
- Model inference metrics
- System resource monitoring
- Custom metrics registration
- Prometheus-compatible export format
"""

import logging
import os
import platform
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabel:
    """Labels for a metric."""
    name: str
    value: str


@dataclass
class MetricValue:
    """A metric value with labels and timestamp."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Metric:
    """A metric definition."""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    values: List[MetricValue] = field(default_factory=list)

    # For histograms
    buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    )
    bucket_counts: Dict[Tuple[str, ...], Dict[float, int]] = field(default_factory=dict)
    sum_values: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    count_values: Dict[Tuple[str, ...], int] = field(default_factory=dict)


class MetricsRegistry:
    """
    Central registry for all metrics.

    Thread-safe metrics collection with support for counters,
    gauges, histograms, and summaries.
    """

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()

        # Register default metrics
        self._register_default_metrics()
        logger.info("MetricsRegistry initialized")

    def _register_default_metrics(self):
        """Register default system metrics."""
        # Request metrics
        self.register(
            "http_requests_total",
            MetricType.COUNTER,
            "Total HTTP requests",
            labels=["method", "endpoint", "status"],
        )
        self.register(
            "http_request_duration_seconds",
            MetricType.HISTOGRAM,
            "HTTP request duration in seconds",
            unit="seconds",
            labels=["method", "endpoint"],
        )

        # Training metrics
        self.register(
            "training_jobs_total",
            MetricType.COUNTER,
            "Total training jobs",
            labels=["status"],
        )
        self.register(
            "training_job_duration_seconds",
            MetricType.HISTOGRAM,
            "Training job duration in seconds",
            unit="seconds",
            labels=["model"],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
        )
        self.register(
            "training_job_progress",
            MetricType.GAUGE,
            "Current training job progress",
            labels=["job_id"],
        )

        # Inference metrics
        self.register(
            "inference_requests_total",
            MetricType.COUNTER,
            "Total inference requests",
            labels=["model", "status"],
        )
        self.register(
            "inference_latency_seconds",
            MetricType.HISTOGRAM,
            "Inference latency in seconds",
            unit="seconds",
            labels=["model"],
        )
        self.register(
            "inference_tokens_total",
            MetricType.COUNTER,
            "Total tokens processed",
            labels=["model", "direction"],  # direction: input/output
        )

        # Model metrics
        self.register(
            "models_loaded",
            MetricType.GAUGE,
            "Number of models currently loaded",
        )
        self.register(
            "model_load_duration_seconds",
            MetricType.HISTOGRAM,
            "Model load duration in seconds",
            unit="seconds",
            labels=["model"],
        )

        # System metrics
        self.register(
            "process_uptime_seconds",
            MetricType.GAUGE,
            "Process uptime in seconds",
            unit="seconds",
        )
        self.register(
            "active_connections",
            MetricType.GAUGE,
            "Number of active connections",
        )

    def register(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str = "",
        labels: List[str] = None,
        buckets: List[float] = None,
    ) -> Metric:
        """
        Register a new metric.

        Args:
            name: Metric name (should follow naming conventions)
            metric_type: Type of metric
            description: Human-readable description
            unit: Unit of measurement
            labels: Label names for this metric
            buckets: Histogram buckets (for histogram type)

        Returns:
            The registered Metric.
        """
        with self._lock:
            if name in self._metrics:
                return self._metrics[name]

            metric = Metric(
                name=name,
                type=metric_type,
                description=description,
                unit=unit,
                labels=labels or [],
            )
            if buckets and metric_type == MetricType.HISTOGRAM:
                metric.buckets = buckets

            self._metrics[name] = metric
            logger.debug(f"Registered metric: {name}")
            return metric

    def get(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Dict[str, str] = None,
    ) -> None:
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to add
            labels: Label values
        """
        metric = self._metrics.get(name)
        if not metric:
            logger.warning(f"Metric not found: {name}")
            return

        if metric.type != MetricType.COUNTER:
            logger.warning(f"Cannot increment non-counter metric: {name}")
            return

        with self._lock:
            metric.values.append(MetricValue(value=value, labels=labels or {}))

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Current value
            labels: Label values
        """
        metric = self._metrics.get(name)
        if not metric:
            logger.warning(f"Metric not found: {name}")
            return

        if metric.type != MetricType.GAUGE:
            logger.warning(f"Cannot set gauge on non-gauge metric: {name}")
            return

        with self._lock:
            # For gauges, we replace the value for the given labels
            labels = labels or {}
            label_key = tuple(sorted(labels.items()))

            # Remove old value with same labels
            metric.values = [
                v for v in metric.values
                if tuple(sorted(v.labels.items())) != label_key
            ]
            metric.values.append(MetricValue(value=value, labels=labels))

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Dict[str, str] = None,
    ) -> None:
        """
        Observe a value for a histogram metric.

        Args:
            name: Metric name
            value: Observed value
            labels: Label values
        """
        metric = self._metrics.get(name)
        if not metric:
            logger.warning(f"Metric not found: {name}")
            return

        if metric.type != MetricType.HISTOGRAM:
            logger.warning(f"Cannot observe on non-histogram metric: {name}")
            return

        with self._lock:
            labels = labels or {}
            label_key = tuple(sorted(labels.items()))

            # Initialize if needed
            if label_key not in metric.bucket_counts:
                metric.bucket_counts[label_key] = {b: 0 for b in metric.buckets}
                metric.bucket_counts[label_key][float("inf")] = 0
                metric.sum_values[label_key] = 0.0
                metric.count_values[label_key] = 0

            # Update buckets
            for bucket in metric.buckets:
                if value <= bucket:
                    metric.bucket_counts[label_key][bucket] += 1
            metric.bucket_counts[label_key][float("inf")] += 1

            # Update sum and count
            metric.sum_values[label_key] += value
            metric.count_values[label_key] += 1

    def get_counter_value(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> float:
        """Get total counter value for given labels."""
        metric = self._metrics.get(name)
        if not metric or metric.type != MetricType.COUNTER:
            return 0.0

        labels = labels or {}
        total = 0.0
        for mv in metric.values:
            if all(mv.labels.get(k) == v for k, v in labels.items()):
                total += mv.value
        return total

    def get_gauge_value(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> Optional[float]:
        """Get current gauge value for given labels."""
        metric = self._metrics.get(name)
        if not metric or metric.type != MetricType.GAUGE:
            return None

        labels = labels or {}
        label_key = tuple(sorted(labels.items()))

        for mv in reversed(metric.values):
            if tuple(sorted(mv.labels.items())) == label_key:
                return mv.value
        return None

    def export_prometheus(self) -> str:
        """
        Export all metrics in Prometheus text format.

        Returns:
            Prometheus-compatible metrics string.
        """
        lines = []

        # Update uptime
        self.set_gauge("process_uptime_seconds", time.time() - self._start_time)

        with self._lock:
            for metric in self._metrics.values():
                # Add help and type
                lines.append(f"# HELP {metric.name} {metric.description}")
                lines.append(f"# TYPE {metric.name} {metric.type.value}")

                if metric.type == MetricType.COUNTER:
                    self._export_counter(metric, lines)
                elif metric.type == MetricType.GAUGE:
                    self._export_gauge(metric, lines)
                elif metric.type == MetricType.HISTOGRAM:
                    self._export_histogram(metric, lines)

                lines.append("")

        return "\n".join(lines)

    def _export_counter(self, metric: Metric, lines: List[str]) -> None:
        """Export counter metric to Prometheus format."""
        # Aggregate values by labels
        aggregated: Dict[Tuple, float] = defaultdict(float)
        for mv in metric.values:
            label_key = tuple(sorted(mv.labels.items()))
            aggregated[label_key] += mv.value

        for label_key, value in aggregated.items():
            label_str = self._format_labels(dict(label_key))
            lines.append(f"{metric.name}{label_str} {value}")

        if not aggregated:
            lines.append(f"{metric.name} 0")

    def _export_gauge(self, metric: Metric, lines: List[str]) -> None:
        """Export gauge metric to Prometheus format."""
        # Get latest values by labels
        latest: Dict[Tuple, float] = {}
        for mv in metric.values:
            label_key = tuple(sorted(mv.labels.items()))
            latest[label_key] = mv.value

        for label_key, value in latest.items():
            label_str = self._format_labels(dict(label_key))
            lines.append(f"{metric.name}{label_str} {value}")

        if not latest:
            lines.append(f"{metric.name} 0")

    def _export_histogram(self, metric: Metric, lines: List[str]) -> None:
        """Export histogram metric to Prometheus format."""
        for label_key, buckets in metric.bucket_counts.items():
            label_dict = dict(label_key)

            # Cumulative bucket counts
            cumulative = 0
            for bucket, count in sorted(buckets.items()):
                cumulative += count
                bucket_labels = {**label_dict, "le": str(bucket) if bucket != float("inf") else "+Inf"}
                label_str = self._format_labels(bucket_labels)
                lines.append(f"{metric.name}_bucket{label_str} {cumulative}")

            # Sum and count
            label_str = self._format_labels(label_dict)
            lines.append(f"{metric.name}_sum{label_str} {metric.sum_values.get(label_key, 0)}")
            lines.append(f"{metric.name}_count{label_str} {metric.count_values.get(label_key, 0)}")

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(parts) + "}"

    def export_json(self) -> Dict[str, Any]:
        """
        Export all metrics as JSON.

        Returns:
            Dict with all metrics data.
        """
        # Update uptime
        self.set_gauge("process_uptime_seconds", time.time() - self._start_time)

        result = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics": {},
        }

        with self._lock:
            for metric in self._metrics.values():
                metric_data = {
                    "type": metric.type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                }

                if metric.type == MetricType.COUNTER:
                    aggregated = defaultdict(float)
                    for mv in metric.values:
                        label_key = str(mv.labels) if mv.labels else "default"
                        aggregated[label_key] += mv.value
                    metric_data["values"] = dict(aggregated)

                elif metric.type == MetricType.GAUGE:
                    latest = {}
                    for mv in metric.values:
                        label_key = str(mv.labels) if mv.labels else "default"
                        latest[label_key] = mv.value
                    metric_data["values"] = latest

                elif metric.type == MetricType.HISTOGRAM:
                    metric_data["buckets"] = {}
                    for label_key, buckets in metric.bucket_counts.items():
                        key_str = str(dict(label_key)) if label_key else "default"
                        metric_data["buckets"][key_str] = {
                            "counts": {str(k): v for k, v in buckets.items()},
                            "sum": metric.sum_values.get(label_key, 0),
                            "count": metric.count_values.get(label_key, 0),
                        }

                result["metrics"][metric.name] = metric_data

        return result


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer(registry, "my_metric", {"label": "value"}):
            # code to time
    """

    def __init__(
        self,
        registry: MetricsRegistry,
        metric_name: str,
        labels: Dict[str, str] = None,
    ):
        self.registry = registry
        self.metric_name = metric_name
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.registry.observe_histogram(self.metric_name, duration, self.labels)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for health checks.

    Returns:
        Dict with system information.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory = process.memory_info()
        cpu_percent = process.cpu_percent()

        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "cpu_percent": cpu_percent,
            "memory_rss_mb": memory.rss / 1024 / 1024,
            "memory_vms_mb": memory.vms / 1024 / 1024,
            "pid": os.getpid(),
        }
    except ImportError:
        return {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "pid": os.getpid(),
        }


# Global metrics registry
_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


# Convenience functions

def increment_counter(name: str, value: float = 1.0, labels: Dict[str, str] = None):
    """Increment a counter metric."""
    get_metrics_registry().increment(name, value, labels)


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    """Set a gauge metric value."""
    get_metrics_registry().set_gauge(name, value, labels)


def observe_histogram(name: str, value: float, labels: Dict[str, str] = None):
    """Observe a value for a histogram metric."""
    get_metrics_registry().observe_histogram(name, value, labels)


def time_function(metric_name: str, labels: Dict[str, str] = None) -> Timer:
    """Create a timer context manager for a function."""
    return Timer(get_metrics_registry(), metric_name, labels)
