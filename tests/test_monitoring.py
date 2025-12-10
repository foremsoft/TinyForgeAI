"""Tests for monitoring and metrics system."""

import pytest
import time
from unittest.mock import patch, MagicMock

from backend.monitoring import (
    MetricType,
    MetricValue,
    Metric,
    MetricsRegistry,
    Timer,
    get_system_info,
    get_metrics_registry,
    increment_counter,
    set_gauge,
    observe_histogram,
    time_function,
)


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_types(self):
        """Test all metric types exist."""
        assert MetricType.COUNTER == "counter"
        assert MetricType.GAUGE == "gauge"
        assert MetricType.HISTOGRAM == "histogram"
        assert MetricType.SUMMARY == "summary"


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_basic_creation(self):
        """Test basic metric value creation."""
        mv = MetricValue(value=42.0)
        assert mv.value == 42.0
        assert mv.labels == {}
        assert mv.timestamp > 0

    def test_with_labels(self):
        """Test metric value with labels."""
        mv = MetricValue(value=10.0, labels={"method": "GET", "status": "200"})
        assert mv.labels["method"] == "GET"
        assert mv.labels["status"] == "200"

    def test_custom_timestamp(self):
        """Test metric value with custom timestamp."""
        ts = 1704067200.0  # 2024-01-01 00:00:00
        mv = MetricValue(value=5.0, timestamp=ts)
        assert mv.timestamp == ts


class TestMetric:
    """Tests for Metric dataclass."""

    def test_counter_creation(self):
        """Test counter metric creation."""
        metric = Metric(
            name="test_counter",
            type=MetricType.COUNTER,
            description="Test counter",
        )
        assert metric.name == "test_counter"
        assert metric.type == MetricType.COUNTER
        assert metric.labels == []
        assert metric.values == []

    def test_gauge_creation(self):
        """Test gauge metric creation."""
        metric = Metric(
            name="test_gauge",
            type=MetricType.GAUGE,
            description="Test gauge",
            unit="bytes",
        )
        assert metric.unit == "bytes"

    def test_histogram_creation(self):
        """Test histogram metric creation."""
        metric = Metric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            description="Test histogram",
            labels=["method", "endpoint"],
        )
        assert len(metric.buckets) > 0  # Default buckets
        assert "method" in metric.labels

    def test_custom_buckets(self):
        """Test histogram with custom buckets."""
        buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
        metric = Metric(
            name="custom_histogram",
            type=MetricType.HISTOGRAM,
            description="Custom histogram",
            buckets=buckets,
        )
        assert metric.buckets == buckets


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a fresh metrics registry."""
        return MetricsRegistry()

    def test_default_metrics_registered(self, registry):
        """Test that default metrics are registered."""
        # HTTP metrics
        assert registry.get("http_requests_total") is not None
        assert registry.get("http_request_duration_seconds") is not None

        # Training metrics
        assert registry.get("training_jobs_total") is not None
        assert registry.get("training_job_duration_seconds") is not None

        # Inference metrics
        assert registry.get("inference_requests_total") is not None
        assert registry.get("inference_latency_seconds") is not None

    def test_register_metric(self, registry):
        """Test registering a new metric."""
        metric = registry.register(
            "custom_metric",
            MetricType.COUNTER,
            "A custom metric",
        )
        assert metric.name == "custom_metric"
        assert registry.get("custom_metric") is not None

    def test_register_duplicate(self, registry):
        """Test registering duplicate metric returns existing."""
        metric1 = registry.register(
            "dup_metric",
            MetricType.COUNTER,
            "First",
        )
        metric2 = registry.register(
            "dup_metric",
            MetricType.GAUGE,  # Different type
            "Second",
        )
        # Should return the first one
        assert metric1 is metric2
        assert metric1.type == MetricType.COUNTER

    def test_increment_counter(self, registry):
        """Test incrementing a counter."""
        registry.register("test_count", MetricType.COUNTER, "Test")
        registry.increment("test_count", 1)
        registry.increment("test_count", 2)

        value = registry.get_counter_value("test_count")
        assert value == 3

    def test_increment_with_labels(self, registry):
        """Test incrementing counter with labels."""
        registry.register(
            "labeled_count",
            MetricType.COUNTER,
            "Test",
            labels=["method"],
        )
        registry.increment("labeled_count", 1, {"method": "GET"})
        registry.increment("labeled_count", 2, {"method": "GET"})
        registry.increment("labeled_count", 1, {"method": "POST"})

        get_value = registry.get_counter_value("labeled_count", {"method": "GET"})
        post_value = registry.get_counter_value("labeled_count", {"method": "POST"})
        assert get_value == 3
        assert post_value == 1

    def test_set_gauge(self, registry):
        """Test setting gauge value."""
        registry.register("test_gauge", MetricType.GAUGE, "Test")
        registry.set_gauge("test_gauge", 100)

        value = registry.get_gauge_value("test_gauge")
        assert value == 100

        # Update gauge
        registry.set_gauge("test_gauge", 200)
        value = registry.get_gauge_value("test_gauge")
        assert value == 200

    def test_set_gauge_with_labels(self, registry):
        """Test setting gauge with labels."""
        registry.register(
            "labeled_gauge",
            MetricType.GAUGE,
            "Test",
            labels=["instance"],
        )
        registry.set_gauge("labeled_gauge", 10, {"instance": "a"})
        registry.set_gauge("labeled_gauge", 20, {"instance": "b"})

        value_a = registry.get_gauge_value("labeled_gauge", {"instance": "a"})
        value_b = registry.get_gauge_value("labeled_gauge", {"instance": "b"})
        assert value_a == 10
        assert value_b == 20

    def test_observe_histogram(self, registry):
        """Test observing histogram values."""
        registry.register(
            "test_hist",
            MetricType.HISTOGRAM,
            "Test",
            buckets=[0.1, 0.5, 1.0],
        )

        registry.observe_histogram("test_hist", 0.05)  # In 0.1 bucket
        registry.observe_histogram("test_hist", 0.3)   # In 0.5 bucket
        registry.observe_histogram("test_hist", 0.8)   # In 1.0 bucket
        registry.observe_histogram("test_hist", 2.0)   # In +Inf bucket

        metric = registry.get("test_hist")
        label_key = ()
        assert metric.count_values[label_key] == 4
        assert metric.sum_values[label_key] == 0.05 + 0.3 + 0.8 + 2.0

    def test_observe_histogram_with_labels(self, registry):
        """Test observing histogram with labels."""
        registry.register(
            "labeled_hist",
            MetricType.HISTOGRAM,
            "Test",
            labels=["endpoint"],
        )

        registry.observe_histogram("labeled_hist", 0.1, {"endpoint": "/api"})
        registry.observe_histogram("labeled_hist", 0.2, {"endpoint": "/api"})
        registry.observe_histogram("labeled_hist", 0.5, {"endpoint": "/health"})

        metric = registry.get("labeled_hist")
        api_key = (("endpoint", "/api"),)
        health_key = (("endpoint", "/health"),)

        assert metric.count_values[api_key] == 2
        assert metric.count_values[health_key] == 1

    def test_increment_nonexistent(self, registry):
        """Test incrementing nonexistent metric."""
        # Should not raise, just log warning
        registry.increment("nonexistent", 1)

    def test_increment_wrong_type(self, registry):
        """Test incrementing non-counter metric."""
        registry.register("a_gauge", MetricType.GAUGE, "Test")
        # Should not raise, just log warning
        registry.increment("a_gauge", 1)

    def test_export_prometheus(self, registry):
        """Test Prometheus format export."""
        registry.register("simple_counter", MetricType.COUNTER, "A simple counter")
        registry.increment("simple_counter", 5)

        output = registry.export_prometheus()

        assert "# HELP simple_counter A simple counter" in output
        assert "# TYPE simple_counter counter" in output
        assert "simple_counter 5" in output

    def test_export_prometheus_with_labels(self, registry):
        """Test Prometheus export with labels."""
        registry.register(
            "http_total",
            MetricType.COUNTER,
            "HTTP requests",
            labels=["method"],
        )
        registry.increment("http_total", 10, {"method": "GET"})

        output = registry.export_prometheus()

        assert 'http_total{method="GET"} 10' in output

    def test_export_prometheus_histogram(self, registry):
        """Test Prometheus export for histogram."""
        registry.register(
            "duration",
            MetricType.HISTOGRAM,
            "Duration",
            buckets=[0.1, 0.5, 1.0],
        )
        registry.observe_histogram("duration", 0.2)
        registry.observe_histogram("duration", 0.7)

        output = registry.export_prometheus()

        assert "duration_bucket" in output
        assert "duration_sum" in output
        assert "duration_count" in output
        assert 'le="0.1"' in output
        assert 'le="+Inf"' in output

    def test_export_json(self, registry):
        """Test JSON export."""
        registry.register("json_counter", MetricType.COUNTER, "Test")
        registry.increment("json_counter", 3)

        output = registry.export_json()

        assert "timestamp" in output
        assert "metrics" in output
        assert "json_counter" in output["metrics"]
        assert output["metrics"]["json_counter"]["type"] == "counter"

    def test_format_labels(self, registry):
        """Test label formatting."""
        labels = {"method": "GET", "status": "200"}
        formatted = registry._format_labels(labels)
        assert formatted == '{method="GET",status="200"}'

    def test_format_empty_labels(self, registry):
        """Test formatting empty labels."""
        formatted = registry._format_labels({})
        assert formatted == ""


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_basic(self):
        """Test basic timer usage."""
        registry = MetricsRegistry()
        registry.register(
            "timed_operation",
            MetricType.HISTOGRAM,
            "Timed operation",
        )

        with Timer(registry, "timed_operation"):
            time.sleep(0.01)  # 10ms

        metric = registry.get("timed_operation")
        label_key = ()
        assert metric.count_values[label_key] == 1
        assert metric.sum_values[label_key] >= 0.01

    def test_timer_with_labels(self):
        """Test timer with labels."""
        registry = MetricsRegistry()
        registry.register(
            "labeled_timer",
            MetricType.HISTOGRAM,
            "Labeled timer",
            labels=["operation"],
        )

        with Timer(registry, "labeled_timer", {"operation": "read"}):
            pass

        metric = registry.get("labeled_timer")
        label_key = (("operation", "read"),)
        assert metric.count_values[label_key] == 1

    def test_timer_returns_self(self):
        """Test timer returns self from __enter__."""
        registry = MetricsRegistry()
        registry.register("t", MetricType.HISTOGRAM, "Test")

        with Timer(registry, "t") as timer:
            assert timer.start_time is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_metrics_registry_singleton(self):
        """Test get_metrics_registry returns singleton."""
        import backend.monitoring as monitoring_module
        monitoring_module._metrics_registry = None

        reg1 = get_metrics_registry()
        reg2 = get_metrics_registry()
        assert reg1 is reg2

    def test_increment_counter_convenience(self):
        """Test increment_counter convenience function."""
        import backend.monitoring as monitoring_module
        monitoring_module._metrics_registry = None

        registry = get_metrics_registry()
        registry.register("conv_counter", MetricType.COUNTER, "Test")

        increment_counter("conv_counter", 5)
        value = registry.get_counter_value("conv_counter")
        assert value == 5

    def test_set_gauge_convenience(self):
        """Test set_gauge convenience function."""
        import backend.monitoring as monitoring_module
        monitoring_module._metrics_registry = None

        registry = get_metrics_registry()
        registry.register("conv_gauge", MetricType.GAUGE, "Test")

        set_gauge("conv_gauge", 42)
        value = registry.get_gauge_value("conv_gauge")
        assert value == 42

    def test_observe_histogram_convenience(self):
        """Test observe_histogram convenience function."""
        import backend.monitoring as monitoring_module
        monitoring_module._metrics_registry = None

        registry = get_metrics_registry()
        registry.register("conv_hist", MetricType.HISTOGRAM, "Test")

        observe_histogram("conv_hist", 0.5)
        metric = registry.get("conv_hist")
        label_key = ()
        assert metric.count_values[label_key] == 1

    def test_time_function_convenience(self):
        """Test time_function convenience function."""
        import backend.monitoring as monitoring_module
        monitoring_module._metrics_registry = None

        registry = get_metrics_registry()
        registry.register("func_timer", MetricType.HISTOGRAM, "Test")

        with time_function("func_timer"):
            pass

        metric = registry.get("func_timer")
        label_key = ()
        assert metric.count_values[label_key] == 1


class TestSystemInfo:
    """Tests for system info function."""

    def test_get_system_info_basic(self):
        """Test get_system_info returns expected fields."""
        info = get_system_info()

        assert "platform" in info
        assert "python_version" in info
        assert "cpu_count" in info
        assert "pid" in info

    def test_get_system_info_with_psutil(self):
        """Test get_system_info with psutil available."""
        try:
            import psutil
            info = get_system_info()
            assert "memory_rss_mb" in info
            assert "cpu_percent" in info
        except ImportError:
            pytest.skip("psutil not installed")

    def test_get_system_info_without_psutil(self):
        """Test get_system_info without psutil."""
        with patch.dict("sys.modules", {"psutil": None}):
            # Force reimport
            import importlib
            import backend.monitoring as monitoring
            importlib.reload(monitoring)

            info = monitoring.get_system_info()
            # Should still have basic info
            assert "platform" in info
            assert "python_version" in info


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_increments(self):
        """Test concurrent counter increments."""
        import threading

        registry = MetricsRegistry()
        registry.register("concurrent_counter", MetricType.COUNTER, "Test")

        def increment_many():
            for _ in range(100):
                registry.increment("concurrent_counter", 1)

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        value = registry.get_counter_value("concurrent_counter")
        assert value == 1000

    def test_concurrent_gauge_updates(self):
        """Test concurrent gauge updates."""
        import threading

        registry = MetricsRegistry()
        registry.register("concurrent_gauge", MetricType.GAUGE, "Test")

        def update_gauge(value):
            for _ in range(100):
                registry.set_gauge("concurrent_gauge", value)

        threads = [
            threading.Thread(target=update_gauge, args=(i,))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some value (last write wins)
        value = registry.get_gauge_value("concurrent_gauge")
        assert value is not None
