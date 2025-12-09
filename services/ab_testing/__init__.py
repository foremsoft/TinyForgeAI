"""
A/B Testing Module for TinyForgeAI

Provides A/B testing capabilities for comparing model performance.
"""

from services.ab_testing.experiment import (
    Experiment,
    ExperimentStatus,
    Variant,
    TrafficAllocation,
    ExperimentConfig,
)
from services.ab_testing.manager import ABTestManager
from services.ab_testing.metrics import (
    ExperimentMetrics,
    VariantMetrics,
    MetricsCollector,
)
from services.ab_testing.analysis import (
    StatisticalAnalysis,
    SignificanceResult,
    analyze_experiment,
)

__all__ = [
    "Experiment",
    "ExperimentStatus",
    "Variant",
    "TrafficAllocation",
    "ExperimentConfig",
    "ABTestManager",
    "ExperimentMetrics",
    "VariantMetrics",
    "MetricsCollector",
    "StatisticalAnalysis",
    "SignificanceResult",
    "analyze_experiment",
]
