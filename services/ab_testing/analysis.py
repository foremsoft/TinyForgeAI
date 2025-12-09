"""
A/B Testing Statistical Analysis

Provides statistical analysis for A/B test experiments.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.ab_testing.experiment import Experiment, Variant
from services.ab_testing.metrics import ExperimentMetrics, VariantMetrics


@dataclass
class SignificanceResult:
    """Result of statistical significance testing."""
    is_significant: bool
    confidence_level: float
    p_value: float
    effect_size: float
    effect_size_ci_lower: float
    effect_size_ci_upper: float
    winner: Optional[str] = None  # variant_id of winner
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "p_value": round(self.p_value, 4),
            "effect_size": round(self.effect_size, 4),
            "effect_size_ci_lower": round(self.effect_size_ci_lower, 4),
            "effect_size_ci_upper": round(self.effect_size_ci_upper, 4),
            "winner": self.winner,
            "interpretation": self.interpretation,
        }


@dataclass
class StatisticalAnalysis:
    """Complete statistical analysis of an experiment."""
    experiment_id: str
    primary_metric: str
    control_variant_id: str
    sample_sizes: Dict[str, int]
    comparisons: Dict[str, SignificanceResult]  # variant_id -> result
    recommendation: str
    confidence_level: float
    sufficient_sample_size: bool
    min_sample_size_required: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "primary_metric": self.primary_metric,
            "control_variant_id": self.control_variant_id,
            "sample_sizes": self.sample_sizes,
            "comparisons": {
                vid: r.to_dict() for vid, r in self.comparisons.items()
            },
            "recommendation": self.recommendation,
            "confidence_level": self.confidence_level,
            "sufficient_sample_size": self.sufficient_sample_size,
            "min_sample_size_required": self.min_sample_size_required,
        }


def _normal_cdf(x: float) -> float:
    """Approximate cumulative distribution function for standard normal."""
    # Approximation using error function
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _inverse_normal_cdf(p: float) -> float:
    """Approximate inverse CDF (quantile function) for standard normal."""
    # Rational approximation (Abramowitz and Stegun)
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -_inverse_normal_cdf(1 - p)

    # Approximation for p > 0.5
    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _calculate_z_score(
    mean1: float,
    mean2: float,
    var1: float,
    var2: float,
    n1: int,
    n2: int,
) -> float:
    """Calculate z-score for two-sample comparison."""
    if n1 == 0 or n2 == 0:
        return 0.0

    se = math.sqrt(var1 / n1 + var2 / n2)
    if se == 0:
        return 0.0

    return (mean1 - mean2) / se


def _calculate_variance(samples: List[float]) -> float:
    """Calculate sample variance."""
    if len(samples) < 2:
        return 0.0

    mean = sum(samples) / len(samples)
    return sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)


def _calculate_pooled_std(
    var1: float,
    var2: float,
    n1: int,
    n2: int,
) -> float:
    """Calculate pooled standard deviation."""
    if n1 + n2 <= 2:
        return 0.0
    return math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))


def _cohens_d(
    mean1: float,
    mean2: float,
    pooled_std: float,
) -> float:
    """Calculate Cohen's d effect size."""
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def compare_variants(
    control_metrics: VariantMetrics,
    treatment_metrics: VariantMetrics,
    metric: str = "latency_ms",
    confidence_level: float = 0.95,
    control_samples: Optional[List[float]] = None,
    treatment_samples: Optional[List[float]] = None,
) -> SignificanceResult:
    """
    Compare a treatment variant against the control.

    Args:
        control_metrics: Metrics for control variant
        treatment_metrics: Metrics for treatment variant
        metric: Metric to compare
        confidence_level: Required confidence level (e.g., 0.95)
        control_samples: Optional raw samples for more accurate variance
        treatment_samples: Optional raw samples for more accurate variance

    Returns:
        SignificanceResult with analysis
    """
    n1 = control_metrics.request_count
    n2 = treatment_metrics.request_count

    if n1 < 2 or n2 < 2:
        return SignificanceResult(
            is_significant=False,
            confidence_level=confidence_level,
            p_value=1.0,
            effect_size=0.0,
            effect_size_ci_lower=0.0,
            effect_size_ci_upper=0.0,
            interpretation="Insufficient data for analysis",
        )

    # Get metric values
    if metric == "latency_ms":
        mean1 = control_metrics.avg_latency_ms
        mean2 = treatment_metrics.avg_latency_ms
        # For latency, lower is better
        lower_is_better = True
    elif metric == "success_rate":
        mean1 = control_metrics.success_rate
        mean2 = treatment_metrics.success_rate
        # For success rate, higher is better
        lower_is_better = False
    elif metric == "tokens_per_second":
        mean1 = control_metrics.tokens_per_second
        mean2 = treatment_metrics.tokens_per_second
        # For throughput, higher is better
        lower_is_better = False
    else:
        return SignificanceResult(
            is_significant=False,
            confidence_level=confidence_level,
            p_value=1.0,
            effect_size=0.0,
            effect_size_ci_lower=0.0,
            effect_size_ci_upper=0.0,
            interpretation=f"Unknown metric: {metric}",
        )

    # Calculate variance
    if control_samples and treatment_samples:
        var1 = _calculate_variance(control_samples)
        var2 = _calculate_variance(treatment_samples)
    else:
        # Estimate variance from aggregate metrics
        # Using coefficient of variation approximation
        if metric == "latency_ms":
            # Assume CV of ~0.3 for latency
            var1 = (mean1 * 0.3) ** 2 if mean1 > 0 else 1.0
            var2 = (mean2 * 0.3) ** 2 if mean2 > 0 else 1.0
        elif metric == "success_rate":
            # Binomial variance
            var1 = mean1 * (1 - mean1)
            var2 = mean2 * (1 - mean2)
        else:
            var1 = (mean1 * 0.3) ** 2 if mean1 > 0 else 1.0
            var2 = (mean2 * 0.3) ** 2 if mean2 > 0 else 1.0

    # Calculate z-score
    z_score = _calculate_z_score(mean2, mean1, var2, var1, n2, n1)

    # Calculate p-value (two-tailed)
    p_value = 2 * (1 - _normal_cdf(abs(z_score)))

    # Calculate effect size (Cohen's d)
    pooled_std = _calculate_pooled_std(var1, var2, n1, n2)
    effect_size = _cohens_d(mean2, mean1, pooled_std)

    # Calculate confidence interval for effect size
    se_d = math.sqrt((n1 + n2) / (n1 * n2) + effect_size ** 2 / (2 * (n1 + n2)))
    z_crit = _inverse_normal_cdf((1 + confidence_level) / 2)
    ci_lower = effect_size - z_crit * se_d
    ci_upper = effect_size + z_crit * se_d

    # Determine significance and winner
    alpha = 1 - confidence_level
    is_significant = p_value < alpha

    winner = None
    if is_significant:
        if lower_is_better:
            # For latency: negative effect = treatment is faster (winner)
            winner = treatment_metrics.variant_id if effect_size < 0 else control_metrics.variant_id
        else:
            # For success rate/throughput: positive effect = treatment is better
            winner = treatment_metrics.variant_id if effect_size > 0 else control_metrics.variant_id

    # Generate interpretation
    if not is_significant:
        interpretation = f"No statistically significant difference detected (p={p_value:.4f})"
    else:
        improvement_pct = abs((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0
        direction = "improvement" if winner == treatment_metrics.variant_id else "degradation"
        interpretation = (
            f"Treatment shows {improvement_pct:.1f}% {direction} in {metric} "
            f"(p={p_value:.4f}, effect size d={effect_size:.3f})"
        )

    return SignificanceResult(
        is_significant=is_significant,
        confidence_level=confidence_level,
        p_value=p_value,
        effect_size=effect_size,
        effect_size_ci_lower=ci_lower,
        effect_size_ci_upper=ci_upper,
        winner=winner,
        interpretation=interpretation,
    )


def analyze_experiment(
    experiment: Experiment,
    metrics: ExperimentMetrics,
    latency_samples: Optional[Dict[str, List[float]]] = None,
) -> StatisticalAnalysis:
    """
    Perform complete statistical analysis of an experiment.

    Args:
        experiment: The experiment to analyze
        metrics: Collected metrics
        latency_samples: Optional dict of variant_id -> latency samples

    Returns:
        Complete StatisticalAnalysis
    """
    control = experiment.get_control_variant()
    control_metrics = metrics.get_variant_metrics(control.id)

    # Get sample sizes
    sample_sizes = {
        vm.variant_id: vm.request_count
        for vm in metrics.variant_metrics.values()
    }

    # Check if we have sufficient sample size
    min_required = experiment.config.min_sample_size
    sufficient = all(n >= min_required for n in sample_sizes.values())

    # Compare each treatment against control
    comparisons: Dict[str, SignificanceResult] = {}
    for variant in experiment.variants:
        if variant.is_control:
            continue

        treatment_metrics = metrics.get_variant_metrics(variant.id)

        control_samples = latency_samples.get(control.id) if latency_samples else None
        treatment_samples = latency_samples.get(variant.id) if latency_samples else None

        comparisons[variant.id] = compare_variants(
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            metric=experiment.config.primary_metric,
            confidence_level=experiment.config.confidence_level,
            control_samples=control_samples,
            treatment_samples=treatment_samples,
        )

    # Generate recommendation
    if not sufficient:
        recommendation = (
            f"Continue collecting data. Need at least {min_required} samples per variant. "
            f"Current: {min(sample_sizes.values()) if sample_sizes else 0}"
        )
    else:
        # Find best performing variant
        significant_winners = [
            (vid, result) for vid, result in comparisons.items()
            if result.is_significant and result.winner == vid
        ]

        if significant_winners:
            best_vid, best_result = significant_winners[0]
            best_variant = experiment.get_variant(best_vid)
            recommendation = (
                f"Consider adopting '{best_variant.name}' as the new baseline. "
                f"It shows statistically significant improvement. "
                f"{best_result.interpretation}"
            )
        elif any(r.is_significant for r in comparisons.values()):
            recommendation = (
                "Control variant performs best. Consider stopping the experiment "
                "or trying different configurations."
            )
        else:
            recommendation = (
                "No significant differences detected. Consider extending the experiment "
                "or increasing sample size for more conclusive results."
            )

    return StatisticalAnalysis(
        experiment_id=experiment.id,
        primary_metric=experiment.config.primary_metric,
        control_variant_id=control.id,
        sample_sizes=sample_sizes,
        comparisons=comparisons,
        recommendation=recommendation,
        confidence_level=experiment.config.confidence_level,
        sufficient_sample_size=sufficient,
        min_sample_size_required=min_required,
    )


def calculate_required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size per variant.

    Args:
        baseline_rate: Baseline conversion/success rate (e.g., 0.10 for 10%)
        minimum_detectable_effect: Minimum relative change to detect (e.g., 0.05 for 5%)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80)

    Returns:
        Required sample size per variant
    """
    # Treatment rate
    treatment_rate = baseline_rate * (1 + minimum_detectable_effect)

    # Pooled variance
    p_pooled = (baseline_rate + treatment_rate) / 2
    var_pooled = p_pooled * (1 - p_pooled)

    # Z-scores
    z_alpha = _inverse_normal_cdf(1 - alpha / 2)
    z_beta = _inverse_normal_cdf(power)

    # Effect size
    effect = abs(treatment_rate - baseline_rate)

    if effect == 0:
        return float('inf')

    # Sample size formula
    n = 2 * var_pooled * ((z_alpha + z_beta) / effect) ** 2

    return int(math.ceil(n))
