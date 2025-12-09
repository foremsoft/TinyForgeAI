"""
A/B Testing API Endpoints

FastAPI router for managing A/B test experiments.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from services.ab_testing.experiment import ExperimentStatus
from services.ab_testing.manager import ABTestManager

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DB_PATH = os.getenv("TINYFORGE_AB_TESTING_DB", "data/ab_testing.db")
ADMIN_API_KEY = os.getenv("TINYFORGE_ADMIN_API_KEY", "admin-secret-key")

# Global manager instance
_manager: Optional[ABTestManager] = None


def get_manager() -> ABTestManager:
    """Get or create the A/B test manager."""
    global _manager
    if _manager is None:
        _manager = ABTestManager(db_path=DB_PATH)
    return _manager


def verify_admin(request: Request) -> bool:
    """Verify admin API key."""
    api_key = request.headers.get("X-Admin-Key")
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return True


# =============================================================================
# Request/Response Models
# =============================================================================


class VariantCreate(BaseModel):
    """Variant creation request."""
    name: str = Field(..., description="Variant name")
    model_id: str = Field(..., description="Model ID to use")
    description: str = Field("", description="Variant description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional config")


class ExperimentConfigCreate(BaseModel):
    """Experiment configuration request."""
    min_sample_size: int = Field(100, description="Minimum samples per variant")
    confidence_level: float = Field(0.95, ge=0.8, le=0.99, description="Confidence level")
    primary_metric: str = Field("latency_ms", description="Primary metric to optimize")
    secondary_metrics: List[str] = Field(
        default_factory=lambda: ["success_rate", "tokens_per_second"],
        description="Secondary metrics"
    )
    auto_stop_on_significance: bool = Field(False, description="Auto-stop when significant")
    max_duration_hours: int = Field(0, ge=0, description="Max duration (0=unlimited)")


class ExperimentCreate(BaseModel):
    """Experiment creation request."""
    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    variants: List[VariantCreate] = Field(..., min_length=2, description="At least 2 variants")
    traffic_split: Optional[List[float]] = Field(None, description="Traffic split percentages")
    config: Optional[ExperimentConfigCreate] = Field(None, description="Experiment config")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for scoping")


class TrafficAllocationUpdate(BaseModel):
    """Traffic allocation update request."""
    allocations: Dict[str, float] = Field(..., description="variant_id -> percentage")


class VariantResponse(BaseModel):
    """Variant response."""
    id: str
    name: str
    model_id: str
    description: str
    config: Dict[str, Any]
    is_control: bool


class ExperimentResponse(BaseModel):
    """Experiment response."""
    id: str
    name: str
    description: str
    status: str
    variants: List[VariantResponse]
    traffic_allocation: List[Dict[str, Any]]
    config: Dict[str, Any]
    tenant_id: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class MetricsResponse(BaseModel):
    """Metrics response."""
    experiment_id: str
    variant_metrics: Dict[str, Dict[str, Any]]
    total_requests: int
    started_at: Optional[str]
    last_updated: Optional[str]


class AnalysisResponse(BaseModel):
    """Analysis response."""
    experiment_id: str
    primary_metric: str
    control_variant_id: str
    sample_sizes: Dict[str, int]
    comparisons: Dict[str, Dict[str, Any]]
    recommendation: str
    confidence_level: float
    sufficient_sample_size: bool
    min_sample_size_required: int


class ExperimentSummaryResponse(BaseModel):
    """Full experiment summary response."""
    experiment: Dict[str, Any]
    metrics: Dict[str, Any]
    analysis: Optional[Dict[str, Any]]


class VariantAssignment(BaseModel):
    """Variant assignment response."""
    experiment_id: str
    variant_id: str
    variant_name: str
    model_id: str


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/ab-testing", tags=["A/B Testing"])


# =============================================================================
# Public Endpoints (for inference integration)
# =============================================================================


@router.get("/assign/{experiment_id}", response_model=Optional[VariantAssignment])
async def assign_variant(
    experiment_id: str,
    user_id: str,
    manager: ABTestManager = Depends(get_manager),
):
    """
    Assign a variant to a user for an experiment.

    Uses consistent hashing to ensure the same user always gets the same variant.
    """
    variant = manager.assign_variant(experiment_id, user_id)
    if not variant:
        return None

    return VariantAssignment(
        experiment_id=experiment_id,
        variant_id=variant.id,
        variant_name=variant.name,
        model_id=variant.model_id,
    )


@router.post("/record")
async def record_request(
    experiment_id: str,
    variant_id: str,
    user_id: str,
    success: bool,
    latency_ms: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
    manager: ABTestManager = Depends(get_manager),
):
    """Record a request result for an A/B test."""
    manager.record_request(
        experiment_id=experiment_id,
        variant_id=variant_id,
        user_id=user_id,
        success=success,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    return {"status": "recorded"}


# =============================================================================
# Admin Endpoints
# =============================================================================


@router.post("/experiments", response_model=ExperimentResponse)
async def create_experiment(
    request: ExperimentCreate,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Create a new A/B test experiment (admin only)."""
    variants = [
        {
            "name": v.name,
            "model_id": v.model_id,
            "description": v.description,
            "config": v.config,
        }
        for v in request.variants
    ]

    config = request.config.dict() if request.config else None

    experiment = manager.create_experiment(
        name=request.name,
        description=request.description,
        variants=variants,
        traffic_split=request.traffic_split,
        config=config,
        tenant_id=request.tenant_id,
    )

    return _experiment_to_response(experiment)


@router.get("/experiments", response_model=List[ExperimentResponse])
async def list_experiments(
    status: Optional[str] = None,
    tenant_id: Optional[str] = None,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """List all experiments (admin only)."""
    filter_status = ExperimentStatus(status) if status else None
    experiments = manager.list_experiments(status=filter_status, tenant_id=tenant_id)
    return [_experiment_to_response(e) for e in experiments]


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Get an experiment by ID (admin only)."""
    experiment = manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return _experiment_to_response(experiment)


@router.get("/experiments/{experiment_id}/summary", response_model=ExperimentSummaryResponse)
async def get_experiment_summary(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Get full experiment summary including metrics and analysis (admin only)."""
    summary = manager.get_experiment_summary(experiment_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return ExperimentSummaryResponse(**summary)


@router.get("/experiments/{experiment_id}/metrics", response_model=MetricsResponse)
async def get_experiment_metrics(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Get metrics for an experiment (admin only)."""
    experiment = manager.get_experiment(experiment_id)
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    metrics = manager.get_metrics(experiment_id)
    return MetricsResponse(**metrics.to_dict())


@router.get("/experiments/{experiment_id}/analysis", response_model=AnalysisResponse)
async def get_experiment_analysis(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Get statistical analysis for an experiment (admin only)."""
    analysis = manager.analyze_experiment(experiment_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    return AnalysisResponse(**analysis.to_dict())


@router.post("/experiments/{experiment_id}/start", response_model=ExperimentResponse)
async def start_experiment(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Start an experiment (admin only)."""
    try:
        experiment = manager.start_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/experiments/{experiment_id}/pause", response_model=ExperimentResponse)
async def pause_experiment(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Pause an experiment (admin only)."""
    try:
        experiment = manager.pause_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/experiments/{experiment_id}/resume", response_model=ExperimentResponse)
async def resume_experiment(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Resume a paused experiment (admin only)."""
    try:
        experiment = manager.resume_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/experiments/{experiment_id}/complete", response_model=ExperimentResponse)
async def complete_experiment(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Complete an experiment (admin only)."""
    try:
        experiment = manager.complete_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/experiments/{experiment_id}/cancel", response_model=ExperimentResponse)
async def cancel_experiment(
    experiment_id: str,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Cancel an experiment (admin only)."""
    try:
        experiment = manager.cancel_experiment(experiment_id)
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str,
    force: bool = False,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Delete an experiment (admin only)."""
    try:
        if manager.delete_experiment(experiment_id, force=force):
            return {"message": "Experiment deleted"}
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.patch("/experiments/{experiment_id}/traffic", response_model=ExperimentResponse)
async def update_traffic_allocation(
    experiment_id: str,
    request: TrafficAllocationUpdate,
    _: bool = Depends(verify_admin),
    manager: ABTestManager = Depends(get_manager),
):
    """Update traffic allocation for an experiment (admin only)."""
    try:
        experiment = manager.update_traffic_allocation(
            experiment_id,
            request.allocations,
        )
        return _experiment_to_response(experiment)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _experiment_to_response(experiment) -> ExperimentResponse:
    """Convert experiment to response model."""
    return ExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        description=experiment.description,
        status=experiment.status.value,
        variants=[
            VariantResponse(
                id=v.id,
                name=v.name,
                model_id=v.model_id,
                description=v.description,
                config=v.config,
                is_control=v.is_control,
            )
            for v in experiment.variants
        ],
        traffic_allocation=[ta.to_dict() for ta in experiment.traffic_allocation],
        config=experiment.config.to_dict(),
        tenant_id=experiment.tenant_id,
        created_at=experiment.created_at,
        updated_at=experiment.updated_at,
        started_at=experiment.started_at,
        completed_at=experiment.completed_at,
    )
