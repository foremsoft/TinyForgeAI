"""
Training API routes for TinyForgeAI.

Provides endpoints for submitting, monitoring, and managing training jobs.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.config import settings
from backend.webhooks import (
    emit_training_started,
    emit_training_progress,
    emit_training_completed,
    emit_training_failed,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])


# =============================================================================
# Job State Management
# =============================================================================


class JobStatus(str, Enum):
    """Training job status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# In-memory job store (use database in production)
_jobs: Dict[str, "TrainingJob"] = {}


class TrainingJob:
    """Represents a training job."""

    def __init__(
        self,
        job_id: str,
        model_name: str,
        config: Dict[str, Any],
        data_path: str,
    ):
        self.job_id = job_id
        self.model_name = model_name
        self.config = config
        self.data_path = data_path
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.metrics: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "model_name": self.model_name,
            "config": self.config,
            "data_path": self.data_path,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "progress": self.progress,
            "metrics": self.metrics,
        }


# =============================================================================
# Request/Response Models
# =============================================================================


class TrainingConfig(BaseModel):
    """Training configuration."""
    model_name: str = Field(default="t5-small", description="Base model to fine-tune")
    model_type: str = Field(default="seq2seq", description="Model type: seq2seq or causal")
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=2e-4, gt=0)
    use_peft: bool = Field(default=True, description="Use LoRA/PEFT")
    lora_r: int = Field(default=8, ge=1, le=64)
    lora_alpha: int = Field(default=32, ge=1, le=128)
    max_length: int = Field(default=512, ge=32, le=4096)
    validation_split: float = Field(default=0.1, ge=0, le=0.5)
    output_model_name: Optional[str] = Field(None, description="Name for the trained model")


class TrainingRequest(BaseModel):
    """Training job request."""
    config: TrainingConfig
    data_path: Optional[str] = Field(None, description="Path to training data (if already uploaded)")


class JobResponse(BaseModel):
    """Training job response."""
    job_id: str
    model_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    """List of training jobs."""
    jobs: List[JobResponse]
    total: int


# =============================================================================
# Background Task Functions
# =============================================================================


async def run_training_job(job_id: str):
    """Execute a training job in the background."""
    job = _jobs.get(job_id)
    if not job:
        logger.error(f"Job not found: {job_id}")
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow().isoformat()
    logger.info(f"Starting training job: {job_id}")

    # Emit training started webhook
    try:
        await emit_training_started(job_id, job.model_name, job.config)
    except Exception as e:
        logger.warning(f"Failed to emit training started webhook: {e}")

    try:
        # Check if real training is available
        from backend.training.real_trainer import TRAINING_AVAILABLE, RealTrainer, TrainingConfig as RTConfig

        if not TRAINING_AVAILABLE:
            # Simulate training for demo purposes
            import asyncio
            for i in range(10):
                await asyncio.sleep(0.5)
                job.progress = (i + 1) * 10
                job.metrics["step"] = i + 1

                # Emit progress webhook every 3 steps
                if (i + 1) % 3 == 0:
                    try:
                        await emit_training_progress(job_id, job.progress, job.metrics)
                    except Exception as e:
                        logger.warning(f"Failed to emit progress webhook: {e}")

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow().isoformat()
            job.metrics["train_loss"] = 0.5
            job.metrics["note"] = "Simulated training (dependencies not available)"
            logger.info(f"Job {job_id} completed (simulated)")

            # Emit completed webhook
            try:
                await emit_training_completed(job_id, job.model_name, job.metrics)
            except Exception as e:
                logger.warning(f"Failed to emit completed webhook: {e}")
            return

        # Real training
        config = RTConfig(
            model_name=job.config.get("model_name", "t5-small"),
            model_type=job.config.get("model_type", "seq2seq"),
            epochs=job.config.get("epochs", 3),
            batch_size=job.config.get("batch_size", 4),
            learning_rate=job.config.get("learning_rate", 2e-4),
            use_peft=job.config.get("use_peft", True),
            lora_r=job.config.get("lora_r", 8),
            lora_alpha=job.config.get("lora_alpha", 32),
            max_length=job.config.get("max_length", 512),
            validation_split=job.config.get("validation_split", 0.1),
        )

        # Determine output directory
        output_name = job.config.get("output_model_name") or f"{job.model_name}_{job_id[:8]}"
        output_dir = Path(settings.MODEL_REGISTRY_PATH) / output_name / "v1"
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer = RealTrainer(config)
        result = trainer.train(data_path=job.data_path, output_dir=output_dir)

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow().isoformat()
        job.progress = 100.0
        job.metrics = result.get("training_results", {})
        job.metrics["output_dir"] = str(output_dir)

        # Create current pointer
        model_dir = output_dir.parent
        current_path = model_dir / "current"
        with open(current_path, "w") as f:
            f.write("v1")

        logger.info(f"Job {job_id} completed successfully")

        # Emit completed webhook
        try:
            await emit_training_completed(
                job_id, job.model_name, job.metrics, str(output_dir)
            )
        except Exception as e:
            logger.warning(f"Failed to emit completed webhook: {e}")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.utcnow().isoformat()
        job.error = str(e)
        logger.error(f"Job {job_id} failed: {e}")

        # Emit failed webhook
        try:
            await emit_training_failed(job_id, str(e))
        except Exception as webhook_err:
            logger.warning(f"Failed to emit failed webhook: {webhook_err}")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("", response_model=JobResponse)
async def submit_training_job(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a new training job.

    Returns immediately with job ID. Use GET /training/{job_id} to monitor progress.
    """
    job_id = str(uuid.uuid4())

    if not request.data_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="data_path is required. Upload data first using POST /training/data",
        )

    # Validate data path exists
    if not Path(request.data_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data file not found: {request.data_path}",
        )

    job = TrainingJob(
        job_id=job_id,
        model_name=request.config.model_name,
        config=request.config.model_dump(),
        data_path=request.data_path,
    )
    _jobs[job_id] = job

    # Schedule background task
    background_tasks.add_task(run_training_job, job_id)

    logger.info(f"Submitted training job: {job_id}")

    return JobResponse(
        job_id=job.job_id,
        model_name=job.model_name,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        error=job.error,
        metrics=job.metrics,
    )


@router.post("/data")
async def upload_training_data(file: UploadFile = File(...)):
    """
    Upload training data file (JSONL format).

    Returns the path to use in the training request.
    """
    # Create data directory
    data_dir = Path(settings.MODEL_REGISTRY_PATH).parent / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_id = str(uuid.uuid4())[:8]
    safe_name = file.filename.replace(" ", "_") if file.filename else "data.jsonl"
    data_path = data_dir / f"{file_id}_{safe_name}"

    # Save file
    content = await file.read()
    with open(data_path, "wb") as f:
        f.write(content)

    # Validate JSONL format
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            lines = 0
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if "input" not in record or "output" not in record:
                        raise ValueError("Missing 'input' or 'output' field")
                    lines += 1
    except (json.JSONDecodeError, ValueError) as e:
        data_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSONL format: {e}",
        )

    logger.info(f"Uploaded training data: {data_path} ({lines} records)")

    return {
        "data_path": str(data_path),
        "records": lines,
        "message": f"Successfully uploaded {lines} training records",
    }


@router.get("", response_model=JobListResponse)
async def list_training_jobs(
    status_filter: Optional[str] = None,
    limit: int = 50,
):
    """List training jobs with optional status filter."""
    jobs = list(_jobs.values())

    if status_filter:
        jobs = [j for j in jobs if j.status.value == status_filter]

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    jobs = jobs[:limit]

    return JobListResponse(
        jobs=[
            JobResponse(
                job_id=j.job_id,
                model_name=j.model_name,
                status=j.status.value,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
                progress=j.progress,
                error=j.error,
                metrics=j.metrics,
            )
            for j in jobs
        ],
        total=len(_jobs),
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_training_job(job_id: str):
    """Get status and details of a specific training job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    return JobResponse(
        job_id=job.job_id,
        model_name=job.model_name,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=job.progress,
        error=job.error,
        metrics=job.metrics,
    )


@router.post("/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    """Cancel a pending or running training job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status.value}",
        )

    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow().isoformat()
    logger.info(f"Cancelled training job: {job_id}")

    return {"message": f"Job {job_id} cancelled"}


@router.delete("/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job record."""
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    job = _jobs[job_id]
    if job.status == JobStatus.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running job. Cancel it first.",
        )

    del _jobs[job_id]
    logger.info(f"Deleted training job: {job_id}")

    return {"message": f"Job {job_id} deleted"}
