"""
TinyForgeAI Dashboard API

FastAPI backend for managing training jobs, services, and model inference.
"""

import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="TinyForgeAI Dashboard API",
    description="Backend API for TinyForgeAI SaaS Dashboard",
    version="0.1.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Enums
# ============================================

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


# ============================================
# Pydantic Models
# ============================================

class TrainingJobCreate(BaseModel):
    """Request to create a training job."""
    name: str = Field(..., min_length=1, max_length=100)
    dataset_path: str
    model_name: str = "t5-small"
    model_type: str = "seq2seq"
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=4, ge=1, le=128)
    learning_rate: float = Field(default=2e-4, gt=0)
    use_lora: bool = True
    lora_r: int = Field(default=8, ge=1, le=128)


class TrainingJob(BaseModel):
    """Training job details."""
    id: str
    name: str
    dataset_path: str
    model_name: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    config: Dict[str, Any] = {}


class ServiceCreate(BaseModel):
    """Request to create/deploy a service."""
    name: str = Field(..., min_length=1, max_length=100)
    model_path: str
    port: int = Field(default=8000, ge=1024, le=65535)
    replicas: int = Field(default=1, ge=1, le=10)


class Service(BaseModel):
    """Deployed service details."""
    id: str
    name: str
    model_path: str
    status: ServiceStatus
    port: int
    replicas: int
    endpoint: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0


class PredictRequest(BaseModel):
    """Inference request."""
    input: str = Field(..., min_length=1)
    service_id: Optional[str] = None


class PredictResponse(BaseModel):
    """Inference response."""
    output: str
    confidence: float
    latency_ms: float


class LogEntry(BaseModel):
    """Log entry."""
    timestamp: datetime
    level: str
    source: str
    message: str


class DashboardStats(BaseModel):
    """Dashboard statistics."""
    total_jobs: int
    running_jobs: int
    completed_jobs: int
    total_services: int
    running_services: int
    total_predictions: int


# ============================================
# In-Memory Storage (replace with DB in production)
# ============================================

jobs_db: Dict[str, TrainingJob] = {}
services_db: Dict[str, Service] = {}
logs_db: List[LogEntry] = []


def add_log(level: str, source: str, message: str):
    """Add a log entry."""
    logs_db.append(LogEntry(
        timestamp=datetime.utcnow(),
        level=level,
        source=source,
        message=message
    ))
    # Keep only last 1000 logs
    if len(logs_db) > 1000:
        logs_db.pop(0)


# ============================================
# Training Job Endpoints
# ============================================

@app.post("/api/jobs", response_model=TrainingJob, tags=["Training"])
async def create_training_job(
    job_create: TrainingJobCreate,
    background_tasks: BackgroundTasks
):
    """Create a new training job."""
    job_id = str(uuid.uuid4())[:8]

    job = TrainingJob(
        id=job_id,
        name=job_create.name,
        dataset_path=job_create.dataset_path,
        model_name=job_create.model_name,
        status=JobStatus.PENDING,
        created_at=datetime.utcnow(),
        config={
            "epochs": job_create.epochs,
            "batch_size": job_create.batch_size,
            "learning_rate": job_create.learning_rate,
            "use_lora": job_create.use_lora,
            "lora_r": job_create.lora_r,
            "model_type": job_create.model_type,
        }
    )

    jobs_db[job_id] = job
    add_log("INFO", "jobs", f"Created training job: {job.name} ({job_id})")

    # In a real implementation, this would start the training
    # background_tasks.add_task(run_training_job, job_id)

    return job


@app.get("/api/jobs", response_model=List[TrainingJob], tags=["Training"])
async def list_training_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 50
):
    """List all training jobs."""
    jobs = list(jobs_db.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    jobs.sort(key=lambda x: x.created_at, reverse=True)
    return jobs[:limit]


@app.get("/api/jobs/{job_id}", response_model=TrainingJob, tags=["Training"])
async def get_training_job(job_id: str):
    """Get training job details."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]


@app.post("/api/jobs/{job_id}/cancel", response_model=TrainingJob, tags=["Training"])
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    job.status = JobStatus.CANCELLED
    add_log("WARN", "jobs", f"Cancelled training job: {job.name} ({job_id})")

    return job


@app.delete("/api/jobs/{job_id}", tags=["Training"])
async def delete_training_job(job_id: str):
    """Delete a training job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running job")

    del jobs_db[job_id]
    add_log("INFO", "jobs", f"Deleted training job: {job.name} ({job_id})")

    return {"message": "Job deleted"}


# ============================================
# Service Endpoints
# ============================================

@app.post("/api/services", response_model=Service, tags=["Services"])
async def create_service(service_create: ServiceCreate):
    """Deploy a new inference service."""
    service_id = str(uuid.uuid4())[:8]

    service = Service(
        id=service_id,
        name=service_create.name,
        model_path=service_create.model_path,
        status=ServiceStatus.STOPPED,
        port=service_create.port,
        replicas=service_create.replicas,
        created_at=datetime.utcnow(),
    )

    services_db[service_id] = service
    add_log("INFO", "services", f"Created service: {service.name} ({service_id})")

    return service


@app.get("/api/services", response_model=List[Service], tags=["Services"])
async def list_services(status: Optional[ServiceStatus] = None):
    """List all deployed services."""
    services = list(services_db.values())

    if status:
        services = [s for s in services if s.status == status]

    services.sort(key=lambda x: x.created_at, reverse=True)
    return services


@app.get("/api/services/{service_id}", response_model=Service, tags=["Services"])
async def get_service(service_id: str):
    """Get service details."""
    if service_id not in services_db:
        raise HTTPException(status_code=404, detail="Service not found")
    return services_db[service_id]


@app.post("/api/services/{service_id}/start", response_model=Service, tags=["Services"])
async def start_service(service_id: str):
    """Start a stopped service."""
    if service_id not in services_db:
        raise HTTPException(status_code=404, detail="Service not found")

    service = services_db[service_id]
    if service.status == ServiceStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Service already running")

    service.status = ServiceStatus.RUNNING
    service.started_at = datetime.utcnow()
    service.endpoint = f"http://localhost:{service.port}"
    add_log("INFO", "services", f"Started service: {service.name} ({service_id})")

    return service


@app.post("/api/services/{service_id}/stop", response_model=Service, tags=["Services"])
async def stop_service(service_id: str):
    """Stop a running service."""
    if service_id not in services_db:
        raise HTTPException(status_code=404, detail="Service not found")

    service = services_db[service_id]
    if service.status != ServiceStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Service not running")

    service.status = ServiceStatus.STOPPED
    service.endpoint = None
    add_log("INFO", "services", f"Stopped service: {service.name} ({service_id})")

    return service


@app.delete("/api/services/{service_id}", tags=["Services"])
async def delete_service(service_id: str):
    """Delete a service."""
    if service_id not in services_db:
        raise HTTPException(status_code=404, detail="Service not found")

    service = services_db[service_id]
    if service.status == ServiceStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Stop service before deleting")

    del services_db[service_id]
    add_log("INFO", "services", f"Deleted service: {service.name} ({service_id})")

    return {"message": "Service deleted"}


# ============================================
# Inference Endpoint
# ============================================

@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """Run inference on a deployed service."""
    import time
    start_time = time.time()

    # Stub implementation - in production, route to actual service
    output = request.input[::-1]  # Reverse string as stub
    confidence = 0.85

    latency_ms = (time.time() - start_time) * 1000

    add_log("DEBUG", "inference", f"Prediction completed in {latency_ms:.2f}ms")

    return PredictResponse(
        output=output,
        confidence=confidence,
        latency_ms=latency_ms
    )


# ============================================
# Dashboard Stats & Logs
# ============================================

@app.get("/api/stats", response_model=DashboardStats, tags=["Dashboard"])
async def get_stats():
    """Get dashboard statistics."""
    jobs = list(jobs_db.values())
    services = list(services_db.values())

    return DashboardStats(
        total_jobs=len(jobs),
        running_jobs=len([j for j in jobs if j.status == JobStatus.RUNNING]),
        completed_jobs=len([j for j in jobs if j.status == JobStatus.COMPLETED]),
        total_services=len(services),
        running_services=len([s for s in services if s.status == ServiceStatus.RUNNING]),
        total_predictions=sum(s.request_count for s in services),
    )


@app.get("/api/logs", response_model=List[LogEntry], tags=["Dashboard"])
async def get_logs(
    level: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100
):
    """Get recent logs."""
    logs = list(logs_db)

    if level:
        logs = [l for l in logs if l.level == level.upper()]

    if source:
        logs = [l for l in logs if l.source == source]

    logs.sort(key=lambda x: x.timestamp, reverse=True)
    return logs[:limit]


# ============================================
# Health Check
# ============================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "jobs_count": len(jobs_db),
        "services_count": len(services_db),
    }


@app.get("/", tags=["Root"])
async def root():
    """API root."""
    return {
        "name": "TinyForgeAI Dashboard API",
        "version": "0.1.0",
        "docs": "/docs",
    }


# ============================================
# Run with uvicorn
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
