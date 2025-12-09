"""
TinyForgeAI Dashboard API

FastAPI backend for managing training jobs, services, and model inference.
Includes WebSocket support for real-time training progress updates.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import authentication
from services.dashboard_api.auth import (
    require_auth,
    get_auth_status,
    token_auth,
    verify_basic_credentials,
    AUTH_ENABLED,
)

# Import database
from services.dashboard_api.database import (
    init_db,
    get_db,
    get_db_context,
    JobRepository,
    ServiceRepository,
    ModelRepository,
    LogRepository,
    JobModel,
    ServiceModel,
    ModelRegistryModel,
)
from sqlalchemy.orm import Session

# Import metrics (optional)
from services.dashboard_api.metrics import (
    METRICS_AVAILABLE,
    METRICS_ENABLED,
    MetricsMiddleware,
    get_metrics_router,
    record_training_job_created,
    record_training_job_status_change,
    record_inference_request,
    update_model_counts,
    update_service_counts,
)

# Configuration
USE_DATABASE = os.getenv("TINYFORGE_USE_DATABASE", "true").lower() == "true"

app = FastAPI(
    title="TinyForgeAI Dashboard API",
    description="Backend API for TinyForgeAI SaaS Dashboard",
    version="0.2.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware (middleware only if available)
if METRICS_AVAILABLE and METRICS_ENABLED:
    app.add_middleware(MetricsMiddleware)

# Always include metrics router so /metrics endpoint exists
app.include_router(get_metrics_router())


# ============================================
# Startup Event
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    if USE_DATABASE:
        init_db()
        print("Database initialized")


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
    total_models: int = 0


class ModelCreate(BaseModel):
    """Request to register a model."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    path: str = Field(..., min_length=1)
    size_bytes: int = Field(default=0, ge=0)
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ModelUpdate(BaseModel):
    """Request to update a model."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    model_type: Optional[str] = None
    is_deployed: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class RegisteredModel(BaseModel):
    """Registered model details."""
    id: str
    name: str
    description: Optional[str] = None
    model_type: Optional[str] = None
    base_model: Optional[str] = None
    path: str
    size_bytes: int = 0
    size: str = "N/A"
    created_at: datetime
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    is_deployed: bool = False


# ============================================
# In-Memory Storage (fallback when database is disabled)
# ============================================

jobs_db: Dict[str, TrainingJob] = {}
services_db: Dict[str, Service] = {}
models_db: Dict[str, RegisteredModel] = {}
logs_db: List[LogEntry] = []


# ============================================
# Database Helper Functions
# ============================================

def get_optional_db():
    """Get database session if enabled, else return None."""
    if USE_DATABASE:
        return next(get_db())
    return None


def db_add_log(level: str, source: str, message: str):
    """Add log to database if enabled."""
    if USE_DATABASE:
        with get_db_context() as db:
            LogRepository.create(db, level, source, message)


# ============================================
# WebSocket Connection Manager
# ============================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "jobs": set(),
            "logs": set(),
            "stats": set(),
        }

    async def connect(self, websocket: WebSocket, channel: str = "jobs"):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)

    def disconnect(self, websocket: WebSocket, channel: str = "jobs"):
        """Remove a WebSocket connection."""
        if channel in self.active_connections:
            self.active_connections[channel].discard(websocket)

    async def broadcast(self, message: dict, channel: str = "jobs"):
        """Broadcast a message to all connections in a channel."""
        if channel not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[channel]:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections[channel].discard(conn)

    async def send_job_update(self, job: TrainingJob):
        """Send a job update to all subscribed clients."""
        await self.broadcast({
            "type": "job_update",
            "data": {
                "id": job.id,
                "name": job.name,
                "status": job.status.value,
                "progress": job.progress,
                "model_name": job.model_name,
            }
        }, channel="jobs")

    async def send_log(self, log: LogEntry):
        """Send a log entry to all subscribed clients."""
        await self.broadcast({
            "type": "log",
            "data": {
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "source": log.source,
                "message": log.message,
            }
        }, channel="logs")


# Global connection manager
ws_manager = ConnectionManager()


def add_log(level: str, source: str, message: str):
    """Add a log entry and broadcast to WebSocket clients."""
    log_entry = LogEntry(
        timestamp=datetime.utcnow(),
        level=level,
        source=source,
        message=message
    )

    # Store in database or in-memory
    if USE_DATABASE:
        db_add_log(level, source, message)
    else:
        logs_db.append(log_entry)
        # Keep only last 1000 logs
        if len(logs_db) > 1000:
            logs_db.pop(0)

    # Broadcast log to WebSocket clients (fire and forget)
    asyncio.create_task(ws_manager.send_log(log_entry))


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
    created_at = datetime.utcnow()

    config = {
        "epochs": job_create.epochs,
        "batch_size": job_create.batch_size,
        "learning_rate": job_create.learning_rate,
        "use_lora": job_create.use_lora,
        "lora_r": job_create.lora_r,
        "model_type": job_create.model_type,
    }

    if USE_DATABASE:
        with get_db_context() as db:
            job_data = {
                "id": job_id,
                "name": job_create.name,
                "dataset_path": job_create.dataset_path,
                "model_name": job_create.model_name,
                "status": JobStatus.PENDING.value,
                "created_at": created_at,
                "config": config,
            }
            db_job = JobRepository.create(db, job_data)
            job = TrainingJob(
                id=db_job.id,
                name=db_job.name,
                dataset_path=db_job.dataset_path,
                model_name=db_job.model_name,
                status=JobStatus(db_job.status),
                progress=db_job.progress,
                created_at=db_job.created_at,
                started_at=db_job.started_at,
                completed_at=db_job.completed_at,
                error_message=db_job.error_message,
                output_dir=db_job.output_dir,
                config=db_job.config or {},
            )
    else:
        job = TrainingJob(
            id=job_id,
            name=job_create.name,
            dataset_path=job_create.dataset_path,
            model_name=job_create.model_name,
            status=JobStatus.PENDING,
            created_at=created_at,
            config=config,
        )
        jobs_db[job_id] = job

    add_log("INFO", "jobs", f"Created training job: {job.name} ({job_id})")

    # In a real implementation, this would start the training
    # background_tasks.add_task(run_training_job, job_id)

    return job


def _db_job_to_pydantic(db_job: JobModel) -> TrainingJob:
    """Convert database job model to Pydantic model."""
    return TrainingJob(
        id=db_job.id,
        name=db_job.name,
        dataset_path=db_job.dataset_path,
        model_name=db_job.model_name,
        status=JobStatus(db_job.status),
        progress=db_job.progress,
        created_at=db_job.created_at,
        started_at=db_job.started_at,
        completed_at=db_job.completed_at,
        error_message=db_job.error_message,
        output_dir=db_job.output_dir,
        config=db_job.config or {},
    )


@app.get("/api/jobs", response_model=List[TrainingJob], tags=["Training"])
async def list_training_jobs(
    status: Optional[JobStatus] = None,
    limit: int = 50
):
    """List all training jobs."""
    if USE_DATABASE:
        with get_db_context() as db:
            status_str = status.value if status else None
            db_jobs = JobRepository.list(db, status=status_str, limit=limit)
            return [_db_job_to_pydantic(j) for j in db_jobs]
    else:
        jobs = list(jobs_db.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]


@app.get("/api/jobs/{job_id}", response_model=TrainingJob, tags=["Training"])
async def get_training_job(job_id: str):
    """Get training job details."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_job = JobRepository.get(db, job_id)
            if not db_job:
                raise HTTPException(status_code=404, detail="Job not found")
            return _db_job_to_pydantic(db_job)
    else:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs_db[job_id]


@app.post("/api/jobs/{job_id}/cancel", response_model=TrainingJob, tags=["Training"])
async def cancel_training_job(job_id: str):
    """Cancel a running training job."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_job = JobRepository.get(db, job_id)
            if not db_job:
                raise HTTPException(status_code=404, detail="Job not found")
            if db_job.status not in [JobStatus.PENDING.value, JobStatus.RUNNING.value]:
                raise HTTPException(status_code=400, detail="Job cannot be cancelled")

            db_job = JobRepository.update(db, job_id, {"status": JobStatus.CANCELLED.value})
            add_log("WARN", "jobs", f"Cancelled training job: {db_job.name} ({job_id})")
            return _db_job_to_pydantic(db_job)
    else:
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
    if USE_DATABASE:
        with get_db_context() as db:
            db_job = JobRepository.get(db, job_id)
            if not db_job:
                raise HTTPException(status_code=404, detail="Job not found")
            if db_job.status == JobStatus.RUNNING.value:
                raise HTTPException(status_code=400, detail="Cannot delete running job")

            job_name = db_job.name
            JobRepository.delete(db, job_id)
            add_log("INFO", "jobs", f"Deleted training job: {job_name} ({job_id})")
            return {"message": "Job deleted"}
    else:
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

def _db_service_to_pydantic(db_service: ServiceModel) -> Service:
    """Convert database service model to Pydantic model."""
    return Service(
        id=db_service.id,
        name=db_service.name,
        model_path=db_service.model_path,
        status=ServiceStatus(db_service.status),
        port=db_service.port,
        replicas=db_service.replicas,
        endpoint=db_service.endpoint,
        created_at=db_service.created_at,
        started_at=db_service.started_at,
        request_count=db_service.request_count,
        error_count=db_service.error_count,
    )


@app.post("/api/services", response_model=Service, tags=["Services"])
async def create_service(service_create: ServiceCreate):
    """Deploy a new inference service."""
    service_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow()

    if USE_DATABASE:
        with get_db_context() as db:
            service_data = {
                "id": service_id,
                "name": service_create.name,
                "model_path": service_create.model_path,
                "status": ServiceStatus.STOPPED.value,
                "port": service_create.port,
                "replicas": service_create.replicas,
                "created_at": created_at,
            }
            db_service = ServiceRepository.create(db, service_data)
            service = _db_service_to_pydantic(db_service)
    else:
        service = Service(
            id=service_id,
            name=service_create.name,
            model_path=service_create.model_path,
            status=ServiceStatus.STOPPED,
            port=service_create.port,
            replicas=service_create.replicas,
            created_at=created_at,
        )
        services_db[service_id] = service

    add_log("INFO", "services", f"Created service: {service.name} ({service_id})")
    return service


@app.get("/api/services", response_model=List[Service], tags=["Services"])
async def list_services(status: Optional[ServiceStatus] = None):
    """List all deployed services."""
    if USE_DATABASE:
        with get_db_context() as db:
            status_str = status.value if status else None
            db_services = ServiceRepository.list(db, status=status_str)
            return [_db_service_to_pydantic(s) for s in db_services]
    else:
        services = list(services_db.values())
        if status:
            services = [s for s in services if s.status == status]
        services.sort(key=lambda x: x.created_at, reverse=True)
        return services


@app.get("/api/services/{service_id}", response_model=Service, tags=["Services"])
async def get_service(service_id: str):
    """Get service details."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_service = ServiceRepository.get(db, service_id)
            if not db_service:
                raise HTTPException(status_code=404, detail="Service not found")
            return _db_service_to_pydantic(db_service)
    else:
        if service_id not in services_db:
            raise HTTPException(status_code=404, detail="Service not found")
        return services_db[service_id]


@app.post("/api/services/{service_id}/start", response_model=Service, tags=["Services"])
async def start_service(service_id: str):
    """Start a stopped service."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_service = ServiceRepository.get(db, service_id)
            if not db_service:
                raise HTTPException(status_code=404, detail="Service not found")
            if db_service.status == ServiceStatus.RUNNING.value:
                raise HTTPException(status_code=400, detail="Service already running")

            updates = {
                "status": ServiceStatus.RUNNING.value,
                "started_at": datetime.utcnow(),
                "endpoint": f"http://localhost:{db_service.port}",
            }
            db_service = ServiceRepository.update(db, service_id, updates)
            add_log("INFO", "services", f"Started service: {db_service.name} ({service_id})")
            return _db_service_to_pydantic(db_service)
    else:
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
    if USE_DATABASE:
        with get_db_context() as db:
            db_service = ServiceRepository.get(db, service_id)
            if not db_service:
                raise HTTPException(status_code=404, detail="Service not found")
            if db_service.status != ServiceStatus.RUNNING.value:
                raise HTTPException(status_code=400, detail="Service not running")

            updates = {
                "status": ServiceStatus.STOPPED.value,
                "endpoint": None,
            }
            db_service = ServiceRepository.update(db, service_id, updates)
            add_log("INFO", "services", f"Stopped service: {db_service.name} ({service_id})")
            return _db_service_to_pydantic(db_service)
    else:
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
    if USE_DATABASE:
        with get_db_context() as db:
            db_service = ServiceRepository.get(db, service_id)
            if not db_service:
                raise HTTPException(status_code=404, detail="Service not found")
            if db_service.status == ServiceStatus.RUNNING.value:
                raise HTTPException(status_code=400, detail="Stop service before deleting")

            service_name = db_service.name
            ServiceRepository.delete(db, service_id)
            add_log("INFO", "services", f"Deleted service: {service_name} ({service_id})")
            return {"message": "Service deleted"}
    else:
        if service_id not in services_db:
            raise HTTPException(status_code=404, detail="Service not found")

        service = services_db[service_id]
        if service.status == ServiceStatus.RUNNING:
            raise HTTPException(status_code=400, detail="Stop service before deleting")

        del services_db[service_id]
        add_log("INFO", "services", f"Deleted service: {service.name} ({service_id})")
        return {"message": "Service deleted"}


# ============================================
# Model Registry Endpoints
# ============================================

def _format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    if not bytes_size:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def _db_model_to_pydantic(db_model: ModelRegistryModel) -> RegisteredModel:
    """Convert database model registry model to Pydantic model."""
    return RegisteredModel(
        id=db_model.id,
        name=db_model.name,
        description=db_model.description,
        model_type=db_model.model_type,
        base_model=db_model.base_model,
        path=db_model.path,
        size_bytes=db_model.size_bytes or 0,
        size=_format_size(db_model.size_bytes or 0),
        created_at=db_model.created_at,
        job_id=db_model.job_id,
        metadata=db_model.model_metadata or {},
        is_deployed=db_model.is_deployed or False,
    )


@app.post("/api/models", response_model=RegisteredModel, tags=["Models"])
async def register_model(model_create: ModelCreate):
    """Register a new trained model in the registry."""
    model_id = str(uuid.uuid4())[:8]
    created_at = datetime.utcnow()

    if USE_DATABASE:
        with get_db_context() as db:
            model_data = {
                "id": model_id,
                "name": model_create.name,
                "description": model_create.description,
                "model_type": model_create.model_type,
                "base_model": model_create.base_model,
                "path": model_create.path,
                "size_bytes": model_create.size_bytes,
                "created_at": created_at,
                "job_id": model_create.job_id,
                "model_metadata": model_create.metadata,
                "is_deployed": False,
            }
            db_model = ModelRepository.create(db, model_data)
            model = _db_model_to_pydantic(db_model)
    else:
        model = RegisteredModel(
            id=model_id,
            name=model_create.name,
            description=model_create.description,
            model_type=model_create.model_type,
            base_model=model_create.base_model,
            path=model_create.path,
            size_bytes=model_create.size_bytes,
            size=_format_size(model_create.size_bytes),
            created_at=created_at,
            job_id=model_create.job_id,
            metadata=model_create.metadata,
            is_deployed=False,
        )
        models_db[model_id] = model

    add_log("INFO", "models", f"Registered model: {model.name} ({model_id})")
    return model


@app.get("/api/models", response_model=List[RegisteredModel], tags=["Models"])
async def list_models(model_type: Optional[str] = None):
    """List all registered models."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_models = ModelRepository.list(db, model_type=model_type)
            return [_db_model_to_pydantic(m) for m in db_models]
    else:
        models = list(models_db.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        models.sort(key=lambda x: x.created_at, reverse=True)
        return models


@app.get("/api/models/{model_id}", response_model=RegisteredModel, tags=["Models"])
async def get_model(model_id: str):
    """Get model details by ID."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_model = ModelRepository.get(db, model_id)
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")
            return _db_model_to_pydantic(db_model)
    else:
        if model_id not in models_db:
            raise HTTPException(status_code=404, detail="Model not found")
        return models_db[model_id]


@app.put("/api/models/{model_id}", response_model=RegisteredModel, tags=["Models"])
async def update_model(model_id: str, model_update: ModelUpdate):
    """Update a registered model."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_model = ModelRepository.get(db, model_id)
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")

            updates = {}
            if model_update.name is not None:
                updates["name"] = model_update.name
            if model_update.description is not None:
                updates["description"] = model_update.description
            if model_update.model_type is not None:
                updates["model_type"] = model_update.model_type
            if model_update.is_deployed is not None:
                updates["is_deployed"] = model_update.is_deployed
            if model_update.metadata is not None:
                updates["model_metadata"] = model_update.metadata

            if updates:
                db_model = ModelRepository.update(db, model_id, updates)

            add_log("INFO", "models", f"Updated model: {db_model.name} ({model_id})")
            return _db_model_to_pydantic(db_model)
    else:
        if model_id not in models_db:
            raise HTTPException(status_code=404, detail="Model not found")

        model = models_db[model_id]
        if model_update.name is not None:
            model.name = model_update.name
        if model_update.description is not None:
            model.description = model_update.description
        if model_update.model_type is not None:
            model.model_type = model_update.model_type
        if model_update.is_deployed is not None:
            model.is_deployed = model_update.is_deployed
        if model_update.metadata is not None:
            model.metadata = model_update.metadata

        add_log("INFO", "models", f"Updated model: {model.name} ({model_id})")
        return model


@app.delete("/api/models/{model_id}", tags=["Models"])
async def delete_model(model_id: str):
    """Delete a registered model."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_model = ModelRepository.get(db, model_id)
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")
            if db_model.is_deployed:
                raise HTTPException(status_code=400, detail="Cannot delete deployed model")

            model_name = db_model.name
            ModelRepository.delete(db, model_id)
            add_log("INFO", "models", f"Deleted model: {model_name} ({model_id})")
            return {"message": "Model deleted"}
    else:
        if model_id not in models_db:
            raise HTTPException(status_code=404, detail="Model not found")

        model = models_db[model_id]
        if model.is_deployed:
            raise HTTPException(status_code=400, detail="Cannot delete deployed model")

        del models_db[model_id]
        add_log("INFO", "models", f"Deleted model: {model.name} ({model_id})")
        return {"message": "Model deleted"}


@app.post("/api/models/{model_id}/deploy", response_model=RegisteredModel, tags=["Models"])
async def mark_model_deployed(model_id: str):
    """Mark a model as deployed."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_model = ModelRepository.get(db, model_id)
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")

            db_model = ModelRepository.update(db, model_id, {"is_deployed": True})
            add_log("INFO", "models", f"Model marked as deployed: {db_model.name} ({model_id})")
            return _db_model_to_pydantic(db_model)
    else:
        if model_id not in models_db:
            raise HTTPException(status_code=404, detail="Model not found")

        model = models_db[model_id]
        model.is_deployed = True
        add_log("INFO", "models", f"Model marked as deployed: {model.name} ({model_id})")
        return model


@app.post("/api/models/{model_id}/undeploy", response_model=RegisteredModel, tags=["Models"])
async def mark_model_undeployed(model_id: str):
    """Mark a model as not deployed."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_model = ModelRepository.get(db, model_id)
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")

            db_model = ModelRepository.update(db, model_id, {"is_deployed": False})
            add_log("INFO", "models", f"Model marked as undeployed: {db_model.name} ({model_id})")
            return _db_model_to_pydantic(db_model)
    else:
        if model_id not in models_db:
            raise HTTPException(status_code=404, detail="Model not found")

        model = models_db[model_id]
        model.is_deployed = False
        add_log("INFO", "models", f"Model marked as undeployed: {model.name} ({model_id})")
        return model


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
    if USE_DATABASE:
        with get_db_context() as db:
            total_jobs = JobRepository.count(db)
            running_jobs = JobRepository.count(db, status=JobStatus.RUNNING.value)
            completed_jobs = JobRepository.count(db, status=JobStatus.COMPLETED.value)
            total_services = ServiceRepository.count(db)
            running_services = ServiceRepository.count(db, status=ServiceStatus.RUNNING.value)
            total_models = ModelRepository.count(db)
            # Get total predictions from running services
            db_services = ServiceRepository.list(db)
            total_predictions = sum(s.request_count for s in db_services)

            return DashboardStats(
                total_jobs=total_jobs,
                running_jobs=running_jobs,
                completed_jobs=completed_jobs,
                total_services=total_services,
                running_services=running_services,
                total_predictions=total_predictions,
                total_models=total_models,
            )
    else:
        jobs = list(jobs_db.values())
        services = list(services_db.values())
        models = list(models_db.values())

        return DashboardStats(
            total_jobs=len(jobs),
            running_jobs=len([j for j in jobs if j.status == JobStatus.RUNNING]),
            completed_jobs=len([j for j in jobs if j.status == JobStatus.COMPLETED]),
            total_services=len(services),
            running_services=len([s for s in services if s.status == ServiceStatus.RUNNING]),
            total_predictions=sum(s.request_count for s in services),
            total_models=len(models),
        )


@app.get("/api/logs", response_model=List[LogEntry], tags=["Dashboard"])
async def get_logs(
    level: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100
):
    """Get recent logs."""
    if USE_DATABASE:
        with get_db_context() as db:
            db_logs = LogRepository.list(db, level=level, source=source, limit=limit)
            return [
                LogEntry(
                    timestamp=log.timestamp,
                    level=log.level,
                    source=log.source,
                    message=log.message,
                )
                for log in db_logs
            ]
    else:
        logs = list(logs_db)
        if level:
            logs = [l for l in logs if l.level == level.upper()]
        if source:
            logs = [l for l in logs if l.source == source]
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        return logs[:limit]


# ============================================
# Authentication Endpoints
# ============================================

class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response."""
    token: str
    username: str
    expires_in: int = 86400  # 24 hours


@app.get("/api/auth/status", tags=["Authentication"])
async def auth_status():
    """
    Get authentication status.
    Returns whether auth is enabled and what methods are available.
    """
    return get_auth_status()


@app.post("/api/auth/login", response_model=LoginResponse, tags=["Authentication"])
async def login(request: LoginRequest):
    """
    Login with username and password.
    Returns a session token for subsequent requests.
    """
    if not verify_basic_credentials(request.username, request.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    token = token_auth.create_token(request.username)
    add_log("INFO", "auth", f"User logged in: {request.username}")

    return LoginResponse(
        token=token,
        username=request.username,
    )


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout(token: str):
    """Logout and revoke session token."""
    if token_auth.revoke_token(token):
        add_log("INFO", "auth", "User logged out")
        return {"message": "Logged out successfully"}
    raise HTTPException(status_code=400, detail="Invalid token")


@app.get("/api/auth/verify", tags=["Authentication"])
async def verify_token_endpoint(token: str):
    """Verify if a token is valid."""
    username = token_auth.verify_token(token)
    if username:
        return {"valid": True, "username": username}
    return {"valid": False}


# ============================================
# Health Check
# ============================================

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint (no auth required)."""
    if USE_DATABASE:
        with get_db_context() as db:
            jobs_count = JobRepository.count(db)
            services_count = ServiceRepository.count(db)
            models_count = ModelRepository.count(db)
    else:
        jobs_count = len(jobs_db)
        services_count = len(services_db)
        models_count = len(models_db)

    return {
        "status": "healthy",
        "version": "0.2.0",
        "auth_enabled": AUTH_ENABLED,
        "database_enabled": USE_DATABASE,
        "metrics_enabled": METRICS_AVAILABLE and METRICS_ENABLED,
        "jobs_count": jobs_count,
        "services_count": services_count,
        "models_count": models_count,
    }


@app.get("/", tags=["Root"])
async def root():
    """API root (no auth required)."""
    return {
        "name": "TinyForgeAI Dashboard API",
        "version": "0.2.0",
        "docs": "/docs",
        "websocket": "/ws/{channel}",
        "auth_enabled": AUTH_ENABLED,
    }


# ============================================
# WebSocket Endpoints
# ============================================

@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """
    WebSocket endpoint for real-time updates.

    Channels:
    - jobs: Training job progress updates
    - logs: Real-time log streaming
    - stats: Dashboard statistics updates

    Example client connection (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8001/ws/jobs');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Update:', data);
    };
    ```
    """
    if channel not in ["jobs", "logs", "stats"]:
        await websocket.close(code=4000, reason=f"Invalid channel: {channel}")
        return

    await ws_manager.connect(websocket, channel)
    add_log("INFO", "websocket", f"Client connected to channel: {channel}")

    try:
        # Send initial state based on channel
        if channel == "jobs":
            # Send current jobs list
            if USE_DATABASE:
                with get_db_context() as db:
                    db_jobs = JobRepository.list(db, limit=50)
                    jobs_list = [
                        {
                            "id": j.id,
                            "name": j.name,
                            "status": j.status,
                            "progress": j.progress,
                            "model_name": j.model_name,
                        }
                        for j in db_jobs
                    ]
            else:
                jobs_list = [
                    {
                        "id": j.id,
                        "name": j.name,
                        "status": j.status.value,
                        "progress": j.progress,
                        "model_name": j.model_name,
                    }
                    for j in jobs_db.values()
                ]
            await websocket.send_json({"type": "initial_jobs", "data": jobs_list})

        elif channel == "stats":
            # Send current stats
            if USE_DATABASE:
                with get_db_context() as db:
                    stats = {
                        "total_jobs": JobRepository.count(db),
                        "running_jobs": JobRepository.count(db, status=JobStatus.RUNNING.value),
                        "completed_jobs": JobRepository.count(db, status=JobStatus.COMPLETED.value),
                        "total_services": ServiceRepository.count(db),
                        "running_services": ServiceRepository.count(db, status=ServiceStatus.RUNNING.value),
                    }
            else:
                jobs = list(jobs_db.values())
                services = list(services_db.values())
                stats = {
                    "total_jobs": len(jobs),
                    "running_jobs": len([j for j in jobs if j.status == JobStatus.RUNNING]),
                    "completed_jobs": len([j for j in jobs if j.status == JobStatus.COMPLETED]),
                    "total_services": len(services),
                    "running_services": len([s for s in services if s.status == ServiceStatus.RUNNING]),
                }
            await websocket.send_json({"type": "initial_stats", "data": stats})

        # Keep connection open and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle ping/pong for connection keep-alive
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                # Handle subscription to specific job
                elif message.get("type") == "subscribe_job":
                    job_id = message.get("job_id")
                    job = None
                    if USE_DATABASE:
                        with get_db_context() as db:
                            db_job = JobRepository.get(db, job_id) if job_id else None
                            if db_job:
                                job = _db_job_to_pydantic(db_job)
                    else:
                        if job_id and job_id in jobs_db:
                            job = jobs_db[job_id]

                    if job:
                        await websocket.send_json({
                            "type": "job_detail",
                            "data": {
                                "id": job.id,
                                "name": job.name,
                                "status": job.status.value,
                                "progress": job.progress,
                                "model_name": job.model_name,
                                "config": job.config,
                                "created_at": job.created_at.isoformat(),
                            }
                        })

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, channel)
        add_log("INFO", "websocket", f"Client disconnected from channel: {channel}")


@app.post("/api/jobs/{job_id}/progress", tags=["Training"])
async def update_job_progress(job_id: str, progress: float):
    """
    Update training job progress (called by training worker).

    This endpoint is typically called by the training process to report
    progress updates, which are then broadcast to WebSocket clients.
    """
    progress = min(max(progress, 0.0), 100.0)

    if USE_DATABASE:
        with get_db_context() as db:
            db_job = JobRepository.get(db, job_id)
            if not db_job:
                raise HTTPException(status_code=404, detail="Job not found")

            updates = {"progress": progress}

            # Start job if it was pending
            if db_job.status == JobStatus.PENDING.value and progress > 0:
                updates["status"] = JobStatus.RUNNING.value
                updates["started_at"] = datetime.utcnow()

            # Complete job if progress reaches 100
            if progress >= 100.0:
                updates["status"] = JobStatus.COMPLETED.value
                updates["completed_at"] = datetime.utcnow()
                add_log("INFO", "jobs", f"Training job completed: {db_job.name} ({job_id})")

            db_job = JobRepository.update(db, job_id, updates)
            job = _db_job_to_pydantic(db_job)
    else:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs_db[job_id]
        job.progress = progress

        # Start job if it was pending
        if job.status == JobStatus.PENDING and progress > 0:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

        # Complete job if progress reaches 100
        if progress >= 100.0:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            add_log("INFO", "jobs", f"Training job completed: {job.name} ({job_id})")

    # Broadcast update to WebSocket clients
    await ws_manager.send_job_update(job)

    return {"status": "updated", "progress": job.progress}


@app.post("/api/jobs/{job_id}/fail", tags=["Training"])
async def fail_job(job_id: str, error_message: str = "Unknown error"):
    """
    Mark a training job as failed (called by training worker).
    """
    if USE_DATABASE:
        with get_db_context() as db:
            db_job = JobRepository.get(db, job_id)
            if not db_job:
                raise HTTPException(status_code=404, detail="Job not found")

            updates = {
                "status": JobStatus.FAILED.value,
                "error_message": error_message,
                "completed_at": datetime.utcnow(),
            }
            db_job = JobRepository.update(db, job_id, updates)
            add_log("ERROR", "jobs", f"Training job failed: {db_job.name} ({job_id}) - {error_message}")
            job = _db_job_to_pydantic(db_job)
    else:
        if job_id not in jobs_db:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs_db[job_id]
        job.status = JobStatus.FAILED
        job.error_message = error_message
        job.completed_at = datetime.utcnow()
        add_log("ERROR", "jobs", f"Training job failed: {job.name} ({job_id}) - {error_message}")

    # Broadcast update to WebSocket clients
    await ws_manager.send_job_update(job)

    return {"status": "failed", "error_message": error_message}


# ============================================
# Run with uvicorn
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
