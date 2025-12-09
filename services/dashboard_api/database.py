"""
TinyForgeAI Dashboard Database

SQLite database for persistent storage of jobs, services, and models.
Uses SQLAlchemy for ORM with async support via aiosqlite.

Usage:
    from services.dashboard_api.database import get_db, init_db

    # Initialize on startup
    await init_db()

    # Use in endpoints
    @app.get("/api/jobs")
    async def list_jobs(db: Session = Depends(get_db)):
        return db.query(Job).all()
"""

import os
from datetime import datetime
from typing import Optional, List, Generator
from contextlib import contextmanager

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    JSON,
    Enum as SQLEnum,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database configuration
def get_database_url():
    """Get database URL from environment."""
    return os.getenv("TINYFORGE_DATABASE_URL", "sqlite:///./data/tinyforge.db")

DATABASE_URL = get_database_url()

# Ensure data directory exists (only for file-based databases)
if "memory" not in DATABASE_URL:
    os.makedirs("data", exist_ok=True)

# SQLAlchemy setup
Base = declarative_base()

def create_db_engine(url: str = None):
    """Create database engine."""
    db_url = url or DATABASE_URL
    return create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        echo=os.getenv("TINYFORGE_DB_ECHO", "false").lower() == "true",
    )

engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def reset_engine(url: str = None):
    """Reset database engine (useful for testing)."""
    global engine, SessionLocal, DATABASE_URL
    if url:
        DATABASE_URL = url
    engine = create_db_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# =============================================================================
# Enums
# =============================================================================

import enum

class JobStatusEnum(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceStatusEnum(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


# =============================================================================
# Models
# =============================================================================

class JobModel(Base):
    """Training job database model."""
    __tablename__ = "jobs"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    dataset_path = Column(String(500), nullable=False)
    model_name = Column(String(200), nullable=False)
    status = Column(String(20), default="pending")
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    output_dir = Column(String(500), nullable=True)
    config = Column(JSON, default=dict)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "dataset_path": self.dataset_path,
            "model_name": self.model_name,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "output_dir": self.output_dir,
            "config": self.config or {},
        }


class ServiceModel(Base):
    """Inference service database model."""
    __tablename__ = "services"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    model_path = Column(String(500), nullable=False)
    status = Column(String(20), default="stopped")
    port = Column(Integer, default=8000)
    replicas = Column(Integer, default=1)
    endpoint = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    request_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "model_path": self.model_path,
            "status": self.status,
            "port": self.port,
            "replicas": self.replicas,
            "endpoint": self.endpoint,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "request_count": self.request_count,
            "error_count": self.error_count,
        }


class ModelRegistryModel(Base):
    """Trained model registry database model."""
    __tablename__ = "models"

    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    model_type = Column(String(50), nullable=True)  # seq2seq, classification, etc.
    base_model = Column(String(200), nullable=True)  # Original HuggingFace model
    path = Column(String(500), nullable=False)
    size_bytes = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    job_id = Column(String(50), nullable=True)  # Link to training job
    model_metadata = Column(JSON, default=dict)  # Renamed from 'metadata' (reserved in SQLAlchemy)
    is_deployed = Column(Boolean, default=False)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type,
            "base_model": self.base_model,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size": self._format_size(self.size_bytes),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "job_id": self.job_id,
            "metadata": self.model_metadata or {},  # Return as 'metadata' in API
            "is_deployed": self.is_deployed,
        }

    @staticmethod
    def _format_size(bytes_size):
        """Format bytes to human-readable size."""
        if not bytes_size:
            return "N/A"
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"


class LogModel(Base):
    """Log entry database model."""
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(20), index=True)
    source = Column(String(50), index=True)
    message = Column(Text)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "source": self.source,
            "message": self.message,
        }


# =============================================================================
# Database Functions
# =============================================================================

def init_db():
    """Initialize database and create tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# Repository Functions
# =============================================================================

class JobRepository:
    """Repository for job operations."""

    @staticmethod
    def create(db: Session, job_data: dict) -> JobModel:
        """Create a new job."""
        job = JobModel(**job_data)
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get(db: Session, job_id: str) -> Optional[JobModel]:
        """Get job by ID."""
        return db.query(JobModel).filter(JobModel.id == job_id).first()

    @staticmethod
    def list(db: Session, status: Optional[str] = None, limit: int = 50) -> List[JobModel]:
        """List jobs with optional status filter."""
        query = db.query(JobModel)
        if status:
            query = query.filter(JobModel.status == status)
        return query.order_by(JobModel.created_at.desc()).limit(limit).all()

    @staticmethod
    def update(db: Session, job_id: str, updates: dict) -> Optional[JobModel]:
        """Update a job."""
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if job:
            for key, value in updates.items():
                setattr(job, key, value)
            db.commit()
            db.refresh(job)
        return job

    @staticmethod
    def delete(db: Session, job_id: str) -> bool:
        """Delete a job."""
        job = db.query(JobModel).filter(JobModel.id == job_id).first()
        if job:
            db.delete(job)
            db.commit()
            return True
        return False

    @staticmethod
    def count(db: Session, status: Optional[str] = None) -> int:
        """Count jobs with optional status filter."""
        query = db.query(JobModel)
        if status:
            query = query.filter(JobModel.status == status)
        return query.count()


class ServiceRepository:
    """Repository for service operations."""

    @staticmethod
    def create(db: Session, service_data: dict) -> ServiceModel:
        """Create a new service."""
        service = ServiceModel(**service_data)
        db.add(service)
        db.commit()
        db.refresh(service)
        return service

    @staticmethod
    def get(db: Session, service_id: str) -> Optional[ServiceModel]:
        """Get service by ID."""
        return db.query(ServiceModel).filter(ServiceModel.id == service_id).first()

    @staticmethod
    def list(db: Session, status: Optional[str] = None) -> List[ServiceModel]:
        """List services with optional status filter."""
        query = db.query(ServiceModel)
        if status:
            query = query.filter(ServiceModel.status == status)
        return query.order_by(ServiceModel.created_at.desc()).all()

    @staticmethod
    def update(db: Session, service_id: str, updates: dict) -> Optional[ServiceModel]:
        """Update a service."""
        service = db.query(ServiceModel).filter(ServiceModel.id == service_id).first()
        if service:
            for key, value in updates.items():
                setattr(service, key, value)
            db.commit()
            db.refresh(service)
        return service

    @staticmethod
    def delete(db: Session, service_id: str) -> bool:
        """Delete a service."""
        service = db.query(ServiceModel).filter(ServiceModel.id == service_id).first()
        if service:
            db.delete(service)
            db.commit()
            return True
        return False

    @staticmethod
    def count(db: Session, status: Optional[str] = None) -> int:
        """Count services with optional status filter."""
        query = db.query(ServiceModel)
        if status:
            query = query.filter(ServiceModel.status == status)
        return query.count()


class ModelRepository:
    """Repository for model registry operations."""

    @staticmethod
    def create(db: Session, model_data: dict) -> ModelRegistryModel:
        """Register a new model."""
        model = ModelRegistryModel(**model_data)
        db.add(model)
        db.commit()
        db.refresh(model)
        return model

    @staticmethod
    def get(db: Session, model_id: str) -> Optional[ModelRegistryModel]:
        """Get model by ID."""
        return db.query(ModelRegistryModel).filter(ModelRegistryModel.id == model_id).first()

    @staticmethod
    def list(db: Session, model_type: Optional[str] = None) -> List[ModelRegistryModel]:
        """List models with optional type filter."""
        query = db.query(ModelRegistryModel)
        if model_type:
            query = query.filter(ModelRegistryModel.model_type == model_type)
        return query.order_by(ModelRegistryModel.created_at.desc()).all()

    @staticmethod
    def update(db: Session, model_id: str, updates: dict) -> Optional[ModelRegistryModel]:
        """Update a model."""
        model = db.query(ModelRegistryModel).filter(ModelRegistryModel.id == model_id).first()
        if model:
            for key, value in updates.items():
                setattr(model, key, value)
            db.commit()
            db.refresh(model)
        return model

    @staticmethod
    def delete(db: Session, model_id: str) -> bool:
        """Delete a model."""
        model = db.query(ModelRegistryModel).filter(ModelRegistryModel.id == model_id).first()
        if model:
            db.delete(model)
            db.commit()
            return True
        return False

    @staticmethod
    def count(db: Session) -> int:
        """Count total models."""
        return db.query(ModelRegistryModel).count()


class LogRepository:
    """Repository for log operations."""

    @staticmethod
    def create(db: Session, level: str, source: str, message: str) -> LogModel:
        """Create a log entry."""
        log = LogModel(level=level, source=source, message=message)
        db.add(log)
        db.commit()
        db.refresh(log)
        return log

    @staticmethod
    def list(
        db: Session,
        level: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[LogModel]:
        """List logs with optional filters."""
        query = db.query(LogModel)
        if level:
            query = query.filter(LogModel.level == level.upper())
        if source:
            query = query.filter(LogModel.source == source)
        return query.order_by(LogModel.timestamp.desc()).limit(limit).all()

    @staticmethod
    def cleanup(db: Session, keep_count: int = 1000):
        """Remove old logs, keeping only the most recent ones."""
        subquery = db.query(LogModel.id).order_by(
            LogModel.timestamp.desc()
        ).limit(keep_count).subquery()

        db.query(LogModel).filter(
            ~LogModel.id.in_(subquery)
        ).delete(synchronize_session=False)
        db.commit()
