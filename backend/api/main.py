"""
TinyForgeAI Backend API.

Main FastAPI application providing endpoints for model management,
training, and inference.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.logging_setup import setup_logging
from backend.api.routes.health import router as health_router
from backend.api.routes.models import router as models_router
from backend.api.routes.training import router as training_router
from backend.api.routes.inference import router as inference_router
from backend.api.routes.webhooks import router as webhooks_router
from backend.api.routes.metrics import router as metrics_router
from backend.middleware import MetricsMiddleware

# Configure logging before app creation
setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TinyForgeAI Backend",
    description="API for training small, focused language models and deploying them as microservices",
    version="0.2.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics middleware
app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(health_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(inference_router)
app.include_router(webhooks_router)
app.include_router(metrics_router)


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("TinyForgeAI Backend starting up...")
    logger.info(f"Model registry path: {settings.MODEL_REGISTRY_PATH}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("TinyForgeAI Backend shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
