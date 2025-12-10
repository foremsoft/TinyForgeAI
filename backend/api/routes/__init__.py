"""TinyForgeAI Backend API Routes."""

from backend.api.routes.health import router as health_router
from backend.api.routes.models import router as models_router
from backend.api.routes.training import router as training_router
from backend.api.routes.inference import router as inference_router
from backend.api.routes.webhooks import router as webhooks_router
from backend.api.routes.metrics import router as metrics_router

__all__ = [
    "health_router",
    "models_router",
    "training_router",
    "inference_router",
    "webhooks_router",
    "metrics_router",
]
