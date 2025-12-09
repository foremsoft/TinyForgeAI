from fastapi import FastAPI

from backend.config import settings  # noqa: F401 - ensures env is loaded
from backend.logging_setup import setup_logging
from backend.api.routes.health import router as health_router

# Configure logging before app creation
setup_logging()

app = FastAPI(title="TinyForgeAI Backend")

app.include_router(health_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
