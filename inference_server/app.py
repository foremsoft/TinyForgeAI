"""
Inference server FastAPI application.

Provides a /predict endpoint for running model inference
and a /health endpoint for service health checks.
"""

import os

from fastapi import FastAPI

from inference_server.model_loader import ModelWrapper
from inference_server.schemas import PredictRequest, PredictResponse

app = FastAPI(title="TinyForgeAI Inference Server")

# Initialize model wrapper (lazy loading)
model = ModelWrapper()


@app.get("/health")
async def health_check():
    """Health check endpoint for the inference server."""
    return {"status": "ok", "service": "inference-server"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run inference on the provided input.

    Args:
        request: PredictRequest containing the input text.

    Returns:
        PredictResponse with the model output and confidence score.
    """
    output, confidence = model.predict(request.input)
    return PredictResponse(output=output, confidence=confidence)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("INFERENCE_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
