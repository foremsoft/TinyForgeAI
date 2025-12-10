"""
Inference API routes for TinyForgeAI.

Provides endpoints for running model inference including batch processing.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.config import settings
from backend.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["Inference"])


# =============================================================================
# Request/Response Models
# =============================================================================


class InferenceRequest(BaseModel):
    """Single inference request."""
    input: str = Field(..., description="Input text")
    model_name: Optional[str] = Field(None, description="Model to use (defaults to loaded model)")
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)


class InferenceResponse(BaseModel):
    """Inference response."""
    output: str
    model_name: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class BatchInferenceRequest(BaseModel):
    """Batch inference request."""
    inputs: List[str] = Field(..., min_length=1, max_length=100)
    model_name: Optional[str] = Field(None)
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0, le=2)


class BatchInferenceResponse(BaseModel):
    """Batch inference response."""
    results: List[InferenceResponse]
    total_latency_ms: float
    batch_size: int


class LoadModelRequest(BaseModel):
    """Load model request."""
    model_name: str
    version: Optional[str] = Field(None, description="Specific version to load")


class ModelStatusResponse(BaseModel):
    """Loaded model status."""
    loaded: bool
    model_name: Optional[str] = None
    version: Optional[str] = None
    model_type: Optional[str] = None


# =============================================================================
# Model State Management
# =============================================================================


class ModelManager:
    """Manages loaded models for inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name: Optional[str] = None
        self.version: Optional[str] = None
        self.model_type: Optional[str] = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded(),
            "model_name": self.model_name,
            "version": self.version,
            "model_type": self.model_type,
        }

    def load(self, model_path: str, model_name: str, version: str):
        """Load a model for inference."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
            import json
            from pathlib import Path

            # Load metadata to determine model type
            metadata_path = Path(model_path) / "model_metadata.json"
            model_type = "seq2seq"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    model_type = metadata.get("model_type", "seq2seq")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load model based on type
            if model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

            self.model.eval()
            self.model_name = model_name
            self.version = version
            self.model_type = model_type

            logger.info(f"Loaded model: {model_name} version {version}")

        except ImportError:
            # Stub mode - no transformers installed
            self.model = "stub"
            self.tokenizer = "stub"
            self.model_name = model_name
            self.version = version
            self.model_type = "stub"
            logger.warning(f"Loaded stub model: {model_name} (transformers not available)")

    def unload(self):
        """Unload the current model."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.model_name}")
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.version = None
        self.model_type = None

    def predict(
        self,
        text: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> str:
        """Generate prediction for input text."""
        if not self.is_loaded():
            raise ModelNotLoadedError()

        if self.model == "stub":
            # Stub response
            return f"[{self.model_name}] Response to: {text[:50]}..."

        import torch

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
                num_beams=1,
            )

        if self.model_type == "seq2seq":
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            input_length = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )

        return decoded


# Global model manager
_model_manager = ModelManager()


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get the status of the currently loaded model."""
    return ModelStatusResponse(**_model_manager.get_status())


@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a model for inference."""
    from pathlib import Path

    registry_path = Path(settings.MODEL_REGISTRY_PATH)
    model_dir = registry_path / request.model_name

    if not model_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {request.model_name}",
        )

    # Determine version
    version = request.version
    if not version:
        current_path = model_dir / "current"
        if current_path.exists():
            with open(current_path) as f:
                version = f.read().strip()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current version set for model: {request.model_name}",
            )

    version_path = model_dir / version
    if not version_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {version} not found for model: {request.model_name}",
        )

    # Unload existing model
    _model_manager.unload()

    # Load new model
    try:
        _model_manager.load(str(version_path), request.model_name, version)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )

    return {
        "message": f"Loaded model {request.model_name} version {version}",
        "model_name": request.model_name,
        "version": version,
    }


@router.post("/unload")
async def unload_model():
    """Unload the currently loaded model."""
    if not _model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is currently loaded",
        )

    model_name = _model_manager.model_name
    _model_manager.unload()

    return {"message": f"Unloaded model {model_name}"}


@router.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Run inference on a single input."""
    if not _model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is loaded. Use POST /inference/load first.",
        )

    start_time = time.time()

    try:
        output = _model_manager.predict(
            request.input,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    latency_ms = (time.time() - start_time) * 1000

    # Estimate token counts
    input_tokens = len(request.input.split())
    output_tokens = len(output.split())

    return InferenceResponse(
        output=output,
        model_name=_model_manager.model_name or "unknown",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=round(latency_ms, 2),
    )


@router.post("/batch", response_model=BatchInferenceResponse)
async def batch_predict(request: BatchInferenceRequest):
    """
    Run inference on a batch of inputs.

    More efficient than multiple single requests.
    """
    if not _model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is loaded. Use POST /inference/load first.",
        )

    start_time = time.time()
    results = []

    for text in request.inputs:
        item_start = time.time()

        try:
            output = _model_manager.predict(
                text,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except Exception as e:
            logger.error(f"Batch inference failed on item: {e}")
            output = f"[Error: {str(e)}]"

        item_latency = (time.time() - item_start) * 1000

        results.append(InferenceResponse(
            output=output,
            model_name=_model_manager.model_name or "unknown",
            input_tokens=len(text.split()),
            output_tokens=len(output.split()),
            latency_ms=round(item_latency, 2),
        ))

    total_latency = (time.time() - start_time) * 1000

    return BatchInferenceResponse(
        results=results,
        total_latency_ms=round(total_latency, 2),
        batch_size=len(request.inputs),
    )
