"""
Pydantic schemas for inference server request/response models.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for the /predict endpoint."""

    input: str = Field(..., description="Input text to process")


class PredictResponse(BaseModel):
    """Response schema for the /predict endpoint."""

    output: str = Field(..., description="Model prediction output")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
