"""
Multi-Tenant Inference Service Application

FastAPI application providing tenant-isolated inference endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.inference.tenant import (
    Tenant,
    TenantManager,
    TenantConfig,
    TenantQuota,
    TenantStatus,
    TenantTier,
)
from services.inference.auth import TenantAuth, add_rate_limit_headers

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DB_PATH = os.getenv("TINYFORGE_TENANT_DB", "data/tenants.db")
ADMIN_API_KEY = os.getenv("TINYFORGE_ADMIN_API_KEY", "admin-secret-key")

# =============================================================================
# Request/Response Models
# =============================================================================


class InferenceRequest(BaseModel):
    """Inference request body."""
    input: str = Field(..., description="Input text for inference")
    model: Optional[str] = Field(None, description="Model to use (defaults to tenant's default)")
    max_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="Sampling temperature")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional options")


class InferenceResponse(BaseModel):
    """Inference response body."""
    output: str = Field(..., description="Model output")
    model: str = Field(..., description="Model used")
    tokens_in: int = Field(..., description="Input tokens")
    tokens_out: int = Field(..., description="Output tokens")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    tenant_id: str = Field(..., description="Tenant ID")


class TenantCreateRequest(BaseModel):
    """Create tenant request."""
    name: str = Field(..., description="Tenant name")
    tier: Optional[str] = Field("free", description="Tenant tier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TenantCreateResponse(BaseModel):
    """Create tenant response."""
    tenant_id: str
    name: str
    api_key: str = Field(..., description="API key - store securely, shown only once!")
    tier: str
    status: str


class TenantResponse(BaseModel):
    """Tenant details response."""
    id: str
    name: str
    status: str
    tier: str
    quota: Dict[str, Any]
    usage: Dict[str, Any]
    created_at: str
    updated_at: str


class TenantUpdateRequest(BaseModel):
    """Update tenant request."""
    name: Optional[str] = None
    status: Optional[str] = None
    tier: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class UsageResponse(BaseModel):
    """Usage statistics response."""
    tenant_id: str
    usage: Dict[str, Any]
    quota: Dict[str, Any]
    tier: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    timestamp: str


# =============================================================================
# Application Setup
# =============================================================================

# Global instances
tenant_manager: Optional[TenantManager] = None
tenant_auth: Optional[TenantAuth] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global tenant_manager, tenant_auth

    # Initialize tenant manager
    tenant_manager = TenantManager(db_path=DB_PATH)
    tenant_auth = TenantAuth(tenant_manager)

    logger.info(f"Multi-tenant inference service started with DB: {DB_PATH}")

    yield

    # Cleanup
    logger.info("Shutting down multi-tenant inference service")


app = FastAPI(
    title="TinyForgeAI Multi-Tenant Inference Service",
    description="Tenant-isolated inference endpoints with rate limiting and usage tracking",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependency Functions
# =============================================================================


def get_tenant_manager() -> TenantManager:
    """Get tenant manager dependency."""
    if tenant_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant manager not initialized",
        )
    return tenant_manager


def get_tenant_auth() -> TenantAuth:
    """Get tenant auth dependency."""
    if tenant_auth is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tenant auth not initialized",
        )
    return tenant_auth


async def get_current_tenant(
    request: Request,
    auth: TenantAuth = Depends(get_tenant_auth),
) -> Tenant:
    """Get current authenticated tenant."""
    return await auth.get_current_tenant(request)


def verify_admin(request: Request) -> bool:
    """Verify admin API key."""
    api_key = request.headers.get("X-Admin-Key")
    if api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return True


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="multi-tenant-inference",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/ready", tags=["Health"])
async def readiness_check(manager: TenantManager = Depends(get_tenant_manager)):
    """Readiness check endpoint."""
    return {
        "status": "ready",
        "tenants_loaded": len(manager._tenants),
    }


# =============================================================================
# Inference Endpoints
# =============================================================================


@app.post("/v1/inference", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(
    request: InferenceRequest,
    tenant: Tenant = Depends(get_current_tenant),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Run inference for the authenticated tenant.

    Validates quota limits, runs the model, and tracks usage.
    """
    import time
    start_time = time.time()

    # Validate input length
    input_tokens = len(request.input.split())  # Simple word-based tokenization
    if input_tokens > tenant.quota.max_input_tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input exceeds maximum tokens ({tenant.quota.max_input_tokens})",
        )

    # Determine model to use
    model = request.model or tenant.config.default_model or "default"

    # Check if model is allowed for tenant
    if tenant.config.allowed_models and model not in tenant.config.allowed_models:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Model '{model}' not allowed for this tenant",
        )

    # Stub inference - in production, load actual model
    # For now, return a simple response
    output = f"[{model}] Response to: {request.input[:50]}..."
    output_tokens = len(output.split())

    # Record usage
    manager.record_usage(tenant.id, tokens_in=input_tokens, tokens_out=output_tokens)

    latency_ms = (time.time() - start_time) * 1000

    return InferenceResponse(
        output=output,
        model=model,
        tokens_in=input_tokens,
        tokens_out=output_tokens,
        latency_ms=round(latency_ms, 2),
        tenant_id=tenant.id,
    )


@app.post("/v1/completions", response_model=InferenceResponse, tags=["Inference"])
async def completions(
    request: InferenceRequest,
    tenant: Tenant = Depends(get_current_tenant),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Text completion endpoint (OpenAI-compatible style).

    Alias for /v1/inference.
    """
    return await run_inference(request, tenant, manager)


# =============================================================================
# Tenant Self-Service Endpoints
# =============================================================================


@app.get("/v1/me", response_model=TenantResponse, tags=["Tenant"])
async def get_current_tenant_info(tenant: Tenant = Depends(get_current_tenant)):
    """Get current tenant information."""
    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        status=tenant.status.value,
        tier=tenant.tier.value,
        quota=tenant.quota.__dict__,
        usage=tenant.usage.to_dict(),
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@app.get("/v1/usage", response_model=UsageResponse, tags=["Tenant"])
async def get_usage(
    tenant: Tenant = Depends(get_current_tenant),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Get current tenant's usage statistics."""
    stats = manager.get_usage_stats(tenant.id)
    return UsageResponse(**stats)


@app.post("/v1/api-key/rotate", tags=["Tenant"])
async def rotate_api_key(
    tenant: Tenant = Depends(get_current_tenant),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """
    Rotate API key for current tenant.

    Returns the new API key - store securely!
    """
    new_key = manager.rotate_api_key(tenant.id)
    return {
        "message": "API key rotated successfully",
        "api_key": new_key,
        "warning": "Store this key securely - it will not be shown again!",
    }


# =============================================================================
# Admin Endpoints
# =============================================================================


@app.post("/admin/tenants", response_model=TenantCreateResponse, tags=["Admin"])
async def create_tenant(
    request: TenantCreateRequest,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Create a new tenant (admin only)."""
    try:
        tier = TenantTier(request.tier) if request.tier else TenantTier.FREE
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier. Must be one of: {[t.value for t in TenantTier]}",
        )

    tenant, api_key = manager.create_tenant(
        name=request.name,
        tier=tier,
        metadata=request.metadata,
    )

    return TenantCreateResponse(
        tenant_id=tenant.id,
        name=tenant.name,
        api_key=api_key,
        tier=tenant.tier.value,
        status=tenant.status.value,
    )


@app.get("/admin/tenants", response_model=List[TenantResponse], tags=["Admin"])
async def list_tenants(
    status: Optional[str] = None,
    tier: Optional[str] = None,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """List all tenants (admin only)."""
    filter_status = TenantStatus(status) if status else None
    filter_tier = TenantTier(tier) if tier else None

    tenants = manager.list_tenants(status=filter_status, tier=filter_tier)

    return [
        TenantResponse(
            id=t.id,
            name=t.name,
            status=t.status.value,
            tier=t.tier.value,
            quota=t.quota.__dict__,
            usage=t.usage.to_dict(),
            created_at=t.created_at,
            updated_at=t.updated_at,
        )
        for t in tenants
    ]


@app.get("/admin/tenants/{tenant_id}", response_model=TenantResponse, tags=["Admin"])
async def get_tenant(
    tenant_id: str,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Get tenant by ID (admin only)."""
    tenant = manager.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        status=tenant.status.value,
        tier=tenant.tier.value,
        quota=tenant.quota.__dict__,
        usage=tenant.usage.to_dict(),
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@app.patch("/admin/tenants/{tenant_id}", response_model=TenantResponse, tags=["Admin"])
async def update_tenant(
    tenant_id: str,
    request: TenantUpdateRequest,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Update tenant (admin only)."""
    update_status = TenantStatus(request.status) if request.status else None
    update_tier = TenantTier(request.tier) if request.tier else None

    tenant = manager.update_tenant(
        tenant_id=tenant_id,
        name=request.name,
        status=update_status,
        tier=update_tier,
        metadata=request.metadata,
    )

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        status=tenant.status.value,
        tier=tenant.tier.value,
        quota=tenant.quota.__dict__,
        usage=tenant.usage.to_dict(),
        created_at=tenant.created_at,
        updated_at=tenant.updated_at,
    )


@app.post("/admin/tenants/{tenant_id}/suspend", tags=["Admin"])
async def suspend_tenant(
    tenant_id: str,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Suspend a tenant (admin only)."""
    tenant = manager.suspend_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    return {"message": f"Tenant {tenant_id} suspended"}


@app.post("/admin/tenants/{tenant_id}/activate", tags=["Admin"])
async def activate_tenant(
    tenant_id: str,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Activate a tenant (admin only)."""
    tenant = manager.activate_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    return {"message": f"Tenant {tenant_id} activated"}


@app.delete("/admin/tenants/{tenant_id}", tags=["Admin"])
async def delete_tenant(
    tenant_id: str,
    hard_delete: bool = False,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Delete a tenant (admin only)."""
    if not manager.delete_tenant(tenant_id, hard_delete=hard_delete):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    return {"message": f"Tenant {tenant_id} deleted (hard={hard_delete})"}


@app.get("/admin/tenants/{tenant_id}/usage", response_model=UsageResponse, tags=["Admin"])
async def get_tenant_usage(
    tenant_id: str,
    _: bool = Depends(verify_admin),
    manager: TenantManager = Depends(get_tenant_manager),
):
    """Get tenant usage statistics (admin only)."""
    stats = manager.get_usage_stats(tenant_id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )
    return UsageResponse(**stats)


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    port = int(os.getenv("INFERENCE_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
