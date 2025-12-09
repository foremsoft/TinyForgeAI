"""
Authentication for Multi-Tenant Inference Service

Provides API key authentication with tenant identification and rate limiting.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader

from services.inference.tenant import Tenant, TenantManager, TenantStatus

logger = logging.getLogger(__name__)

# API key header configuration
API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class TenantAuth:
    """
    Tenant authentication middleware.

    Validates API keys and enforces rate limits.
    """

    def __init__(self, tenant_manager: TenantManager):
        """
        Initialize auth with tenant manager.

        Args:
            tenant_manager: TenantManager instance for tenant lookup.
        """
        self.tenant_manager = tenant_manager

    async def get_current_tenant(
        self,
        request: Request,
        api_key: Optional[str] = None,
    ) -> Tenant:
        """
        Get the current tenant from API key.

        Validates the API key and checks rate limits.

        Args:
            request: FastAPI request object.
            api_key: API key from header (or extracted from request headers).

        Returns:
            Authenticated Tenant.

        Raises:
            HTTPException: If authentication fails or rate limit exceeded.
        """
        # Extract API key from header if not provided
        if api_key is None:
            api_key = request.headers.get(API_KEY_HEADER_NAME)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": f"ApiKey realm='{API_KEY_HEADER_NAME}'"},
            )

        # Look up tenant by API key
        tenant = self.tenant_manager.get_tenant_by_api_key(api_key)

        if not tenant:
            logger.warning(f"Invalid API key attempted from {request.client.host}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": f"ApiKey realm='{API_KEY_HEADER_NAME}'"},
            )

        # Check tenant status
        if tenant.status == TenantStatus.DELETED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant has been deleted",
            )

        if tenant.status == TenantStatus.SUSPENDED:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant has been suspended",
            )

        if tenant.status == TenantStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tenant is pending activation",
            )

        # Check rate limits
        can_request, reason = tenant.can_make_request()
        if not can_request:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=reason,
                headers={
                    "X-RateLimit-Limit-Minute": str(tenant.quota.requests_per_minute),
                    "X-RateLimit-Limit-Day": str(tenant.quota.requests_per_day),
                    "X-RateLimit-Remaining-Minute": str(
                        max(0, tenant.quota.requests_per_minute - tenant.usage.requests_this_minute)
                    ),
                    "X-RateLimit-Remaining-Day": str(
                        max(0, tenant.quota.requests_per_day - tenant.usage.requests_today)
                    ),
                    "Retry-After": "60",
                },
            )

        return tenant

    def create_dependency(self):
        """Create a FastAPI dependency for tenant authentication."""

        async def dependency(
            request: Request,
            api_key: Optional[str] = Security(api_key_header),
        ) -> Tenant:
            return await self.get_current_tenant(request, api_key)

        return dependency


def verify_tenant_api_key(
    tenant_manager: TenantManager,
    api_key: str,
) -> Optional[Tenant]:
    """
    Verify an API key and return the tenant.

    Args:
        tenant_manager: TenantManager instance.
        api_key: API key to verify.

    Returns:
        Tenant if valid, None otherwise.
    """
    tenant = tenant_manager.get_tenant_by_api_key(api_key)
    if tenant and tenant.is_active():
        return tenant
    return None


class RateLimitMiddleware:
    """
    Rate limiting middleware for FastAPI.

    Tracks requests per tenant and enforces limits.
    """

    def __init__(self, tenant_manager: TenantManager):
        """
        Initialize rate limiter.

        Args:
            tenant_manager: TenantManager for usage tracking.
        """
        self.tenant_manager = tenant_manager

    async def __call__(self, request: Request, call_next):
        """Process request through rate limiter."""
        # Extract API key
        api_key = request.headers.get(API_KEY_HEADER_NAME)

        if api_key:
            tenant = self.tenant_manager.get_tenant_by_api_key(api_key)
            if tenant:
                can_request, reason = tenant.can_make_request()
                if not can_request:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=429,
                        content={"detail": reason},
                        headers={
                            "X-RateLimit-Limit-Minute": str(tenant.quota.requests_per_minute),
                            "Retry-After": "60",
                        },
                    )

        response = await call_next(request)
        return response


def add_rate_limit_headers(response, tenant: Tenant) -> None:
    """Add rate limit headers to response."""
    response.headers["X-RateLimit-Limit-Minute"] = str(tenant.quota.requests_per_minute)
    response.headers["X-RateLimit-Limit-Day"] = str(tenant.quota.requests_per_day)
    response.headers["X-RateLimit-Remaining-Minute"] = str(
        max(0, tenant.quota.requests_per_minute - tenant.usage.requests_this_minute)
    )
    response.headers["X-RateLimit-Remaining-Day"] = str(
        max(0, tenant.quota.requests_per_day - tenant.usage.requests_today)
    )
