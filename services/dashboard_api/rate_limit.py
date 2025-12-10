"""
Rate limiting middleware and utilities for TinyForgeAI Dashboard API.

Supports multiple strategies:
- In-memory sliding window (default, single instance)
- Redis-based sliding window (distributed, multi-instance)

Usage:
    from services.dashboard_api.rate_limit import RateLimiter, rate_limit

    # Decorator-based limiting
    @app.get("/api/endpoint")
    @rate_limit(requests=100, window=60)  # 100 requests per 60 seconds
    async def endpoint():
        ...

    # Dependency-based limiting
    @app.get("/api/endpoint")
    async def endpoint(limiter: RateLimiter = Depends(get_rate_limiter)):
        await limiter.check("endpoint", request)
        ...
"""

import asyncio
import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configuration
RATE_LIMIT_ENABLED = os.getenv("TINYFORGE_RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_REDIS_URL = os.getenv("TINYFORGE_REDIS_URL", "")
DEFAULT_REQUESTS_PER_MINUTE = int(os.getenv("TINYFORGE_RATE_LIMIT_RPM", "60"))
DEFAULT_REQUESTS_PER_HOUR = int(os.getenv("TINYFORGE_RATE_LIMIT_RPH", "1000"))

# Try to import Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""
    requests: int  # Max requests allowed
    window: int  # Time window in seconds
    key_prefix: str = ""  # Optional prefix for the key
    by_ip: bool = True  # Rate limit by IP
    by_user: bool = False  # Rate limit by user (requires auth)
    by_endpoint: bool = True  # Rate limit by endpoint
    burst_multiplier: float = 1.5  # Allow burst up to this multiplier


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None
    limit: int = 0
    current: int = 0


class InMemoryStore:
    """In-memory sliding window rate limit store."""

    def __init__(self):
        self._store: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()

    async def _cleanup(self):
        """Remove expired entries."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        async with self._lock:
            cutoff = now - 3600  # Keep last hour of data
            for key in list(self._store.keys()):
                self._store[key] = [t for t in self._store[key] if t > cutoff]
                if not self._store[key]:
                    del self._store[key]
            self._last_cleanup = now

    async def check_and_increment(
        self,
        key: str,
        limit: int,
        window: int,
    ) -> RateLimitResult:
        """Check rate limit and increment counter if allowed."""
        await self._cleanup()

        now = time.time()
        window_start = now - window

        async with self._lock:
            # Get requests in current window
            self._store[key] = [t for t in self._store[key] if t > window_start]
            current = len(self._store[key])

            if current < limit:
                self._store[key].append(now)
                return RateLimitResult(
                    allowed=True,
                    remaining=limit - current - 1,
                    reset_at=window_start + window,
                    limit=limit,
                    current=current + 1,
                )
            else:
                # Find when the oldest request will expire
                oldest = min(self._store[key]) if self._store[key] else now
                retry_after = oldest + window - now
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest + window,
                    retry_after=max(0, retry_after),
                    limit=limit,
                    current=current,
                )

    async def get_count(self, key: str, window: int) -> int:
        """Get current request count for a key."""
        now = time.time()
        window_start = now - window

        async with self._lock:
            return len([t for t in self._store[key] if t > window_start])


if REDIS_AVAILABLE:
    class RedisStore:
        """Redis-based sliding window rate limit store."""

        def __init__(self, redis_url: str):
            self._redis_url = redis_url
            self._client = None

        async def _get_client(self):
            """Get or create Redis client."""
            if self._client is None:
                self._client = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            return self._client

        async def close(self):
            """Close Redis connection."""
            if self._client:
                await self._client.close()
                self._client = None

        async def check_and_increment(
            self,
            key: str,
            limit: int,
            window: int,
        ) -> RateLimitResult:
            """Check rate limit and increment counter if allowed using Redis."""
            client = await self._get_client()
            now = time.time()
            window_start = now - window

            # Use Redis sorted set with timestamps as scores
            pipe = client.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            pipe.zcard(key)
            # Get oldest entry
            pipe.zrange(key, 0, 0, withscores=True)

            results = await pipe.execute()
            current = results[1]
            oldest_entries = results[2]

            if current < limit:
                # Add new entry
                await client.zadd(key, {str(now): now})
                await client.expire(key, window + 60)  # Extra buffer

                return RateLimitResult(
                    allowed=True,
                    remaining=limit - current - 1,
                    reset_at=window_start + window,
                    limit=limit,
                    current=current + 1,
                )
            else:
                oldest = float(oldest_entries[0][1]) if oldest_entries else now
                retry_after = oldest + window - now
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=oldest + window,
                    retry_after=max(0, retry_after),
                    limit=limit,
                    current=current,
                )

        async def get_count(self, key: str, window: int) -> int:
            """Get current request count for a key."""
            client = await self._get_client()
            now = time.time()
            window_start = now - window

            # Remove old entries and count
            await client.zremrangebyscore(key, 0, window_start)
            return await client.zcard(key)
else:
    # Placeholder when Redis is not available
    RedisStore = None  # type: ignore


class RateLimiter:
    """
    Rate limiter with support for multiple strategies.

    Example:
        limiter = RateLimiter()

        # Check a specific limit
        result = await limiter.check(
            "api_calls",
            request,
            RateLimitConfig(requests=100, window=60)
        )

        if not result.allowed:
            raise HTTPException(429, detail="Rate limit exceeded")
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize rate limiter.

        Args:
            redis_url: Redis URL for distributed rate limiting.
                      If None, uses in-memory store.
        """
        if redis_url and REDIS_AVAILABLE:
            self._store = RedisStore(redis_url)
            self._distributed = True
        else:
            self._store = InMemoryStore()
            self._distributed = False

        # Default rate limits by category
        self._default_limits: Dict[str, RateLimitConfig] = {
            "default": RateLimitConfig(
                requests=DEFAULT_REQUESTS_PER_MINUTE,
                window=60,
            ),
            "api": RateLimitConfig(
                requests=DEFAULT_REQUESTS_PER_MINUTE,
                window=60,
            ),
            "auth": RateLimitConfig(
                requests=10,
                window=60,
                burst_multiplier=1.0,  # No burst for auth
            ),
            "inference": RateLimitConfig(
                requests=30,
                window=60,
            ),
            "websocket": RateLimitConfig(
                requests=10,
                window=60,
            ),
            "hourly": RateLimitConfig(
                requests=DEFAULT_REQUESTS_PER_HOUR,
                window=3600,
            ),
        }

    def _build_key(
        self,
        category: str,
        request: Request,
        config: RateLimitConfig,
    ) -> str:
        """Build rate limit key from request."""
        parts = [config.key_prefix or "ratelimit", category]

        if config.by_ip:
            # Get client IP (handle proxies)
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                ip = forwarded.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else "unknown"
            parts.append(f"ip:{ip}")

        if config.by_user:
            # Get user from request state (set by auth middleware)
            user = getattr(request.state, "user", None)
            if user:
                parts.append(f"user:{user}")

        if config.by_endpoint:
            # Hash the endpoint for consistent key length
            endpoint = f"{request.method}:{request.url.path}"
            endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
            parts.append(f"ep:{endpoint_hash}")

        return ":".join(parts)

    async def check(
        self,
        category: str,
        request: Request,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitResult:
        """
        Check rate limit for a request.

        Args:
            category: Rate limit category (e.g., "api", "auth")
            request: FastAPI request object
            config: Optional custom config, uses default for category if None

        Returns:
            RateLimitResult with allowed status and metadata
        """
        if not RATE_LIMIT_ENABLED:
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_at=time.time() + 60,
                limit=999999,
                current=0,
            )

        config = config or self._default_limits.get(
            category,
            self._default_limits["default"],
        )

        key = self._build_key(category, request, config)
        return await self._store.check_and_increment(
            key,
            config.requests,
            config.window,
        )

    async def check_multiple(
        self,
        request: Request,
        categories: List[str],
    ) -> Tuple[bool, Optional[RateLimitResult]]:
        """
        Check multiple rate limits.

        Returns (allowed, failing_result) where failing_result is the
        first limit that was exceeded, or None if all passed.
        """
        for category in categories:
            result = await self.check(category, request)
            if not result.allowed:
                return False, result
        return True, None

    def set_limit(self, category: str, config: RateLimitConfig):
        """Set or update a rate limit configuration."""
        self._default_limits[category] = config

    async def close(self):
        """Close any connections."""
        if hasattr(self._store, "close"):
            await self._store.close()


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        redis_url = RATE_LIMIT_REDIS_URL if REDIS_AVAILABLE else None
        _rate_limiter = RateLimiter(redis_url)
    return _rate_limiter


def rate_limit(
    requests: int = DEFAULT_REQUESTS_PER_MINUTE,
    window: int = 60,
    category: str = "api",
    by_ip: bool = True,
    by_user: bool = False,
    by_endpoint: bool = True,
):
    """
    Decorator to rate limit an endpoint.

    Args:
        requests: Max requests allowed in window
        window: Time window in seconds
        category: Rate limit category
        by_ip: Rate limit by client IP
        by_user: Rate limit by authenticated user
        by_endpoint: Rate limit by endpoint

    Example:
        @app.get("/api/data")
        @rate_limit(requests=100, window=60)
        async def get_data(request: Request):
            ...
    """
    config = RateLimitConfig(
        requests=requests,
        window=window,
        by_ip=by_ip,
        by_user=by_user,
        by_endpoint=by_endpoint,
    )

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find request in args/kwargs
            request = kwargs.get("request")
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is None:
                # No request found, skip rate limiting
                return await func(*args, **kwargs)

            limiter = get_rate_limiter()
            result = await limiter.check(category, request, config)

            if not result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": result.limit,
                        "remaining": result.remaining,
                        "retry_after": result.retry_after,
                    },
                    headers={
                        "X-RateLimit-Limit": str(result.limit),
                        "X-RateLimit-Remaining": str(result.remaining),
                        "X-RateLimit-Reset": str(int(result.reset_at)),
                        "Retry-After": str(int(result.retry_after or 1)),
                    },
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global rate limiting.

    Applies default rate limits to all requests and adds
    rate limit headers to responses.
    """

    def __init__(
        self,
        app,
        limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[List[str]] = None,
    ):
        super().__init__(app)
        self._limiter = limiter or get_rate_limiter()
        self._exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]

    def _get_category(self, path: str, method: str) -> str:
        """Determine rate limit category from path."""
        if "/auth/" in path:
            return "auth"
        if "/predict" in path or "/inference" in path:
            return "inference"
        if path.startswith("/ws"):
            return "websocket"
        return "api"

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        path = request.url.path

        # Skip excluded paths
        if any(path.startswith(p) for p in self._exclude_paths):
            return await call_next(request)

        # Determine category
        category = self._get_category(path, request.method)

        # Check rate limit
        result = await self._limiter.check(category, request)

        if not result.allowed:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                    "Retry-After": str(int(result.retry_after or 1)),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))

        return response


# Endpoint-specific rate limit configurations
ENDPOINT_LIMITS = {
    "/api/auth/login": RateLimitConfig(requests=5, window=60, by_ip=True),
    "/api/auth/logout": RateLimitConfig(requests=10, window=60, by_ip=True),
    "/api/jobs": RateLimitConfig(requests=30, window=60),
    "/api/predict": RateLimitConfig(requests=60, window=60),
    "/api/models": RateLimitConfig(requests=30, window=60),
    "/api/services": RateLimitConfig(requests=30, window=60),
}


async def check_endpoint_limit(request: Request, path: str) -> RateLimitResult:
    """Check rate limit for a specific endpoint."""
    limiter = get_rate_limiter()
    config = ENDPOINT_LIMITS.get(path)

    if config:
        return await limiter.check("endpoint", request, config)

    return await limiter.check("api", request)
