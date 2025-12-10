"""Tests for rate limiting module."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from services.dashboard_api.rate_limit import (
    RateLimitConfig,
    RateLimitResult,
    InMemoryStore,
    RateLimiter,
    get_rate_limiter,
    rate_limit,
    RATE_LIMIT_ENABLED,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig(requests=100, window=60)
        assert config.requests == 100
        assert config.window == 60
        assert config.key_prefix == ""
        assert config.by_ip is True
        assert config.by_user is False
        assert config.by_endpoint is True
        assert config.burst_multiplier == 1.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = RateLimitConfig(
            requests=10,
            window=30,
            key_prefix="custom",
            by_ip=False,
            by_user=True,
            by_endpoint=False,
            burst_multiplier=2.0,
        )
        assert config.requests == 10
        assert config.window == 30
        assert config.key_prefix == "custom"
        assert config.by_ip is False
        assert config.by_user is True
        assert config.by_endpoint is False
        assert config.burst_multiplier == 2.0


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_allowed_result(self):
        """Test allowed rate limit result."""
        result = RateLimitResult(
            allowed=True,
            remaining=99,
            reset_at=time.time() + 60,
            limit=100,
            current=1,
        )
        assert result.allowed is True
        assert result.remaining == 99
        assert result.retry_after is None

    def test_denied_result(self):
        """Test denied rate limit result."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=time.time() + 30,
            retry_after=30.0,
            limit=100,
            current=100,
        )
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30.0


class TestInMemoryStore:
    """Tests for InMemoryStore."""

    @pytest.fixture
    def store(self):
        """Create a fresh in-memory store."""
        return InMemoryStore()

    @pytest.mark.asyncio
    async def test_first_request_allowed(self, store):
        """Test first request is always allowed."""
        result = await store.check_and_increment("test_key", limit=10, window=60)
        assert result.allowed is True
        assert result.remaining == 9
        assert result.current == 1

    @pytest.mark.asyncio
    async def test_within_limit(self, store):
        """Test requests within limit are allowed."""
        for i in range(5):
            result = await store.check_and_increment("test_key", limit=10, window=60)
            assert result.allowed is True
            assert result.remaining == 10 - i - 1

    @pytest.mark.asyncio
    async def test_exceed_limit(self, store):
        """Test requests exceeding limit are denied."""
        # Use up the limit
        for _ in range(10):
            await store.check_and_increment("test_key", limit=10, window=60)

        # Next request should be denied
        result = await store.check_and_increment("test_key", limit=10, window=60)
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    @pytest.mark.asyncio
    async def test_different_keys(self, store):
        """Test different keys have separate limits."""
        # Exhaust limit for key1
        for _ in range(5):
            await store.check_and_increment("key1", limit=5, window=60)

        result1 = await store.check_and_increment("key1", limit=5, window=60)
        assert result1.allowed is False

        # key2 should still be allowed
        result2 = await store.check_and_increment("key2", limit=5, window=60)
        assert result2.allowed is True

    @pytest.mark.asyncio
    async def test_window_expiry(self, store):
        """Test requests are allowed after window expires."""
        # Use a very short window
        for _ in range(5):
            await store.check_and_increment("test_key", limit=5, window=0.1)

        # Wait for window to expire
        await asyncio.sleep(0.15)

        # Should be allowed again
        result = await store.check_and_increment("test_key", limit=5, window=0.1)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_get_count(self, store):
        """Test get_count method."""
        for _ in range(3):
            await store.check_and_increment("test_key", limit=10, window=60)

        count = await store.get_count("test_key", window=60)
        assert count == 3


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self):
        """Create a fresh rate limiter."""
        return RateLimiter()

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        request.method = "GET"
        request.url.path = "/api/test"
        request.state = MagicMock()
        request.state.user = None
        return request

    @pytest.mark.asyncio
    async def test_check_allows_request(self, limiter, mock_request):
        """Test check allows initial request."""
        result = await limiter.check("api", mock_request)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_default_limits_exist(self, limiter):
        """Test default rate limits are configured."""
        assert "default" in limiter._default_limits
        assert "api" in limiter._default_limits
        assert "auth" in limiter._default_limits
        assert "inference" in limiter._default_limits

    @pytest.mark.asyncio
    async def test_auth_limit_stricter(self, limiter, mock_request):
        """Test auth endpoint has stricter limit."""
        api_config = limiter._default_limits["api"]
        auth_config = limiter._default_limits["auth"]
        assert auth_config.requests < api_config.requests

    @pytest.mark.asyncio
    async def test_custom_config(self, limiter, mock_request):
        """Test using custom config."""
        config = RateLimitConfig(requests=1, window=60)
        result1 = await limiter.check("test", mock_request, config)
        assert result1.allowed is True

        result2 = await limiter.check("test", mock_request, config)
        assert result2.allowed is False

    @pytest.mark.asyncio
    async def test_set_limit(self, limiter):
        """Test setting custom limit."""
        config = RateLimitConfig(requests=50, window=30)
        limiter.set_limit("custom", config)
        assert "custom" in limiter._default_limits
        assert limiter._default_limits["custom"].requests == 50

    @pytest.mark.asyncio
    async def test_check_multiple_all_pass(self, limiter, mock_request):
        """Test check_multiple when all limits pass."""
        allowed, result = await limiter.check_multiple(mock_request, ["api", "default"])
        assert allowed is True
        assert result is None

    @pytest.mark.asyncio
    async def test_forwarded_ip(self, limiter, mock_request):
        """Test X-Forwarded-For header is used."""
        mock_request.headers = {"x-forwarded-for": "192.168.1.1, 10.0.0.1"}
        config = RateLimitConfig(requests=1, window=60)

        # First request from forwarded IP
        result1 = await limiter.check("test", mock_request, config)
        assert result1.allowed is True

        # Second request should be denied
        result2 = await limiter.check("test", mock_request, config)
        assert result2.allowed is False

        # Request from different IP should be allowed
        mock_request.headers = {"x-forwarded-for": "192.168.1.2"}
        result3 = await limiter.check("test", mock_request, config)
        assert result3.allowed is True


class TestRateLimitDecorator:
    """Tests for rate_limit decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_request(self):
        """Test decorator allows request within limit."""
        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.state = MagicMock()
        mock_request.state.user = None

        @rate_limit(requests=100, window=60)
        async def test_endpoint(request):
            return {"status": "ok"}

        # Should not raise
        result = await test_endpoint(request=mock_request)
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_decorator_no_request(self):
        """Test decorator works without request object."""
        @rate_limit(requests=100, window=60)
        async def test_endpoint():
            return {"status": "ok"}

        # Should not raise even without request
        result = await test_endpoint()
        assert result == {"status": "ok"}


class TestGlobalRateLimiter:
    """Tests for global rate limiter singleton."""

    def test_get_rate_limiter_returns_instance(self):
        """Test get_rate_limiter returns a RateLimiter instance."""
        import services.dashboard_api.rate_limit as rate_limit_module
        rate_limit_module._rate_limiter = None

        limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_singleton(self):
        """Test get_rate_limiter returns same instance."""
        import services.dashboard_api.rate_limit as rate_limit_module
        rate_limit_module._rate_limiter = None

        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test rate limiting under concurrent load."""
        store = InMemoryStore()

        async def make_request():
            return await store.check_and_increment("concurrent", limit=10, window=60)

        # Make 20 concurrent requests
        tasks = [make_request() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        allowed_count = sum(1 for r in results if r.allowed)
        denied_count = sum(1 for r in results if not r.allowed)

        # Exactly 10 should be allowed
        assert allowed_count == 10
        assert denied_count == 10

    @pytest.mark.asyncio
    async def test_sequential_requests(self):
        """Test rate limiting with sequential requests."""
        store = InMemoryStore()

        allowed = 0
        denied = 0

        for _ in range(15):
            result = await store.check_and_increment("sequential", limit=10, window=60)
            if result.allowed:
                allowed += 1
            else:
                denied += 1

        assert allowed == 10
        assert denied == 5
