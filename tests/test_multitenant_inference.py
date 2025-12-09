"""
Tests for Multi-Tenant Inference Service.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from services.inference.tenant import (
    Tenant,
    TenantManager,
    TenantConfig,
    TenantQuota,
    TenantUsage,
    TenantStatus,
    TenantTier,
)
from services.inference.auth import TenantAuth, verify_tenant_api_key


class TestTenantQuota:
    """Test TenantQuota class."""

    def test_default_quota(self):
        """Test default quota values."""
        quota = TenantQuota()
        assert quota.requests_per_minute == 60
        assert quota.requests_per_day == 1000
        assert quota.max_input_tokens == 4096
        assert quota.max_output_tokens == 2048
        assert quota.max_models == 1
        assert quota.max_concurrent_requests == 5

    def test_quota_for_free_tier(self):
        """Test quota for free tier."""
        quota = TenantQuota.for_tier(TenantTier.FREE)
        assert quota.requests_per_minute == 10
        assert quota.requests_per_day == 100
        assert quota.max_models == 1

    def test_quota_for_starter_tier(self):
        """Test quota for starter tier."""
        quota = TenantQuota.for_tier(TenantTier.STARTER)
        assert quota.requests_per_minute == 30
        assert quota.requests_per_day == 1000
        assert quota.max_models == 3

    def test_quota_for_professional_tier(self):
        """Test quota for professional tier."""
        quota = TenantQuota.for_tier(TenantTier.PROFESSIONAL)
        assert quota.requests_per_minute == 100
        assert quota.requests_per_day == 10000
        assert quota.max_models == 10

    def test_quota_for_enterprise_tier(self):
        """Test quota for enterprise tier."""
        quota = TenantQuota.for_tier(TenantTier.ENTERPRISE)
        assert quota.requests_per_minute == 1000
        assert quota.requests_per_day == 100000
        assert quota.max_models == 100


class TestTenantUsage:
    """Test TenantUsage class."""

    def test_default_usage(self):
        """Test default usage values."""
        usage = TenantUsage()
        assert usage.total_requests == 0
        assert usage.total_tokens_in == 0
        assert usage.total_tokens_out == 0
        assert usage.requests_today == 0
        assert usage.requests_this_minute == 0

    def test_usage_to_dict(self):
        """Test usage serialization."""
        usage = TenantUsage(total_requests=100, total_tokens_in=5000)
        d = usage.to_dict()
        assert d["total_requests"] == 100
        assert d["total_tokens_in"] == 5000

    def test_usage_from_dict(self):
        """Test usage deserialization."""
        data = {"total_requests": 50, "requests_today": 10}
        usage = TenantUsage.from_dict(data)
        assert usage.total_requests == 50
        assert usage.requests_today == 10


class TestTenantConfig:
    """Test TenantConfig class."""

    def test_default_config(self):
        """Test default config values."""
        config = TenantConfig()
        assert config.default_model is None
        assert config.allowed_models == []
        assert config.custom_settings == {}
        assert config.webhook_url is None

    def test_config_with_values(self):
        """Test config with custom values."""
        config = TenantConfig(
            default_model="gpt2",
            allowed_models=["gpt2", "flan-t5"],
            webhook_url="https://example.com/callback",
        )
        assert config.default_model == "gpt2"
        assert len(config.allowed_models) == 2
        assert config.webhook_url == "https://example.com/callback"


class TestTenant:
    """Test Tenant class."""

    def test_tenant_creation(self):
        """Test tenant creation."""
        tenant = Tenant(
            id="test123",
            name="Test Tenant",
            api_key_hash="abc123",
        )
        assert tenant.id == "test123"
        assert tenant.name == "Test Tenant"
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.tier == TenantTier.FREE

    def test_tenant_is_active(self):
        """Test is_active method."""
        tenant = Tenant(id="t1", name="Test", api_key_hash="hash")
        assert tenant.is_active() is True

        tenant.status = TenantStatus.SUSPENDED
        assert tenant.is_active() is False

    def test_tenant_can_make_request_active(self):
        """Test can_make_request for active tenant."""
        tenant = Tenant(id="t1", name="Test", api_key_hash="hash")
        can_request, reason = tenant.can_make_request()
        assert can_request is True
        assert reason is None

    def test_tenant_can_make_request_suspended(self):
        """Test can_make_request for suspended tenant."""
        tenant = Tenant(id="t1", name="Test", api_key_hash="hash")
        tenant.status = TenantStatus.SUSPENDED
        can_request, reason = tenant.can_make_request()
        assert can_request is False
        assert "suspended" in reason

    def test_tenant_record_request(self):
        """Test record_request updates counters."""
        tenant = Tenant(id="t1", name="Test", api_key_hash="hash")
        assert tenant.usage.total_requests == 0

        tenant.record_request(tokens_in=100, tokens_out=50)

        assert tenant.usage.total_requests == 1
        assert tenant.usage.requests_this_minute == 1
        assert tenant.usage.requests_today == 1
        assert tenant.usage.total_tokens_in == 100
        assert tenant.usage.total_tokens_out == 50

    def test_tenant_rate_limit_exceeded(self):
        """Test rate limit enforcement."""
        tenant = Tenant(
            id="t1",
            name="Test",
            api_key_hash="hash",
            quota=TenantQuota(requests_per_minute=2, requests_per_day=10),
        )

        # First two requests should pass
        tenant.record_request()
        can_request, _ = tenant.can_make_request()
        assert can_request is True

        tenant.record_request()
        can_request, reason = tenant.can_make_request()
        assert can_request is False
        assert "per minute" in reason

    def test_tenant_to_dict(self):
        """Test tenant serialization."""
        tenant = Tenant(id="t1", name="Test", api_key_hash="hash")
        d = tenant.to_dict()

        assert d["id"] == "t1"
        assert d["name"] == "Test"
        assert d["status"] == "active"
        assert d["tier"] == "free"
        assert "quota" in d
        assert "usage" in d

    def test_tenant_from_dict(self):
        """Test tenant deserialization."""
        data = {
            "id": "t2",
            "name": "From Dict",
            "api_key_hash": "hash123",
            "status": "active",
            "tier": "starter",
            "quota": {"requests_per_minute": 30},
            "usage": {"total_requests": 5},
            "config": {},
            "metadata": {},
        }
        tenant = Tenant.from_dict(data)

        assert tenant.id == "t2"
        assert tenant.name == "From Dict"
        assert tenant.tier == TenantTier.STARTER
        assert tenant.quota.requests_per_minute == 30
        assert tenant.usage.total_requests == 5


class TestTenantManager:
    """Test TenantManager class."""

    def test_manager_creation(self):
        """Test manager creation without persistence."""
        manager = TenantManager()
        assert len(manager._tenants) == 0

    def test_manager_with_db(self, tmp_path):
        """Test manager with SQLite persistence."""
        db_path = str(tmp_path / "tenants.db")
        manager = TenantManager(db_path=db_path)
        assert (tmp_path / "tenants.db").exists()

    def test_create_tenant(self):
        """Test creating a tenant."""
        manager = TenantManager()
        tenant, api_key = manager.create_tenant("Test Tenant")

        assert tenant.name == "Test Tenant"
        assert tenant.id is not None
        assert api_key.startswith("tf_")
        assert len(api_key) > 20

    def test_create_tenant_with_tier(self):
        """Test creating tenant with specific tier."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Premium", tier=TenantTier.PROFESSIONAL)

        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.quota.requests_per_minute == 100

    def test_get_tenant_by_id(self):
        """Test getting tenant by ID."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")

        retrieved = manager.get_tenant(tenant.id)
        assert retrieved is not None
        assert retrieved.id == tenant.id

    def test_get_tenant_by_api_key(self):
        """Test getting tenant by API key."""
        manager = TenantManager()
        tenant, api_key = manager.create_tenant("Test")

        retrieved = manager.get_tenant_by_api_key(api_key)
        assert retrieved is not None
        assert retrieved.id == tenant.id

    def test_get_tenant_invalid_api_key(self):
        """Test getting tenant with invalid API key."""
        manager = TenantManager()
        manager.create_tenant("Test")

        retrieved = manager.get_tenant_by_api_key("invalid_key")
        assert retrieved is None

    def test_list_tenants(self):
        """Test listing tenants."""
        manager = TenantManager()
        manager.create_tenant("Tenant 1")
        manager.create_tenant("Tenant 2")
        manager.create_tenant("Tenant 3")

        tenants = manager.list_tenants()
        assert len(tenants) == 3

    def test_list_tenants_by_status(self):
        """Test listing tenants filtered by status."""
        manager = TenantManager()
        t1, _ = manager.create_tenant("Active 1")
        t2, _ = manager.create_tenant("Active 2")
        t3, _ = manager.create_tenant("Suspended")
        manager.suspend_tenant(t3.id)

        active = manager.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active) == 2

        suspended = manager.list_tenants(status=TenantStatus.SUSPENDED)
        assert len(suspended) == 1

    def test_list_tenants_by_tier(self):
        """Test listing tenants filtered by tier."""
        manager = TenantManager()
        manager.create_tenant("Free", tier=TenantTier.FREE)
        manager.create_tenant("Pro", tier=TenantTier.PROFESSIONAL)

        free_tenants = manager.list_tenants(tier=TenantTier.FREE)
        assert len(free_tenants) == 1

    def test_update_tenant(self):
        """Test updating tenant."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Original")

        updated = manager.update_tenant(tenant.id, name="Updated")
        assert updated.name == "Updated"

    def test_update_tenant_tier(self):
        """Test updating tenant tier."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test", tier=TenantTier.FREE)

        updated = manager.update_tenant(tenant.id, tier=TenantTier.PROFESSIONAL)
        assert updated.tier == TenantTier.PROFESSIONAL
        assert updated.quota.requests_per_minute == 100

    def test_suspend_tenant(self):
        """Test suspending tenant."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")

        suspended = manager.suspend_tenant(tenant.id)
        assert suspended.status == TenantStatus.SUSPENDED

    def test_activate_tenant(self):
        """Test activating tenant."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")
        manager.suspend_tenant(tenant.id)

        activated = manager.activate_tenant(tenant.id)
        assert activated.status == TenantStatus.ACTIVE

    def test_rotate_api_key(self):
        """Test rotating API key."""
        manager = TenantManager()
        tenant, old_key = manager.create_tenant("Test")

        new_key = manager.rotate_api_key(tenant.id)
        assert new_key != old_key

        # Old key should not work
        assert manager.get_tenant_by_api_key(old_key) is None

        # New key should work
        assert manager.get_tenant_by_api_key(new_key) is not None

    def test_delete_tenant_soft(self):
        """Test soft deleting tenant."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")

        result = manager.delete_tenant(tenant.id, hard_delete=False)
        assert result is True

        # Tenant still exists but marked as deleted
        t = manager.get_tenant(tenant.id)
        assert t.status == TenantStatus.DELETED

    def test_delete_tenant_hard(self):
        """Test hard deleting tenant."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")

        result = manager.delete_tenant(tenant.id, hard_delete=True)
        assert result is True

        # Tenant should not exist
        assert manager.get_tenant(tenant.id) is None

    def test_record_usage(self):
        """Test recording usage."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")

        manager.record_usage(tenant.id, tokens_in=100, tokens_out=50)

        t = manager.get_tenant(tenant.id)
        assert t.usage.total_requests == 1
        assert t.usage.total_tokens_in == 100

    def test_get_usage_stats(self):
        """Test getting usage stats."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")
        manager.record_usage(tenant.id, tokens_in=100, tokens_out=50)

        stats = manager.get_usage_stats(tenant.id)
        assert stats["tenant_id"] == tenant.id
        assert stats["usage"]["total_requests"] == 1


class TestTenantManagerPersistence:
    """Test TenantManager persistence."""

    def test_persistence_save_and_load(self, tmp_path):
        """Test saving and loading tenants."""
        db_path = str(tmp_path / "tenants.db")

        # Create manager and tenant
        manager1 = TenantManager(db_path=db_path)
        tenant, api_key = manager1.create_tenant("Persistent", tier=TenantTier.STARTER)
        tenant_id = tenant.id

        # Create new manager instance
        manager2 = TenantManager(db_path=db_path)

        # Tenant should be loaded
        loaded = manager2.get_tenant(tenant_id)
        assert loaded is not None
        assert loaded.name == "Persistent"
        assert loaded.tier == TenantTier.STARTER

        # API key should still work
        assert manager2.get_tenant_by_api_key(api_key) is not None


class TestTenantAuth:
    """Test TenantAuth class."""

    def test_verify_tenant_api_key_valid(self):
        """Test verifying valid API key."""
        manager = TenantManager()
        tenant, api_key = manager.create_tenant("Test")

        result = verify_tenant_api_key(manager, api_key)
        assert result is not None
        assert result.id == tenant.id

    def test_verify_tenant_api_key_invalid(self):
        """Test verifying invalid API key."""
        manager = TenantManager()
        manager.create_tenant("Test")

        result = verify_tenant_api_key(manager, "invalid_key")
        assert result is None

    def test_verify_tenant_api_key_suspended(self):
        """Test verifying API key for suspended tenant."""
        manager = TenantManager()
        tenant, api_key = manager.create_tenant("Test")
        manager.suspend_tenant(tenant.id)

        result = verify_tenant_api_key(manager, api_key)
        assert result is None


class TestInferenceAPI:
    """Test inference API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from services.inference.app import app, tenant_manager, tenant_auth
        from services.inference.tenant import TenantManager
        from services.inference.auth import TenantAuth

        # Initialize for testing
        import services.inference.app as app_module
        app_module.tenant_manager = TenantManager()
        app_module.tenant_auth = TenantAuth(app_module.tenant_manager)

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_inference_without_api_key(self, client):
        """Test inference without API key returns 401."""
        response = client.post("/v1/inference", json={"input": "test"})
        assert response.status_code == 401

    def test_inference_with_invalid_api_key(self, client):
        """Test inference with invalid API key returns 401."""
        response = client.post(
            "/v1/inference",
            json={"input": "test"},
            headers={"X-API-Key": "invalid_key"},
        )
        assert response.status_code == 401

    def test_inference_success(self, client):
        """Test successful inference."""
        import services.inference.app as app_module

        # Create tenant
        tenant, api_key = app_module.tenant_manager.create_tenant("Test")

        response = client.post(
            "/v1/inference",
            json={"input": "Hello world"},
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert data["tenant_id"] == tenant.id

    def test_admin_create_tenant(self, client):
        """Test admin create tenant endpoint."""
        response = client.post(
            "/admin/tenants",
            json={"name": "New Tenant", "tier": "starter"},
            headers={"X-Admin-Key": "admin-secret-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Tenant"
        assert data["tier"] == "starter"
        assert "api_key" in data

    def test_admin_create_tenant_no_auth(self, client):
        """Test admin endpoint without auth."""
        response = client.post(
            "/admin/tenants",
            json={"name": "New Tenant"},
        )
        assert response.status_code == 403

    def test_admin_list_tenants(self, client):
        """Test admin list tenants endpoint."""
        import services.inference.app as app_module
        app_module.tenant_manager.create_tenant("Tenant 1")
        app_module.tenant_manager.create_tenant("Tenant 2")

        response = client.get(
            "/admin/tenants",
            headers={"X-Admin-Key": "admin-secret-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

    def test_tenant_self_service_me(self, client):
        """Test tenant /me endpoint."""
        import services.inference.app as app_module
        tenant, api_key = app_module.tenant_manager.create_tenant("Self Service")

        response = client.get(
            "/v1/me",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == tenant.id
        assert data["name"] == "Self Service"

    def test_tenant_usage_tracking(self, client):
        """Test usage is tracked after inference."""
        import services.inference.app as app_module
        tenant, api_key = app_module.tenant_manager.create_tenant("Usage Test")

        # Make a request
        client.post(
            "/v1/inference",
            json={"input": "Test input"},
            headers={"X-API-Key": api_key},
        )

        # Check usage
        response = client.get(
            "/v1/usage",
            headers={"X-API-Key": api_key},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["usage"]["total_requests"] == 1


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_per_minute(self):
        """Test per-minute rate limiting."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant(
            "Rate Limited",
            tier=TenantTier.FREE,  # 10 req/min
        )

        # Override quota for testing
        tenant.quota.requests_per_minute = 3

        # First 3 requests should pass
        for _ in range(3):
            can_request, _ = tenant.can_make_request()
            assert can_request is True
            tenant.record_request()

        # 4th request should fail
        can_request, reason = tenant.can_make_request()
        assert can_request is False
        assert "per minute" in reason

    def test_rate_limit_reset_after_minute(self):
        """Test rate limit resets after a minute."""
        manager = TenantManager()
        tenant, _ = manager.create_tenant("Test")
        tenant.quota.requests_per_minute = 2

        # Use up quota
        tenant.record_request()
        tenant.record_request()

        # Simulate time passing
        past_time = (datetime.utcnow() - timedelta(minutes=2)).isoformat()
        tenant.usage.last_minute_reset = past_time
        tenant.usage.requests_this_minute = 0

        # Should be able to make requests again
        can_request, _ = tenant.can_make_request()
        assert can_request is True
