"""Tests for webhook system."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.webhooks import (
    WebhookEventType,
    WebhookEndpoint,
    WebhookEvent,
    WebhookDelivery,
    WebhookManager,
    get_webhook_manager,
    emit_training_started,
    emit_training_progress,
    emit_training_completed,
    emit_training_failed,
    emit_model_deployed,
)


class TestWebhookEventType:
    """Tests for WebhookEventType enum."""

    def test_training_events(self):
        """Test training event types exist."""
        assert WebhookEventType.TRAINING_STARTED == "training.started"
        assert WebhookEventType.TRAINING_PROGRESS == "training.progress"
        assert WebhookEventType.TRAINING_COMPLETED == "training.completed"
        assert WebhookEventType.TRAINING_FAILED == "training.failed"
        assert WebhookEventType.TRAINING_CANCELLED == "training.cancelled"

    def test_model_events(self):
        """Test model event types exist."""
        assert WebhookEventType.MODEL_CREATED == "model.created"
        assert WebhookEventType.MODEL_UPDATED == "model.updated"
        assert WebhookEventType.MODEL_DELETED == "model.deleted"
        assert WebhookEventType.MODEL_DEPLOYED == "model.deployed"

    def test_inference_events(self):
        """Test inference event types exist."""
        assert WebhookEventType.INFERENCE_ERROR == "inference.error"
        assert WebhookEventType.MODEL_LOADED == "model.loaded"
        assert WebhookEventType.MODEL_UNLOADED == "model.unloaded"

    def test_system_events(self):
        """Test system event types exist."""
        assert WebhookEventType.SYSTEM_ERROR == "system.error"
        assert WebhookEventType.SYSTEM_WARNING == "system.warning"


class TestWebhookEndpoint:
    """Tests for WebhookEndpoint dataclass."""

    def test_basic_creation(self):
        """Test basic endpoint creation."""
        endpoint = WebhookEndpoint(
            id="ep-1",
            url="https://example.com/webhook",
        )
        assert endpoint.id == "ep-1"
        assert endpoint.url == "https://example.com/webhook"
        assert endpoint.enabled is True
        assert endpoint.secret is None

    def test_with_secret(self):
        """Test endpoint with secret."""
        endpoint = WebhookEndpoint(
            id="ep-2",
            url="https://example.com/webhook",
            secret="my-secret-key",
        )
        assert endpoint.secret == "my-secret-key"

    def test_with_events(self):
        """Test endpoint with specific events."""
        endpoint = WebhookEndpoint(
            id="ep-3",
            url="https://example.com/webhook",
            events=[WebhookEventType.TRAINING_COMPLETED],
        )
        assert len(endpoint.events) == 1
        assert WebhookEventType.TRAINING_COMPLETED in endpoint.events

    def test_default_events(self):
        """Test endpoint subscribes to all events by default."""
        endpoint = WebhookEndpoint(
            id="ep-4",
            url="https://example.com/webhook",
        )
        # Should have all event types
        assert len(endpoint.events) == len(WebhookEventType)

    def test_with_headers(self):
        """Test endpoint with custom headers."""
        headers = {"X-Custom": "value"}
        endpoint = WebhookEndpoint(
            id="ep-5",
            url="https://example.com/webhook",
            headers=headers,
        )
        assert endpoint.headers == headers

    def test_retry_settings(self):
        """Test endpoint retry settings."""
        endpoint = WebhookEndpoint(
            id="ep-6",
            url="https://example.com/webhook",
            max_retries=5,
            retry_delay=2.0,
            timeout=60,
        )
        assert endpoint.max_retries == 5
        assert endpoint.retry_delay == 2.0
        assert endpoint.timeout == 60


class TestWebhookEvent:
    """Tests for WebhookEvent dataclass."""

    def test_event_creation(self):
        """Test event creation."""
        event = WebhookEvent(
            id="evt-1",
            type=WebhookEventType.TRAINING_STARTED,
            timestamp="2024-01-01T00:00:00Z",
            data={"job_id": "job-1"},
        )
        assert event.id == "evt-1"
        assert event.type == WebhookEventType.TRAINING_STARTED
        assert event.data["job_id"] == "job-1"
        assert event.source == "tinyforgeai"

    def test_to_dict(self):
        """Test event to_dict method."""
        event = WebhookEvent(
            id="evt-2",
            type=WebhookEventType.TRAINING_COMPLETED,
            timestamp="2024-01-01T00:00:00Z",
            data={"metrics": {"loss": 0.5}},
        )
        d = event.to_dict()
        assert d["id"] == "evt-2"
        assert d["type"] == "training.completed"
        assert d["data"]["metrics"]["loss"] == 0.5

    def test_to_json(self):
        """Test event to_json method."""
        event = WebhookEvent(
            id="evt-3",
            type=WebhookEventType.MODEL_DEPLOYED,
            timestamp="2024-01-01T00:00:00Z",
            data={"model": "test"},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "evt-3"
        assert parsed["type"] == "model.deployed"


class TestWebhookDelivery:
    """Tests for WebhookDelivery dataclass."""

    def test_delivery_creation(self):
        """Test delivery creation."""
        delivery = WebhookDelivery(
            id="del-1",
            endpoint_id="ep-1",
            event_id="evt-1",
            status="pending",
        )
        assert delivery.id == "del-1"
        assert delivery.status == "pending"
        assert delivery.attempts == 0

    def test_delivery_success(self):
        """Test successful delivery."""
        delivery = WebhookDelivery(
            id="del-2",
            endpoint_id="ep-1",
            event_id="evt-1",
            status="success",
            attempts=1,
            response_code=200,
        )
        assert delivery.status == "success"
        assert delivery.response_code == 200

    def test_delivery_failed(self):
        """Test failed delivery."""
        delivery = WebhookDelivery(
            id="del-3",
            endpoint_id="ep-1",
            event_id="evt-1",
            status="failed",
            attempts=3,
            error="Connection refused",
        )
        assert delivery.status == "failed"
        assert delivery.error == "Connection refused"


class TestWebhookManager:
    """Tests for WebhookManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh webhook manager."""
        return WebhookManager()

    def test_register_endpoint(self, manager):
        """Test registering an endpoint."""
        endpoint = WebhookEndpoint(
            id="ep-1",
            url="https://example.com/webhook",
        )
        endpoint_id = manager.register_endpoint(endpoint)
        assert endpoint_id == "ep-1"
        assert manager.get_endpoint("ep-1") is not None

    def test_unregister_endpoint(self, manager):
        """Test unregistering an endpoint."""
        endpoint = WebhookEndpoint(
            id="ep-1",
            url="https://example.com/webhook",
        )
        manager.register_endpoint(endpoint)
        assert manager.unregister_endpoint("ep-1") is True
        assert manager.get_endpoint("ep-1") is None

    def test_unregister_nonexistent(self, manager):
        """Test unregistering nonexistent endpoint."""
        assert manager.unregister_endpoint("nonexistent") is False

    def test_list_endpoints(self, manager):
        """Test listing endpoints."""
        for i in range(3):
            manager.register_endpoint(WebhookEndpoint(
                id=f"ep-{i}",
                url=f"https://example.com/webhook{i}",
            ))
        endpoints = manager.list_endpoints()
        assert len(endpoints) == 3

    def test_add_handler(self, manager):
        """Test adding local event handler."""
        events_received = []

        def handler(event):
            events_received.append(event)

        manager.add_handler(WebhookEventType.TRAINING_STARTED, handler)
        event = manager.create_event(
            WebhookEventType.TRAINING_STARTED,
            {"job_id": "test"},
        )
        assert len(events_received) == 1
        assert events_received[0].type == WebhookEventType.TRAINING_STARTED

    def test_create_event(self, manager):
        """Test creating an event."""
        event = manager.create_event(
            WebhookEventType.MODEL_DEPLOYED,
            {"model": "test-model", "version": "v1"},
        )
        assert event.type == WebhookEventType.MODEL_DEPLOYED
        assert event.data["model"] == "test-model"
        assert event.id is not None
        assert event.timestamp is not None

    def test_generate_signature(self, manager):
        """Test signature generation."""
        payload = '{"test": "data"}'
        secret = "my-secret"
        sig = manager._generate_signature(payload, secret)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex digest

    def test_signature_consistency(self, manager):
        """Test signature is consistent for same input."""
        payload = '{"test": "data"}'
        secret = "my-secret"
        sig1 = manager._generate_signature(payload, secret)
        sig2 = manager._generate_signature(payload, secret)
        assert sig1 == sig2

    def test_signature_differs_for_different_secret(self, manager):
        """Test signature differs with different secret."""
        payload = '{"test": "data"}'
        sig1 = manager._generate_signature(payload, "secret1")
        sig2 = manager._generate_signature(payload, "secret2")
        assert sig1 != sig2

    @pytest.mark.asyncio
    async def test_emit_no_subscribers(self, manager):
        """Test emit with no subscribers."""
        event = await manager.emit(
            WebhookEventType.TRAINING_STARTED,
            {"job_id": "test"},
        )
        assert event is not None
        assert event.type == WebhookEventType.TRAINING_STARTED

    def test_list_deliveries(self, manager):
        """Test listing deliveries."""
        # Add some mock deliveries
        manager._deliveries["d1"] = WebhookDelivery(
            id="d1", endpoint_id="ep-1", event_id="e1", status="success"
        )
        manager._deliveries["d2"] = WebhookDelivery(
            id="d2", endpoint_id="ep-1", event_id="e2", status="failed"
        )
        manager._deliveries["d3"] = WebhookDelivery(
            id="d3", endpoint_id="ep-2", event_id="e3", status="success"
        )

        # Filter by endpoint
        deliveries = manager.list_deliveries(endpoint_id="ep-1")
        assert len(deliveries) == 2

        # Filter by status
        deliveries = manager.list_deliveries(status="success")
        assert len(deliveries) == 2

    def test_get_delivery(self, manager):
        """Test getting a delivery."""
        delivery = WebhookDelivery(
            id="d1", endpoint_id="ep-1", event_id="e1", status="success"
        )
        manager._deliveries["d1"] = delivery
        assert manager.get_delivery("d1") == delivery
        assert manager.get_delivery("nonexistent") is None


class TestWebhookManagerAsync:
    """Async tests for WebhookManager."""

    @pytest.fixture
    def manager(self):
        """Create a fresh webhook manager."""
        return WebhookManager()

    @pytest.mark.asyncio
    async def test_emit_with_subscriber(self, manager):
        """Test emit with a subscriber (mocked HTTP)."""
        endpoint = WebhookEndpoint(
            id="ep-1",
            url="https://example.com/webhook",
            events=[WebhookEventType.TRAINING_COMPLETED],
        )
        manager.register_endpoint(endpoint)

        # Mock httpx
        with patch("backend.webhooks._HTTPX_AVAILABLE", True):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "OK"
            mock_response.headers = {}

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.is_closed = False

            manager._client = mock_client

            event = await manager.emit(
                WebhookEventType.TRAINING_COMPLETED,
                {"job_id": "test", "metrics": {"loss": 0.1}},
                wait=True,
            )

            assert event is not None
            # Client should have been called
            assert mock_client.post.called

    @pytest.mark.asyncio
    async def test_context_manager(self, manager):
        """Test async context manager."""
        with patch("backend.webhooks._HTTPX_AVAILABLE", True):
            with patch("backend.webhooks.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.is_closed = False
                mock_httpx.AsyncClient.return_value = mock_client
                mock_httpx.Limits = MagicMock()

                async with manager:
                    pass

                # Client should be closed after context
                mock_client.aclose.assert_called()


class TestConvenienceFunctions:
    """Tests for convenience emission functions."""

    @pytest.mark.asyncio
    async def test_emit_training_started(self):
        """Test emit_training_started function."""
        with patch("backend.webhooks.get_webhook_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.emit = AsyncMock(return_value=WebhookEvent(
                id="e1",
                type=WebhookEventType.TRAINING_STARTED,
                timestamp="2024-01-01T00:00:00Z",
                data={},
            ))
            mock_get.return_value = mock_manager

            event = await emit_training_started(
                job_id="job-1",
                model_name="test-model",
                config={"epochs": 3},
            )

            mock_manager.emit.assert_called_once()
            call_args = mock_manager.emit.call_args
            assert call_args[0][0] == WebhookEventType.TRAINING_STARTED

    @pytest.mark.asyncio
    async def test_emit_training_progress(self):
        """Test emit_training_progress function."""
        with patch("backend.webhooks.get_webhook_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.emit = AsyncMock(return_value=WebhookEvent(
                id="e1",
                type=WebhookEventType.TRAINING_PROGRESS,
                timestamp="2024-01-01T00:00:00Z",
                data={},
            ))
            mock_get.return_value = mock_manager

            event = await emit_training_progress(
                job_id="job-1",
                progress=50.0,
                metrics={"loss": 0.5},
            )

            mock_manager.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_training_completed(self):
        """Test emit_training_completed function."""
        with patch("backend.webhooks.get_webhook_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.emit = AsyncMock(return_value=WebhookEvent(
                id="e1",
                type=WebhookEventType.TRAINING_COMPLETED,
                timestamp="2024-01-01T00:00:00Z",
                data={},
            ))
            mock_get.return_value = mock_manager

            event = await emit_training_completed(
                job_id="job-1",
                model_name="test-model",
                metrics={"final_loss": 0.1},
                output_path="/models/test",
            )

            mock_manager.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_training_failed(self):
        """Test emit_training_failed function."""
        with patch("backend.webhooks.get_webhook_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.emit = AsyncMock(return_value=WebhookEvent(
                id="e1",
                type=WebhookEventType.TRAINING_FAILED,
                timestamp="2024-01-01T00:00:00Z",
                data={},
            ))
            mock_get.return_value = mock_manager

            event = await emit_training_failed(
                job_id="job-1",
                error="Out of memory",
                details={"gpu": 0},
            )

            mock_manager.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_model_deployed(self):
        """Test emit_model_deployed function."""
        with patch("backend.webhooks.get_webhook_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.emit = AsyncMock(return_value=WebhookEvent(
                id="e1",
                type=WebhookEventType.MODEL_DEPLOYED,
                timestamp="2024-01-01T00:00:00Z",
                data={},
            ))
            mock_get.return_value = mock_manager

            event = await emit_model_deployed(
                model_name="test-model",
                version="v1",
                endpoint="/inference/test",
            )

            mock_manager.emit.assert_called_once()


class TestGlobalManager:
    """Tests for global webhook manager."""

    def test_get_webhook_manager_singleton(self):
        """Test that get_webhook_manager returns singleton."""
        # Reset global
        import backend.webhooks as webhooks_module
        webhooks_module._webhook_manager = None

        manager1 = get_webhook_manager()
        manager2 = get_webhook_manager()
        assert manager1 is manager2
