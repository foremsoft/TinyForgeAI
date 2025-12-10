"""
Webhook system for TinyForgeAI.

Provides functionality to send webhook notifications for various events
such as training job state changes, model deployments, and errors.

Supports:
- Multiple webhook endpoints per event type
- Retry with exponential backoff
- Async notification delivery
- Signature verification for security
- Event filtering
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check for httpx
_HTTPX_AVAILABLE = False
try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    pass


class WebhookEventType(str, Enum):
    """Types of webhook events."""
    # Training events
    TRAINING_STARTED = "training.started"
    TRAINING_PROGRESS = "training.progress"
    TRAINING_COMPLETED = "training.completed"
    TRAINING_FAILED = "training.failed"
    TRAINING_CANCELLED = "training.cancelled"

    # Model events
    MODEL_CREATED = "model.created"
    MODEL_UPDATED = "model.updated"
    MODEL_DELETED = "model.deleted"
    MODEL_DEPLOYED = "model.deployed"

    # Inference events
    INFERENCE_ERROR = "inference.error"
    MODEL_LOADED = "model.loaded"
    MODEL_UNLOADED = "model.unloaded"

    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"


@dataclass
class WebhookEndpoint:
    """Configuration for a webhook endpoint."""
    id: str
    url: str
    secret: Optional[str] = None
    events: List[WebhookEventType] = field(default_factory=list)
    enabled: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30

    def __post_init__(self):
        if not self.events:
            # Subscribe to all events by default
            self.events = list(WebhookEventType)


@dataclass
class WebhookEvent:
    """A webhook event to be delivered."""
    id: str
    type: WebhookEventType
    timestamp: str
    data: Dict[str, Any]
    source: str = "tinyforgeai"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "source": self.source,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    id: str
    endpoint_id: str
    event_id: str
    status: str  # "pending", "success", "failed"
    attempts: int = 0
    last_attempt: Optional[str] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None


class WebhookManager:
    """
    Manages webhook subscriptions and event delivery.

    Handles registration of webhook endpoints, event creation,
    and async delivery with retry logic.
    """

    def __init__(self):
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._event_handlers: Dict[WebhookEventType, List[Callable]] = {}
        self._client: Optional[httpx.AsyncClient] = None
        logger.info("WebhookManager initialized")

    async def __aenter__(self) -> "WebhookManager":
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _get_client(self) -> Optional[httpx.AsyncClient]:
        """Get or create HTTP client."""
        if not _HTTPX_AVAILABLE:
            logger.warning("httpx not available, webhooks will not be delivered")
            return None

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def register_endpoint(self, endpoint: WebhookEndpoint) -> str:
        """
        Register a webhook endpoint.

        Args:
            endpoint: WebhookEndpoint configuration.

        Returns:
            The endpoint ID.
        """
        self._endpoints[endpoint.id] = endpoint
        logger.info(f"Registered webhook endpoint: {endpoint.id} -> {endpoint.url}")
        return endpoint.id

    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """
        Unregister a webhook endpoint.

        Args:
            endpoint_id: ID of endpoint to remove.

        Returns:
            True if removed, False if not found.
        """
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            logger.info(f"Unregistered webhook endpoint: {endpoint_id}")
            return True
        return False

    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all registered endpoints."""
        return list(self._endpoints.values())

    def add_handler(
        self,
        event_type: WebhookEventType,
        handler: Callable[[WebhookEvent], Any],
    ) -> None:
        """
        Add a local handler for an event type.

        Handlers are called synchronously when events are created,
        in addition to webhook delivery.

        Args:
            event_type: Type of event to handle.
            handler: Callable that receives the event.
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate HMAC-SHA256 signature for webhook payload.

        Args:
            payload: JSON payload string.
            secret: Webhook secret.

        Returns:
            Hex-encoded signature.
        """
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    def create_event(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
    ) -> WebhookEvent:
        """
        Create a webhook event.

        Args:
            event_type: Type of event.
            data: Event data payload.

        Returns:
            The created WebhookEvent.
        """
        event = WebhookEvent(
            id=str(uuid.uuid4()),
            type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            data=data,
        )

        # Call local handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

        return event

    async def emit(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
        wait: bool = False,
    ) -> WebhookEvent:
        """
        Emit a webhook event to all subscribed endpoints.

        Args:
            event_type: Type of event.
            data: Event data payload.
            wait: If True, wait for all deliveries to complete.

        Returns:
            The emitted WebhookEvent.
        """
        event = self.create_event(event_type, data)

        # Find subscribed endpoints
        subscribed = [
            ep for ep in self._endpoints.values()
            if ep.enabled and event_type in ep.events
        ]

        if not subscribed:
            logger.debug(f"No endpoints subscribed to {event_type}")
            return event

        logger.info(f"Emitting {event_type} to {len(subscribed)} endpoints")

        # Create delivery tasks
        tasks = [
            self._deliver(endpoint, event)
            for endpoint in subscribed
        ]

        if wait:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Fire and forget
            for task in tasks:
                asyncio.create_task(task)

        return event

    async def _deliver(
        self,
        endpoint: WebhookEndpoint,
        event: WebhookEvent,
    ) -> WebhookDelivery:
        """
        Deliver an event to an endpoint with retries.

        Args:
            endpoint: Target endpoint.
            event: Event to deliver.

        Returns:
            WebhookDelivery record.
        """
        delivery = WebhookDelivery(
            id=str(uuid.uuid4()),
            endpoint_id=endpoint.id,
            event_id=event.id,
            status="pending",
        )
        self._deliveries[delivery.id] = delivery

        client = await self._get_client()
        if client is None:
            delivery.status = "failed"
            delivery.error = "HTTP client not available"
            return delivery

        payload = event.to_json()

        # Build headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TinyForgeAI-Webhook/1.0",
            "X-Webhook-Event": event.type.value,
            "X-Webhook-ID": event.id,
            "X-Webhook-Timestamp": event.timestamp,
            **endpoint.headers,
        }

        # Add signature if secret is configured
        if endpoint.secret:
            signature = self._generate_signature(payload, endpoint.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        # Retry loop
        for attempt in range(endpoint.max_retries):
            delivery.attempts = attempt + 1
            delivery.last_attempt = datetime.utcnow().isoformat()

            try:
                response = await client.post(
                    endpoint.url,
                    content=payload,
                    headers=headers,
                    timeout=endpoint.timeout,
                )

                delivery.response_code = response.status_code
                delivery.response_body = response.text[:1000]  # Truncate

                if 200 <= response.status_code < 300:
                    delivery.status = "success"
                    logger.debug(
                        f"Webhook delivered: {event.type} -> {endpoint.url} "
                        f"(status={response.status_code})"
                    )
                    return delivery
                else:
                    logger.warning(
                        f"Webhook failed: {endpoint.url} returned {response.status_code}"
                    )

            except httpx.TimeoutException as e:
                delivery.error = f"Timeout: {e}"
                logger.warning(f"Webhook timeout: {endpoint.url}")
            except httpx.ConnectError as e:
                delivery.error = f"Connection error: {e}"
                logger.warning(f"Webhook connection error: {endpoint.url}")
            except Exception as e:
                delivery.error = str(e)
                logger.error(f"Webhook error: {endpoint.url} - {e}")

            # Wait before retry
            if attempt < endpoint.max_retries - 1:
                delay = endpoint.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)

        delivery.status = "failed"
        logger.error(
            f"Webhook delivery failed after {endpoint.max_retries} attempts: "
            f"{event.type} -> {endpoint.url}"
        )
        return delivery

    def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery record by ID."""
        return self._deliveries.get(delivery_id)

    def list_deliveries(
        self,
        endpoint_id: Optional[str] = None,
        event_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """
        List delivery records with optional filters.

        Args:
            endpoint_id: Filter by endpoint.
            event_id: Filter by event.
            status: Filter by status.
            limit: Maximum results.

        Returns:
            List of matching deliveries.
        """
        deliveries = list(self._deliveries.values())

        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]
        if event_id:
            deliveries = [d for d in deliveries if d.event_id == event_id]
        if status:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by most recent first
        deliveries.sort(key=lambda d: d.last_attempt or "", reverse=True)

        return deliveries[:limit]


# Global webhook manager instance
_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


# Convenience functions for emitting common events

async def emit_training_started(
    job_id: str,
    model_name: str,
    config: Dict[str, Any],
) -> WebhookEvent:
    """Emit training started event."""
    manager = get_webhook_manager()
    return await manager.emit(
        WebhookEventType.TRAINING_STARTED,
        {
            "job_id": job_id,
            "model_name": model_name,
            "config": config,
        },
    )


async def emit_training_progress(
    job_id: str,
    progress: float,
    metrics: Dict[str, Any],
) -> WebhookEvent:
    """Emit training progress event."""
    manager = get_webhook_manager()
    return await manager.emit(
        WebhookEventType.TRAINING_PROGRESS,
        {
            "job_id": job_id,
            "progress": progress,
            "metrics": metrics,
        },
    )


async def emit_training_completed(
    job_id: str,
    model_name: str,
    metrics: Dict[str, Any],
    output_path: Optional[str] = None,
) -> WebhookEvent:
    """Emit training completed event."""
    manager = get_webhook_manager()
    return await manager.emit(
        WebhookEventType.TRAINING_COMPLETED,
        {
            "job_id": job_id,
            "model_name": model_name,
            "metrics": metrics,
            "output_path": output_path,
        },
    )


async def emit_training_failed(
    job_id: str,
    error: str,
    details: Optional[Dict[str, Any]] = None,
) -> WebhookEvent:
    """Emit training failed event."""
    manager = get_webhook_manager()
    return await manager.emit(
        WebhookEventType.TRAINING_FAILED,
        {
            "job_id": job_id,
            "error": error,
            "details": details or {},
        },
    )


async def emit_model_deployed(
    model_name: str,
    version: str,
    endpoint: Optional[str] = None,
) -> WebhookEvent:
    """Emit model deployed event."""
    manager = get_webhook_manager()
    return await manager.emit(
        WebhookEventType.MODEL_DEPLOYED,
        {
            "model_name": model_name,
            "version": version,
            "endpoint": endpoint,
        },
    )
