"""
Webhook management API routes for TinyForgeAI.

Provides endpoints for managing webhook subscriptions and viewing delivery history.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from backend.webhooks import (
    WebhookEndpoint,
    WebhookEventType,
    get_webhook_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# =============================================================================
# Request/Response Models
# =============================================================================


class WebhookEndpointCreate(BaseModel):
    """Request to create a webhook endpoint."""
    url: HttpUrl = Field(..., description="URL to send webhook events to")
    secret: Optional[str] = Field(None, description="Secret for HMAC signature verification")
    events: Optional[List[str]] = Field(
        None,
        description="List of event types to subscribe to (defaults to all)",
    )
    headers: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Custom headers to include in webhook requests",
    )
    enabled: bool = Field(default=True, description="Whether the endpoint is active")


class WebhookEndpointUpdate(BaseModel):
    """Request to update a webhook endpoint."""
    url: Optional[HttpUrl] = None
    secret: Optional[str] = None
    events: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None


class WebhookEndpointResponse(BaseModel):
    """Webhook endpoint details."""
    id: str
    url: str
    events: List[str]
    enabled: bool
    headers: Dict[str, str]
    has_secret: bool


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery record."""
    id: str
    endpoint_id: str
    event_id: str
    status: str
    attempts: int
    last_attempt: Optional[str]
    response_code: Optional[int]
    error: Optional[str]


class WebhookEventTypeInfo(BaseModel):
    """Information about a webhook event type."""
    type: str
    description: str


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/events", response_model=List[WebhookEventTypeInfo])
async def list_event_types():
    """List all available webhook event types."""
    event_descriptions = {
        WebhookEventType.TRAINING_STARTED: "Emitted when a training job starts",
        WebhookEventType.TRAINING_PROGRESS: "Emitted periodically during training with progress updates",
        WebhookEventType.TRAINING_COMPLETED: "Emitted when a training job completes successfully",
        WebhookEventType.TRAINING_FAILED: "Emitted when a training job fails",
        WebhookEventType.TRAINING_CANCELLED: "Emitted when a training job is cancelled",
        WebhookEventType.MODEL_CREATED: "Emitted when a new model is registered",
        WebhookEventType.MODEL_UPDATED: "Emitted when model metadata is updated",
        WebhookEventType.MODEL_DELETED: "Emitted when a model is deleted",
        WebhookEventType.MODEL_DEPLOYED: "Emitted when a model is deployed to production",
        WebhookEventType.INFERENCE_ERROR: "Emitted when an inference request fails",
        WebhookEventType.MODEL_LOADED: "Emitted when a model is loaded for inference",
        WebhookEventType.MODEL_UNLOADED: "Emitted when a model is unloaded",
        WebhookEventType.SYSTEM_ERROR: "Emitted for critical system errors",
        WebhookEventType.SYSTEM_WARNING: "Emitted for system warnings",
    }

    return [
        WebhookEventTypeInfo(type=et.value, description=event_descriptions.get(et, ""))
        for et in WebhookEventType
    ]


@router.post("", response_model=WebhookEndpointResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(request: WebhookEndpointCreate):
    """
    Create a new webhook endpoint subscription.

    The webhook will receive POST requests for subscribed events with:
    - X-Webhook-Event: Event type
    - X-Webhook-ID: Unique event ID
    - X-Webhook-Timestamp: ISO timestamp
    - X-Webhook-Signature: HMAC-SHA256 signature (if secret configured)
    """
    manager = get_webhook_manager()

    # Parse event types
    events = []
    if request.events:
        for event_str in request.events:
            try:
                events.append(WebhookEventType(event_str))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type: {event_str}",
                )
    else:
        events = list(WebhookEventType)

    endpoint = WebhookEndpoint(
        id=str(uuid.uuid4()),
        url=str(request.url),
        secret=request.secret,
        events=events,
        headers=request.headers or {},
        enabled=request.enabled,
    )

    manager.register_endpoint(endpoint)
    logger.info(f"Created webhook endpoint: {endpoint.id}")

    return WebhookEndpointResponse(
        id=endpoint.id,
        url=endpoint.url,
        events=[e.value for e in endpoint.events],
        enabled=endpoint.enabled,
        headers=endpoint.headers,
        has_secret=endpoint.secret is not None,
    )


@router.get("", response_model=List[WebhookEndpointResponse])
async def list_webhooks():
    """List all registered webhook endpoints."""
    manager = get_webhook_manager()
    endpoints = manager.list_endpoints()

    return [
        WebhookEndpointResponse(
            id=ep.id,
            url=ep.url,
            events=[e.value for e in ep.events],
            enabled=ep.enabled,
            headers=ep.headers,
            has_secret=ep.secret is not None,
        )
        for ep in endpoints
    ]


@router.get("/{webhook_id}", response_model=WebhookEndpointResponse)
async def get_webhook(webhook_id: str):
    """Get details of a specific webhook endpoint."""
    manager = get_webhook_manager()
    endpoint = manager.get_endpoint(webhook_id)

    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    return WebhookEndpointResponse(
        id=endpoint.id,
        url=endpoint.url,
        events=[e.value for e in endpoint.events],
        enabled=endpoint.enabled,
        headers=endpoint.headers,
        has_secret=endpoint.secret is not None,
    )


@router.patch("/{webhook_id}", response_model=WebhookEndpointResponse)
async def update_webhook(webhook_id: str, request: WebhookEndpointUpdate):
    """Update a webhook endpoint."""
    manager = get_webhook_manager()
    endpoint = manager.get_endpoint(webhook_id)

    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    # Update fields
    if request.url is not None:
        endpoint.url = str(request.url)
    if request.secret is not None:
        endpoint.secret = request.secret
    if request.events is not None:
        events = []
        for event_str in request.events:
            try:
                events.append(WebhookEventType(event_str))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid event type: {event_str}",
                )
        endpoint.events = events
    if request.headers is not None:
        endpoint.headers = request.headers
    if request.enabled is not None:
        endpoint.enabled = request.enabled

    logger.info(f"Updated webhook endpoint: {webhook_id}")

    return WebhookEndpointResponse(
        id=endpoint.id,
        url=endpoint.url,
        events=[e.value for e in endpoint.events],
        enabled=endpoint.enabled,
        headers=endpoint.headers,
        has_secret=endpoint.secret is not None,
    )


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str):
    """Delete a webhook endpoint."""
    manager = get_webhook_manager()

    if not manager.unregister_endpoint(webhook_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    logger.info(f"Deleted webhook endpoint: {webhook_id}")
    return {"message": f"Webhook {webhook_id} deleted"}


@router.get("/{webhook_id}/deliveries", response_model=List[WebhookDeliveryResponse])
async def list_webhook_deliveries(
    webhook_id: str,
    status_filter: Optional[str] = None,
    limit: int = 50,
):
    """List delivery history for a webhook endpoint."""
    manager = get_webhook_manager()
    endpoint = manager.get_endpoint(webhook_id)

    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    deliveries = manager.list_deliveries(
        endpoint_id=webhook_id,
        status=status_filter,
        limit=limit,
    )

    return [
        WebhookDeliveryResponse(
            id=d.id,
            endpoint_id=d.endpoint_id,
            event_id=d.event_id,
            status=d.status,
            attempts=d.attempts,
            last_attempt=d.last_attempt,
            response_code=d.response_code,
            error=d.error,
        )
        for d in deliveries
    ]


@router.post("/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    """
    Send a test event to a webhook endpoint.

    Sends a test ping event to verify the webhook is working.
    """
    manager = get_webhook_manager()
    endpoint = manager.get_endpoint(webhook_id)

    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    # Create and emit a test event
    event = await manager.emit(
        WebhookEventType.SYSTEM_WARNING,
        {"message": "This is a test webhook event", "webhook_id": webhook_id},
        wait=True,
    )

    # Get delivery result
    deliveries = manager.list_deliveries(
        endpoint_id=webhook_id,
        event_id=event.id,
        limit=1,
    )

    if deliveries:
        delivery = deliveries[0]
        return {
            "success": delivery.status == "success",
            "event_id": event.id,
            "delivery_id": delivery.id,
            "status": delivery.status,
            "response_code": delivery.response_code,
            "error": delivery.error,
        }

    return {
        "success": False,
        "event_id": event.id,
        "error": "No delivery record found",
    }
