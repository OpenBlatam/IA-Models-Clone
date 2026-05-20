"""
Webhooks API Functions
Convenience functions for webhook operations with type hints and error handling
"""

import logging
from typing import Optional, List, Dict, Any
from ..models import WebhookEvent, WebhookEndpoint, WebhookDelivery
from ..factory import get_default_webhook_manager

logger = logging.getLogger(__name__)


async def send_webhook(
    event: WebhookEvent,
    data: Dict[str, Any],
    request_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send webhook for an event - returns status information.
    
    Args:
        event: Webhook event type
        data: Payload data dictionary
        request_id: Request ID for tracking (optional)
        user_id: User ID for context (optional)
    
    Returns:
        Status information dictionary with delivery status
    
    Raises:
        ValueError: If event or data is invalid
        RuntimeError: If webhook manager is unavailable
    """
    if not event:
        raise ValueError("Webhook event is required")
    
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    
    try:
        manager = get_default_webhook_manager()
        return await manager.send_webhook(event, data, request_id, user_id)
    except Exception as e:
        logger.error(f"Failed to send webhook for event {event}: {e}", exc_info=True)
        raise


def register_webhook_endpoint(endpoint: WebhookEndpoint) -> None:
    """
    Register a webhook endpoint.
    
    Args:
        endpoint: Webhook endpoint configuration
    """
    manager = get_default_webhook_manager()
    manager.register_endpoint_sync(endpoint)


def unregister_webhook_endpoint(endpoint_id: str) -> bool:
    """
    Unregister a webhook endpoint.
    
    Args:
        endpoint_id: Endpoint identifier
    
    Returns:
        True if unregistered successfully
    """
    manager = get_default_webhook_manager()
    return manager.unregister_endpoint(endpoint_id)


def get_webhook_endpoints() -> List[WebhookEndpoint]:
    """
    Get all registered webhook endpoints.
    
    Returns:
        List of webhook endpoints
    """
    manager = get_default_webhook_manager()
    return manager.get_endpoints()


def get_webhook_deliveries(limit: int = 100) -> List[WebhookDelivery]:
    """
    Get webhook delivery history.
    
    Args:
        limit: Maximum number of deliveries to return
    
    Returns:
        List of webhook deliveries
    """
    manager = get_default_webhook_manager()
    return manager.get_deliveries(limit)


def get_webhook_stats() -> dict:
    """
    Get webhook system statistics.
    
    Returns:
        Statistics dictionary
    """
    manager = get_default_webhook_manager()
    return manager.get_delivery_stats()


async def get_webhook_health() -> dict:
    """
    Get webhook system health status.
    
    Returns:
        Health status dictionary
    """
    manager = get_default_webhook_manager()
    return await manager.get_health()


def get_rate_limit_status(endpoint_id: str) -> dict:
    """
    Get rate limit status for an endpoint.
    
    Args:
        endpoint_id: Endpoint identifier
    
    Returns:
        Rate limit status dictionary
    """
    manager = get_default_webhook_manager()
    return manager.get_rate_limit_status(endpoint_id)


def configure_rate_limit(
    endpoint_id: str,
    max_requests: int,
    window_seconds: int = 60,
    burst_allowance: int = 10
) -> None:
    """
    Configure rate limit for an endpoint.
    
    Args:
        endpoint_id: Endpoint identifier
        max_requests: Maximum requests per window
        window_seconds: Time window in seconds
        burst_allowance: Burst allowance for rate limiting
    """
    manager = get_default_webhook_manager()
    manager.configure_rate_limit(
        endpoint_id,
        max_requests,
        window_seconds,
        burst_allowance
    )

