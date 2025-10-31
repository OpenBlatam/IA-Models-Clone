"""
Webhook Service Dependencies
Dependency injection for webhook services
"""

from functools import lru_cache
from ...application.services.webhook_service import WebhookService
from ...infrastructure.adapters.webhook_adapters import (
    InMemoryWebhookRepository,
    InMemoryWebhookDeliveryRepository,
    HTTPWebhookSender,
    HMACWebhookSigner
)


@lru_cache()
def get_webhook_endpoint_repository():
    """Get webhook endpoint repository"""
    return InMemoryWebhookRepository()


@lru_cache()
def get_webhook_delivery_repository():
    """Get webhook delivery repository"""
    return InMemoryWebhookDeliveryRepository()


@lru_cache()
def get_webhook_sender():
    """Get webhook sender (HTTP client)"""
    return HTTPWebhookSender(timeout=30)


@lru_cache()
def get_webhook_signer():
    """Get webhook signer (HMAC)"""
    return HMACWebhookSigner()


async def get_webhook_service() -> WebhookService:
    """Get webhook service with all dependencies"""
    return WebhookService(
        endpoint_repository=get_webhook_endpoint_repository(),
        delivery_repository=get_webhook_delivery_repository(),
        sender=get_webhook_sender(),
        signer=get_webhook_signer(),
        messaging_service=None  # Optional: inject if available
    )






