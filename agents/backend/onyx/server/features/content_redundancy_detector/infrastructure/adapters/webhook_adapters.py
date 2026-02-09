"""
Webhook Infrastructure Adapters
Concrete implementations of webhook interfaces
"""

import hmac
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional
import httpx
from uuid import uuid4

from ...domain.entities.webhook import (
    WebhookEndpoint, WebhookEventType, WebhookDelivery, WebhookDeliveryStatus
)
from ...domain.interfaces.webhook import (
    IWebhookRepository, IWebhookDeliveryRepository, IWebhookSender, IWebhookSigner
)

logger = logging.getLogger(__name__)


class InMemoryWebhookRepository(IWebhookRepository):
    """In-memory implementation of webhook repository"""
    
    def __init__(self):
        self._endpoints: Dict[str, WebhookEndpoint] = {}
    
    async def save_endpoint(self, endpoint: WebhookEndpoint) -> None:
        """Save webhook endpoint"""
        self._endpoints[endpoint.id] = endpoint
    
    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint by ID"""
        return self._endpoints.get(endpoint_id)
    
    async def get_endpoints_by_event(self, event: WebhookEventType) -> List[WebhookEndpoint]:
        """Get endpoints that support event type"""
        return [
            endpoint for endpoint in self._endpoints.values()
            if endpoint.supports_event(event)
        ]
    
    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint"""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            return True
        return False
    
    async def list_endpoints(self, user_id: Optional[str] = None) -> List[WebhookEndpoint]:
        """List all endpoints"""
        endpoints = list(self._endpoints.values())
        if user_id:
            return [e for e in endpoints if e.user_id == user_id]
        return endpoints


class InMemoryWebhookDeliveryRepository(IWebhookDeliveryRepository):
    """In-memory implementation of delivery repository"""
    
    def __init__(self):
        self._deliveries: Dict[str, WebhookDelivery] = {}
    
    async def save_delivery(self, delivery: WebhookDelivery) -> None:
        """Save webhook delivery"""
        self._deliveries[delivery.id] = delivery
    
    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get webhook delivery by ID"""
        return self._deliveries.get(delivery_id)
    
    async def update_delivery(self, delivery: WebhookDelivery) -> None:
        """Update webhook delivery"""
        self._deliveries[delivery.id] = delivery
    
    async def list_pending_deliveries(self, limit: int = 100) -> List[WebhookDelivery]:
        """List pending deliveries"""
        pending = [
            d for d in self._deliveries.values()
            if d.status in [WebhookDeliveryStatus.PENDING, WebhookDeliveryStatus.RETRYING]
        ]
        return sorted(pending, key=lambda x: x.created_at)[:limit]


class HTTPWebhookSender(IWebhookSender):
    """HTTP client implementation for sending webhooks"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def send(
        self,
        endpoint: WebhookEndpoint,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send webhook HTTP request"""
        try:
            client = await self._get_client()
            response = await client.post(
                endpoint.url,
                json=payload,
                headers=headers or {}
            )
            
            # Consider 2xx status codes as success
            success = 200 <= response.status_code < 300
            
            if not success:
                logger.warning(
                    f"Webhook delivery failed: {endpoint.url} - "
                    f"Status: {response.status_code}"
                )
            
            return success
            
        except httpx.TimeoutException:
            logger.error(f"Webhook timeout: {endpoint.url}")
            return False
        except Exception as e:
            logger.error(f"Webhook send error: {endpoint.url} - {e}")
            return False
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None


class HMACWebhookSigner(IWebhookSigner):
    """HMAC-SHA256 implementation for webhook signatures"""
    
    def sign(self, payload: str, secret: str) -> str:
        """Generate webhook signature"""
        if isinstance(payload, dict):
            payload = json.dumps(payload, sort_keys=True)
        
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self.sign(payload, secret)
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_signature, signature.replace('sha256=', ''))






