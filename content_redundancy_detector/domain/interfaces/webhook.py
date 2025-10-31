"""
Webhook Domain Interfaces
Ports for webhook functionality
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from ..entities.webhook import WebhookEndpoint, WebhookEventType, WebhookDelivery


class IWebhookRepository(ABC):
    """Repository interface for webhook endpoints"""
    
    @abstractmethod
    async def save_endpoint(self, endpoint: WebhookEndpoint) -> None:
        """Save webhook endpoint"""
        pass
    
    @abstractmethod
    async def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get webhook endpoint by ID"""
        pass
    
    @abstractmethod
    async def get_endpoints_by_event(self, event: WebhookEventType) -> List[WebhookEndpoint]:
        """Get endpoints that support event type"""
        pass
    
    @abstractmethod
    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint"""
        pass
    
    @abstractmethod
    async def list_endpoints(self, user_id: Optional[str] = None) -> List[WebhookEndpoint]:
        """List all endpoints"""
        pass


class IWebhookDeliveryRepository(ABC):
    """Repository interface for webhook deliveries"""
    
    @abstractmethod
    async def save_delivery(self, delivery: WebhookDelivery) -> None:
        """Save webhook delivery"""
        pass
    
    @abstractmethod
    async def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get webhook delivery by ID"""
        pass
    
    @abstractmethod
    async def update_delivery(self, delivery: WebhookDelivery) -> None:
        """Update webhook delivery"""
        pass
    
    @abstractmethod
    async def list_pending_deliveries(self, limit: int = 100) -> List[WebhookDelivery]:
        """List pending deliveries"""
        pass


class IWebhookSender(ABC):
    """Interface for sending webhooks"""
    
    @abstractmethod
    async def send(
        self,
        endpoint: WebhookEndpoint,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """Send webhook HTTP request"""
        pass


class IWebhookSigner(ABC):
    """Interface for webhook signature generation"""
    
    @abstractmethod
    def sign(self, payload: str, secret: str) -> str:
        """Generate webhook signature"""
        pass
    
    @abstractmethod
    def verify(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        pass






