"""Advanced webhook system with retry and events"""
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import httpx
import asyncio
import logging
import json

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Webhook event types"""
    CONVERSION_STARTED = "conversion.started"
    CONVERSION_COMPLETED = "conversion.completed"
    CONVERSION_FAILED = "conversion.failed"
    DOCUMENT_SIGNED = "document.signed"
    DOCUMENT_REVIEWED = "document.reviewed"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    url: str
    secret: Optional[str] = None
    events: List[WebhookEvent] = None
    retry_count: int = 3
    retry_delay: int = 5
    timeout: int = 30
    enabled: bool = True
    
    def __post_init__(self):
        if self.events is None:
            self.events = list(WebhookEvent)


@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    webhook_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    status: str  # pending, delivered, failed
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class AdvancedWebhookClient:
    """Advanced webhook client with retry and event filtering"""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.deliveries: List[WebhookDelivery] = []
        self.max_deliveries = 1000  # Keep last 1000 deliveries
    
    def register_webhook(
        self,
        webhook_id: str,
        config: WebhookConfig
    ) -> bool:
        """
        Register a webhook
        
        Args:
            webhook_id: Webhook ID
            config: Webhook configuration
            
        Returns:
            True if successful
        """
        try:
            self.webhooks[webhook_id] = config
            return True
        except Exception as e:
            logger.error(f"Error registering webhook: {e}")
            return False
    
    async def send_webhook(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Send webhook for event
        
        Args:
            event: Event type
            payload: Event payload
            
        Returns:
            List of delivery results
        """
        results = []
        
        for webhook_id, config in self.webhooks.items():
            if not config.enabled:
                continue
            
            if event not in config.events:
                continue
            
            # Create delivery
            delivery = WebhookDelivery(
                webhook_id=webhook_id,
                event=event,
                payload=payload,
                status="pending"
            )
            
            # Send with retry
            success = await self._send_with_retry(delivery, config)
            
            if success:
                delivery.status = "delivered"
            else:
                delivery.status = "failed"
            
            # Store delivery
            self.deliveries.append(delivery)
            if len(self.deliveries) > self.max_deliveries:
                self.deliveries.pop(0)
            
            results.append({
                "webhook_id": webhook_id,
                "status": delivery.status,
                "attempts": delivery.attempts,
                "response_code": delivery.response_code
            })
        
        return results
    
    async def _send_with_retry(
        self,
        delivery: WebhookDelivery,
        config: WebhookConfig
    ) -> bool:
        """
        Send webhook with retry logic
        
        Args:
            delivery: Delivery record
            config: Webhook configuration
            
        Returns:
            True if successful
        """
        payload = {
            "event": delivery.event.value,
            "timestamp": datetime.now().isoformat(),
            "data": delivery.payload
        }
        
        # Add signature if secret provided
        if config.secret:
            import hmac
            import hashlib
            import base64
            signature = hmac.new(
                config.secret.encode(),
                json.dumps(payload).encode(),
                hashlib.sha256
            ).hexdigest()
            payload["signature"] = signature
        
        for attempt in range(config.retry_count):
            delivery.attempts = attempt + 1
            delivery.last_attempt = datetime.now()
            
            try:
                async with httpx.AsyncClient(timeout=config.timeout) as client:
                    response = await client.post(
                        config.url,
                        json=payload,
                        headers={
                            "User-Agent": "MarkdownToDocsAI-Webhook/1.9.0",
                            "X-Webhook-Event": delivery.event.value
                        }
                    )
                    
                    delivery.response_code = response.status_code
                    delivery.response_body = response.text[:500]  # Limit response body
                    
                    if response.status_code < 400:
                        return True
                    else:
                        logger.warning(
                            f"Webhook failed with status {response.status_code}: {response.text}"
                        )
            except Exception as e:
                logger.error(f"Webhook delivery error (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < config.retry_count - 1:
                await asyncio.sleep(config.retry_delay * (attempt + 1))
        
        return False
    
    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        event: Optional[WebhookEvent] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        Get webhook deliveries
        
        Args:
            webhook_id: Optional webhook ID filter
            event: Optional event filter
            limit: Maximum number of results
            
        Returns:
            List of deliveries
        """
        deliveries = self.deliveries
        
        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
        
        if event:
            deliveries = [d for d in deliveries if d.event == event]
        
        return sorted(deliveries, key=lambda x: x.created_at, reverse=True)[:limit]
    
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all registered webhooks"""
        return [
            {
                "id": webhook_id,
                "url": config.url,
                "events": [e.value for e in config.events],
                "enabled": config.enabled
            }
            for webhook_id, config in self.webhooks.items()
        ]


# Global webhook client
_webhook_client: Optional[AdvancedWebhookClient] = None


def get_advanced_webhook_client() -> AdvancedWebhookClient:
    """Get global advanced webhook client"""
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = AdvancedWebhookClient()
    return _webhook_client

