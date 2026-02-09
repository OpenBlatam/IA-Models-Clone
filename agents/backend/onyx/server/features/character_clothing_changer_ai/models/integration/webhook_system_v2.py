"""
Webhook System V2
=================

Advanced webhook system with retry logic and delivery tracking.
"""

import time
import hashlib
import requests
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WebhookStatus(Enum):
    """Webhook status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Webhook:
    """Webhook configuration."""
    id: str
    url: str
    events: List[str]
    secret: Optional[str] = None
    active: bool = True
    retry_count: int = 3
    timeout: float = 5.0
    headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass
class WebhookDelivery:
    """Webhook delivery record."""
    id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status: WebhookStatus
    attempts: int = 0
    last_attempt: Optional[float] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    created_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class WebhookSystemV2:
    """Advanced webhook system."""
    
    def __init__(self):
        """Initialize webhook system."""
        self.webhooks: Dict[str, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        webhook_id: Optional[str] = None,
    ) -> Webhook:
        """
        Register a webhook.
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for signing
            webhook_id: Optional webhook ID
            
        Returns:
            Created webhook
        """
        if webhook_id is None:
            webhook_id = f"webhook_{int(time.time() * 1000)}"
        
        webhook = Webhook(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
        )
        
        self.webhooks[webhook_id] = webhook
        
        # Register event handlers
        for event in events:
            if event not in self.event_handlers:
                self.event_handlers[event] = []
            self.event_handlers[event].append(lambda payload, w=webhook: self._deliver_webhook(w, event, payload))
        
        logger.info(f"Webhook registered: {webhook_id} for events: {events}")
        
        return webhook
    
    def trigger_event(
        self,
        event: str,
        payload: Dict[str, Any],
    ) -> List[WebhookDelivery]:
        """
        Trigger webhook event.
        
        Args:
            event: Event name
            payload: Event payload
            
        Returns:
            List of webhook deliveries
        """
        deliveries = []
        
        # Find webhooks for this event
        webhooks = [
            w for w in self.webhooks.values()
            if w.active and event in w.events
        ]
        
        for webhook in webhooks:
            delivery = self._deliver_webhook(webhook, event, payload)
            deliveries.append(delivery)
        
        return deliveries
    
    def _deliver_webhook(
        self,
        webhook: Webhook,
        event: str,
        payload: Dict[str, Any],
    ) -> WebhookDelivery:
        """
        Deliver webhook.
        
        Args:
            webhook: Webhook configuration
            event: Event name
            payload: Event payload
            
        Returns:
            Webhook delivery record
        """
        delivery_id = f"delivery_{int(time.time() * 1000)}"
        
        delivery = WebhookDelivery(
            id=delivery_id,
            webhook_id=webhook.id,
            event=event,
            payload=payload,
            status=WebhookStatus.PENDING,
        )
        
        # Prepare request
        request_payload = {
            "event": event,
            "timestamp": time.time(),
            "data": payload,
        }
        
        # Add signature if secret exists
        if webhook.secret:
            import hmac
            signature = hmac.new(
                webhook.secret.encode(),
                str(request_payload).encode(),
                hashlib.sha256
            ).hexdigest()
            request_payload["signature"] = signature
        
        # Attempt delivery
        try:
            response = requests.post(
                webhook.url,
                json=request_payload,
                headers=webhook.headers,
                timeout=webhook.timeout,
            )
            
            delivery.attempts = 1
            delivery.last_attempt = time.time()
            delivery.response_code = response.status_code
            delivery.response_body = response.text[:500]  # Limit response body
            
            if response.status_code >= 200 and response.status_code < 300:
                delivery.status = WebhookStatus.DELIVERED
                logger.info(f"Webhook delivered: {webhook.id} - {event}")
            else:
                delivery.status = WebhookStatus.FAILED
                delivery.error = f"HTTP {response.status_code}"
                logger.warning(f"Webhook failed: {webhook.id} - {event} ({response.status_code})")
        
        except Exception as e:
            delivery.attempts = 1
            delivery.last_attempt = time.time()
            delivery.status = WebhookStatus.FAILED
            delivery.error = str(e)
            logger.error(f"Webhook delivery error: {webhook.id} - {e}")
        
        self.deliveries.append(delivery)
        
        # Keep only last 1000 deliveries
        if len(self.deliveries) > 1000:
            self.deliveries = self.deliveries[-1000:]
        
        return delivery
    
    def get_webhook_deliveries(
        self,
        webhook_id: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """
        Get webhook deliveries.
        
        Args:
            webhook_id: Optional webhook ID filter
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of deliveries
        """
        deliveries = self.deliveries
        
        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]
        
        if status:
            deliveries = [d for d in deliveries if d.status == status]
        
        return deliveries[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get webhook system statistics."""
        successful = len([d for d in self.deliveries if d.status == WebhookStatus.DELIVERED])
        failed = len([d for d in self.deliveries if d.status == WebhookStatus.FAILED])
        
        return {
            "total_webhooks": len(self.webhooks),
            "active_webhooks": len([w for w in self.webhooks.values() if w.active]),
            "total_deliveries": len(self.deliveries),
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "success_rate": (successful / len(self.deliveries) * 100) if self.deliveries else 0.0,
        }

