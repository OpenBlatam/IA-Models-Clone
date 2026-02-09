"""
Webhook System for Flux2 Clothing Changer
==========================================

Webhook management and delivery system.
"""

import requests
import time
import hashlib
import hmac
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Webhook:
    """Webhook configuration."""
    webhook_id: str
    url: str
    events: List[str]
    secret: Optional[str] = None
    headers: Dict[str, str] = None
    timeout: float = 30.0
    retries: int = 3
    enabled: bool = True
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass
class WebhookDelivery:
    """Webhook delivery record."""
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status: WebhookStatus
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = 0.0
    attempts: int = 0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class WebhookSystem:
    """Webhook management and delivery system."""
    
    def __init__(self):
        """Initialize webhook system."""
        self.webhooks: Dict[str, Webhook] = {}
        self.delivery_history: List[WebhookDelivery] = []
        
        # Statistics
        self.stats = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "pending_deliveries": 0,
        }
    
    def register_webhook(
        self,
        webhook_id: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        retries: int = 3,
    ) -> Webhook:
        """
        Register a webhook.
        
        Args:
            webhook_id: Unique webhook identifier
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for signature
            headers: Optional headers
            timeout: Request timeout
            retries: Number of retries
            
        Returns:
            Created webhook
        """
        webhook = Webhook(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            headers=headers or {},
            timeout=timeout,
            retries=retries,
        )
        
        self.webhooks[webhook_id] = webhook
        logger.info(f"Registered webhook: {webhook_id} for events: {events}")
        return webhook
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Unregister a webhook.
        
        Args:
            webhook_id: Webhook identifier
            
        Returns:
            True if unregistered
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook: {webhook_id}")
            return True
        return False
    
    def trigger_webhook(
        self,
        event: str,
        payload: Dict[str, Any],
        webhook_id: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        """
        Trigger webhooks for an event.
        
        Args:
            event: Event name
            payload: Event payload
            webhook_id: Optional specific webhook ID
            
        Returns:
            List of delivery records
        """
        # Find webhooks for this event
        if webhook_id:
            webhooks = [self.webhooks[webhook_id]] if webhook_id in self.webhooks else []
        else:
            webhooks = [
                webhook for webhook in self.webhooks.values()
                if event in webhook.events and webhook.enabled
            ]
        
        deliveries = []
        
        for webhook in webhooks:
            delivery = self._deliver_webhook(webhook, event, payload)
            deliveries.append(delivery)
            
            # Update statistics
            self.stats["total_deliveries"] += 1
            if delivery.status == WebhookStatus.SUCCESS:
                self.stats["successful_deliveries"] += 1
            elif delivery.status == WebhookStatus.FAILED:
                self.stats["failed_deliveries"] += 1
            else:
                self.stats["pending_deliveries"] += 1
        
        return deliveries
    
    def _deliver_webhook(
        self,
        webhook: Webhook,
        event: str,
        payload: Dict[str, Any],
    ) -> WebhookDelivery:
        """Deliver webhook to URL."""
        delivery = WebhookDelivery(
            webhook_id=webhook.webhook_id,
            event=event,
            payload=payload,
            status=WebhookStatus.PENDING,
        )
        
        # Prepare payload with metadata
        webhook_payload = {
            "event": event,
            "timestamp": time.time(),
            "data": payload,
        }
        
        # Add signature if secret provided
        headers = webhook.headers.copy()
        if webhook.secret:
            signature = self._generate_signature(
                json.dumps(webhook_payload, sort_keys=True),
                webhook.secret,
            )
            headers["X-Webhook-Signature"] = signature
        
        # Deliver with retries
        last_error = None
        for attempt in range(webhook.retries):
            delivery.attempts = attempt + 1
            
            try:
                response = requests.post(
                    webhook.url,
                    json=webhook_payload,
                    headers=headers,
                    timeout=webhook.timeout,
                )
                
                delivery.response_code = response.status_code
                delivery.response_body = response.text[:500]  # Limit length
                
                if response.status_code < 400:
                    delivery.status = WebhookStatus.SUCCESS
                    self.delivery_history.append(delivery)
                    logger.info(f"Webhook {webhook.webhook_id} delivered successfully")
                    return delivery
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:100]}"
                    delivery.status = WebhookStatus.RETRYING
                    
            except Exception as e:
                last_error = str(e)
                delivery.status = WebhookStatus.RETRYING
                logger.warning(f"Webhook delivery attempt {attempt + 1} failed: {e}")
            
            # Wait before retry
            if attempt < webhook.retries - 1:
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
        
        # All retries failed
        delivery.status = WebhookStatus.FAILED
        delivery.error = last_error
        self.delivery_history.append(delivery)
        logger.error(f"Webhook {webhook.webhook_id} delivery failed after {webhook.retries} attempts")
        
        return delivery
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature."""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Payload string
            signature: Received signature
            secret: Secret key
            
        Returns:
            True if signature is valid
        """
        expected = self._generate_signature(payload, secret)
        return hmac.compare_digest(expected, signature)
    
    def get_delivery_history(
        self,
        webhook_id: Optional[str] = None,
        event: Optional[str] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """
        Get delivery history.
        
        Args:
            webhook_id: Filter by webhook ID
            event: Filter by event
            limit: Maximum number of records
            
        Returns:
            List of delivery records
        """
        history = self.delivery_history
        
        if webhook_id:
            history = [d for d in history if d.webhook_id == webhook_id]
        
        if event:
            history = [d for d in history if d.event == event]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda d: d.timestamp, reverse=True)
        
        return history[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        success_rate = (
            self.stats["successful_deliveries"] / self.stats["total_deliveries"]
            if self.stats["total_deliveries"] > 0 else 0.0
        )
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "total_webhooks": len(self.webhooks),
            "enabled_webhooks": sum(1 for w in self.webhooks.values() if w.enabled),
        }


