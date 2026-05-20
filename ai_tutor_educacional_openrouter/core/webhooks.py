"""
Webhook system for event notifications and integrations.
"""

import logging
import httpx
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Types of webhook events."""
    QUESTION_ASKED = "question.asked"
    ANSWER_RECEIVED = "answer.received"
    QUIZ_COMPLETED = "quiz.completed"
    ACHIEVEMENT_UNLOCKED = "achievement.unlocked"
    STUDENT_REGISTERED = "student.registered"
    REPORT_GENERATED = "report.generated"
    EVALUATION_COMPLETED = "evaluation.completed"


@dataclass
class Webhook:
    """Represents a webhook configuration."""
    webhook_id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class WebhookManager:
    """
    Manages webhooks for event notifications.
    """
    
    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        self.client = httpx.AsyncClient(timeout=10.0)
        self.event_history: List[Dict[str, Any]] = []
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None
    ) -> Webhook:
        """
        Register a new webhook.
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for webhook verification
        
        Returns:
            Created webhook
        """
        webhook_id = f"wh_{datetime.now().timestamp()}"
        
        webhook = Webhook(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret
        )
        
        self.webhooks[webhook_id] = webhook
        
        logger.info(f"Registered webhook {webhook_id} for {len(events)} events")
        
        return webhook
    
    def unregister_webhook(self, webhook_id: str):
        """Unregister a webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
    
    async def trigger_event(
        self,
        event: WebhookEvent,
        data: Dict[str, Any]
    ):
        """
        Trigger a webhook event.
        
        Args:
            event: Event type
            data: Event data
        """
        payload = {
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Find webhooks subscribed to this event
        relevant_webhooks = [
            wh for wh in self.webhooks.values()
            if wh.active and event in wh.events
        ]
        
        # Send to all relevant webhooks
        tasks = [
            self._send_webhook(webhook, payload)
            for webhook in relevant_webhooks
        ]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log results
            for webhook, result in zip(relevant_webhooks, results):
                if isinstance(result, Exception):
                    logger.error(f"Webhook {webhook.webhook_id} failed: {result}")
                else:
                    logger.debug(f"Webhook {webhook.webhook_id} sent successfully")
        
        # Store in history
        self.event_history.append({
            "event": event.value,
            "timestamp": datetime.now().isoformat(),
            "webhooks_triggered": len(relevant_webhooks),
            "data": data
        })
        
        # Keep only last 1000 events
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-1000:]
    
    async def _send_webhook(self, webhook: Webhook, payload: Dict[str, Any]):
        """Send webhook payload to URL."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Tutor-Educacional/1.0"
        }
        
        if webhook.secret:
            import hmac
            import hashlib
            signature = hmac.new(
                webhook.secret.encode(),
                str(payload).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        try:
            response = await self.client.post(
                webhook.url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook.url}: {e}")
            raise
    
    def get_webhooks(self) -> List[Webhook]:
        """Get all registered webhooks."""
        return list(self.webhooks.values())
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history."""
        return self.event_history[-limit:]






