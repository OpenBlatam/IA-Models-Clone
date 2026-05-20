"""
Webhook Manager for Imagen Video Enhancer AI
============================================

Manages webhook notifications for task completion.
"""

import asyncio
import logging
import httpx
import hmac
import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Webhook event types."""
    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    BATCH_COMPLETED = "batch.completed"


@dataclass
class Webhook:
    """Webhook configuration."""
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    timeout: float = 10.0
    retries: int = 3
    enabled: bool = True


@dataclass
class WebhookPayload:
    """Webhook payload."""
    event: WebhookEvent
    task_id: Optional[str] = None
    data: Dict[str, Any] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event": self.event.value,
            "task_id": self.task_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class WebhookManager:
    """
    Manages webhook notifications.
    
    Features:
    - Multiple webhook endpoints
    - Event filtering
    - Retry logic
    - Async delivery
    - Signature verification
    """
    
    def __init__(self):
        """Initialize webhook manager."""
        self._webhooks: List[Webhook] = []
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        
        self._stats = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def register(self, webhook: Webhook):
        """
        Register a webhook.
        
        Args:
            webhook: Webhook configuration
        """
        self._webhooks.append(webhook)
        logger.info(f"Registered webhook: {webhook.url} for events: {[e.value for e in webhook.events]}")
    
    def unregister(self, url: str):
        """
        Unregister a webhook.
        
        Args:
            url: Webhook URL to remove
        """
        self._webhooks = [w for w in self._webhooks if w.url != url]
        logger.info(f"Unregistered webhook: {url}")
    
    async def send(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        task_id: Optional[str] = None
    ):
        """
        Send webhook notifications for an event.
        
        Args:
            event: Event type
            payload: Payload data
            task_id: Optional task ID
        """
        webhook_payload = WebhookPayload(
            event=event,
            task_id=task_id,
            data=payload
        )
        
        # Find webhooks for this event
        relevant_webhooks = [
            w for w in self._webhooks
            if w.enabled and event in w.events
        ]
        
        if not relevant_webhooks:
            return
        
        # Send to all relevant webhooks
        tasks = [
            self._send_to_webhook(webhook, webhook_payload)
            for webhook in relevant_webhooks
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_webhook(
        self,
        webhook: Webhook,
        payload: WebhookPayload
    ):
        """Send webhook to a single endpoint."""
        client = await self._get_client()
        payload_dict = payload.to_dict()
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Imagen-Video-Enhancer-AI/1.0"
        }
        
        # Add signature if secret is provided
        if webhook.secret:
            signature = hmac.new(
                webhook.secret.encode(),
                json.dumps(payload_dict, sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        # Retry logic
        last_error = None
        for attempt in range(webhook.retries):
            try:
                response = await client.post(
                    webhook.url,
                    json=payload_dict,
                    headers=headers,
                    timeout=webhook.timeout
                )
                response.raise_for_status()
                
                async with self._lock:
                    self._stats["total_sent"] += 1
                    self._stats["successful"] += 1
                    if attempt > 0:
                        self._stats["retries"] += 1
                
                logger.debug(f"Webhook sent successfully: {webhook.url}")
                return
                
            except Exception as e:
                last_error = e
                if attempt < webhook.retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(delay)
                    async with self._lock:
                        self._stats["retries"] += 1
                else:
                    async with self._lock:
                        self._stats["total_sent"] += 1
                        self._stats["failed"] += 1
                    logger.error(f"Failed to send webhook to {webhook.url}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        return {
            **self._stats,
            "registered_webhooks": len(self._webhooks),
        }




