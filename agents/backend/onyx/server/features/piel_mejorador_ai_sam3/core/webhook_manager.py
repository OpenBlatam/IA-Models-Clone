"""
Webhook Manager for Piel Mejorador AI SAM3
==========================================

Manages webhook notifications for task completion.
"""

import asyncio
import logging
import httpx
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .common.base_manager import BaseManager
from .common.manager_mixin import StatsMixin, ClientMixin, LockMixin

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


class WebhookManager(BaseManager, StatsMixin, ClientMixin, LockMixin):
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
        BaseManager.__init__(self, "WebhookManager")
        StatsMixin.__init__(self)
        ClientMixin.__init__(self)
        LockMixin.__init__(self)
        
        self._webhooks: List[Webhook] = []
        
        # Override stats with webhook-specific stats
        self._stats = {
            "total_sent": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        await self._close_client()
        await self.stop()
    
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
        """Send webhook to a specific endpoint."""
        client = await self._get_client()
        
        # Prepare payload
        payload_dict = {
            "event": payload.event.value,
            "task_id": payload.task_id,
            "data": payload.data,
            "timestamp": payload.timestamp,
        }
        
        # Add signature if secret is provided
        headers = {"Content-Type": "application/json"}
        if webhook.secret:
            import hmac
            import hashlib
            payload_str = str(payload_dict)
            signature = hmac.new(
                webhook.secret.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature
        
        # Retry logic
        for attempt in range(webhook.retries + 1):
            try:
                response = await client.post(
                    webhook.url,
                    json=payload_dict,
                    headers=headers,
                    timeout=webhook.timeout
                )
                response.raise_for_status()
                
                self._stats["successful"] += 1
                self._stats["total_sent"] += 1
                logger.debug(f"Webhook sent successfully: {webhook.url}")
                return
                
            except Exception as e:
                if attempt < webhook.retries:
                    self._stats["retries"] += 1
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(
                        f"Webhook failed (attempt {attempt + 1}/{webhook.retries + 1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._stats["failed"] += 1
                    self._stats["total_sent"] += 1
                    logger.error(f"Webhook failed after {webhook.retries + 1} attempts: {webhook.url} - {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        base_stats = BaseManager.get_stats(self)
        success_rate = (
            self._stats["successful"] / self._stats["total_sent"]
            if self._stats["total_sent"] > 0 else 0
        )
        
        webhook_stats = {
            **self._stats,
            "success_rate": success_rate,
            "registered_webhooks": len(self._webhooks),
        }
        return {**base_stats, **webhook_stats}

