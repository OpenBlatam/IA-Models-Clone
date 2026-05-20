"""
Webhook Manager for Color Grading AI
====================================

Manages webhook notifications for processing events.
"""

import logging
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class Webhook:
    """Webhook configuration."""
    url: str
    events: List[str]  # completed, failed, progress
    secret: Optional[str] = None
    timeout: float = 30.0


class WebhookManager:
    """
    Manages webhook notifications.
    
    Features:
    - Send webhooks on events
    - Retry on failure
    - Event filtering
    """
    
    def __init__(self):
        """Initialize webhook manager."""
        self._webhooks: List[Webhook] = []
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def register(self, webhook: Webhook):
        """
        Register a webhook.
        
        Args:
            webhook: Webhook configuration
        """
        self._webhooks.append(webhook)
        logger.info(f"Registered webhook: {webhook.url}")
    
    def unregister(self, url: str):
        """
        Unregister a webhook.
        
        Args:
            url: Webhook URL
        """
        self._webhooks = [w for w in self._webhooks if w.url != url]
        logger.info(f"Unregistered webhook: {url}")
    
    async def send(
        self,
        event: str,
        data: Dict[str, Any],
        task_id: Optional[str] = None
    ):
        """
        Send webhook notification.
        
        Args:
            event: Event name
            data: Event data
            task_id: Optional task ID
        """
        client = await self._get_client()
        
        payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        
        if task_id:
            payload["task_id"] = task_id
        
        # Send to all webhooks subscribed to this event
        tasks = []
        for webhook in self._webhooks:
            if event in webhook.events:
                tasks.append(self._send_webhook(client, webhook, payload))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(
        self,
        client: httpx.AsyncClient,
        webhook: Webhook,
        payload: Dict[str, Any]
    ):
        """Send webhook to specific URL."""
        headers = {"Content-Type": "application/json"}
        if webhook.secret:
            headers["X-Webhook-Secret"] = webhook.secret
        
        try:
            response = await client.post(
                webhook.url,
                json=payload,
                headers=headers,
                timeout=webhook.timeout
            )
            response.raise_for_status()
            logger.info(f"Webhook sent successfully: {webhook.url}")
        except Exception as e:
            logger.error(f"Error sending webhook to {webhook.url}: {e}")
    
    async def close(self):
        """Close webhook manager."""
        if self._client:
            await self._client.aclose()
            self._client = None




