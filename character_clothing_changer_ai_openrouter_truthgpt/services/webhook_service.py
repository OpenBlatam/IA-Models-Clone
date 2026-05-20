"""
Webhook Service
===============

Service for sending webhook notifications on workflow completion.
"""

import logging
import httpx
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEvent:
    """Webhook event data"""
    event_type: str  # "workflow_completed", "workflow_failed", "batch_completed"
    prompt_id: Optional[str] = None
    batch_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    url: str
    secret: Optional[str] = None
    timeout: float = DEFAULT_TIMEOUT
    retries: int = MAX_RETRIES
    retry_delay: float = RETRY_DELAY
    enabled: bool = True
    events: List[str] = None
    
    def __post_init__(self):
        if self.events is None:
            self.events = ["workflow_completed", "workflow_failed", "batch_completed"]


class WebhookService:
    """
    Service for sending webhook notifications.
    
    Features:
    - Async webhook delivery
    - Retry logic with exponential backoff
    - Event filtering
    - Signature verification
    - Delivery tracking
    """
    
    def __init__(self):
        """Initialize webhook service"""
        self.webhooks: List[WebhookConfig] = []
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._http_client is None:
            timeout = httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0)
            self._http_client = httpx.AsyncClient(
                timeout=timeout,
                http2=True
            )
        return self._http_client
    
    async def close(self) -> None:
        """Close HTTP client"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    def register_webhook(self, config: WebhookConfig) -> None:
        """
        Register a webhook configuration.
        
        Args:
            config: WebhookConfig object
        """
        if config.enabled:
            self.webhooks.append(config)
            logger.info(f"Registered webhook: {config.url} (events: {config.events})")
    
    def unregister_webhook(self, url: str) -> bool:
        """
        Unregister a webhook by URL.
        
        Args:
            url: Webhook URL to unregister
            
        Returns:
            True if webhook was removed, False if not found
        """
        initial_count = len(self.webhooks)
        self.webhooks = [w for w in self.webhooks if w.url != url]
        removed = len(self.webhooks) < initial_count
        
        if removed:
            logger.info(f"Unregistered webhook: {url}")
        
        return removed
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate webhook signature.
        
        Args:
            payload: JSON payload string
            secret: Secret key
            
        Returns:
            Signature string
        """
        import hmac
        import hashlib
        
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def _send_webhook(
        self,
        config: WebhookConfig,
        event: WebhookEvent
    ) -> Dict[str, Any]:
        """
        Send webhook notification.
        
        Args:
            config: WebhookConfig
            event: WebhookEvent
            
        Returns:
            Dictionary with delivery result
        """
        client = await self._get_http_client()
        
        # Prepare payload
        payload = {
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data or {}
        }
        
        if event.prompt_id:
            payload["prompt_id"] = event.prompt_id
        
        if event.batch_id:
            payload["batch_id"] = event.batch_id
        
        import json
        payload_json = json.dumps(payload)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CharacterClothingChanger/1.0"
        }
        
        # Add signature if secret is provided
        if config.secret:
            signature = self._generate_signature(payload_json, config.secret)
            headers["X-Webhook-Signature"] = signature
        
        # Retry logic
        last_error = None
        for attempt in range(config.retries):
            try:
                response = await client.post(
                    config.url,
                    content=payload_json,
                    headers=headers
                )
                response.raise_for_status()
                
                logger.info(f"Webhook sent successfully: {config.url} (attempt {attempt + 1})")
                
                return {
                    "success": True,
                    "url": config.url,
                    "status_code": response.status_code,
                    "attempt": attempt + 1
                }
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code >= 400 and e.response.status_code < 500:
                    # Client error, don't retry
                    logger.error(f"Webhook client error: {config.url} - {e.response.status_code}")
                    break
                
                if attempt < config.retries - 1:
                    wait_time = config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Webhook failed (attempt {attempt + 1}/{config.retries}): "
                        f"{config.url} - {e.response.status_code}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Webhook failed after {config.retries} attempts: {config.url}")
                    
            except Exception as e:
                last_error = e
                if attempt < config.retries - 1:
                    wait_time = config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Webhook error (attempt {attempt + 1}/{config.retries}): "
                        f"{config.url} - {str(e)}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Webhook error after {config.retries} attempts: {config.url} - {str(e)}")
        
        return {
            "success": False,
            "url": config.url,
            "error": str(last_error) if last_error else "Unknown error",
            "attempts": config.retries
        }
    
    async def send_event(
        self,
        event: WebhookEvent
    ) -> List[Dict[str, Any]]:
        """
        Send webhook event to all registered webhooks.
        
        Args:
            event: WebhookEvent to send
            
        Returns:
            List of delivery results
        """
        if not self.webhooks:
            return []
        
        # Filter webhooks by event type
        relevant_webhooks = [
            w for w in self.webhooks
            if w.enabled and event.event_type in w.events
        ]
        
        if not relevant_webhooks:
            logger.debug(f"No webhooks registered for event type: {event.event_type}")
            return []
        
        logger.info(f"Sending webhook event '{event.event_type}' to {len(relevant_webhooks)} webhook(s)")
        
        # Send to all relevant webhooks in parallel
        tasks = [
            self._send_webhook(config, event)
            for config in relevant_webhooks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        delivery_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                delivery_results.append({
                    "success": False,
                    "url": relevant_webhooks[i].url,
                    "error": str(result)
                })
            else:
                delivery_results.append(result)
        
        successful = sum(1 for r in delivery_results if r.get("success"))
        logger.info(f"Webhook delivery: {successful}/{len(delivery_results)} successful")
        
        return delivery_results
    
    def get_webhooks(self) -> List[Dict[str, Any]]:
        """
        Get list of registered webhooks.
        
        Returns:
            List of webhook configurations (without secrets)
        """
        return [
            {
                "url": w.url,
                "enabled": w.enabled,
                "events": w.events,
                "timeout": w.timeout,
                "retries": w.retries
            }
            for w in self.webhooks
        ]


# Global webhook service instance
_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Get or create webhook service instance"""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service
