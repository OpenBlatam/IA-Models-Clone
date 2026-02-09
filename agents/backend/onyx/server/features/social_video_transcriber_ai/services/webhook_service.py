"""
Webhook Service for Social Video Transcriber AI
Handles webhook notifications for job completion and status updates
"""

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

import httpx

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Webhook event types"""
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    BATCH_STARTED = "batch.started"
    BATCH_PROGRESS = "batch.progress"
    BATCH_COMPLETED = "batch.completed"
    ANALYSIS_COMPLETED = "analysis.completed"
    VARIANTS_GENERATED = "variants.generated"


@dataclass
class WebhookRegistration:
    """Webhook registration"""
    webhook_id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered_at: Optional[datetime] = None
    failure_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "webhook_id": self.webhook_id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "failure_count": self.failure_count,
        }


@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    delivery_id: str
    webhook_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: Optional[datetime] = None
    error: Optional[str] = None
    retries: int = 0


class WebhookService:
    """Service for managing and sending webhooks"""
    
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 5, 30]  # Seconds between retries
    TIMEOUT = 30.0  # Request timeout
    
    def __init__(self):
        self.settings = get_settings()
        self._registrations: Dict[str, WebhookRegistration] = {}
        self._job_webhooks: Dict[UUID, str] = {}  # job_id -> webhook_url
        self._delivery_history: List[WebhookDelivery] = []
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
    ) -> WebhookRegistration:
        """
        Register a webhook endpoint
        
        Args:
            url: Webhook URL
            events: List of events to subscribe to
            secret: Optional secret for signing payloads
            
        Returns:
            WebhookRegistration object
        """
        import secrets
        
        webhook_id = secrets.token_hex(16)
        
        registration = WebhookRegistration(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
        )
        
        self._registrations[webhook_id] = registration
        logger.info(f"Registered webhook {webhook_id}: {url}")
        
        return registration
    
    def register_job_webhook(self, job_id: UUID, webhook_url: str):
        """Register a webhook for a specific job"""
        self._job_webhooks[job_id] = webhook_url
        logger.debug(f"Registered job webhook: {job_id} -> {webhook_url}")
    
    def unregister_webhook(self, webhook_id: str):
        """Unregister a webhook"""
        if webhook_id in self._registrations:
            del self._registrations[webhook_id]
            logger.info(f"Unregistered webhook: {webhook_id}")
    
    def unregister_job_webhook(self, job_id: UUID):
        """Unregister a job webhook"""
        if job_id in self._job_webhooks:
            del self._job_webhooks[job_id]
    
    async def trigger(
        self,
        event: WebhookEvent,
        data: Dict[str, Any],
        job_id: Optional[UUID] = None,
    ):
        """
        Trigger webhooks for an event
        
        Args:
            event: Event type
            data: Event data
            job_id: Optional job ID for job-specific webhooks
        """
        # Build payload
        payload = {
            "event": event.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        
        if job_id:
            payload["job_id"] = str(job_id)
        
        # Get all relevant webhooks
        webhooks_to_notify = []
        
        # Check registered webhooks
        for reg in self._registrations.values():
            if reg.active and event in reg.events:
                webhooks_to_notify.append((reg.webhook_id, reg.url, reg.secret))
        
        # Check job-specific webhook
        if job_id and job_id in self._job_webhooks:
            webhook_url = self._job_webhooks[job_id]
            webhooks_to_notify.append((f"job_{job_id}", webhook_url, None))
        
        # Send webhooks concurrently
        if webhooks_to_notify:
            tasks = [
                self._send_webhook(webhook_id, url, payload, secret)
                for webhook_id, url, secret in webhooks_to_notify
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_webhook(
        self,
        webhook_id: str,
        url: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None,
    ) -> bool:
        """
        Send a webhook notification with retry logic
        
        Args:
            webhook_id: Webhook identifier
            url: Target URL
            payload: Payload to send
            secret: Optional secret for signing
            
        Returns:
            True if successful, False otherwise
        """
        import secrets as secrets_module
        
        delivery_id = secrets_module.token_hex(8)
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=webhook_id,
            event=WebhookEvent(payload["event"]),
            payload=payload,
        )
        
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-ID": webhook_id,
            "X-Delivery-ID": delivery_id,
        }
        
        # Add signature if secret is provided
        if secret:
            payload_bytes = json.dumps(payload, sort_keys=True).encode()
            signature = hmac.new(
                secret.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        # Retry logic
        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    
                    delivery.response_status = response.status_code
                    delivery.response_body = response.text[:1000]  # Limit stored response
                    delivery.delivered_at = datetime.utcnow()
                    
                    if response.status_code < 400:
                        logger.info(f"Webhook delivered: {webhook_id} -> {url}")
                        self._record_success(webhook_id)
                        self._delivery_history.append(delivery)
                        return True
                    
                    logger.warning(
                        f"Webhook response {response.status_code}: {webhook_id}"
                    )
                    
            except Exception as e:
                delivery.error = str(e)
                logger.warning(f"Webhook failed (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(self.RETRY_DELAYS[attempt])
            
            delivery.retries = attempt + 1
        
        # All retries failed
        logger.error(f"Webhook failed after {self.MAX_RETRIES} attempts: {webhook_id}")
        self._record_failure(webhook_id)
        self._delivery_history.append(delivery)
        
        return False
    
    def _record_success(self, webhook_id: str):
        """Record successful delivery"""
        if webhook_id in self._registrations:
            reg = self._registrations[webhook_id]
            reg.last_triggered_at = datetime.utcnow()
            reg.failure_count = 0
    
    def _record_failure(self, webhook_id: str):
        """Record failed delivery"""
        if webhook_id in self._registrations:
            reg = self._registrations[webhook_id]
            reg.failure_count += 1
            
            # Disable webhook after too many failures
            if reg.failure_count >= 10:
                reg.active = False
                logger.warning(f"Webhook disabled due to failures: {webhook_id}")
    
    def list_webhooks(self) -> List[WebhookRegistration]:
        """List all registered webhooks"""
        return list(self._registrations.values())
    
    def get_delivery_history(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get webhook delivery history"""
        history = self._delivery_history
        
        if webhook_id:
            history = [d for d in history if d.webhook_id == webhook_id]
        
        # Sort by delivery time (most recent first)
        history.sort(
            key=lambda d: d.delivered_at or datetime.min,
            reverse=True
        )
        
        return [
            {
                "delivery_id": d.delivery_id,
                "webhook_id": d.webhook_id,
                "event": d.event.value,
                "response_status": d.response_status,
                "delivered_at": d.delivered_at.isoformat() if d.delivered_at else None,
                "error": d.error,
                "retries": d.retries,
            }
            for d in history[:limit]
        ]


_webhook_service: Optional[WebhookService] = None


def get_webhook_service() -> WebhookService:
    """Get webhook service singleton"""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service












