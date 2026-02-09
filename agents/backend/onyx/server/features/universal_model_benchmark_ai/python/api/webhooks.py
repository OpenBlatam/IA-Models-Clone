"""
Webhook Notifications - Async delivery with structured tracking.

Provides:
- Webhook registration and lifecycle management
- Concurrent asynchronous deliveries with retries
- HMAC signatures for payload integrity
- Delivery tracking with rich metadata
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Supported webhook events."""

    EXPERIMENT_STARTED = "experiment.started"
    EXPERIMENT_COMPLETED = "experiment.completed"
    EXPERIMENT_FAILED = "experiment.failed"
    BENCHMARK_STARTED = "benchmark.started"
    BENCHMARK_COMPLETED = "benchmark.completed"
    BENCHMARK_FAILED = "benchmark.failed"
    MODEL_REGISTERED = "model.registered"
    COST_THRESHOLD = "cost.threshold"
    ALERT_TRIGGERED = "alert.triggered"


class WebhookDeliveryStatus(str, Enum):
    """Delivery lifecycle states."""

    PENDING = "pending"
    RETRYING = "retrying"
    DELIVERED = "delivered"
    FAILED = "failed"


@dataclass
class Webhook:
    """Webhook configuration entry."""

    id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    active: bool = True
    retry_count: Optional[int] = None
    timeout: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "active": self.active,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "created_at": self.created_at,
        }


@dataclass
class WebhookDelivery:
    """Delivery log entry."""

    id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status: WebhookDeliveryStatus = WebhookDeliveryStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[str] = None
    response_code: Optional[int] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class WebhookManager:
    """Manage webhook registrations and deliveries."""

    def __init__(
        self,
        *,
        default_timeout: float = 5.0,
        default_retry_count: int = 3,
        max_concurrency: int = 5,
    ) -> None:
        self.webhooks: Dict[str, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self._default_timeout = default_timeout
        self._default_retry_count = default_retry_count
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()

    def register_webhook(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: Optional[str] = None,
        retry_count: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Webhook:
        webhook_id = f"wh_{uuid4().hex}"
        webhook = Webhook(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            retry_count=retry_count,
            timeout=timeout,
        )
        self.webhooks[webhook_id] = webhook
        logger.info("Registered webhook %s for %d events", webhook_id, len(events))
        return webhook

    def unregister_webhook(self, webhook_id: str) -> None:
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info("Unregistered webhook %s", webhook_id)

    def _generate_signature(self, payload: str, secret: str) -> str:
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    async def _deliver_with_semaphore(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any],
    ) -> None:
        async with self._semaphore:
            await self._deliver_webhook(webhook, event, payload)

    async def _deliver_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        payload: Dict[str, Any],
    ) -> None:
        delivery = WebhookDelivery(
            id=f"delivery_{uuid4().hex}",
            webhook_id=webhook.id,
            event=event.value,
            payload=payload,
        )
        async with self._lock:
            self.deliveries.append(delivery)

        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event.value,
            "X-Webhook-ID": webhook.id,
        }

        if webhook.secret:
            signature = self._generate_signature(payload_json, webhook.secret)
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        attempts = webhook.retry_count or self._default_retry_count
        timeout = webhook.timeout or self._default_timeout

        for attempt in range(attempts):
            delivery.attempts = attempt + 1
            delivery.last_attempt = datetime.now().isoformat()
            if attempt > 0:
                delivery.status = WebhookDeliveryStatus.RETRYING

            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        webhook.url,
                        content=payload_json,
                        headers=headers,
                    )
                delivery.response_code = response.status_code

                if 200 <= response.status_code < 300:
                    delivery.status = WebhookDeliveryStatus.DELIVERED
                    logger.info(
                        "Delivered webhook %s for %s (attempt %d)",
                        webhook.id,
                        event.value,
                        attempt + 1,
                    )
                    return

                delivery.error = f"HTTP {response.status_code}"
                logger.warning(
                    "Webhook %s failed with status %s",
                    webhook.id,
                    response.status_code,
                )
            except httpx.HTTPError as exc:
                delivery.error = str(exc)
                logger.error("Webhook %s delivery error: %s", webhook.id, exc)

            if attempt < attempts - 1:
                backoff = min(2 ** attempt, 30)
                await asyncio.sleep(backoff)

        delivery.status = WebhookDeliveryStatus.FAILED

    async def trigger_event(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        *,
        async_delivery: bool = True,
    ) -> None:
        matching = [wh for wh in self.webhooks.values() if wh.active and event in wh.events]
        if not matching:
            return

        for webhook in matching:
            if async_delivery:
                asyncio.create_task(self._deliver_with_semaphore(webhook, event, payload))
            else:
                await self._deliver_with_semaphore(webhook, event, payload)

    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[WebhookDelivery]:
        deliveries = list(self.deliveries)

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        if status:
            deliveries = [d for d in deliveries if d.status.value == status]

        return sorted(deliveries, key=lambda d: d.created_at, reverse=True)


__all__ = [
    "WebhookEvent",
    "WebhookDeliveryStatus",
    "Webhook",
    "WebhookDelivery",
    "WebhookManager",
]

