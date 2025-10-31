#!/usr/bin/env python3
"""
Webhook Manager for Enhanced HeyGen AI
Handles real-time notifications for video generation tasks and system events.
"""

import asyncio
import json
import time
import hashlib
import hmac
from typing import Any, Dict, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import aiohttp
from datetime import datetime, timedelta
import ssl

logger = structlog.get_logger()

class WebhookEventType(Enum):
    """Types of webhook events."""
    VIDEO_COMPLETED = "video.completed"
    VIDEO_FAILED = "video.failed"
    VOICE_GENERATED = "voice.generated"
    AVATAR_GENERATED = "avatar.generated"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    SYSTEM_HEALTH = "system.health"
    MODEL_LOADED = "model.loaded"
    CACHE_UPDATED = "cache.updated"

class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class WebhookEndpoint:
    """Represents a webhook endpoint configuration."""
    id: str
    url: str
    secret: Optional[str]
    events: List[WebhookEventType]
    is_active: bool
    retry_count: int
    max_retries: int
    timeout_seconds: int
    created_at: float
    last_sent: Optional[float]
    success_count: int
    failure_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def should_retry(self) -> bool:
        """Check if webhook should be retried."""
        return self.retry_count < self.max_retries

@dataclass
class WebhookEvent:
    """Represents a webhook event."""
    id: str
    type: WebhookEventType
    data: Dict[str, Any]
    timestamp: float
    source: str
    version: str

@dataclass
class WebhookDelivery:
    """Represents a webhook delivery attempt."""
    id: str
    endpoint_id: str
    event_id: str
    status: WebhookStatus
    attempt_count: int
    sent_at: Optional[float]
    response_code: Optional[int]
    response_body: Optional[str]
    error_message: Optional[str]
    retry_after: Optional[float]

class WebhookManager:
    """Manages webhook endpoints and event delivery."""
    
    def __init__(
        self,
        max_concurrent_deliveries: int = 10,
        default_timeout: int = 30,
        retry_delay_base: float = 1.0,
        max_retry_delay: float = 300.0
    ):
        self.max_concurrent_deliveries = max_concurrent_deliveries
        self.default_timeout = default_timeout
        self.retry_delay_base = retry_delay_base
        self.max_retry_delay = max_retry_delay
        
        # Storage
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.events: Dict[str, WebhookEvent] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        
        # Delivery management
        self.delivery_semaphore = asyncio.Semaphore(max_concurrent_deliveries)
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        
        # HTTP session
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Event handlers
        self.event_handlers: Dict[WebhookEventType, List[Callable]] = {}
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "active_endpoints": 0
        }
    
    async def start(self):
        """Start the webhook manager."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=self.default_timeout)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=20,
            ssl=ssl.create_default_context()
        )
        
        self.http_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
        
        # Start delivery workers
        asyncio.create_task(self._delivery_worker())
        
        logger.info("Webhook manager started")
    
    async def stop(self):
        """Stop the webhook manager."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
        
        logger.info("Webhook manager stopped")
    
    def register_endpoint(
        self,
        url: str,
        events: List[WebhookEventType],
        secret: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: Optional[int] = None
    ) -> str:
        """Register a new webhook endpoint."""
        endpoint_id = str(hashlib.md5(url.encode()).hexdigest())
        
        endpoint = WebhookEndpoint(
            id=endpoint_id,
            url=url,
            secret=secret,
            events=events,
            is_active=True,
            retry_count=0,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds or self.default_timeout,
            created_at=time.time(),
            last_sent=None,
            success_count=0,
            failure_count=0
        )
        
        self.endpoints[endpoint_id] = endpoint
        self.stats["active_endpoints"] = len([e for e in self.endpoints.values() if e.is_active])
        
        logger.info(f"Webhook endpoint registered", 
                   endpoint_id=endpoint_id,
                   url=url,
                   events=[e.value for e in events])
        
        return endpoint_id
    
    def unregister_endpoint(self, endpoint_id: str) -> bool:
        """Unregister a webhook endpoint."""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            self.stats["active_endpoints"] = len([e for e in self.endpoints.values() if e.is_active])
            logger.info(f"Webhook endpoint unregistered", endpoint_id=endpoint_id)
            return True
        return False
    
    def update_endpoint(
        self,
        endpoint_id: str,
        events: Optional[List[WebhookEventType]] = None,
        secret: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """Update webhook endpoint configuration."""
        if endpoint_id not in self.endpoints:
            return False
        
        endpoint = self.endpoints[endpoint_id]
        
        if events is not None:
            endpoint.events = events
        if secret is not None:
            endpoint.secret = secret
        if is_active is not None:
            endpoint.is_active = is_active
        
        self.stats["active_endpoints"] = len([e for e in self.endpoints.values() if e.is_active])
        
        logger.info(f"Webhook endpoint updated", endpoint_id=endpoint_id)
        return True
    
    async def send_event(
        self,
        event_type: WebhookEventType,
        data: Dict[str, Any],
        source: str = "heygen_ai",
        version: str = "2.0.0"
    ) -> str:
        """Send a webhook event to all registered endpoints."""
        # Create event
        event = WebhookEvent(
            id=str(hashlib.md5(f"{event_type.value}:{time.time()}".encode()).hexdigest()),
            type=event_type,
            data=data,
            timestamp=time.time(),
            source=source,
            version=version
        )
        
        self.events[event.id] = event
        self.stats["total_events"] += 1
        
        # Find endpoints that should receive this event
        target_endpoints = [
            endpoint for endpoint in self.endpoints.values()
            if endpoint.is_active and event_type in endpoint.events
        ]
        
        # Create deliveries for each endpoint
        for endpoint in target_endpoints:
            delivery = WebhookDelivery(
                id=str(hashlib.md5(f"{event.id}:{endpoint.id}".encode()).hexdigest()),
                endpoint_id=endpoint.id,
                event_id=event.id,
                status=WebhookStatus.PENDING,
                attempt_count=0,
                sent_at=None,
                response_code=None,
                response_body=None,
                error_message=None,
                retry_after=None
            )
            
            self.deliveries[delivery.id] = delivery
            self.stats["total_deliveries"] += 1
            
            # Queue for delivery
            await self.delivery_queue.put(delivery)
        
        logger.info(f"Webhook event queued", 
                   event_id=event.id,
                   type=event_type.value,
                   target_endpoints=len(target_endpoints))
        
        return event.id
    
    async def _delivery_worker(self):
        """Background worker for processing webhook deliveries."""
        while self.is_running:
            try:
                # Get delivery from queue
                try:
                    delivery = await asyncio.wait_for(
                        self.delivery_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process delivery
                await self._process_delivery(delivery)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Delivery worker error: {e}")
                await asyncio.sleep(1)
    
    async def _process_delivery(self, delivery: WebhookDelivery):
        """Process a single webhook delivery."""
        async with self.delivery_semaphore:
            try:
                # Get endpoint and event
                endpoint = self.endpoints.get(delivery.endpoint_id)
                event = self.events.get(delivery.event_id)
                
                if not endpoint or not event:
                    delivery.status = WebhookStatus.FAILED
                    delivery.error_message = "Endpoint or event not found"
                    return
                
                # Prepare payload
                payload = {
                    "event": {
                        "id": event.id,
                        "type": event.type.value,
                        "timestamp": event.timestamp,
                        "source": event.source,
                        "version": event.version
                    },
                    "data": event.data
                }
                
                # Add signature if secret is configured
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "HeyGen-AI-Webhooks/2.0.0",
                    "X-Webhook-ID": delivery.id,
                    "X-Event-Type": event.type.value
                }
                
                if endpoint.secret:
                    signature = self._generate_signature(payload, endpoint.secret)
                    headers["X-Webhook-Signature"] = signature
                
                # Send webhook
                success = await self._send_webhook(
                    delivery, endpoint, payload, headers
                )
                
                if success:
                    delivery.status = WebhookStatus.SENT
                    delivery.sent_at = time.time()
                    endpoint.last_sent = time.time()
                    endpoint.success_count += 1
                    self.stats["successful_deliveries"] += 1
                else:
                    await self._handle_delivery_failure(delivery, endpoint)
                
            except Exception as e:
                logger.error(f"Delivery processing failed: {e}")
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = str(e)
                await self._handle_delivery_failure(delivery, endpoint)
    
    async def _send_webhook(
        self,
        delivery: WebhookDelivery,
        endpoint: WebhookEndpoint,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> bool:
        """Send webhook to endpoint."""
        try:
            delivery.attempt_count += 1
            
            async with self.http_session.post(
                endpoint.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout_seconds)
            ) as response:
                
                delivery.response_code = response.status
                delivery.response_body = await response.text()
                
                if 200 <= response.status < 300:
                    return True
                else:
                    delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"
                    return False
                    
        except asyncio.TimeoutError:
            delivery.error_message = "Request timeout"
            return False
        except Exception as e:
            delivery.error_message = str(e)
            return False
    
    async def _handle_delivery_failure(self, delivery: WebhookDelivery, endpoint: WebhookEndpoint):
        """Handle webhook delivery failure."""
        endpoint.failure_count += 1
        self.stats["failed_deliveries"] += 1
        
        if delivery.attempt_count < endpoint.max_retries:
            # Schedule retry
            delivery.status = WebhookStatus.RETRYING
            delay = min(
                self.retry_delay_base * (2 ** delivery.attempt_count),
                self.max_retry_delay
            )
            delivery.retry_after = time.time() + delay
            
            # Re-queue for retry
            asyncio.create_task(self._schedule_retry(delivery, delay))
            
            logger.warning(f"Webhook delivery failed, scheduling retry", 
                          delivery_id=delivery.id,
                          endpoint_id=endpoint.id,
                          attempt=delivery.attempt_count,
                          retry_after=delay)
        else:
            # Max retries exceeded
            delivery.status = WebhookStatus.FAILED
            logger.error(f"Webhook delivery failed permanently", 
                        delivery_id=delivery.id,
                        endpoint_id=endpoint.id,
                        max_retries=endpoint.max_retries)
    
    async def _schedule_retry(self, delivery: WebhookDelivery, delay: float):
        """Schedule a webhook delivery retry."""
        await asyncio.sleep(delay)
        
        if not self.is_running:
            return
        
        # Reset delivery for retry
        delivery.status = WebhookStatus.PENDING
        delivery.retry_after = None
        
        # Re-queue for delivery
        await self.delivery_queue.put(delivery)
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def get_endpoints(self) -> List[Dict[str, Any]]:
        """Get all registered webhook endpoints."""
        return [endpoint.to_dict() for endpoint in self.endpoints.values()]
    
    def get_endpoint(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific webhook endpoint."""
        endpoint = self.endpoints.get(endpoint_id)
        return endpoint.to_dict() if endpoint else None
    
    def get_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent webhook events."""
        sorted_events = sorted(
            self.events.values(),
            key=lambda e: e.timestamp,
            reverse=True
        )
        return [event.__dict__ for event in sorted_events[:limit]]
    
    def get_deliveries(self, endpoint_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get webhook deliveries."""
        deliveries = list(self.deliveries.values())
        
        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]
        
        sorted_deliveries = sorted(
            deliveries,
            key=lambda d: d.sent_at or 0,
            reverse=True
        )
        
        return [delivery.__dict__ for delivery in sorted_deliveries[:limit]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get webhook manager statistics."""
        return {
            **self.stats,
            "endpoints": len(self.endpoints),
            "events": len(self.events),
            "deliveries": len(self.deliveries),
            "pending_deliveries": self.delivery_queue.qsize(),
            "is_running": self.is_running
        }
    
    def register_event_handler(self, event_type: WebhookEventType, handler: Callable):
        """Register an event handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
        logger.info(f"Event handler registered for {event_type.value}")
    
    async def trigger_event_handlers(self, event_type: WebhookEventType, data: Dict[str, Any]):
        """Trigger registered event handlers."""
        handlers = self.event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type.value}: {e}")

# Global webhook manager instance
webhook_manager: Optional[WebhookManager] = None

def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance."""
    global webhook_manager
    if webhook_manager is None:
        webhook_manager = WebhookManager()
    return webhook_manager

async def shutdown_webhook_manager():
    """Shutdown global webhook manager."""
    global webhook_manager
    if webhook_manager:
        await webhook_manager.stop()
        webhook_manager = None

