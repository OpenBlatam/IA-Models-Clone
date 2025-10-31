"""
Enhanced Webhook Delivery System
Optimized delivery with intelligent retries, batching, and queue management
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import httpx

from .models import WebhookEvent, WebhookPayload, WebhookEndpoint, WebhookDelivery

logger = logging.getLogger(__name__)


class DeliveryStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"


@dataclass
class DeliveryMetrics:
    """Delivery metrics for monitoring"""
    total_attempts: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    total_latency: float = 0.0
    last_delivery_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_deliveries / self.total_attempts) * 100
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency"""
        if self.successful_deliveries == 0:
            return 0.0
        return self.total_latency / self.successful_deliveries


class RetryStrategy:
    """Intelligent retry strategy"""
    
    @staticmethod
    def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Exponential backoff with jitter"""
        import random
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0, delay * 0.1)  # 10% jitter
        return delay + jitter
    
    @staticmethod
    def linear_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
        """Linear backoff"""
        return min(base_delay * attempt, max_delay)
    
    @staticmethod
    def should_retry(status_code: int, attempt: int, max_attempts: int = 3) -> bool:
        """Determine if should retry based on status code"""
        if attempt >= max_attempts:
            return False
        
        # Retry on transient errors
        retryable_status_codes = [408, 429, 500, 502, 503, 504]
        return status_code in retryable_status_codes or status_code == 0  # Network error


class WebhookQueue:
    """Queue for webhook delivery with batching and prioritization"""
    
    def __init__(self, max_batch_size: int = 10, batch_timeout: float = 0.5):
        self.queue: deque = deque()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.processing = False
        self.metrics = DeliveryMetrics()
    
    async def add(self, delivery: WebhookDelivery):
        """Add webhook to queue"""
        self.queue.append(delivery)
        logger.debug(f"Added webhook to queue: {delivery.id}")
    
    async def get_batch(self, max_items: Optional[int] = None) -> List[WebhookDelivery]:
        """Get batch of webhooks from queue"""
        if not self.queue:
            return []
        
        batch_size = min(max_items or self.max_batch_size, len(self.queue))
        batch = []
        
        for _ in range(batch_size):
            if self.queue:
                batch.append(self.queue.popleft())
        
        return batch
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)


class EnhancedWebhookDelivery:
    """Enhanced webhook delivery system with intelligent retries and batching"""
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        batch_size: int = 10,
        batch_timeout: float = 0.5
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.queue = WebhookQueue(max_batch_size=batch_size, batch_timeout=batch_timeout)
        self.metrics = DeliveryMetrics()
        self.active_deliveries: Dict[str, WebhookDelivery] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.processing_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize delivery system"""
        self.http_client = httpx.AsyncClient(timeout=self.timeout)
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info("Enhanced webhook delivery system initialized")
    
    async def shutdown(self):
        """Shutdown delivery system"""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        if self.http_client:
            await self.http_client.aclose()
        
        logger.info("Enhanced webhook delivery system shutdown")
    
    async def deliver(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload,
        delivery_id: str
    ) -> Dict[str, Any]:
        """
        Deliver webhook with intelligent retries
        
        Returns:
            Delivery result with status and metrics
        """
        delivery = WebhookDelivery(
            id=delivery_id,
            endpoint_id=endpoint.id,
            event=payload.event,
            payload=payload,
            status=DeliveryStatus.PENDING.value
        )
        
        # Add to queue for batch processing
        await self.queue.add(delivery)
        self.active_deliveries[delivery_id] = delivery
        
        return {
            "delivery_id": delivery_id,
            "status": "queued",
            "queue_size": self.queue.size()
        }
    
    async def _process_queue(self):
        """Background task to process webhook queue"""
        logger.info("Webhook queue processor started")
        
        while True:
            try:
                # Get batch of webhooks
                batch = await self.queue.get_batch()
                
                if batch:
                    # Process batch concurrently
                    tasks = [self._deliver_webhook(delivery) for delivery in batch]
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    # No webhooks, wait a bit
                    await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing webhook queue: {e}", exc_info=True)
                await asyncio.sleep(1)
    
    async def _deliver_webhook(self, delivery: WebhookDelivery) -> None:
        """Deliver a single webhook with retries"""
        endpoint = None  # Would be retrieved from storage
        attempt = 0
        
        while attempt < self.max_retries:
            attempt += 1
            delivery.attempts = attempt
            delivery.status = DeliveryStatus.DELIVERING.value
            delivery.last_attempt = time.time()
            
            start_time = time.time()
            
            try:
                # Make HTTP request
                response = await self._make_request(endpoint, delivery.payload)
                
                latency = time.time() - start_time
                self.metrics.total_attempts += 1
                self.metrics.total_latency += latency
                
                if response.status_code == 200:
                    # Success
                    delivery.status = DeliveryStatus.DELIVERED.value
                    self.metrics.successful_deliveries += 1
                    self.metrics.last_delivery_time = time.time()
                    
                    logger.info(
                        f"Webhook delivered: {delivery.id} - {latency:.3f}s",
                        extra={"delivery_id": delivery.id, "attempt": attempt}
                    )
                    return
                else:
                    # Check if should retry
                    if RetryStrategy.should_retry(response.status_code, attempt, self.max_retries):
                        retry_delay = RetryStrategy.exponential_backoff(attempt - 1)
                        delivery.status = DeliveryStatus.RETRYING.value
                        delivery.next_retry = time.time() + retry_delay
                        
                        logger.warning(
                            f"Webhook delivery failed, retrying: {delivery.id} - "
                            f"Status: {response.status_code}, Attempt: {attempt}",
                            extra={
                                "delivery_id": delivery.id,
                                "status_code": response.status_code,
                                "attempt": attempt
                            }
                        )
                        
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # Final failure
                        delivery.status = DeliveryStatus.FAILED.value
                        self.metrics.failed_deliveries += 1
                        
                        logger.error(
                            f"Webhook delivery failed permanently: {delivery.id} - "
                            f"Status: {response.status_code}",
                            extra={
                                "delivery_id": delivery.id,
                                "status_code": response.status_code,
                                "attempt": attempt
                            }
                        )
                        return
            
            except Exception as e:
                latency = time.time() - start_time
                self.metrics.total_attempts += 1
                
                logger.error(
                    f"Webhook delivery error: {delivery.id} - {str(e)}",
                    exc_info=True,
                    extra={"delivery_id": delivery.id, "attempt": attempt}
                )
                
                # Check if should retry on exception
                if attempt < self.max_retries:
                    retry_delay = RetryStrategy.exponential_backoff(attempt - 1)
                    delivery.status = DeliveryStatus.RETRYING.value
                    delivery.next_retry = time.time() + retry_delay
                    
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    delivery.status = DeliveryStatus.FAILED.value
                    self.metrics.failed_deliveries += 1
                    return
    
    async def _make_request(
        self,
        endpoint: WebhookEndpoint,
        payload: WebhookPayload
    ) -> httpx.Response:
        """Make HTTP request to webhook endpoint"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ContentRedundancyDetector/2.0"
        }
        
        if endpoint.secret:
            # Add signature header (simplified)
            import hmac
            import hashlib
            import json as json_lib
            
            payload_json = json_lib.dumps(payload.__dict__, default=str)
            signature = hmac.new(
                endpoint.secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"
        
        response = await self.http_client.post(
            endpoint.url,
            json=payload.__dict__,
            headers=headers
        )
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get delivery metrics"""
        return {
            "total_attempts": self.metrics.total_attempts,
            "successful_deliveries": self.metrics.successful_deliveries,
            "failed_deliveries": self.metrics.failed_deliveries,
            "success_rate": round(self.metrics.success_rate, 2),
            "average_latency": round(self.metrics.average_latency, 3),
            "queue_size": self.queue.size(),
            "active_deliveries": len(self.active_deliveries),
            "last_delivery_time": self.metrics.last_delivery_time
        }






