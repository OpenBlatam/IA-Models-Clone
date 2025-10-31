"""
Gamma App - Webhook Service
Advanced webhook management and delivery service
"""

import asyncio
import json
import hashlib
import hmac
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis
from datetime import datetime, timedelta
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class WebhookEvent(Enum):
    """Webhook event types"""
    CONTENT_CREATED = "content.created"
    CONTENT_UPDATED = "content.updated"
    CONTENT_DELETED = "content.deleted"
    CONTENT_EXPORTED = "content.exported"
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    COLLABORATION_STARTED = "collaboration.started"
    COLLABORATION_ENDED = "collaboration.ended"
    EXPORT_COMPLETED = "export.completed"
    EXPORT_FAILED = "export.failed"
    SYSTEM_ALERT = "system.alert"
    PAYMENT_COMPLETED = "payment.completed"
    PAYMENT_FAILED = "payment.failed"

class WebhookStatus(Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"

@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: str
    active: bool = True
    retry_count: int = 3
    timeout: int = 30
    headers: Dict[str, str] = None
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    id: str
    endpoint_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    status: WebhookStatus
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    delivered_at: Optional[datetime] = None
    created_at: datetime = None
    next_retry_at: Optional[datetime] = None

class WebhookService:
    """Advanced webhook service"""
    
    def __init__(self, redis_client: redis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.endpoints: Dict[str, WebhookEndpoint] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.delivery_queue = asyncio.Queue()
        self.retry_delays = [1, 5, 15, 60, 300]  # Exponential backoff
        
        # Load endpoints
        self._load_endpoints()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _load_endpoints(self):
        """Load webhook endpoints from configuration"""
        # This would load from database or configuration
        # For now, create some default endpoints
        default_endpoints = [
            WebhookEndpoint(
                id="default_webhook",
                url="https://webhook.site/your-webhook-url",
                events=[WebhookEvent.CONTENT_CREATED, WebhookEvent.CONTENT_UPDATED],
                secret="your-webhook-secret",
                active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        for endpoint in default_endpoints:
            self.endpoints[endpoint.id] = endpoint
    
    async def create_endpoint(
        self,
        url: str,
        events: List[WebhookEvent],
        secret: str,
        headers: Optional[Dict[str, str]] = None
    ) -> WebhookEndpoint:
        """Create a new webhook endpoint"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid webhook URL")
            
            # Generate endpoint ID
            endpoint_id = hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()
            
            # Create endpoint
            endpoint = WebhookEndpoint(
                id=endpoint_id,
                url=url,
                events=events,
                secret=secret,
                headers=headers or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Store endpoint
            self.endpoints[endpoint_id] = endpoint
            await self._store_endpoint(endpoint)
            
            logger.info(f"Created webhook endpoint: {endpoint_id}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error creating webhook endpoint: {e}")
            raise
    
    async def update_endpoint(
        self,
        endpoint_id: str,
        **updates
    ) -> Optional[WebhookEndpoint]:
        """Update webhook endpoint"""
        try:
            if endpoint_id not in self.endpoints:
                return None
            
            endpoint = self.endpoints[endpoint_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(endpoint, key):
                    setattr(endpoint, key, value)
            
            endpoint.updated_at = datetime.now()
            
            # Store updated endpoint
            await self._store_endpoint(endpoint)
            
            logger.info(f"Updated webhook endpoint: {endpoint_id}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error updating webhook endpoint: {e}")
            return None
    
    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete webhook endpoint"""
        try:
            if endpoint_id not in self.endpoints:
                return False
            
            # Remove from memory
            del self.endpoints[endpoint_id]
            
            # Remove from storage
            await self._delete_endpoint(endpoint_id)
            
            logger.info(f"Deleted webhook endpoint: {endpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting webhook endpoint: {e}")
            return False
    
    async def trigger_webhook(
        self,
        event: WebhookEvent,
        payload: Dict[str, Any],
        endpoint_id: Optional[str] = None
    ) -> List[WebhookDelivery]:
        """Trigger webhook for event"""
        try:
            deliveries = []
            
            # Find endpoints that should receive this event
            target_endpoints = []
            if endpoint_id:
                if endpoint_id in self.endpoints:
                    target_endpoints = [self.endpoints[endpoint_id]]
            else:
                target_endpoints = [
                    endpoint for endpoint in self.endpoints.values()
                    if event in endpoint.events and endpoint.active
                ]
            
            # Create deliveries for each endpoint
            for endpoint in target_endpoints:
                delivery = WebhookDelivery(
                    id=hashlib.md5(f"{endpoint.id}{event.value}{time.time()}".encode()).hexdigest(),
                    endpoint_id=endpoint.id,
                    event=event,
                    payload=payload,
                    status=WebhookStatus.PENDING,
                    created_at=datetime.now()
                )
                
                deliveries.append(delivery)
                
                # Store delivery
                await self._store_delivery(delivery)
                
                # Add to delivery queue
                await self.delivery_queue.put(delivery)
            
            logger.info(f"Triggered {len(deliveries)} webhook deliveries for event: {event.value}")
            return deliveries
            
        except Exception as e:
            logger.error(f"Error triggering webhook: {e}")
            return []
    
    async def deliver_webhook(self, delivery: WebhookDelivery) -> bool:
        """Deliver webhook to endpoint"""
        try:
            endpoint = self.endpoints.get(delivery.endpoint_id)
            if not endpoint:
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = "Endpoint not found"
                await self._update_delivery(delivery)
                return False
            
            # Prepare payload
            webhook_payload = {
                "event": delivery.event.value,
                "data": delivery.payload,
                "timestamp": delivery.created_at.isoformat(),
                "delivery_id": delivery.id
            }
            
            # Create signature
            signature = self._create_signature(
                json.dumps(webhook_payload, default=str),
                endpoint.secret
            )
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-Event": delivery.event.value,
                "X-Webhook-Delivery": delivery.id,
                "User-Agent": "Gamma-App-Webhook/1.0"
            }
            
            # Add custom headers
            if endpoint.headers:
                headers.update(endpoint.headers)
            
            # Make request
            async with self.session.post(
                endpoint.url,
                json=webhook_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            ) as response:
                delivery.response_code = response.status
                delivery.response_body = await response.text()
                
                if response.status >= 200 and response.status < 300:
                    delivery.status = WebhookStatus.DELIVERED
                    delivery.delivered_at = datetime.now()
                    logger.info(f"Webhook delivered successfully: {delivery.id}")
                else:
                    delivery.status = WebhookStatus.FAILED
                    delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"
                    logger.warning(f"Webhook delivery failed: {delivery.id} - {delivery.error_message}")
            
            # Update delivery record
            await self._update_delivery(delivery)
            
            return delivery.status == WebhookStatus.DELIVERED
            
        except asyncio.TimeoutError:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = "Request timeout"
            await self._update_delivery(delivery)
            logger.error(f"Webhook delivery timeout: {delivery.id}")
            return False
        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            await self._update_delivery(delivery)
            logger.error(f"Webhook delivery error: {delivery.id} - {e}")
            return False
    
    async def retry_failed_deliveries(self) -> int:
        """Retry failed webhook deliveries"""
        try:
            retry_count = 0
            
            # Get failed deliveries that are ready for retry
            failed_deliveries = await self._get_failed_deliveries()
            
            for delivery in failed_deliveries:
                if delivery.retry_count < delivery.max_retries:
                    # Calculate next retry time
                    delay = self.retry_delays[min(delivery.retry_count, len(self.retry_delays) - 1)]
                    next_retry = delivery.created_at + timedelta(seconds=delay)
                    
                    if datetime.now() >= next_retry:
                        delivery.retry_count += 1
                        delivery.status = WebhookStatus.RETRYING
                        delivery.next_retry_at = None
                        
                        # Update delivery record
                        await self._update_delivery(delivery)
                        
                        # Retry delivery
                        success = await self.deliver_webhook(delivery)
                        if success:
                            retry_count += 1
                        else:
                            # Schedule next retry
                            if delivery.retry_count < delivery.max_retries:
                                delay = self.retry_delays[min(delivery.retry_count, len(self.retry_delays) - 1)]
                                delivery.next_retry_at = datetime.now() + timedelta(seconds=delay)
                                await self._update_delivery(delivery)
            
            return retry_count
            
        except Exception as e:
            logger.error(f"Error retrying failed deliveries: {e}")
            return 0
    
    async def process_delivery_queue(self):
        """Process webhook delivery queue"""
        try:
            while True:
                try:
                    # Get delivery from queue
                    delivery = await asyncio.wait_for(
                        self.delivery_queue.get(), timeout=1.0
                    )
                    
                    # Deliver webhook
                    await self.deliver_webhook(delivery)
                    
                    # Mark task as done
                    self.delivery_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No deliveries in queue, continue
                    continue
                except Exception as e:
                    logger.error(f"Error processing delivery queue: {e}")
                    
        except Exception as e:
            logger.error(f"Error in delivery queue processor: {e}")
    
    def _create_signature(self, payload: str, secret: str) -> str:
        """Create webhook signature"""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        try:
            expected_signature = self._create_signature(payload, secret)
            return hmac.compare_digest(signature, expected_signature)
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    async def _store_endpoint(self, endpoint: WebhookEndpoint):
        """Store webhook endpoint"""
        try:
            key = f"webhook_endpoint:{endpoint.id}"
            data = asdict(endpoint)
            self.redis.setex(key, 86400 * 30, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Error storing endpoint: {e}")
    
    async def _delete_endpoint(self, endpoint_id: str):
        """Delete webhook endpoint"""
        try:
            key = f"webhook_endpoint:{endpoint_id}"
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")
    
    async def _store_delivery(self, delivery: WebhookDelivery):
        """Store webhook delivery"""
        try:
            key = f"webhook_delivery:{delivery.id}"
            data = asdict(delivery)
            self.redis.setex(key, 86400 * 7, json.dumps(data, default=str))
            
            # Add to deliveries list
            self.redis.lpush("webhook_deliveries", delivery.id)
            self.redis.ltrim("webhook_deliveries", 0, 9999)  # Keep last 10000
        except Exception as e:
            logger.error(f"Error storing delivery: {e}")
    
    async def _update_delivery(self, delivery: WebhookDelivery):
        """Update webhook delivery"""
        try:
            key = f"webhook_delivery:{delivery.id}"
            data = asdict(delivery)
            self.redis.setex(key, 86400 * 7, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Error updating delivery: {e}")
    
    async def _get_failed_deliveries(self) -> List[WebhookDelivery]:
        """Get failed deliveries ready for retry"""
        try:
            deliveries = []
            
            # Get all delivery IDs
            delivery_ids = self.redis.lrange("webhook_deliveries", 0, 99)
            
            for delivery_id in delivery_ids:
                key = f"webhook_delivery:{delivery_id.decode()}"
                data = self.redis.get(key)
                
                if data:
                    delivery_data = json.loads(data)
                    delivery = WebhookDelivery(**delivery_data)
                    
                    if delivery.status == WebhookStatus.FAILED:
                        deliveries.append(delivery)
            
            return deliveries
            
        except Exception as e:
            logger.error(f"Error getting failed deliveries: {e}")
            return []
    
    async def get_delivery_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        try:
            stats = {
                "total_deliveries": 0,
                "delivered": 0,
                "failed": 0,
                "pending": 0,
                "retrying": 0,
                "by_event": {},
                "by_endpoint": {}
            }
            
            # Get all delivery IDs
            delivery_ids = self.redis.lrange("webhook_deliveries", 0, 999)
            
            for delivery_id in delivery_ids:
                key = f"webhook_delivery:{delivery_id.decode()}"
                data = self.redis.get(key)
                
                if data:
                    delivery_data = json.loads(data)
                    delivery = WebhookDelivery(**delivery_data)
                    
                    stats["total_deliveries"] += 1
                    
                    # Count by status
                    if delivery.status == WebhookStatus.DELIVERED:
                        stats["delivered"] += 1
                    elif delivery.status == WebhookStatus.FAILED:
                        stats["failed"] += 1
                    elif delivery.status == WebhookStatus.PENDING:
                        stats["pending"] += 1
                    elif delivery.status == WebhookStatus.RETRYING:
                        stats["retrying"] += 1
                    
                    # Count by event
                    event = delivery.event.value
                    stats["by_event"][event] = stats["by_event"].get(event, 0) + 1
                    
                    # Count by endpoint
                    endpoint_id = delivery.endpoint_id
                    stats["by_endpoint"][endpoint_id] = stats["by_endpoint"].get(endpoint_id, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting delivery stats: {e}")
            return {}
    
    async def get_endpoint_stats(self, endpoint_id: str) -> Dict[str, Any]:
        """Get endpoint statistics"""
        try:
            stats = {
                "total_deliveries": 0,
                "delivered": 0,
                "failed": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0,
                "last_delivery": None
            }
            
            # Get deliveries for this endpoint
            delivery_ids = self.redis.lrange("webhook_deliveries", 0, 999)
            
            response_times = []
            last_delivery = None
            
            for delivery_id in delivery_ids:
                key = f"webhook_delivery:{delivery_id.decode()}"
                data = self.redis.get(key)
                
                if data:
                    delivery_data = json.loads(data)
                    delivery = WebhookDelivery(**delivery_data)
                    
                    if delivery.endpoint_id == endpoint_id:
                        stats["total_deliveries"] += 1
                        
                        if delivery.status == WebhookStatus.DELIVERED:
                            stats["delivered"] += 1
                        elif delivery.status == WebhookStatus.FAILED:
                            stats["failed"] += 1
                        
                        # Track response time
                        if delivery.delivered_at:
                            response_time = (delivery.delivered_at - delivery.created_at).total_seconds()
                            response_times.append(response_time)
                            
                            if not last_delivery or delivery.delivered_at > last_delivery:
                                last_delivery = delivery.delivered_at
            
            # Calculate success rate
            if stats["total_deliveries"] > 0:
                stats["success_rate"] = stats["delivered"] / stats["total_deliveries"]
            
            # Calculate average response time
            if response_times:
                stats["average_response_time"] = sum(response_times) / len(response_times)
            
            stats["last_delivery"] = last_delivery.isoformat() if last_delivery else None
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting endpoint stats: {e}")
            return {}
    
    async def test_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Test webhook endpoint"""
        try:
            endpoint = self.endpoints.get(endpoint_id)
            if not endpoint:
                return {"success": False, "error": "Endpoint not found"}
            
            # Create test payload
            test_payload = {
                "event": "test",
                "data": {"message": "This is a test webhook"},
                "timestamp": datetime.now().isoformat(),
                "delivery_id": "test"
            }
            
            # Create signature
            signature = self._create_signature(
                json.dumps(test_payload, default=str),
                endpoint.secret
            )
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-Event": "test",
                "X-Webhook-Delivery": "test",
                "User-Agent": "Gamma-App-Webhook/1.0"
            }
            
            # Make test request
            start_time = time.time()
            async with self.session.post(
                endpoint.url,
                json=test_payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
            ) as response:
                response_time = time.time() - start_time
                response_body = await response.text()
                
                return {
                    "success": response.status >= 200 and response.status < 300,
                    "status_code": response.status,
                    "response_time": response_time,
                    "response_body": response_body,
                    "headers": dict(response.headers)
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

























