"""
BUL Webhook System
=================

Real-time webhook notifications for document generation events.
"""

import asyncio
import json
import hashlib
import hmac
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import httpx
import aiohttp

from ..utils import get_logger, get_cache_manager
from ..config import get_config
from ..security import get_encryption

logger = get_logger(__name__)

class WebhookEventType(str, Enum):
    """Webhook event types"""
    DOCUMENT_GENERATED = "document.generated"
    DOCUMENT_FAILED = "document.failed"
    DOCUMENT_STARTED = "document.started"
    DOCUMENT_COMPLETED = "document.completed"
    AGENT_SELECTED = "agent.selected"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    SYSTEM_HEALTH_CHANGED = "system.health_changed"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"
    ERROR_OCCURRED = "error.occurred"

class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"

@dataclass
class WebhookEvent:
    """Webhook event data structure"""
    id: str
    event_type: WebhookEventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    version: str = "1.0"

@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    id: str
    webhook_id: str
    event_id: str
    url: str
    status: WebhookStatus
    attempts: int
    last_attempt: Optional[datetime]
    next_retry: Optional[datetime]
    response_code: Optional[int]
    response_body: Optional[str]
    error_message: Optional[str]
    created_at: datetime

class WebhookSubscription(BaseModel):
    """Webhook subscription model"""
    id: str = Field(..., description="Unique subscription ID")
    url: str = Field(..., description="Webhook URL")
    events: List[WebhookEventType] = Field(..., description="Subscribed events")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
    active: bool = Field(True, description="Subscription status")
    retry_policy: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_attempts": 3,
            "retry_delays": [60, 300, 900],  # 1min, 5min, 15min
            "timeout": 30
        },
        description="Retry policy configuration"
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Event filters")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class WebhookManager:
    """Webhook management system"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.cache_manager = get_cache_manager()
        self.config = get_config()
        self.encryption = get_encryption()
        
        # In-memory storage (in production, use database)
        self.subscriptions: Dict[str, WebhookSubscription] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()
        
        # HTTP clients
        self.http_client: Optional[httpx.AsyncClient] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Background tasks
        self.delivery_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize webhook system"""
        try:
            # Initialize HTTP clients
            timeout = httpx.Timeout(30.0, connect=10.0)
            self.http_client = httpx.AsyncClient(timeout=timeout)
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
            
            # Start background tasks
            self.delivery_task = asyncio.create_task(self._delivery_worker())
            self.cleanup_task = asyncio.create_task(self._cleanup_worker())
            
            self.logger.info("Webhook system initialized successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to initialize webhook system: {e}")
            return False
    
    async def close(self):
        """Close webhook system"""
        try:
            # Cancel background tasks
            if self.delivery_task:
                self.delivery_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Close HTTP clients
            if self.http_client:
                await self.http_client.aclose()
            if self.session:
                await self.session.close()
            
            self.logger.info("Webhook system closed")
        
        except Exception as e:
            self.logger.error(f"Error closing webhook system: {e}")
    
    async def create_subscription(self, subscription: WebhookSubscription) -> WebhookSubscription:
        """Create a new webhook subscription"""
        try:
            # Validate URL
            if not self._validate_url(subscription.url):
                raise ValueError("Invalid webhook URL")
            
            # Generate ID if not provided
            if not subscription.id:
                subscription.id = self._generate_id()
            
            # Encrypt secret if provided
            if subscription.secret:
                subscription.secret = self.encryption.encrypt_data(subscription.secret)
            
            # Store subscription
            self.subscriptions[subscription.id] = subscription
            
            self.logger.info(f"Created webhook subscription: {subscription.id}")
            return subscription
        
        except Exception as e:
            self.logger.error(f"Error creating webhook subscription: {e}")
            raise
    
    async def update_subscription(self, subscription_id: str, updates: Dict[str, Any]) -> WebhookSubscription:
        """Update webhook subscription"""
        try:
            if subscription_id not in self.subscriptions:
                raise ValueError("Subscription not found")
            
            subscription = self.subscriptions[subscription_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(subscription, key):
                    setattr(subscription, key, value)
            
            subscription.updated_at = datetime.now()
            
            # Re-encrypt secret if updated
            if "secret" in updates and subscription.secret:
                subscription.secret = self.encryption.encrypt_data(subscription.secret)
            
            self.logger.info(f"Updated webhook subscription: {subscription_id}")
            return subscription
        
        except Exception as e:
            self.logger.error(f"Error updating webhook subscription: {e}")
            raise
    
    async def delete_subscription(self, subscription_id: str) -> bool:
        """Delete webhook subscription"""
        try:
            if subscription_id not in self.subscriptions:
                return False
            
            del self.subscriptions[subscription_id]
            self.logger.info(f"Deleted webhook subscription: {subscription_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error deleting webhook subscription: {e}")
            return False
    
    async def get_subscription(self, subscription_id: str) -> Optional[WebhookSubscription]:
        """Get webhook subscription"""
        return self.subscriptions.get(subscription_id)
    
    async def list_subscriptions(self) -> List[WebhookSubscription]:
        """List all webhook subscriptions"""
        return list(self.subscriptions.values())
    
    async def trigger_event(self, event_type: WebhookEventType, data: Dict[str, Any], source: str = "system"):
        """Trigger a webhook event"""
        try:
            # Create event
            event = WebhookEvent(
                id=self._generate_id(),
                event_type=event_type,
                timestamp=datetime.now(),
                data=data,
                source=source
            )
            
            # Add to queue
            await self.event_queue.put(event)
            
            self.logger.debug(f"Triggered webhook event: {event_type} from {source}")
        
        except Exception as e:
            self.logger.error(f"Error triggering webhook event: {e}")
    
    async def _delivery_worker(self):
        """Background worker for webhook delivery"""
        while True:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Find matching subscriptions
                matching_subscriptions = [
                    sub for sub in self.subscriptions.values()
                    if sub.active and event.event_type in sub.events
                ]
                
                # Create deliveries for each subscription
                for subscription in matching_subscriptions:
                    await self._create_delivery(event, subscription)
                
                # Mark task as done
                self.event_queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in webhook delivery worker: {e}")
                await asyncio.sleep(1)
    
    async def _create_delivery(self, event: WebhookEvent, subscription: WebhookSubscription):
        """Create webhook delivery"""
        try:
            delivery = WebhookDelivery(
                id=self._generate_id(),
                webhook_id=subscription.id,
                event_id=event.id,
                url=subscription.url,
                status=WebhookStatus.PENDING,
                attempts=0,
                last_attempt=None,
                next_retry=datetime.now(),
                response_code=None,
                response_body=None,
                error_message=None,
                created_at=datetime.now()
            )
            
            self.deliveries[delivery.id] = delivery
            
            # Attempt delivery
            await self._attempt_delivery(delivery, event, subscription)
        
        except Exception as e:
            self.logger.error(f"Error creating webhook delivery: {e}")
    
    async def _attempt_delivery(self, delivery: WebhookDelivery, event: WebhookEvent, subscription: WebhookSubscription):
        """Attempt webhook delivery"""
        try:
            delivery.attempts += 1
            delivery.last_attempt = datetime.now()
            delivery.status = WebhookStatus.RETRYING
            
            # Prepare payload
            payload = {
                "id": event.id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data,
                "source": event.source,
                "version": event.version
            }
            
            # Create signature
            signature = None
            if subscription.secret:
                secret = self.encryption.decrypt_data(subscription.secret)
                signature = self._create_signature(json.dumps(payload), secret)
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "BUL-Webhook/1.0",
                "X-Webhook-Event": event.event_type.value,
                "X-Webhook-Delivery": delivery.id
            }
            
            if signature:
                headers["X-Webhook-Signature"] = f"sha256={signature}"
            
            # Send webhook
            timeout = subscription.retry_policy.get("timeout", 30)
            
            try:
                response = await self.http_client.post(
                    delivery.url,
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                
                delivery.response_code = response.status_code
                delivery.response_body = response.text[:1000]  # Limit response body
                
                if 200 <= response.status_code < 300:
                    delivery.status = WebhookStatus.DELIVERED
                    self.logger.info(f"Webhook delivered successfully: {delivery.id}")
                else:
                    delivery.status = WebhookStatus.FAILED
                    delivery.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
                    await self._schedule_retry(delivery, subscription)
                
            except httpx.TimeoutException:
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = "Request timeout"
                await self._schedule_retry(delivery, subscription)
            
            except httpx.RequestError as e:
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = str(e)
                await self._schedule_retry(delivery, subscription)
        
        except Exception as e:
            self.logger.error(f"Error attempting webhook delivery: {e}")
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            await self._schedule_retry(delivery, subscription)
    
    async def _schedule_retry(self, delivery: WebhookDelivery, subscription: WebhookSubscription):
        """Schedule webhook retry"""
        try:
            max_attempts = subscription.retry_policy.get("max_attempts", 3)
            retry_delays = subscription.retry_policy.get("retry_delays", [60, 300, 900])
            
            if delivery.attempts >= max_attempts:
                delivery.status = WebhookStatus.FAILED
                delivery.next_retry = None
                self.logger.warning(f"Webhook delivery failed after {max_attempts} attempts: {delivery.id}")
                return
            
            # Calculate next retry time
            delay_index = min(delivery.attempts - 1, len(retry_delays) - 1)
            delay_seconds = retry_delays[delay_index]
            delivery.next_retry = datetime.now() + timedelta(seconds=delay_seconds)
            
            self.logger.info(f"Scheduled webhook retry in {delay_seconds}s: {delivery.id}")
        
        except Exception as e:
            self.logger.error(f"Error scheduling webhook retry: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup tasks"""
        while True:
            try:
                # Clean up old deliveries (older than 7 days)
                cutoff_date = datetime.now() - timedelta(days=7)
                old_deliveries = [
                    delivery_id for delivery_id, delivery in self.deliveries.items()
                    if delivery.created_at < cutoff_date
                ]
                
                for delivery_id in old_deliveries:
                    del self.deliveries[delivery_id]
                
                if old_deliveries:
                    self.logger.info(f"Cleaned up {len(old_deliveries)} old webhook deliveries")
                
                # Wait 1 hour before next cleanup
                await asyncio.sleep(3600)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in webhook cleanup worker: {e}")
                await asyncio.sleep(60)
    
    def _validate_url(self, url: str) -> bool:
        """Validate webhook URL"""
        try:
            import re
            url_pattern = re.compile(
                r'^https?://'  # http:// or https://
                r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
                r'localhost|'  # localhost...
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
                r'(?::\d+)?'  # optional port
                r'(?:/?|[/?]\S+)$', re.IGNORECASE)
            return url_pattern.match(url) is not None
        except Exception:
            return False
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _create_signature(self, payload: str, secret: str) -> str:
        """Create webhook signature"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def get_delivery_stats(self) -> Dict[str, Any]:
        """Get webhook delivery statistics"""
        try:
            total_deliveries = len(self.deliveries)
            successful_deliveries = len([d for d in self.deliveries.values() if d.status == WebhookStatus.DELIVERED])
            failed_deliveries = len([d for d in self.deliveries.values() if d.status == WebhookStatus.FAILED])
            pending_deliveries = len([d for d in self.deliveries.values() if d.status == WebhookStatus.PENDING])
            retrying_deliveries = len([d for d in self.deliveries.values() if d.status == WebhookStatus.RETRYING])
            
            success_rate = (successful_deliveries / max(total_deliveries, 1)) * 100
            
            return {
                "total_deliveries": total_deliveries,
                "successful_deliveries": successful_deliveries,
                "failed_deliveries": failed_deliveries,
                "pending_deliveries": pending_deliveries,
                "retrying_deliveries": retrying_deliveries,
                "success_rate": round(success_rate, 2),
                "active_subscriptions": len([s for s in self.subscriptions.values() if s.active])
            }
        
        except Exception as e:
            self.logger.error(f"Error getting webhook delivery stats: {e}")
            return {}

# Global webhook manager
_webhook_manager: Optional[WebhookManager] = None

def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager"""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager

# Webhook router
webhook_router = APIRouter(prefix="/webhooks", tags=["Webhooks"])

@webhook_router.post("/subscriptions")
async def create_webhook_subscription(subscription: WebhookSubscription):
    """Create a new webhook subscription"""
    try:
        webhook_manager = get_webhook_manager()
        result = await webhook_manager.create_subscription(subscription)
        return {"subscription": result, "success": True}
    
    except Exception as e:
        logger.error(f"Error creating webhook subscription: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@webhook_router.get("/subscriptions")
async def list_webhook_subscriptions():
    """List all webhook subscriptions"""
    try:
        webhook_manager = get_webhook_manager()
        subscriptions = await webhook_manager.list_subscriptions()
        return {"subscriptions": subscriptions, "count": len(subscriptions)}
    
    except Exception as e:
        logger.error(f"Error listing webhook subscriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@webhook_router.get("/subscriptions/{subscription_id}")
async def get_webhook_subscription(subscription_id: str):
    """Get webhook subscription by ID"""
    try:
        webhook_manager = get_webhook_manager()
        subscription = await webhook_manager.get_subscription(subscription_id)
        
        if not subscription:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return {"subscription": subscription}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting webhook subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@webhook_router.put("/subscriptions/{subscription_id}")
async def update_webhook_subscription(subscription_id: str, updates: Dict[str, Any]):
    """Update webhook subscription"""
    try:
        webhook_manager = get_webhook_manager()
        result = await webhook_manager.update_subscription(subscription_id, updates)
        return {"subscription": result, "success": True}
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating webhook subscription: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@webhook_router.delete("/subscriptions/{subscription_id}")
async def delete_webhook_subscription(subscription_id: str):
    """Delete webhook subscription"""
    try:
        webhook_manager = get_webhook_manager()
        success = await webhook_manager.delete_subscription(subscription_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Subscription not found")
        
        return {"success": True, "message": "Subscription deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting webhook subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@webhook_router.get("/stats")
async def get_webhook_stats():
    """Get webhook delivery statistics"""
    try:
        webhook_manager = get_webhook_manager()
        stats = await webhook_manager.get_delivery_stats()
        return {"stats": stats}
    
    except Exception as e:
        logger.error(f"Error getting webhook stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@webhook_router.post("/test")
async def test_webhook(test_data: Dict[str, Any]):
    """Test webhook delivery"""
    try:
        webhook_manager = get_webhook_manager()
        
        # Trigger test event
        await webhook_manager.trigger_event(
            WebhookEventType.DOCUMENT_GENERATED,
            test_data.get("data", {"test": True}),
            "test"
        )
        
        return {"success": True, "message": "Test webhook triggered"}
    
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


