"""
Webhook System for OpusClip Improved
===================================

Advanced webhook system for real-time notifications and integrations.
"""

import asyncio
import logging
import json
import hmac
import hashlib
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from uuid import UUID, uuid4

from .schemas import get_settings
from .exceptions import WebhookError, create_webhook_error

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Webhook event types"""
    VIDEO_ANALYSIS_STARTED = "video.analysis.started"
    VIDEO_ANALYSIS_COMPLETED = "video.analysis.completed"
    VIDEO_ANALYSIS_FAILED = "video.analysis.failed"
    
    CLIP_GENERATION_STARTED = "clip.generation.started"
    CLIP_GENERATION_COMPLETED = "clip.generation.completed"
    CLIP_GENERATION_FAILED = "clip.generation.failed"
    
    CLIP_EXPORT_STARTED = "clip.export.started"
    CLIP_EXPORT_COMPLETED = "clip.export.completed"
    CLIP_EXPORT_FAILED = "clip.export.failed"
    
    BATCH_PROCESSING_STARTED = "batch.processing.started"
    BATCH_PROCESSING_PROGRESS = "batch.processing.progress"
    BATCH_PROCESSING_COMPLETED = "batch.processing.completed"
    BATCH_PROCESSING_FAILED = "batch.processing.failed"
    
    PROJECT_CREATED = "project.created"
    PROJECT_UPDATED = "project.updated"
    PROJECT_DELETED = "project.deleted"
    
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    
    SYSTEM_ALERT = "system.alert"
    SYSTEM_HEALTH_CHECK = "system.health_check"


class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"


@dataclass
class WebhookConfig:
    """Webhook configuration"""
    webhook_id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    retry_delay: int = 60
    enabled: bool = True
    headers: Dict[str, str] = None
    filters: Dict[str, Any] = None


@dataclass
class WebhookPayload:
    """Webhook payload"""
    event: WebhookEvent
    data: Dict[str, Any]
    timestamp: datetime
    webhook_id: str
    attempt: int = 1
    metadata: Dict[str, Any] = None


@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    delivery_id: str
    webhook_id: str
    event: WebhookEvent
    url: str
    payload: Dict[str, Any]
    status: WebhookStatus
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None
    created_at: datetime = None
    retry_count: int = 0


class WebhookManager:
    """Advanced webhook management system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.delivery_history: List[WebhookDelivery] = []
        self.event_handlers: Dict[WebhookEvent, List[Callable]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        
        self._initialize_default_webhooks()
    
    def _initialize_default_webhooks(self):
        """Initialize default webhook configurations"""
        # Add default webhook configurations here
        pass
    
    async def start(self):
        """Start webhook manager"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        
        # Start delivery worker
        asyncio.create_task(self._delivery_worker())
        
        self.logger.info("Webhook manager started")
    
    async def stop(self):
        """Stop webhook manager"""
        if self.session:
            await self.session.close()
        
        self.logger.info("Webhook manager stopped")
    
    def register_webhook(self, config: WebhookConfig):
        """Register a webhook configuration"""
        self.webhooks[config.webhook_id] = config
        self.logger.info(f"Webhook registered: {config.webhook_id}")
    
    def unregister_webhook(self, webhook_id: str):
        """Unregister a webhook configuration"""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            self.logger.info(f"Webhook unregistered: {webhook_id}")
    
    def register_event_handler(self, event: WebhookEvent, handler: Callable):
        """Register an event handler"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        
        self.event_handlers[event].append(handler)
        self.logger.info(f"Event handler registered for: {event}")
    
    async def trigger_event(self, event: WebhookEvent, data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Trigger a webhook event"""
        try:
            # Call registered event handlers
            if event in self.event_handlers:
                for handler in self.event_handlers[event]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event, data, metadata)
                        else:
                            handler(event, data, metadata)
                    except Exception as e:
                        self.logger.error(f"Event handler failed for {event}: {e}")
            
            # Find webhooks that should receive this event
            target_webhooks = [
                webhook for webhook in self.webhooks.values()
                if event in webhook.events and webhook.enabled
            ]
            
            # Apply filters
            filtered_webhooks = []
            for webhook in target_webhooks:
                if self._should_deliver_webhook(webhook, event, data, metadata):
                    filtered_webhooks.append(webhook)
            
            # Create webhook payloads
            for webhook in filtered_webhooks:
                payload = WebhookPayload(
                    event=event,
                    data=data,
                    timestamp=datetime.utcnow(),
                    webhook_id=webhook.webhook_id,
                    metadata=metadata or {}
                )
                
                # Add to delivery queue
                await self.delivery_queue.put(payload)
            
            self.logger.info(f"Event {event} triggered for {len(filtered_webhooks)} webhooks")
            
        except Exception as e:
            self.logger.error(f"Failed to trigger event {event}: {e}")
            raise create_webhook_error("event_trigger", event.value, e)
    
    def _should_deliver_webhook(self, webhook: WebhookConfig, event: WebhookEvent, data: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
        """Check if webhook should be delivered based on filters"""
        if not webhook.filters:
            return True
        
        # Apply filters (simplified implementation)
        for filter_key, filter_value in webhook.filters.items():
            if filter_key in data:
                if data[filter_key] != filter_value:
                    return False
            elif filter_key in metadata:
                if metadata[filter_key] != filter_value:
                    return False
        
        return True
    
    async def _delivery_worker(self):
        """Background worker for webhook delivery"""
        while True:
            try:
                # Get payload from queue
                payload = await self.delivery_queue.get()
                
                # Deliver webhook
                await self._deliver_webhook(payload)
                
                # Mark task as done
                self.delivery_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Webhook delivery worker error: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_webhook(self, payload: WebhookPayload):
        """Deliver a webhook"""
        webhook = self.webhooks.get(payload.webhook_id)
        if not webhook:
            self.logger.error(f"Webhook not found: {payload.webhook_id}")
            return
        
        delivery_id = str(uuid4())
        
        # Create delivery record
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=payload.webhook_id,
            event=payload.event,
            url=webhook.url,
            payload=asdict(payload),
            status=WebhookStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        # Attempt delivery
        for attempt in range(webhook.retry_count + 1):
            try:
                delivery.retry_count = attempt
                delivery.status = WebhookStatus.RETRYING if attempt > 0 else WebhookStatus.PENDING
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "OpusClip-Webhook/1.0",
                    "X-Webhook-Event": payload.event.value,
                    "X-Webhook-Delivery": delivery_id,
                    "X-Webhook-Timestamp": payload.timestamp.isoformat()
                }
                
                # Add custom headers
                if webhook.headers:
                    headers.update(webhook.headers)
                
                # Add signature if secret is provided
                if webhook.secret:
                    signature = self._generate_signature(webhook.secret, json.dumps(asdict(payload)))
                    headers["X-Webhook-Signature"] = f"sha256={signature}"
                
                # Make HTTP request
                async with self.session.post(
                    webhook.url,
                    json=asdict(payload),
                    headers=headers,
                    timeout=webhook.timeout
                ) as response:
                    delivery.response_code = response.status
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        delivery.status = WebhookStatus.DELIVERED
                        delivery.delivered_at = datetime.utcnow()
                        self.logger.info(f"Webhook delivered successfully: {delivery_id}")
                        break
                    else:
                        delivery.error_message = f"HTTP {response.status}: {delivery.response_body}"
                        self.logger.warning(f"Webhook delivery failed: {delivery_id} - {delivery.error_message}")
                
            except asyncio.TimeoutError:
                delivery.error_message = "Request timeout"
                self.logger.warning(f"Webhook delivery timeout: {delivery_id}")
            except Exception as e:
                delivery.error_message = str(e)
                self.logger.error(f"Webhook delivery error: {delivery_id} - {e}")
            
            # Wait before retry
            if attempt < webhook.retry_count:
                await asyncio.sleep(webhook.retry_delay)
        
        # Mark as failed if all attempts failed
        if delivery.status != WebhookStatus.DELIVERED:
            delivery.status = WebhookStatus.FAILED
        
        # Store delivery record
        self.delivery_history.append(delivery)
        
        # Keep only last 1000 delivery records
        if len(self.delivery_history) > 1000:
            self.delivery_history = self.delivery_history[-1000:]
    
    def _generate_signature(self, secret: str, payload: str) -> str:
        """Generate webhook signature"""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, secret: str, payload: str, signature: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self._generate_signature(secret, payload)
        return hmac.compare_digest(signature, expected_signature)
    
    def get_webhook_config(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook configuration"""
        return self.webhooks.get(webhook_id)
    
    def list_webhooks(self) -> List[WebhookConfig]:
        """List all webhook configurations"""
        return list(self.webhooks.values())
    
    def get_delivery_history(self, webhook_id: str = None, limit: int = 100) -> List[WebhookDelivery]:
        """Get webhook delivery history"""
        history = self.delivery_history
        
        if webhook_id:
            history = [d for d in history if d.webhook_id == webhook_id]
        
        return history[-limit:]
    
    def get_webhook_stats(self) -> Dict[str, Any]:
        """Get webhook statistics"""
        total_deliveries = len(self.delivery_history)
        successful_deliveries = len([d for d in self.delivery_history if d.status == WebhookStatus.DELIVERED])
        failed_deliveries = len([d for d in self.delivery_history if d.status == WebhookStatus.FAILED])
        
        success_rate = (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0
        
        # Group by event type
        event_stats = {}
        for delivery in self.delivery_history:
            event = delivery.event.value
            if event not in event_stats:
                event_stats[event] = {"total": 0, "successful": 0, "failed": 0}
            
            event_stats[event]["total"] += 1
            if delivery.status == WebhookStatus.DELIVERED:
                event_stats[event]["successful"] += 1
            else:
                event_stats[event]["failed"] += 1
        
        return {
            "total_webhooks": len(self.webhooks),
            "active_webhooks": len([w for w in self.webhooks.values() if w.enabled]),
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful_deliveries,
            "failed_deliveries": failed_deliveries,
            "success_rate": round(success_rate, 2),
            "queue_size": self.delivery_queue.qsize(),
            "event_stats": event_stats
        }
    
    async def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Test webhook delivery"""
        webhook = self.webhooks.get(webhook_id)
        if not webhook:
            raise ValueError(f"Webhook not found: {webhook_id}")
        
        # Create test payload
        test_payload = WebhookPayload(
            event=WebhookEvent.SYSTEM_HEALTH_CHECK,
            data={
                "test": True,
                "message": "This is a test webhook delivery",
                "timestamp": datetime.utcnow().isoformat()
            },
            timestamp=datetime.utcnow(),
            webhook_id=webhook_id,
            metadata={"test": True}
        )
        
        # Deliver test webhook
        await self._deliver_webhook(test_payload)
        
        return {
            "webhook_id": webhook_id,
            "url": webhook.url,
            "test_payload": asdict(test_payload),
            "status": "delivered"
        }


class WebhookEventEmitter:
    """Webhook event emitter for easy integration"""
    
    def __init__(self, webhook_manager: WebhookManager):
        self.webhook_manager = webhook_manager
    
    async def emit_video_analysis_started(self, analysis_id: str, user_id: str, video_url: str):
        """Emit video analysis started event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.VIDEO_ANALYSIS_STARTED,
            {
                "analysis_id": analysis_id,
                "user_id": user_id,
                "video_url": video_url,
                "status": "started"
            }
        )
    
    async def emit_video_analysis_completed(self, analysis_id: str, user_id: str, results: Dict[str, Any]):
        """Emit video analysis completed event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.VIDEO_ANALYSIS_COMPLETED,
            {
                "analysis_id": analysis_id,
                "user_id": user_id,
                "results": results,
                "status": "completed"
            }
        )
    
    async def emit_video_analysis_failed(self, analysis_id: str, user_id: str, error: str):
        """Emit video analysis failed event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.VIDEO_ANALYSIS_FAILED,
            {
                "analysis_id": analysis_id,
                "user_id": user_id,
                "error": error,
                "status": "failed"
            }
        )
    
    async def emit_clip_generation_started(self, generation_id: str, user_id: str, analysis_id: str):
        """Emit clip generation started event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.CLIP_GENERATION_STARTED,
            {
                "generation_id": generation_id,
                "user_id": user_id,
                "analysis_id": analysis_id,
                "status": "started"
            }
        )
    
    async def emit_clip_generation_completed(self, generation_id: str, user_id: str, clips: List[Dict[str, Any]]):
        """Emit clip generation completed event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.CLIP_GENERATION_COMPLETED,
            {
                "generation_id": generation_id,
                "user_id": user_id,
                "clips": clips,
                "status": "completed"
            }
        )
    
    async def emit_clip_export_started(self, export_id: str, user_id: str, generation_id: str, platform: str):
        """Emit clip export started event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.CLIP_EXPORT_STARTED,
            {
                "export_id": export_id,
                "user_id": user_id,
                "generation_id": generation_id,
                "platform": platform,
                "status": "started"
            }
        )
    
    async def emit_clip_export_completed(self, export_id: str, user_id: str, download_urls: List[str]):
        """Emit clip export completed event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.CLIP_EXPORT_COMPLETED,
            {
                "export_id": export_id,
                "user_id": user_id,
                "download_urls": download_urls,
                "status": "completed"
            }
        )
    
    async def emit_batch_processing_progress(self, batch_id: str, user_id: str, progress: Dict[str, Any]):
        """Emit batch processing progress event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.BATCH_PROCESSING_PROGRESS,
            {
                "batch_id": batch_id,
                "user_id": user_id,
                "progress": progress,
                "status": "in_progress"
            }
        )
    
    async def emit_batch_processing_completed(self, batch_id: str, user_id: str, results: Dict[str, Any]):
        """Emit batch processing completed event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.BATCH_PROCESSING_COMPLETED,
            {
                "batch_id": batch_id,
                "user_id": user_id,
                "results": results,
                "status": "completed"
            }
        )
    
    async def emit_project_created(self, project_id: str, user_id: str, project_data: Dict[str, Any]):
        """Emit project created event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.PROJECT_CREATED,
            {
                "project_id": project_id,
                "user_id": user_id,
                "project": project_data,
                "status": "created"
            }
        )
    
    async def emit_user_registered(self, user_id: str, user_data: Dict[str, Any]):
        """Emit user registered event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.USER_REGISTERED,
            {
                "user_id": user_id,
                "user": user_data,
                "status": "registered"
            }
        )
    
    async def emit_system_alert(self, alert_data: Dict[str, Any]):
        """Emit system alert event"""
        await self.webhook_manager.trigger_event(
            WebhookEvent.SYSTEM_ALERT,
            {
                "alert": alert_data,
                "status": "alert"
            }
        )


# Global webhook manager and event emitter
webhook_manager = WebhookManager()
webhook_emitter = WebhookEventEmitter(webhook_manager)





























