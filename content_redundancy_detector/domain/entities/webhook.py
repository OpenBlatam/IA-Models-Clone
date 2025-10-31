"""
Webhook Domain Entities
Business entities for webhook management
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime


class WebhookEventType(Enum):
    """Webhook event types"""
    ANALYSIS_COMPLETED = "analysis_completed"
    SIMILARITY_COMPLETED = "similarity_completed"
    QUALITY_COMPLETED = "quality_completed"
    BATCH_COMPLETED = "batch_completed"
    BATCH_FAILED = "batch_failed"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class WebhookDeliveryStatus(Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class WebhookEndpoint:
    """Webhook endpoint entity"""
    id: str
    url: str
    events: List[WebhookEventType]
    secret: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    
    def supports_event(self, event: WebhookEventType) -> bool:
        """Check if endpoint supports event type"""
        return event in self.events and self.is_active
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "is_active": self.is_active,
            "created_at": self.created_at,
            "timeout": self.timeout,
            "retry_count": self.retry_count
        }


@dataclass
class WebhookPayload:
    """Webhook payload entity"""
    event: WebhookEventType
    timestamp: float
    data: Dict[str, Any]
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event": self.event.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "request_id": self.request_id,
            "user_id": self.user_id
        }


@dataclass
class WebhookDelivery:
    """Webhook delivery entity"""
    id: str
    endpoint_id: str
    event: WebhookEventType
    payload: WebhookPayload
    status: WebhookDeliveryStatus = WebhookDeliveryStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[float] = None
    next_retry: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    delivered_at: Optional[float] = None
    error_message: Optional[str] = None
    
    def can_retry(self, max_attempts: int) -> bool:
        """Check if delivery can be retried"""
        return (
            self.status in [WebhookDeliveryStatus.PENDING, WebhookDeliveryStatus.FAILED]
            and self.attempts < max_attempts
        )
    
    def mark_delivered(self):
        """Mark delivery as delivered"""
        self.status = WebhookDeliveryStatus.DELIVERED
        self.delivered_at = time.time()
    
    def mark_failed(self, error_message: str):
        """Mark delivery as failed"""
        self.status = WebhookDeliveryStatus.FAILED
        self.error_message = error_message
        self.last_attempt = time.time()
    
    def increment_attempt(self):
        """Increment attempt count"""
        self.attempts += 1
        self.last_attempt = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "endpoint_id": self.endpoint_id,
            "event": self.event.value,
            "status": self.status.value,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "error_message": self.error_message
        }






