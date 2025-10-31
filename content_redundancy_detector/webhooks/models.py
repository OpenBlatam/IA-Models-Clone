from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import orjson


class WebhookEvent(BaseModel):
    id: str = Field(..., description="Unique event id")
    type: str = Field(..., description="Event type")
    payload: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    timestamp: Optional[str] = None  # header-provided timestamp for HMAC

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_loads = orjson.loads
        json_dumps = lambda v, **kwargs: orjson.dumps(v, **kwargs).decode()


class WebhookResult(BaseModel):
    id: str
    success: bool
    attempts: int
    error: Optional[str] = None
    processed_at: datetime = Field(default_factory=datetime.utcnow)

"""
Webhook Models - Data structures and types
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class WebhookEvent(Enum):
    """Webhook event types"""
    ANALYSIS_COMPLETED = "analysis_completed"
    SIMILARITY_COMPLETED = "similarity_completed"
    QUALITY_COMPLETED = "quality_completed"
    BATCH_COMPLETED = "batch_completed"
    BATCH_FAILED = "batch_failed"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class WebhookPayload:
    """Webhook payload structure"""
    event: str
    timestamp: float
    data: Dict[str, Any]
    request_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    id: str
    url: str
    events: List[WebhookEvent]
    secret: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3
    is_active: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class WebhookDelivery:
    """Webhook delivery record"""
    id: str
    endpoint_id: str
    event: str
    payload: WebhookPayload
    status: str = "pending"  # pending, delivered, failed
    attempts: int = 0
    last_attempt: Optional[float] = None
    next_retry: Optional[float] = None
    created_at: float = field(default_factory=time.time)

