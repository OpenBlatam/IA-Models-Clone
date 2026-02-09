"""
Common Models
Common models used across multiple domains
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .base import BaseResponse
from .enums import ContentType
from .validators import sanitize_string_field, sanitize_optional_string_field

class Template(BaseModel):
    """Template model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    content_type: ContentType
    category: str
    template_data: Dict[str, Any]
    is_public: bool = False
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    usage_count: int = 0
    rating: float = 0.0

class TemplateCreate(BaseModel):
    """Template creation model"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    content_type: ContentType
    category: str = Field(..., min_length=1, max_length=50)
    template_data: Dict[str, Any]
    is_public: bool = False

    @field_validator('name', 'description', 'category')
    @classmethod
    def sanitize_string_fields(cls, v: str) -> str:
        """Sanitize string fields"""
        return sanitize_string_field(v)

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=200)
    content_type: Optional[ContentType] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Sanitize search query"""
        return sanitize_string_field(v)

class SearchResponse(BaseResponse):
    """Search response model"""
    results: List[Dict[str, Any]]
    total: int
    query: str
    filters: Dict[str, Any]

class Notification(BaseModel):
    """Notification model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: str
    message: str
    type: str = Field(..., pattern="^(info|success|warning|error)$")
    is_read: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)

class NotificationCreate(BaseModel):
    """Notification creation model"""
    user_id: str
    title: str
    message: str
    type: str = "info"
    data: Dict[str, Any] = Field(default_factory=dict)

class UserSettings(BaseModel):
    """User settings model"""
    user_id: str
    theme: str = "light"
    language: str = "en"
    notifications: Dict[str, bool] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=datetime.now)

class SystemSettings(BaseModel):
    """System settings model"""
    maintenance_mode: bool = False
    max_file_size: int = 100 * 1024 * 1024
    allowed_formats: List[str] = Field(default_factory=lambda: ["pdf", "pptx", "docx", "html"])
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    features: Dict[str, bool] = Field(default_factory=dict)

class HealthCheck(BaseModel):
    """Health check model"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str
    uptime: float

class ServiceStatus(BaseModel):
    """Service status model"""
    name: str
    status: str
    response_time: float
    last_check: datetime
    details: Dict[str, Any] = Field(default_factory=dict)

class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None

class WebSocketResponse(BaseModel):
    """WebSocket response model"""
    type: str
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)







