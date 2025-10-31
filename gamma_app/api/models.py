"""
Gamma App - API Models
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

class ContentType(str, Enum):
    """Content types"""
    PRESENTATION = "presentation"
    DOCUMENT = "document"
    WEB_PAGE = "web_page"
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    REPORT = "report"
    PROPOSAL = "proposal"

class OutputFormat(str, Enum):
    """Output formats"""
    PDF = "pdf"
    PPTX = "pptx"
    HTML = "html"
    DOCX = "docx"
    MD = "markdown"
    JSON = "json"
    PNG = "png"
    JPG = "jpg"

class DesignStyle(str, Enum):
    """Design styles"""
    MODERN = "modern"
    MINIMALIST = "minimalist"
    CORPORATE = "corporate"
    CREATIVE = "creative"
    ACADEMIC = "academic"
    CASUAL = "casual"
    PROFESSIONAL = "professional"

class UserRole(str, Enum):
    """User roles"""
    ADMIN = "admin"
    USER = "user"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"

class SessionStatus(str, Enum):
    """Collaboration session status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"

# Base Models
class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: str = "Success"
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error: str
    status_code: int
    details: Optional[Dict[str, Any]] = None

# User Models
class User(BaseModel):
    """User model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    email: str
    name: str
    role: UserRole = UserRole.USER
    created_at: datetime = Field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    subscription_plan: str = "free"
    is_active: bool = True

class UserCreate(BaseModel):
    """User creation model"""
    email: str
    name: str
    password: str
    role: UserRole = UserRole.USER

class UserUpdate(BaseModel):
    """User update model"""
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    subscription_plan: Optional[str] = None

# Project Models
class Project(BaseModel):
    """Project model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    owner_id: str
    collaborators: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    settings: Dict[str, Any] = Field(default_factory=dict)
    is_public: bool = False

class ProjectCreate(BaseModel):
    """Project creation model"""
    name: str
    description: Optional[str] = None
    is_public: bool = False
    settings: Dict[str, Any] = Field(default_factory=dict)

class ProjectUpdate(BaseModel):
    """Project update model"""
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None

# Content Models
class ContentRequest(BaseModel):
    """Content generation request"""
    content_type: ContentType
    topic: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    target_audience: str = Field(..., min_length=1, max_length=100)
    length: str = Field("medium", regex="^(short|medium|long)$")
    style: DesignStyle = DesignStyle.PROFESSIONAL
    output_format: OutputFormat = OutputFormat.HTML
    include_images: bool = True
    include_charts: bool = False
    language: str = Field("en", regex="^[a-z]{2}$")
    tone: str = Field("professional", regex="^(professional|casual|friendly|formal|persuasive|informative|conversational|authoritative|creative|technical)$")
    keywords: List[str] = Field(default_factory=list, max_items=10)
    custom_instructions: str = Field("", max_length=2000)
    user_id: str = ""
    project_id: str = ""

    @validator('keywords')
    def validate_keywords(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 keywords allowed')
        return [keyword.strip() for keyword in v if keyword.strip()]

class ContentResponse(BaseResponse):
    """Content generation response"""
    content_id: str
    content_type: ContentType
    title: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    generated_at: datetime
    processing_time: float
    quality_score: float
    suggestions: List[str]
    export_urls: Dict[str, str]

class ContentUpdate(BaseModel):
    """Content update model"""
    title: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

# Collaboration Models
class CollaborationSession(BaseModel):
    """Collaboration session model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    session_name: str
    creator_id: str
    participants: List[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    settings: Dict[str, Any] = Field(default_factory=dict)

class CollaborationMessage(BaseModel):
    """Collaboration message model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    message_type: str = Field(..., regex="^(text|edit|comment|cursor|selection)$")
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class CollaborationEvent(BaseModel):
    """Collaboration event model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    user_id: str
    event_type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# Export Models
class ExportRequest(BaseModel):
    """Export request model"""
    content: Dict[str, Any]
    output_format: OutputFormat
    theme: Optional[str] = None
    template: Optional[str] = None
    style: Optional[str] = None
    document_type: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

class ExportResponse(BaseResponse):
    """Export response model"""
    export_id: str
    download_url: str
    file_size: int
    expires_at: datetime

# Analytics Models
class AnalyticsRequest(BaseModel):
    """Analytics request model"""
    user_id: str
    content_id: Optional[str] = None
    time_period: str = Field("7d", regex="^(1d|7d|30d|90d|1y)$")
    metrics: List[str] = Field(default_factory=list)

class AnalyticsResponse(BaseResponse):
    """Analytics response model"""
    data: Dict[str, Any]
    time_period: str
    generated_at: datetime

class DashboardData(BaseModel):
    """Dashboard data model"""
    total_content: int
    total_exports: int
    active_collaborations: int
    recent_activity: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    content_types: Dict[str, int]
    export_formats: Dict[str, int]

class ContentPerformance(BaseModel):
    """Content performance model"""
    content_id: str
    views: int
    exports: int
    shares: int
    engagement_score: float
    quality_score: float
    feedback: List[Dict[str, Any]]

class CollaborationStats(BaseModel):
    """Collaboration statistics model"""
    total_sessions: int
    active_sessions: int
    total_participants: int
    average_session_duration: float
    most_active_users: List[Dict[str, Any]]
    session_types: Dict[str, int]

# Template Models
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
    name: str
    description: str
    content_type: ContentType
    category: str
    template_data: Dict[str, Any]
    is_public: bool = False

# Search Models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=200)
    content_type: Optional[ContentType] = None
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

class SearchResponse(BaseResponse):
    """Search response model"""
    results: List[Dict[str, Any]]
    total: int
    query: str
    filters: Dict[str, Any]

# Notification Models
class Notification(BaseModel):
    """Notification model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    title: str
    message: str
    type: str = Field(..., regex="^(info|success|warning|error)$")
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

# Settings Models
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
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_formats: List[str] = Field(default_factory=lambda: ["pdf", "pptx", "docx", "html"])
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    features: Dict[str, bool] = Field(default_factory=dict)

# Health Check Models
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

# WebSocket Models
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



























