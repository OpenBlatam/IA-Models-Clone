"""
PDF Variantes API Models
Pydantic models for API requests and responses
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, Field, validator
from ..models import (
    PDFDocument, PDFVariant, TopicItem, BrainstormIdea,
    VariantConfiguration, CollaborationInvite, Annotation,
    Feedback, SearchRequest, SearchResponse, SearchResult,
    BatchProcessingRequest, BatchProcessingResponse,
    ExportRequest, ExportResponse, SystemHealth, AnalyticsReport
)

# Authentication Models
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class TokenData(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None
    permissions: List[str] = []

class User(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[str] = []

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[str] = Field(None, regex=r'^[^@]+@[^@]+\.[^@]+$')
    full_name: Optional[str] = None
    is_active: Optional[bool] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    user: User
    token: Token

# Enhanced Request/Response Models
class PDFUploadRequest(BaseModel):
    """Enhanced PDF upload request"""
    filename: Optional[str] = None
    auto_process: bool = True
    extract_text: bool = True
    detect_language: bool = True
    extract_topics: bool = True
    generate_summary: bool = False
    custom_metadata: Optional[Dict[str, Any]] = None
    processing_options: Optional[Dict[str, Any]] = None

class PDFUploadResponse(BaseModel):
    """Enhanced PDF upload response"""
    success: bool
    document: PDFDocument
    message: str
    processing_started: bool
    estimated_processing_time: Optional[float] = None
    processing_job_id: Optional[str] = None

class VariantGenerateRequest(BaseModel):
    """Enhanced variant generation request"""
    document_id: str
    number_of_variants: int = Field(default=10, ge=1, le=1000)
    continuous_generation: bool = True
    configuration: Optional[VariantConfiguration] = None
    stop_condition: Optional[str] = None
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    callback_url: Optional[str] = None
    custom_prompts: Optional[List[str]] = None

class VariantGenerateResponse(BaseModel):
    """Enhanced variant generation response"""
    success: bool
    variants: List[PDFVariant]
    total_generated: int
    generation_time: float
    message: str
    is_stopped: bool
    job_id: Optional[str] = None
    estimated_completion: Optional[datetime] = None

class TopicExtractRequest(BaseModel):
    """Enhanced topic extraction request"""
    document_id: str
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    max_topics: int = Field(default=50, ge=1, le=200)
    include_sentiment: bool = True
    include_entities: bool = True
    language: Optional[str] = None
    custom_categories: Optional[List[str]] = None

class TopicExtractResponse(BaseModel):
    """Enhanced topic extraction response"""
    success: bool
    topics: List[TopicItem]
    main_topic: Optional[str] = None
    total_topics: int
    extraction_time: float
    sentiment_analysis: Optional[Dict[str, Any]] = None
    named_entities: Optional[List[Dict[str, Any]]] = None

class BrainstormGenerateRequest(BaseModel):
    """Enhanced brainstorm generation request"""
    document_id: str
    topics: Optional[List[str]] = None
    number_of_ideas: int = Field(default=20, ge=1, le=500)
    diversity_level: float = Field(default=0.7, ge=0.0, le=1.0)
    creativity_level: float = Field(default=0.8, ge=0.0, le=1.0)
    focus_areas: Optional[List[str]] = None
    exclude_topics: Optional[List[str]] = None
    target_audience: Optional[str] = None

class BrainstormGenerateResponse(BaseModel):
    """Enhanced brainstorm generation response"""
    success: bool
    ideas: List[BrainstormIdea]
    total_ideas: int
    generation_time: float
    categories: List[str]
    creativity_score: Optional[float] = None
    diversity_score: Optional[float] = None

# Collaboration Models
class CollaborationRequest(BaseModel):
    """Collaboration request"""
    document_id: str
    action: str = Field(..., regex="^(invite|remove|update_permissions|join|leave)$")
    target_user_id: Optional[str] = None
    permissions: Optional[List[str]] = None
    message: Optional[str] = None

class CollaborationResponse(BaseModel):
    """Collaboration response"""
    success: bool
    message: str
    collaboration_data: Optional[Dict[str, Any]] = None

class RealTimeUpdate(BaseModel):
    """Real-time update for WebSocket"""
    type: str = Field(..., regex="^(cursor|selection|edit|comment|annotation)$")
    document_id: str
    user_id: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Export Models
class ExportRequest(BaseModel):
    """Enhanced export request"""
    document_id: str
    variant_ids: Optional[List[str]] = None
    export_format: str = Field(..., regex="^(pdf|docx|txt|html|json|zip|pptx)$")
    include_metadata: bool = True
    include_statistics: bool = True
    compress: bool = False
    custom_styling: Optional[Dict[str, Any]] = None
    watermark: Optional[str] = None
    password_protect: Optional[str] = None

class ExportResponse(BaseModel):
    """Enhanced export response"""
    success: bool
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    record_count: int
    export_time: float
    file_id: Optional[str] = None

# Analytics Models
class DashboardData(BaseModel):
    """Dashboard analytics data"""
    total_documents: int
    total_variants: int
    total_topics: int
    total_brainstorm_ideas: int
    recent_activity: List[Dict[str, Any]]
    usage_statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    system_status: Dict[str, Any]

class AnalyticsRequest(BaseModel):
    """Analytics request"""
    start_date: datetime
    end_date: datetime
    metrics: Optional[List[str]] = None
    group_by: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class AnalyticsResponse(BaseModel):
    """Analytics response"""
    success: bool
    data: Dict[str, Any]
    generated_at: datetime
    query_time: float

# Search Models
class SearchRequest(BaseModel):
    """Enhanced search request"""
    query: str = Field(..., min_length=1, max_length=500)
    document_id: Optional[str] = None
    variant_ids: Optional[List[str]] = None
    search_fields: List[str] = Field(default=["content"])
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=50, ge=1, le=500)
    use_fuzzy_search: bool = True
    use_semantic_search: bool = False
    filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")

class SearchResponse(BaseModel):
    """Enhanced search response"""
    success: bool
    query: str
    total_results: int
    results: List[SearchResult]
    search_time: float
    facets: Dict[str, Any]
    suggestions: Optional[List[str]] = None
    related_searches: Optional[List[str]] = None

# Batch Processing Models
class BatchProcessingRequest(BaseModel):
    """Enhanced batch processing request"""
    document_ids: List[str] = Field(..., min_items=1, max_items=100)
    operation: str = Field(..., regex="^(generate_variants|extract_topics|generate_brainstorm|export|analyze)$")
    configuration: Optional[Dict[str, Any]] = None
    optimization_settings: Optional[Dict[str, Any]] = None
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    callback_url: Optional[str] = None
    notify_on_completion: bool = True

class BatchProcessingResponse(BaseModel):
    """Enhanced batch processing response"""
    success: bool
    total_documents: int
    successful: int
    failed: int
    results: Dict[str, Any]
    processing_time: float
    metrics: Dict[str, Any]
    message: str
    job_id: Optional[str] = None
    estimated_completion: Optional[datetime] = None

# Notification Models
class NotificationRequest(BaseModel):
    """Notification request"""
    type: str = Field(..., regex="^(email|sms|push|webhook|in_app)$")
    recipient: str
    subject: Optional[str] = None
    message: str
    data: Optional[Dict[str, Any]] = None
    priority: str = Field(default="normal", regex="^(low|normal|high|urgent)$")
    scheduled_at: Optional[datetime] = None

class NotificationResponse(BaseModel):
    """Notification response"""
    success: bool
    notification_id: str
    message: str
    sent_at: Optional[datetime] = None

# File Management Models
class FileUploadRequest(BaseModel):
    """File upload request"""
    file_type: str = Field(..., regex="^(pdf|image|document|archive)$")
    max_size_mb: int = Field(default=100, ge=1, le=1000)
    allowed_extensions: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None

class FileUploadResponse(BaseModel):
    """File upload response"""
    success: bool
    file_id: str
    filename: str
    file_size: int
    file_type: str
    upload_url: Optional[str] = None
    expires_at: Optional[datetime] = None

# Configuration Models
class SystemConfigRequest(BaseModel):
    """System configuration request"""
    config_type: str = Field(..., regex="^(ai|security|performance|cache|export)$")
    settings: Dict[str, Any]
    apply_globally: bool = False
    user_id: Optional[str] = None

class SystemConfigResponse(BaseModel):
    """System configuration response"""
    success: bool
    config_id: str
    applied_settings: Dict[str, Any]
    message: str

# Error Models
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ValidationError(BaseModel):
    """Validation error model"""
    field: str
    message: str
    value: Any

class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    errors: List[ValidationError]
    message: str = "Validation failed"

# Success Models
class SuccessResponse(BaseModel):
    """Success response model"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Pagination Models
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")

class PaginatedResponse(BaseModel):
    """Paginated response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

# WebSocket Models
class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str

class WebSocketResponse(BaseModel):
    """WebSocket response model"""
    type: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Health Check Models
class HealthCheck(BaseModel):
    """Health check model"""
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str]
    version: str
    uptime: float

class ServiceHealth(BaseModel):
    """Service health model"""
    service_name: str
    status: str = Field(..., regex="^(healthy|degraded|unhealthy|unknown)$")
    response_time: Optional[float] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = None

# Export all models
__all__ = [
    # Authentication
    "Token", "TokenData", "User", "UserCreate", "UserUpdate", 
    "LoginRequest", "LoginResponse",
    
    # Enhanced Requests/Responses
    "PDFUploadRequest", "PDFUploadResponse",
    "VariantGenerateRequest", "VariantGenerateResponse",
    "TopicExtractRequest", "TopicExtractResponse",
    "BrainstormGenerateRequest", "BrainstormGenerateResponse",
    
    # Collaboration
    "CollaborationRequest", "CollaborationResponse", "RealTimeUpdate",
    
    # Export
    "ExportRequest", "ExportResponse",
    
    # Analytics
    "DashboardData", "AnalyticsRequest", "AnalyticsResponse",
    
    # Search
    "SearchRequest", "SearchResponse",
    
    # Batch Processing
    "BatchProcessingRequest", "BatchProcessingResponse",
    
    # Notifications
    "NotificationRequest", "NotificationResponse",
    
    # File Management
    "FileUploadRequest", "FileUploadResponse",
    
    # Configuration
    "SystemConfigRequest", "SystemConfigResponse",
    
    # Error Handling
    "ErrorResponse", "ValidationError", "ValidationErrorResponse",
    
    # Success
    "SuccessResponse",
    
    # Pagination
    "PaginationParams", "PaginatedResponse",
    
    # WebSocket
    "WebSocketMessage", "WebSocketResponse",
    
    # Health
    "HealthCheck", "ServiceHealth"
]
