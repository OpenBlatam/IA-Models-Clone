"""Enhanced schemas with comprehensive validation."""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import re

# Re-import existing models from .models to avoid duplication
from .models import (
    VariantStatus, PDFProcessingStatus, TopicCategory, PDFMetadata, EditedPage,
    PDFDocument, VariantConfiguration, PDFVariant, TopicItem, BrainstormIdea,
    CollaborationInvite, Revision, Annotation, Tag, DocumentTag, Feedback,
    BackupJob, RestoreRequest, TransformationRule, Template, AIRecommendation,
    Workflow, WorkflowExecution, ContentModeration, AITranslation,
    ContentSummarization, ContentEnhancement, PlagiarismCheck, ContentAnalysis,
    ContentTemplate, StyleGuide, ProcessingPipeline, ProcessingStage, ProcessingResult,
    DashboardWidget, Dashboard, TestSuite, TestResult, CacheConfiguration, RateLimitConfig,
    SecuritySettings, ScheduledJob, IntegrationConfig, SyncJob, UserPermission, AuditLog
)


# --- Enhanced Request Models ---
class PDFUploadRequest(BaseModel):
    """Enhanced PDF upload request with validation."""
    filename: Optional[str] = Field(None, description="Custom filename for the uploaded PDF")
    auto_process: bool = Field(True, description="Auto-process PDF after upload")
    extract_text: bool = Field(True, description="Extract text content from PDF")
    detect_language: bool = Field(True, description="Detect language of PDF content")
    generate_preview: bool = Field(True, description="Generate preview images")
    extract_metadata: bool = Field(True, description="Extract PDF metadata")
    max_file_size_mb: int = Field(100, ge=1, le=500, description="Maximum file size in MB")
    
    @validator('filename')
    def validate_filename(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9._-]+\.pdf$', v):
            raise ValueError('Filename must be a valid PDF filename')
        return v


class PDFEditRequest(BaseModel):
    """Enhanced PDF edit request."""
    page_number: int = Field(..., ge=1, description="Page number to edit")
    new_content: str = Field(..., min_length=1, max_length=10000, description="New content for the page")
    preserve_formatting: bool = Field(True, description="Preserve original formatting")
    add_annotations: bool = Field(False, description="Add annotations to the edit")
    backup_original: bool = Field(True, description="Create backup of original page")
    
    @validator('new_content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class VariantGenerateRequest(BaseModel):
    """Enhanced variant generation request."""
    variant_type: str = Field(..., description="Type of variant to generate")
    options: Dict[str, Any] = Field(default_factory=dict, description="Variant generation options")
    quality: Literal["draft", "standard", "high"] = Field("standard", description="Output quality")
    include_metadata: bool = Field(True, description="Include metadata in variant")
    optimize_for_web: bool = Field(False, description="Optimize for web display")
    
    @validator('variant_type')
    def validate_variant_type(cls, v):
        allowed_types = ["summary", "outline", "highlights", "translation", "simplified"]
        if v not in allowed_types:
            raise ValueError(f'Variant type must be one of: {", ".join(allowed_types)}')
        return v


class TopicExtractRequest(BaseModel):
    """Enhanced topic extraction request."""
    min_relevance: float = Field(0.5, ge=0.0, le=1.0, description="Minimum relevance score")
    max_topics: int = Field(50, ge=1, le=200, description="Maximum number of topics")
    include_similarities: bool = Field(True, description="Include topic similarities")
    group_related: bool = Field(True, description="Group related topics")
    language: Optional[str] = Field(None, description="Language for topic extraction")
    
    @validator('language')
    def validate_language(cls, v):
        if v and len(v) != 2:
            raise ValueError('Language must be a 2-letter ISO code')
        return v


class BrainstormGenerateRequest(BaseModel):
    """Enhanced brainstorming request."""
    topics: List[str] = Field(..., min_items=1, max_items=10, description="Topics for brainstorming")
    number_of_ideas: int = Field(20, ge=1, le=100, description="Number of ideas to generate")
    creativity_level: Literal["conservative", "balanced", "creative"] = Field("balanced", description="Creativity level")
    include_examples: bool = Field(True, description="Include examples in ideas")
    filter_duplicates: bool = Field(True, description="Filter duplicate ideas")
    
    @validator('topics')
    def validate_topics(cls, v):
        if not all(topic.strip() for topic in v):
            raise ValueError('All topics must be non-empty')
        return [topic.strip() for topic in v]


class CollaborationRequest(BaseModel):
    """Enhanced collaboration request."""
    session_name: str = Field(..., min_length=1, max_length=100, description="Name of collaboration session")
    participants: List[str] = Field(..., min_items=1, max_items=20, description="List of participant emails")
    permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Permissions for each participant")
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")
    is_public: bool = Field(False, description="Whether session is public")
    
    @validator('participants')
    def validate_participants(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        for email in v:
            if not re.match(email_pattern, email):
                raise ValueError(f'Invalid email format: {email}')
        return v


# --- Enhanced Response Models ---
class APIResponse(BaseModel):
    """Enhanced generic API response."""
    success: bool = Field(..., description="Indicates if the request was successful")
    message: str = Field(..., description="Human-readable message")
    data: Optional[Any] = Field(None, description="Response data")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    warnings: List[str] = Field(default_factory=list, description="List of warning messages")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique request ID")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class PDFUploadResponse(APIResponse):
    """Enhanced PDF upload response."""
    document_id: Optional[str] = Field(None, description="ID of the uploaded document")
    metadata: Optional[PDFMetadata] = Field(None, description="PDF metadata")
    processing_started: bool = Field(False, description="Whether background processing started")
    file_size: Optional[str] = Field(None, description="Human-readable file size")
    preview_url: Optional[str] = Field(None, description="URL to preview the PDF")


class PDFEditResponse(APIResponse):
    """Enhanced PDF edit response."""
    edited_page: Optional[EditedPage] = Field(None, description="Details of the edited page")
    backup_id: Optional[str] = Field(None, description="ID of the backup created")
    changes_applied: bool = Field(False, description="Whether changes were applied successfully")


class VariantGenerateResponse(APIResponse):
    """Enhanced variant generation response."""
    variant: Optional[PDFVariant] = Field(None, description="Generated variant")
    generation_time: Optional[float] = Field(None, description="Time taken to generate variant")
    quality_score: Optional[float] = Field(None, description="Quality score of the variant")


class TopicExtractResponse(APIResponse):
    """Enhanced topic extraction response."""
    topics: List[TopicItem] = Field(default_factory=list, description="Extracted topics")
    total_count: int = Field(0, description="Total number of topics")
    main_topic: Optional[str] = Field(None, description="Main topic identified")
    confidence_score: Optional[float] = Field(None, description="Overall confidence score")


class BrainstormGenerateResponse(APIResponse):
    """Enhanced brainstorming response."""
    ideas: List[BrainstormIdea] = Field(default_factory=list, description="Generated ideas")
    total_count: int = Field(0, description="Total number of ideas generated")
    creativity_score: Optional[float] = Field(None, description="Creativity score of ideas")


class CollaborationResponse(APIResponse):
    """Enhanced collaboration response."""
    session_id: Optional[str] = Field(None, description="ID of the collaboration session")
    session_name: str = Field(..., description="Name of the session")
    participants: List[str] = Field(default_factory=list, description="List of participants")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")


class HealthCheckResponse(BaseModel):
    """Enhanced health check response."""
    status: Literal["healthy", "unhealthy", "degraded"] = Field(..., description="System status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    services: Dict[str, bool] = Field(default_factory=dict, description="Service status")
    database_status: Literal["connected", "disconnected", "error"] = Field(..., description="Database status")
    message: Optional[str] = Field(default=None, description="Status message")
    uptime_seconds: Optional[float] = Field(None, description="System uptime")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")


class MetricsResponse(BaseModel):
    """Enhanced metrics response."""
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metrics timestamp")
    counters: Dict[str, int] = Field(default_factory=dict, description="Counter metrics")
    timers: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Timer metrics")
    gauges: Dict[str, float] = Field(default_factory=dict, description="Gauge metrics")
    system_info: Dict[str, Any] = Field(default_factory=dict, description="System information")


class BatchProcessResponse(BaseModel):
    """Enhanced batch processing response."""
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successfully processed files")
    failed: int = Field(..., description="Number of failed files")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Processing results")
    processing_time: float = Field(..., description="Total processing time")
    average_time_per_file: float = Field(..., description="Average processing time per file")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")


class FeatureToggleRequest(BaseModel):
    """Feature toggle request."""
    feature_name: str = Field(..., description="Name of the feature to toggle")
    enabled: bool = Field(..., description="Whether to enable or disable the feature")
    reason: Optional[str] = Field(None, description="Reason for the change")
    
    @validator('feature_name')
    def validate_feature_name(cls, v):
        allowed_features = [
            "pdf_upload", "variant_generation", "topic_extraction", "brainstorming",
            "ai_enhancement", "collaboration", "analytics", "real_time_processing"
        ]
        if v not in allowed_features:
            raise ValueError(f'Feature must be one of: {", ".join(allowed_features)}')
        return v


class ConfigurationResponse(BaseModel):
    """Enhanced configuration response."""
    environment: str = Field(..., description="Current environment")
    debug: bool = Field(..., description="Debug mode status")
    features: Dict[str, bool] = Field(..., description="Feature flags")
    performance: Dict[str, Any] = Field(..., description="Performance settings")
    security: Dict[str, Any] = Field(..., description="Security settings")
    monitoring: Dict[str, Any] = Field(..., description="Monitoring settings")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Configuration timestamp")


# --- Validation Utilities ---
def validate_file_size(file_size: int, max_size_mb: int = 100) -> bool:
    """Validate file size."""
    max_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_bytes and file_size > 0


def validate_email_list(emails: List[str]) -> bool:
    """Validate list of email addresses."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return all(re.match(email_pattern, email) for email in emails)


def validate_permissions(permissions: Dict[str, List[str]]) -> bool:
    """Validate permission structure."""
    allowed_permissions = ["read", "write", "admin", "delete", "share"]
    for user_permissions in permissions.values():
        if not all(perm in allowed_permissions for perm in user_permissions):
            return False
    return True


# --- Custom Validators ---
class CustomValidator:
    """Custom validation utilities."""
    
    @staticmethod
    def validate_pdf_content(content: str) -> bool:
        """Validate PDF content is not empty."""
        return bool(content and content.strip())
    
    @staticmethod
    def validate_topic_relevance(relevance: float) -> bool:
        """Validate topic relevance score."""
        return 0.0 <= relevance <= 1.0
    
    @staticmethod
    def validate_idea_creativity(creativity: str) -> bool:
        """Validate creativity level."""
        return creativity in ["conservative", "balanced", "creative"]
    
    @staticmethod
    def validate_session_name(name: str) -> bool:
        """Validate collaboration session name."""
        return bool(name and len(name.strip()) >= 1 and len(name.strip()) <= 100)
