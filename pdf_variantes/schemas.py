"""
Pydantic Schemas
================

Pydantic schemas for request/response validation and serialization.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum


class AnnotationType(str, Enum):
    """Annotation types."""
    TEXT = "text"
    HIGHLIGHT = "highlight"
    NOTE = "note"
    COMMENT = "comment"
    QUESTION = "question"
    SUGGESTION = "suggestion"


class VariantType(str, Enum):
    """Variant types with descriptions."""
    SUMMARY = "summary"
    OUTLINE = "outline"
    HIGHLIGHTS = "highlights"
    NOTES = "notes"
    QUIZ = "quiz"
    PRESENTATION = "presentation"
    
    @property
    def description(self) -> str:
        """Get human-readable description."""
        descriptions = {
            "summary": "Generate a condensed summary of the document",
            "outline": "Extract document structure and main points",
            "highlights": "Extract key points and important information",
            "notes": "Generate study notes from the content",
            "quiz": "Create quiz questions based on the content",
            "presentation": "Generate presentation slides from the document"
        }
        return descriptions.get(self.value, "Unknown variant type")


class PDFUploadSchema(BaseModel):
    """Schema for PDF upload request."""
    filename: Optional[str] = Field(None, description="Custom filename")
    auto_process: bool = Field(True, description="Auto-process PDF on upload")
    extract_text: bool = Field(True, description="Extract text content")
    detect_language: bool = Field(True, description="Detect document language")
    
    @validator('filename')
    def validate_filename(cls, v):
        if v and len(v) > 255:
            raise ValueError('Filename too long')
        return v


class PDFEditSchema(BaseModel):
    """Schema for PDF editing operations."""
    page_number: int = Field(..., ge=1, description="Page number to edit")
    annotation_type: AnnotationType = Field(..., description="Type of annotation")
    content: str = Field(..., min_length=1, max_length=5000, description="Annotation content")
    position: Dict[str, Any] = Field(..., description="Annotation position")
    properties: Optional[Dict[str, Any]] = Field(None, description="Additional properties")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @validator('position')
    def validate_position(cls, v):
        required_fields = ['x', 'y', 'width', 'height']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Position must include {field}')
            if not isinstance(v[field], (int, float)) or v[field] < 0:
                raise ValueError(f'{field} must be a positive number')
        return v


class VariantGenerateSchema(BaseModel):
    """Schema for variant generation request."""
    variant_type: VariantType = Field(..., description="Type of variant to generate")
    options: Optional[Dict[str, Any]] = Field(None, description="Generation options")
    preserve_structure: bool = Field(True, description="Preserve document structure")
    preserve_meaning: bool = Field(True, description="Preserve core meaning")
    creativity_level: float = Field(0.6, ge=0.0, le=1.0, description="Creativity level")
    
    @validator('options')
    def validate_options(cls, v):
        if v is None:
            return {}
        
        # Validate specific options
        if 'similarity_level' in v:
            similarity = v['similarity_level']
            if not isinstance(similarity, (int, float)) or not 0 <= similarity <= 1:
                raise ValueError('similarity_level must be between 0 and 1')
        
        return v


class TopicExtractSchema(BaseModel):
    """Schema for topic extraction request."""
    min_relevance: float = Field(0.5, ge=0.0, le=1.0, description="Minimum relevance score")
    max_topics: int = Field(50, ge=1, le=200, description="Maximum number of topics")
    include_context: bool = Field(True, description="Include context examples")
    categorize_topics: bool = Field(True, description="Categorize topics")
    
    @validator('max_topics')
    def validate_max_topics(cls, v):
        if v > 200:
            raise ValueError('Maximum topics cannot exceed 200')
        return v


class BrainstormGenerateSchema(BaseModel):
    """Schema for brainstorm generation request."""
    number_of_ideas: int = Field(20, ge=1, le=500, description="Number of ideas to generate")
    diversity_level: float = Field(0.7, ge=0.0, le=1.0, description="Diversity level")
    focus_areas: Optional[List[str]] = Field(None, description="Specific focus areas")
    creativity_mode: str = Field("balanced", description="Creativity mode")
    
    @validator('creativity_mode')
    def validate_creativity_mode(cls, v):
        valid_modes = ["conservative", "balanced", "creative", "experimental"]
        if v not in valid_modes:
            raise ValueError(f'creativity_mode must be one of {valid_modes}')
        return v
    
    @validator('focus_areas')
    def validate_focus_areas(cls, v):
        if v and len(v) > 10:
            raise ValueError('Maximum 10 focus areas allowed')
        return v


class SearchSchema(BaseModel):
    """Schema for search requests."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    search_fields: List[str] = Field(["content"], description="Fields to search")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Similarity threshold")
    max_results: int = Field(50, ge=1, le=500, description="Maximum results")
    use_fuzzy_search: bool = Field(True, description="Use fuzzy search")
    
    @validator('search_fields')
    def validate_search_fields(cls, v):
        valid_fields = ["content", "topics", "metadata", "differences", "annotations"]
        for field in v:
            if field not in valid_fields:
                raise ValueError(f'Invalid search field: {field}')
        return v


class BatchProcessingSchema(BaseModel):
    """Schema for batch processing requests."""
    file_ids: List[str] = Field(..., min_items=1, max_items=100, description="File IDs to process")
    operation: str = Field(..., description="Operation to perform")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Custom configuration")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    
    @validator('operation')
    def validate_operation(cls, v):
        valid_operations = [
            "generate_variants", "extract_topics", "generate_brainstorm",
            "add_annotations", "extract_images", "search_content"
        ]
        if v not in valid_operations:
            raise ValueError(f'Invalid operation: {v}')
        return v


class CollaborationSchema(BaseModel):
    """Schema for collaboration requests."""
    session_name: str = Field(..., min_length=1, max_length=100, description="Session name")
    participants: List[str] = Field(..., min_items=1, max_items=20, description="Participant user IDs")
    permissions: Dict[str, List[str]] = Field(..., description="Permissions per user")
    expires_at: Optional[datetime] = Field(None, description="Session expiration")
    
    @validator('permissions')
    def validate_permissions(cls, v):
        valid_permissions = ["view", "edit", "delete", "share", "comment"]
        for user_id, perms in v.items():
            for perm in perms:
                if perm not in valid_permissions:
                    raise ValueError(f'Invalid permission: {perm}')
        return v


class AnalyticsSchema(BaseModel):
    """Schema for analytics requests."""
    date_range_start: Optional[datetime] = Field(None, description="Start date")
    date_range_end: Optional[datetime] = Field(None, description="End date")
    metrics: List[str] = Field(["usage", "performance"], description="Metrics to include")
    group_by: Optional[str] = Field(None, description="Group results by field")
    
    @validator('metrics')
    def validate_metrics(cls, v):
        valid_metrics = ["usage", "performance", "quality", "errors", "collaboration"]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f'Invalid metric: {metric}')
        return v
    
    @root_validator
    def validate_date_range(cls, values):
        start = values.get('date_range_start')
        end = values.get('date_range_end')
        
        if start and end and start >= end:
            raise ValueError('Start date must be before end date')
        
        return values


class ConfigurationSchema(BaseModel):
    """Schema for configuration updates."""
    feature_name: str = Field(..., description="Feature name")
    enabled: bool = Field(..., description="Enable/disable feature")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Feature configuration")
    
    @validator('feature_name')
    def validate_feature_name(cls, v):
        valid_features = [
            "pdf_upload", "variant_generation", "topic_extraction",
            "brainstorming", "ai_enhancement", "collaboration",
            "analytics", "monitoring"
        ]
        if v not in valid_features:
            raise ValueError(f'Invalid feature: {v}')
        return v


class FilterSchema(BaseModel):
    """Schema for filtering requests."""
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity")
    max_similarity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum similarity")
    min_quality: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum quality")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    exclude_patterns: Optional[List[str]] = Field(None, description="Exclude patterns")
    include_keywords: Optional[List[str]] = Field(None, description="Include keywords")
    date_range_start: Optional[datetime] = Field(None, description="Date range start")
    date_range_end: Optional[datetime] = Field(None, description="Date range end")
    sort_by: Optional[str] = Field("similarity", description="Sort criteria")
    sort_order: str = Field("desc", description="Sort order")
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    
    @validator('sort_by')
    def validate_sort_by(cls, v):
        valid_sort_fields = ["similarity", "quality", "date", "relevance", "name"]
        if v not in valid_sort_fields:
            raise ValueError(f'Invalid sort field: {v}')
        return v
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v not in ["asc", "desc"]:
            raise ValueError('sort_order must be "asc" or "desc"')
        return v
    
    @root_validator
    def validate_similarity_range(cls, values):
        min_sim = values.get('min_similarity')
        max_sim = values.get('max_similarity')
        
        if min_sim is not None and max_sim is not None and min_sim > max_sim:
            raise ValueError('min_similarity cannot be greater than max_similarity')
        
        return values


class ExportSchema(BaseModel):
    """Schema for export requests."""
    file_ids: List[str] = Field(..., min_items=1, max_items=100, description="File IDs to export")
    export_format: str = Field("json", description="Export format")
    include_metadata: bool = Field(True, description="Include metadata")
    include_statistics: bool = Field(True, description="Include statistics")
    compress: bool = Field(False, description="Compress export file")
    
    @validator('export_format')
    def validate_export_format(cls, v):
        valid_formats = ["json", "csv", "xlsx", "txt", "pdf"]
        if v not in valid_formats:
            raise ValueError(f'Invalid export format: {v}')
        return v


class WebhookSchema(BaseModel):
    """Schema for webhook configuration."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook secret")
    enabled: bool = Field(True, description="Whether webhook is enabled")
    retry_on_failure: bool = Field(True, description="Retry on failure")
    timeout_seconds: int = Field(30, ge=5, le=300, description="Timeout in seconds")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('events')
    def validate_events(cls, v):
        valid_events = [
            "pdf_uploaded", "pdf_processed", "variant_generated",
            "topics_extracted", "brainstorm_completed", "annotation_added",
            "collaboration_started", "error_occurred"
        ]
        for event in v:
            if event not in valid_events:
                raise ValueError(f'Invalid event: {event}')
        return v


class NotificationSchema(BaseModel):
    """Schema for notification requests."""
    recipient_email: str = Field(..., description="Recipient email address")
    subject: str = Field(..., min_length=1, max_length=200, description="Email subject")
    body: str = Field(..., min_length=1, max_length=5000, description="Email body")
    notification_type: str = Field("user", description="Notification type")
    priority: str = Field("normal", description="Notification priority")
    
    @validator('recipient_email')
    def validate_email(cls, v):
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v
    
    @validator('notification_type')
    def validate_notification_type(cls, v):
        valid_types = ["system", "user", "alert", "marketing"]
        if v not in valid_types:
            raise ValueError(f'Invalid notification type: {v}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "normal", "high", "urgent"]
        if v not in valid_priorities:
            raise ValueError(f'Invalid priority: {v}')
        return v
