"""Ultra-efficient schemas with minimal overhead."""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, Field, validator
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


# --- Ultra-Fast Request Models ---
class UltraFastPDFUploadRequest(BaseModel):
    """Ultra-fast PDF upload request."""
    filename: Optional[str] = Field(None, description="PDF filename")
    auto_process: bool = Field(True, description="Auto-process after upload")
    extract_text: bool = Field(True, description="Extract text content")
    max_file_size_mb: int = Field(100, ge=1, le=500, description="Max file size")
    
    @validator('filename')
    def validate_filename(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9._-]+\.pdf$', v):
            raise ValueError('Invalid PDF filename')
        return v


class UltraFastVariantRequest(BaseModel):
    """Ultra-fast variant generation request."""
    variant_type: str = Field(..., description="Variant type")
    quality: Literal["draft", "standard", "high"] = Field("standard", description="Quality")
    
    @validator('variant_type')
    def validate_variant_type(cls, v):
        allowed_types = ["summary", "outline", "highlights", "translation"]
        if v not in allowed_types:
            raise ValueError(f'Variant type must be one of: {", ".join(allowed_types)}')
        return v


class UltraFastTopicRequest(BaseModel):
    """Ultra-fast topic extraction request."""
    min_relevance: float = Field(0.5, ge=0.0, le=1.0, description="Min relevance")
    max_topics: int = Field(50, ge=1, le=200, description="Max topics")


class UltraFastBatchRequest(BaseModel):
    """Ultra-fast batch processing request."""
    max_concurrent: int = Field(50, ge=1, le=100, description="Max concurrent")
    include_topics: bool = Field(True, description="Include topics")
    include_variants: bool = Field(True, description="Include variants")


# --- Ultra-Fast Response Models ---
class UltraFastResponse(BaseModel):
    """Ultra-fast generic response."""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: float = Field(..., description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID")


class UltraFastUploadResponse(UltraFastResponse):
    """Ultra-fast upload response."""
    file_id: Optional[str] = Field(None, description="File ID")
    file_size: Optional[str] = Field(None, description="File size")
    processing_started: bool = Field(False, description="Processing started")


class UltraFastVariantResponse(UltraFastResponse):
    """Ultra-fast variant response."""
    variant: Optional[Dict[str, Any]] = Field(None, description="Generated variant")
    generation_time: Optional[float] = Field(None, description="Generation time")


class UltraFastTopicResponse(UltraFastResponse):
    """Ultra-fast topic response."""
    topics: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted topics")
    total_count: int = Field(0, description="Total topics")


class UltraFastBatchResponse(UltraFastResponse):
    """Ultra-fast batch response."""
    total_files: int = Field(..., description="Total files")
    successful: int = Field(..., description="Successful files")
    failed: int = Field(..., description="Failed files")
    processing_time: float = Field(..., description="Processing time")


class UltraFastHealthResponse(BaseModel):
    """Ultra-fast health response."""
    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    timestamp: float = Field(..., description="Check timestamp")
    uptime: float = Field(..., description="System uptime")
    services: Dict[str, str] = Field(default_factory=dict, description="Service status")


class UltraFastMetricsResponse(BaseModel):
    """Ultra-fast metrics response."""
    timestamp: float = Field(..., description="Metrics timestamp")
    requests: int = Field(..., description="Total requests")
    errors: int = Field(..., description="Total errors")
    uptime: float = Field(..., description="System uptime")
    avg_response_time: float = Field(..., description="Average response time")
    requests_per_second: float = Field(..., description="Requests per second")


class UltraFastStatusResponse(BaseModel):
    """Ultra-fast status response."""
    file_id: str = Field(..., description="File ID")
    status: str = Field(..., description="Processing status")
    has_topics: bool = Field(False, description="Has topics")
    has_variants: bool = Field(False, description="Has variants")
    last_processed: float = Field(..., description="Last processed timestamp")


class UltraFastFeatureResponse(BaseModel):
    """Ultra-fast feature response."""
    file_id: str = Field(..., description="File ID")
    features: Dict[str, Any] = Field(default_factory=dict, description="Extracted features")
    extracted_at: float = Field(..., description="Extraction timestamp")


# --- Ultra-Fast Validation Utilities ---
def ultra_fast_validate_file_size(file_size: int, max_size_mb: int = 100) -> bool:
    """Ultra-fast file size validation."""
    max_bytes = max_size_mb * 1024 * 1024
    return 0 < file_size <= max_bytes


def ultra_fast_validate_filename(filename: str) -> bool:
    """Ultra-fast filename validation."""
    if not filename:
        return False
    
    # Check extension
    if not filename.lower().endswith('.pdf'):
        return False
    
    # Check length
    if len(filename) > 255:
        return False
    
    # Check characters
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        return False
    
    return True


def ultra_fast_validate_email(email: str) -> bool:
    """Ultra-fast email validation."""
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def ultra_fast_validate_url(url: str) -> bool:
    """Ultra-fast URL validation."""
    if not url:
        return False
    
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def ultra_fast_sanitize_text(text: str, max_length: int = 1000) -> str:
    """Ultra-fast text sanitization."""
    if not text:
        return ""
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip and limit length
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    
    return text


def ultra_fast_format_size(size_bytes: int) -> str:
    """Ultra-fast size formatting."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def ultra_fast_format_time(seconds: float) -> str:
    """Ultra-fast time formatting."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds / 60:.1f}m"


def ultra_fast_generate_id(prefix: str = "") -> str:
    """Ultra-fast ID generation."""
    import secrets
    random_part = secrets.token_hex(8)
    return f"{prefix}{random_part}" if prefix else random_part


# --- Ultra-Fast Serialization ---
def ultra_fast_serialize(data: Any) -> str:
    """Ultra-fast serialization."""
    if isinstance(data, str):
        return data
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, dict):
        return str(data)
    return str(data)


def ultra_fast_deserialize(data: str, target_type: type) -> Any:
    """Ultra-fast deserialization."""
    if target_type == str:
        return data
    if target_type == int:
        return int(data)
    if target_type == float:
        return float(data)
    if target_type == bool:
        return data.lower() == 'true'
    return data


# --- Ultra-Fast Content Validation ---
def ultra_fast_validate_content(content: str) -> bool:
    """Ultra-fast content validation."""
    if not content:
        return False
    
    # Check length
    if len(content) > 1000000:  # 1MB limit
        return False
    
    # Check for valid characters
    if not content.isprintable():
        return False
    
    return True


def ultra_fast_validate_topics(topics: List[str]) -> bool:
    """Ultra-fast topics validation."""
    if not topics:
        return False
    
    # Check length
    if len(topics) > 100:
        return False
    
    # Check each topic
    for topic in topics:
        if not topic or len(topic) > 100:
            return False
    
    return True


def ultra_fast_validate_variants(variants: Dict[str, Any]) -> bool:
    """Ultra-fast variants validation."""
    if not variants:
        return False
    
    # Check number of variants
    if len(variants) > 10:
        return False
    
    # Check each variant
    for key, value in variants.items():
        if not key or len(key) > 50:
            return False
        if not value or len(str(value)) > 10000:
            return False
    
    return True


# --- Ultra-Fast Error Models ---
class UltraFastError(BaseModel):
    """Ultra-fast error model."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID")


class UltraFastValidationError(UltraFastError):
    """Ultra-fast validation error."""
    field: str = Field(..., description="Field name")
    value: Any = Field(..., description="Invalid value")


class UltraFastProcessingError(UltraFastError):
    """Ultra-fast processing error."""
    file_id: str = Field(..., description="File ID")
    processing_step: str = Field(..., description="Processing step")


# --- Ultra-Fast Configuration Models ---
class UltraFastConfigResponse(BaseModel):
    """Ultra-fast configuration response."""
    environment: str = Field(..., description="Environment")
    debug: bool = Field(..., description="Debug mode")
    features: Dict[str, bool] = Field(..., description="Feature flags")
    performance: Dict[str, Any] = Field(..., description="Performance settings")
    timestamp: float = Field(..., description="Configuration timestamp")


class UltraFastFeatureToggleRequest(BaseModel):
    """Ultra-fast feature toggle request."""
    feature_name: str = Field(..., description="Feature name")
    enabled: bool = Field(..., description="Enable/disable feature")
    
    @validator('feature_name')
    def validate_feature_name(cls, v):
        allowed_features = [
            "pdf_upload", "variant_generation", "topic_extraction",
            "batch_processing", "caching", "compression"
        ]
        if v not in allowed_features:
            raise ValueError(f'Feature must be one of: {", ".join(allowed_features)}')
        return v
