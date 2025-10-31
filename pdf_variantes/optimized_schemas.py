"""Optimized schemas with functional validation patterns."""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from uuid import uuid4
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import re

# Re-import existing models
from .models import (
    VariantStatus, PDFProcessingStatus, TopicCategory, PDFMetadata, EditedPage,
    PDFDocument, VariantConfiguration, PDFVariant, TopicItem, BrainstormIdea
)


# --- Optimized Request Models ---
class OptimizedPDFUploadRequest(BaseModel):
    """Optimized PDF upload request with early validation."""
    filename: Optional[str] = Field(None, description="PDF filename")
    auto_process: bool = Field(True, description="Auto-process after upload")
    extract_text: bool = Field(True, description="Extract text content")
    max_file_size_mb: int = Field(100, ge=1, le=500, description="Max file size")
    
    @validator('filename')
    def validate_filename(cls, v):
        if v and not re.match(r'^[a-zA-Z0-9._-]+\.pdf$', v):
            raise ValueError('Invalid PDF filename')
        return v
    
    @validator('max_file_size_mb')
    def validate_file_size(cls, v):
        if v > 500:
            raise ValueError('File size too large')
        return v


class OptimizedVariantRequest(BaseModel):
    """Optimized variant generation request."""
    variant_type: str = Field(..., description="Variant type")
    quality: Literal["draft", "standard", "high"] = Field("standard", description="Quality")
    include_summary: bool = Field(True, description="Include summary")
    include_outline: bool = Field(True, description="Include outline")
    include_highlights: bool = Field(True, description="Include highlights")
    
    @validator('variant_type')
    def validate_variant_type(cls, v):
        allowed_types = ["summary", "outline", "highlights", "all"]
        if v not in allowed_types:
            raise ValueError(f'Variant type must be one of: {", ".join(allowed_types)}')
        return v


class OptimizedTopicRequest(BaseModel):
    """Optimized topic extraction request."""
    min_relevance: float = Field(0.5, ge=0.0, le=1.0, description="Min relevance score")
    max_topics: int = Field(50, ge=1, le=200, description="Max topics")
    methods: List[str] = Field(["rake", "spacy", "tfidf"], description="Extraction methods")
    
    @validator('methods')
    def validate_methods(cls, v):
        allowed_methods = ["rake", "spacy", "tfidf", "lda"]
        for method in v:
            if method not in allowed_methods:
                raise ValueError(f'Method must be one of: {", ".join(allowed_methods)}')
        return v


class OptimizedBatchRequest(BaseModel):
    """Optimized batch processing request."""
    max_concurrent: int = Field(20, ge=1, le=50, description="Max concurrent processing")
    include_topics: bool = Field(True, description="Include topic extraction")
    include_variants: bool = Field(True, description="Include variant generation")
    include_quality: bool = Field(True, description="Include quality analysis")
    
    @validator('max_concurrent')
    def validate_concurrency(cls, v):
        if v > 50:
            raise ValueError('Max concurrent processing limited to 50')
        return v


# --- Optimized Response Models ---
class OptimizedResponse(BaseModel):
    """Optimized generic response."""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid4: lambda v: str(v)
        }


class OptimizedUploadResponse(OptimizedResponse):
    """Optimized upload response."""
    file_id: Optional[str] = Field(None, description="File ID")
    file_size: Optional[str] = Field(None, description="File size")
    processing_started: bool = Field(False, description="Processing started")


class OptimizedProcessResponse(OptimizedResponse):
    """Optimized processing response."""
    file_id: str = Field(..., description="File ID")
    topics: Optional[Dict[str, Any]] = Field(None, description="Extracted topics")
    variants: Optional[Dict[str, Any]] = Field(None, description="Generated variants")
    quality: Optional[Dict[str, Any]] = Field(None, description="Quality analysis")
    processing_time: Optional[float] = Field(None, description="Processing time")


class OptimizedTopicResponse(OptimizedResponse):
    """Optimized topic response."""
    file_id: str = Field(..., description="File ID")
    topics: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted topics")
    total_count: int = Field(0, description="Total topics")
    main_topic: Optional[str] = Field(None, description="Main topic")
    confidence: float = Field(0.0, description="Confidence score")


class OptimizedBatchResponse(OptimizedResponse):
    """Optimized batch response."""
    total_files: int = Field(..., description="Total files")
    successful: int = Field(..., description="Successful files")
    failed: int = Field(..., description="Failed files")
    success_rate: float = Field(..., description="Success rate")
    processing_time: float = Field(..., description="Total processing time")


class OptimizedQualityResponse(OptimizedResponse):
    """Optimized quality response."""
    file_id: str = Field(..., description="File ID")
    quality_score: float = Field(..., description="Quality score")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quality metrics")
    quality_factors: Dict[str, float] = Field(default_factory=dict, description="Quality factors")


class OptimizedHealthResponse(BaseModel):
    """Optimized health response."""
    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    uptime_seconds: float = Field(..., description="System uptime")
    services: Dict[str, str] = Field(default_factory=dict, description="Service status")
    features: Dict[str, int] = Field(default_factory=dict, description="Feature counts")


class OptimizedStatusResponse(BaseModel):
    """Optimized status response."""
    file_id: str = Field(..., description="File ID")
    status: str = Field(..., description="Processing status")
    has_topics: bool = Field(False, description="Has topics")
    has_quality: bool = Field(False, description="Has quality analysis")
    has_preview: bool = Field(False, description="Has preview")
    last_checked: datetime = Field(default_factory=datetime.utcnow, description="Last checked")


# --- Functional Validation Utilities ---
def validate_file_size(file_size: int, max_size_mb: int = 100) -> bool:
    """Validate file size with early return."""
    if file_size <= 0:
        return False
    
    max_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_bytes


def validate_filename(filename: str) -> bool:
    """Validate filename with early returns."""
    if not filename:
        return False
    
    if len(filename) > 255:
        return False
    
    if not filename.lower().endswith('.pdf'):
        return False
    
    if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
        return False
    
    return True


def validate_email(email: str) -> bool:
    """Validate email with early return."""
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL with early return."""
    if not url:
        return False
    
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))


def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize text with early returns."""
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


def format_file_size(size_bytes: int) -> str:
    """Format file size with early returns."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def format_duration(seconds: float) -> str:
    """Format duration with early returns."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def generate_id(prefix: str = "") -> str:
    """Generate unique ID."""
    import secrets
    random_part = secrets.token_hex(8)
    return f"{prefix}{random_part}" if prefix else random_part


# --- Content Validation ---
def validate_content_length(content: str, max_length: int = 1000000) -> bool:
    """Validate content length with early return."""
    if not content:
        return False
    
    return len(content) <= max_length


def validate_topic_count(topics: List[str], max_count: int = 100) -> bool:
    """Validate topic count with early return."""
    if not topics:
        return False
    
    return len(topics) <= max_count


def validate_variant_count(variants: Dict[str, Any], max_count: int = 10) -> bool:
    """Validate variant count with early return."""
    if not variants:
        return False
    
    return len(variants) <= max_count


# --- Error Models ---
class OptimizedError(BaseModel):
    """Optimized error model."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID")


class OptimizedValidationError(OptimizedError):
    """Optimized validation error."""
    field: str = Field(..., description="Field name")
    value: Any = Field(..., description="Invalid value")


class OptimizedProcessingError(OptimizedError):
    """Optimized processing error."""
    file_id: str = Field(..., description="File ID")
    processing_step: str = Field(..., description="Processing step")


# --- Configuration Models ---
class OptimizedConfigResponse(BaseModel):
    """Optimized configuration response."""
    environment: str = Field(..., description="Environment")
    debug: bool = Field(..., description="Debug mode")
    features: Dict[str, bool] = Field(..., description="Feature flags")
    performance: Dict[str, Any] = Field(..., description="Performance settings")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Configuration timestamp")


class OptimizedFeatureToggleRequest(BaseModel):
    """Optimized feature toggle request."""
    feature_name: str = Field(..., description="Feature name")
    enabled: bool = Field(..., description="Enable/disable feature")
    
    @validator('feature_name')
    def validate_feature_name(cls, v):
        allowed_features = [
            "pdf_upload", "variant_generation", "topic_extraction",
            "batch_processing", "quality_analysis", "caching"
        ]
        if v not in allowed_features:
            raise ValueError(f'Feature must be one of: {", ".join(allowed_features)}')
        return v
