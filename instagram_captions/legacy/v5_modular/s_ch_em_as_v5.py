from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
    from .config_v5 import config
    from config_v5 import config
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v5.0 - Schemas Module

Pydantic models for ultra-fast mass processing with advanced validation.
"""

try:
except ImportError:


class UltraFastCaptionRequest(BaseModel):
    """Ultra-fast caption request with optimized validation."""
    
    content_description: str = Field(
        ..., 
        min_length=5, 
        max_length=1000,
        description="Content description for caption generation"
    )
    style: str = Field(
        default="casual",
        pattern="^(casual|professional|playful|inspirational|educational|promotional)$",
        description="Caption style"
    )
    audience: str = Field(
        default="general",
        pattern="^(general|business|millennials|gen_z|creators|lifestyle)$",
        description="Target audience"
    )
    include_hashtags: bool = Field(
        default=True,
        description="Include hashtags in response"
    )
    hashtag_count: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of hashtags to generate"
    )
    content_type: str = Field(
        default="post",
        pattern="^(post|story|reel|carousel)$",
        description="Instagram content type"
    )
    priority: str = Field(
        default="normal",
        pattern="^(low|normal|high|urgent)$",
        description="Processing priority"
    )
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Client identifier for tracking"
    )
    
    @field_validator('content_description')
    @classmethod
    def validate_content_description(cls, v: str) -> str:
        """Validate and sanitize content description."""
        if not v.strip():
            raise ValueError("Content description cannot be empty")
        
        # Security: Remove potentially dangerous content
        dangerous_patterns = [
            '<script>', '</script>', '<iframe>', '</iframe>',
            'javascript:', 'onload=', 'onerror=', 'onclick='
        ]
        
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Content contains potentially dangerous pattern: {pattern}")
        
        return v.strip()
    
    @field_validator('client_id')
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        """Validate client ID format."""
        if not v.strip():
            raise ValueError("Client ID cannot be empty")
        
        # Remove special characters for security
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        if not all(c in allowed_chars for c in v):
            raise ValueError("Client ID contains invalid characters")
        
        return v.strip()


class BatchCaptionRequest(BaseModel):
    """Batch processing request for mass caption generation."""
    
    requests: List[UltraFastCaptionRequest] = Field(
        ...,
        max_length=config.MAX_BATCH_SIZE,
        description=f"List of caption requests (max {config.MAX_BATCH_SIZE})"
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique batch identifier"
    )
    
    @field_validator('requests')
    @classmethod
    async def validate_requests(cls, v: List[UltraFastCaptionRequest]) -> List[UltraFastCaptionRequest]:
        """Validate batch requests."""
        if not v:
            raise ValueError("Batch cannot be empty")
        
        if len(v) > config.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {config.MAX_BATCH_SIZE}")
        
        return v
    
    @field_validator('batch_id')
    @classmethod
    def validate_batch_id(cls, v: str) -> str:
        """Validate batch ID format."""
        if not v.strip():
            raise ValueError("Batch ID cannot be empty")
        
        return v.strip()


class UltraFastCaptionResponse(BaseModel):
    """Ultra-fast caption response with comprehensive metadata."""
    
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Response status")
    caption: str = Field(..., description="Generated caption")
    hashtags: List[str] = Field(..., description="Generated hashtags")
    quality_score: float = Field(..., ge=0, le=100, description="Quality score out of 100")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Response timestamp")
    cache_hit: bool = Field(..., description="Whether response came from cache")
    api_version: str = Field(..., description="API version")


class BatchCaptionResponse(BaseModel):
    """Batch processing response with aggregated metrics."""
    
    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Batch processing status")
    results: List[UltraFastCaptionResponse] = Field(..., description="Individual caption results")
    total_processed: int = Field(..., ge=0, description="Total captions processed")
    total_time_ms: float = Field(..., ge=0, description="Total processing time")
    avg_quality_score: float = Field(..., ge=0, le=100, description="Average quality score")
    cache_hits: int = Field(..., ge=0, description="Number of cache hits")
    timestamp: datetime = Field(..., description="Batch completion timestamp")
    api_version: str = Field(..., description="API version")


class UltraHealthResponse(BaseModel):
    """Ultra-fast health check response."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    performance_grade: str = Field(..., description="Performance grade (A+, A, B, C)")


class MetricsResponse(BaseModel):
    """Detailed metrics response."""
    
    api_version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Metrics timestamp")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    configuration: Dict[str, Any] = Field(..., description="Configuration settings")
    capabilities: Dict[str, str] = Field(..., description="API capabilities")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    error: Dict[str, Any] = Field(..., description="Error details")
    
    @classmethod
    def create(cls, message: str, status_code: int, request_id: str = "unknown") -> "ErrorResponse":
        """Create standardized error response."""
        return cls(
            error={
                "message": message,
                "status_code": status_code,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "api_version": config.API_VERSION
            }
        )


# Validation utilities
class ValidationUtils:
    """Utility functions for advanced validation."""
    
    @staticmethod
    def is_valid_content_length(content: str, min_length: int = 5, max_length: int = 1000) -> bool:
        """Check if content length is within valid range."""
        return min_length <= len(content.strip()) <= max_length
    
    @staticmethod
    def has_valid_characters(text: str) -> bool:
        """Check if text contains only valid characters."""
        # Allow alphanumeric, spaces, and common punctuation
        valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_()[]{}:;"\'@#$%&*+=/')
        return all(c in valid_chars or ord(c) > 127 for c in text)  # Allow emojis (ord > 127)
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text by removing potentially dangerous content."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove JavaScript-like patterns
        js_patterns = [
            r'javascript:',
            r'on\w+\s*=',
            r'<script.*?</script>',
            r'<iframe.*?</iframe>'
        ]
        
        for pattern in js_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()


# Export all schemas
__all__ = [
    'UltraFastCaptionRequest',
    'BatchCaptionRequest', 
    'UltraFastCaptionResponse',
    'BatchCaptionResponse',
    'UltraHealthResponse',
    'MetricsResponse',
    'ErrorResponse',
    'ValidationUtils'
] 