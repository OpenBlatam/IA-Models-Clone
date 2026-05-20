"""
Advanced Data Validation with Pydantic v2
Enhanced validation, custom validators, and error handling
"""

from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator, constr, conint, confloat
from pydantic.validators import str_validator
import re
import logging

logger = logging.getLogger(__name__)


class ContentAnalysisRequest(BaseModel):
    """Advanced validation for content analysis requests"""
    
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=1000000,
        description="Content to analyze"
    )
    analysis_type: Literal["redundancy", "quality", "similarity", "comprehensive"] = Field(
        default="comprehensive",
        description="Type of analysis to perform"
    )
    language: Optional[str] = Field(
        default="auto",
        regex=r"^[a-z]{2}(-[A-Z]{2})?$|^auto$",
        description="Language code (ISO 639-1) or 'auto' for detection"
    )
    threshold: confloat(ge=0.0, le=1.0) = Field(
        default=0.7,
        description="Similarity threshold (0.0 to 1.0)"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in response"
    )
    cache_result: bool = Field(
        default=True,
        description="Cache the analysis result"
    )
    priority: conint(ge=1, le=10) = Field(
        default=5,
        description="Processing priority (1=low, 10=high)"
    )
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content quality and format"""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'data:text/html',  # Data URLs
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                logger.warning(f"Suspicious content detected: {pattern}")
                # Don't raise error, just log warning
        
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        """Validate language code"""
        if v == "auto":
            return v
        
        # Common language codes
        valid_languages = {
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
            'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no'
        }
        
        if v not in valid_languages:
            logger.warning(f"Unsupported language: {v}, defaulting to 'auto'")
            return "auto"
        
        return v
    
    @root_validator
    def validate_analysis_configuration(cls, values):
        """Validate overall analysis configuration"""
        analysis_type = values.get('analysis_type')
        threshold = values.get('threshold')
        
        # Adjust threshold based on analysis type
        if analysis_type == "redundancy" and threshold < 0.5:
            logger.warning("Low threshold for redundancy analysis may produce many false positives")
        elif analysis_type == "similarity" and threshold > 0.9:
            logger.warning("High threshold for similarity analysis may miss similar content")
        
        return values


class SimilarityRequest(BaseModel):
    """Advanced validation for similarity comparison requests"""
    
    text1: str = Field(..., min_length=1, max_length=500000)
    text2: str = Field(..., min_length=1, max_length=500000)
    algorithm: Literal["cosine", "jaccard", "levenshtein", "semantic", "hybrid"] = Field(
        default="hybrid",
        description="Similarity algorithm to use"
    )
    threshold: confloat(ge=0.0, le=1.0) = Field(
        default=0.7,
        description="Similarity threshold"
    )
    normalize: bool = Field(
        default=True,
        description="Normalize text before comparison"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Case-sensitive comparison"
    )
    
    @validator('text1', 'text2')
    def validate_texts(cls, v):
        """Validate text content"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Check text length ratio
        return v.strip()
    
    @root_validator
    def validate_texts_compatibility(cls, values):
        """Validate text compatibility"""
        text1 = values.get('text1', '')
        text2 = values.get('text2', '')
        
        # Check if texts are too different in length
        len1, len2 = len(text1), len(text2)
        if len1 > 0 and len2 > 0:
            ratio = min(len1, len2) / max(len1, len2)
            if ratio < 0.1:
                logger.warning("Texts have very different lengths, similarity may be low")
        
        return values


class QualityRequest(BaseModel):
    """Advanced validation for quality analysis requests"""
    
    content: str = Field(..., min_length=1, max_length=1000000)
    quality_metrics: List[Literal["readability", "grammar", "coherence", "completeness", "clarity"]] = Field(
        default=["readability", "grammar", "coherence"],
        description="Quality metrics to analyze"
    )
    target_audience: Literal["general", "academic", "technical", "casual"] = Field(
        default="general",
        description="Target audience for quality assessment"
    )
    min_score: confloat(ge=0.0, le=1.0) = Field(
        default=0.0,
        description="Minimum acceptable quality score"
    )
    
    @validator('content')
    def validate_content_quality(cls, v):
        """Validate content for quality analysis"""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        
        # Basic quality checks
        word_count = len(v.split())
        if word_count < 10:
            logger.warning("Content is very short, quality analysis may be limited")
        elif word_count > 10000:
            logger.warning("Content is very long, analysis may take longer")
        
        return v.strip()


class BatchRequest(BaseModel):
    """Advanced validation for batch processing requests"""
    
    items: List[Dict[str, Any]] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of items to process"
    )
    batch_size: conint(ge=1, le=100) = Field(
        default=10,
        description="Number of items to process in parallel"
    )
    priority: conint(ge=1, le=10) = Field(
        default=5,
        description="Batch processing priority"
    )
    callback_url: Optional[str] = Field(
        default=None,
        regex=r"^https?://.+",
        description="Callback URL for batch completion notification"
    )
    
    @validator('items')
    def validate_items(cls, v):
        """Validate batch items"""
        if not v:
            raise ValueError("Items list cannot be empty")
        
        # Check for duplicate items
        item_hashes = set()
        for item in v:
            item_str = str(sorted(item.items()))
            item_hash = hash(item_str)
            if item_hash in item_hashes:
                logger.warning("Duplicate items detected in batch")
            item_hashes.add(item_hash)
        
        return v


class WebhookRequest(BaseModel):
    """Advanced validation for webhook requests"""
    
    url: str = Field(
        ...,
        regex=r"^https?://.+",
        description="Webhook URL"
    )
    events: List[Literal["analysis.completed", "similarity.completed", "quality.completed", "batch.completed"]] = Field(
        ...,
        min_items=1,
        description="Events to subscribe to"
    )
    secret: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=256,
        description="Webhook secret for verification"
    )
    retry_attempts: conint(ge=0, le=5) = Field(
        default=3,
        description="Number of retry attempts"
    )
    timeout: conint(ge=5, le=300) = Field(
        default=30,
        description="Request timeout in seconds"
    )
    
    @validator('url')
    def validate_webhook_url(cls, v):
        """Validate webhook URL"""
        # Check for localhost/private IPs in production
        if re.match(r'https?://(localhost|127\.0\.0\.1|192\.168\.|10\.|172\.)', v):
            logger.warning("Webhook URL points to local/private network")
        
        return v


class APIResponse(BaseModel):
    """Standardized API response model"""
    
    success: bool = Field(..., description="Request success status")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error information")
    message: Optional[str] = Field(default=None, description="Human-readable message")
    timestamp: float = Field(..., description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp"""
        import time
        current_time = time.time()
        if v > current_time + 1:  # Allow 1 second tolerance
            raise ValueError("Timestamp cannot be in the future")
        return v


class ErrorResponse(BaseModel):
    """Standardized error response model"""
    
    success: bool = Field(default=False, description="Always false for errors")
    data: Optional[Any] = Field(default=None, description="Always null for errors")
    error: Dict[str, Any] = Field(..., description="Error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "data": None,
                "error": {
                    "message": "Validation error",
                    "status_code": 400,
                    "type": "ValidationError",
                    "detail": {
                        "field": "content",
                        "issue": "Content cannot be empty"
                    }
                },
                "timestamp": 1640995200.0,
                "request_id": "req_123456789"
            }
        }


# Custom validators
def validate_content_length(content: str, min_length: int = 1, max_length: int = 1000000) -> str:
    """Custom validator for content length"""
    if len(content) < min_length:
        raise ValueError(f"Content must be at least {min_length} characters long")
    if len(content) > max_length:
        raise ValueError(f"Content must be no more than {max_length} characters long")
    return content


def validate_similarity_threshold(threshold: float) -> float:
    """Custom validator for similarity threshold"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be between 0.0 and 1.0")
    return threshold


def validate_language_code(language: str) -> str:
    """Custom validator for language codes"""
    if language == "auto":
        return language
    
    # ISO 639-1 language codes
    valid_codes = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh',
        'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no',
        'fi', 'cs', 'hu', 'ro', 'bg', 'hr', 'sk', 'sl', 'et', 'lv',
        'lt', 'mt', 'ga', 'cy', 'eu', 'ca', 'gl', 'is', 'mk', 'sq'
    }
    
    if language not in valid_codes:
        raise ValueError(f"Invalid language code: {language}")
    
    return language





