from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import re
import html
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.types import StrictStr, StrictInt, StrictFloat
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive Validation System for Instagram Captions API

This module provides:
- Pydantic v2 models with field validators
- Custom validation functions
- Input sanitization utilities
- Content validation for Instagram captions
- Security validation for user inputs
"""



# ============================================================================
# VALIDATION CONSTANTS
# ============================================================================

class ContentType(str, Enum):
    """Instagram content types with their character limits."""
    POST = "post"
    STORY = "story"
    REEL = "reel"
    CAROUSEL = "carousel"
    IGTV = "igtv"

CONTENT_LIMITS = {
    ContentType.POST: 2200,
    ContentType.STORY: 500,
    ContentType.REEL: 1000,
    ContentType.CAROUSEL: 2200,
    ContentType.IGTV: 2200
}

class ToneType(str, Enum):
    """Types for caption generation."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    HUMOROUS = "humorous"
    INSPIRATIONAL = "inspirational"
    EDUCATIONAL = "educational"

# ============================================================================
# BASE VALIDATION MODELS
# ============================================================================

class BaseValidationModel(BaseModel):
    """Model with common validation configuration."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        use_enum_values=True
    )

class CaptionRequest(BaseValidationModel):
    """Request model for caption generation."""
    prompt: str = Field(..., min_length=1, max_length=500, description="Caption prompt")
    content_type: ContentType = Field(default=ContentType.POST, description="Instagram content type")
    tone: ToneType = Field(default=ToneType.PROFESSIONAL, description="Desired tone")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags to include")
    max_length: Optional[int] = Field(None, ge=1, le=2200, description="Maximum caption length")
    include_hashtags: bool = Field(default=True, description="Include hashtags in caption")

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate and sanitize prompt."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        # Basic content filtering
        v = html.escape(v)
        
        return v
    
    @field_validator('hashtags')
    @classmethod
    def validate_hashtags(cls, v: List[str]) -> List[str]:
        """Validate and sanitize hashtags."""
        if not v:
            return v
        
        sanitized = []
        for hashtag in v:
            # Remove # if present and clean
            clean_tag = re.sub(r'[^\w\-_]', '', hashtag.strip().lstrip('#'))
            
            if clean_tag and len(clean_tag) <= 30:
                sanitized.append(f"#{clean_tag.lower()}")
        
        # Remove duplicates while preserving order
        unique_hashtags = []
        seen = set()
        for tag in sanitized:
            if tag not in seen:
                unique_hashtags.append(tag)
                seen.add(tag)
        
        return unique_hashtags[:30]  # Instagram limit
    
    @field_validator('max_length')
    @classmethod
    def validate_max_length(cls, v: Optional[int], info) -> Optional[int]:
        """Validate max_length against content type limits."""
        if v is None:
            return v
        
        content_type = info.data.get('content_type', ContentType.POST)
        content_limit = CONTENT_LIMITS[content_type]
        
        if v > content_limit:
            raise ValueError(f"max_length cannot exceed {content_limit} for {content_type}")
        
        return v
    
    @model_validator(mode='after')
    def validate_total_length(self) -> CaptionRequest:
        """Validate total caption length including hashtags."""
        if not self.include_hashtags:
            return self
        
        # Calculate hashtag length
        hashtag_length = sum(len(tag) + 1 for tag in self.hashtags)  # +1 for space
        
        # Get effective max length
        effective_max = self.max_length or CONTENT_LIMITS[self.content_type]
        
        # Check if prompt + hashtags would exceed limit
        total_length = len(self.prompt) + hashtag_length
        if total_length > effective_max:
            raise ValueError(
                f"Total length ({total_length}) would exceed limit ({effective_max})"
                f" for {self.content_type}"
            )
        
        return self

class BatchCaptionRequest(BaseValidationModel):
    """Request model for batch caption generation."""
    requests: List[CaptionRequest] = Field(..., min_length=1, max_length=100, description="Caption requests")
    batch_size: int = Field(default=10, ge=1, le=50, description="Batch processing size")

    @field_validator('requests')
    @classmethod
    async def validate_requests(cls, v: List[CaptionRequest]) -> List[CaptionRequest]:
        """Validate batch requests."""
        if not v:
            raise ValueError("At least one caption request is required")
        
        if len(v) > 100:
            raise ValueError("Maximum 100 requests per batch")
        
        return v

class CaptionResponse(BaseValidationModel):
    """Response model for caption generation."""
    caption: str = Field(..., description="Generated caption")
    content_type: ContentType = Field(..., description="Content type")
    length: int = Field(..., ge=0, description="Caption length")
    hashtags: List[str] = Field(default_factory=list, description="Included hashtags")
    tone: ToneType = Field(..., description="Applied tone")
    generation_time: float = Field(..., ge=0, description="Generation time in seconds")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Quality score")

    @field_validator('caption')
    @classmethod
    def validate_caption(cls, v: str) -> str:
        """Validate generated caption."""
        if not v or not v.strip():
            raise ValueError("Generated caption cannot be empty")
        
        # Check length against content type limits
        content_limit = CONTENT_LIMITS.get(ContentType.POST, 2200) # Default to post limit
        if len(v) > content_limit:
            raise ValueError(f"Generated caption exceeds {content_limit} character limit")
        
        return v.strip()

# ============================================================================
# SECURITY VALIDATION MODELS
# ============================================================================

class SecurityValidationConfig(BaseValidationModel):
    """Configuration for security validation."""
    max_input_length: int = Field(default=1000, description="Maximum input length")
    allowed_tags: List[str] = Field(default_factory=list, description="Allowed HTML tags")
    block_sql_injection: bool = Field(default=True, description="Block SQL injection patterns")
    block_xss: bool = Field(default=True, description="Block XSS patterns")
    block_path_traversal: bool = Field(default=True, description="Block path traversal patterns")

class UserInputValidation(BaseValidationModel):
    """Model for validating user inputs."""
    input_text: str = Field(..., description="User input to validate")
    input_type: str = Field(..., description="Type of input (text, url, email, etc.)")
    max_length: int = Field(default=10, description="Maximum allowed length")

    @field_validator('input_text')
    @classmethod
    def validate_input_text(cls, v: str) -> str:
        """Validate and sanitize user input."""
        if not v or not v.strip():
            raise ValueError("Input text cannot be empty")
        
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        # Basic length check
        if len(v) > 1000: # limit
            raise ValueError("Input text too long")
        
        return v
    
    @field_validator('input_type')
    @classmethod
    def validate_input_type(cls, v: str) -> str:
        """Validate input type."""
        allowed_types = ['text', 'url', 'email', 'username', 'password', 'caption']
        if v not in allowed_types:
            raise ValueError(f"Invalid input type. Allowed: {allowed_types}")
        return v

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_email(*, email: str) -> Dict[str, Any]:
    """Validate email address (RORO)."""
    if not isinstance(email, str):
        return {"is_valid": False, "error": "Email must be a string"}
    if not email or not email.strip():
        return {"is_valid": False, "error": "Email cannot be empty"}
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email.strip()):
        return {"is_valid": False, "error": "Invalid email format"}
    return {"is_valid": True, "email": email.strip().lower()}

def validate_url(*, url: str) -> Dict[str, Any]:
    """Validate URL (RORO)."""
    if not isinstance(url, str):
        return {"is_valid": False, "error": "URL must be a string"}
    if not url or not url.strip():
        return {"is_valid": False, "error": "URL cannot be empty"}
    url_pattern = r'^https?://(?:[-\w.])+(?::\d+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
    if not re.match(url_pattern, url.strip()):
        return {"is_valid": False, "error": "Invalid URL format"}
    return {"is_valid": True, "url": url.strip()}

def validate_instagram_username(*, username: str) -> Dict[str, Any]:
    """Validate Instagram username (RORO)."""
    if not isinstance(username, str):
        return {"is_valid": False, "error": "Username must be a string"}
    if not username or not username.strip():
        return {"is_valid": False, "error": "Username cannot be empty"}
    username = username.strip()
    if len(username) < 1 or len(username) > 30:
        return {"is_valid": False, "error": "Username must be 1-30 characters"}
    username_pattern = r'^[a-zA-Z0-9]{1,30}$'
    if not re.match(username_pattern, username):
        return {"is_valid": False, "error": "Username contains invalid characters"}
    reserved_words = ['admin', 'instagram', 'meta', 'facebook', 'help', 'support']
    if username.lower() in reserved_words:
        return {"is_valid": False, "error": "Username is reserved"}
    return {"is_valid": True, "username": username}

def sanitize_html(*, html_content: str, allowed_tags: List[str] = None) -> Dict[str, Any]:
    """Sanitize HTML content (RORO)."""
    if not isinstance(html_content, str):
        return {"sanitized": "", "removed_tags": ["invalid_type"]}
    if allowed_tags is None:
        allowed_tags = ['b', 'i', 'u', 'strong', 'em']
    if not html_content:
        return {"sanitized": "", "removed_tags": []}
    pattern = r'<(/?)([^>]+)>|<!--.*?-->'
    removed_tags = []
    def replace_tag(match) -> Any:
        tag = match.group(2).split()[0].lower() if match.group(2) else ""
        if tag not in allowed_tags:
            removed_tags.append(tag)
            return ''
        return match.group(0)
    sanitized = re.sub(pattern, replace_tag, html_content)
    return {
        "sanitized": sanitized,
        "removed_tags": list(set(removed_tags))
    }

def validate_caption_content(*, caption: str, content_type: ContentType) -> Dict[str, Any]:
    """Validate caption content (RORO)."""
    if not isinstance(caption, str):
        return {"is_valid": False, "error": "Caption must be a string"}
    if not isinstance(content_type, ContentType):
        return {"is_valid": False, "error": "Invalid content type"}
    if not caption or not caption.strip():
        return {"is_valid": False, "error": "Caption cannot be empty"}
    caption = caption.strip()
    max_length = CONTENT_LIMITS[content_type]
    if len(caption) > max_length:
        return {
            "is_valid": False,
            "error": f"Caption exceeds {max_length} character limit for {content_type}",
            "current_length": len(caption),
            "max_length": max_length
        }
    hashtag_count = len(re.findall(r'#[\w-]+', caption))
    if hashtag_count > 30:
        return {
            "is_valid": False,
            "error": "Too many hashtags (maximum 30)",
            "hashtag_count": hashtag_count
        }
    mention_count = len(re.findall(r'@[\w-]+', caption))
    if mention_count > 20:
        return {
            "is_valid": False,
            "error": "Too many mentions (maximum 20)",
            "mention_count": mention_count
        }
    return {
        "is_valid": True,
        "caption": caption,
        "length": len(caption),
        "hashtag_count": hashtag_count,
        "mention_count": mention_count
    }

# ============================================================================
# VALIDATION DECORATORS
# ============================================================================

async def validate_caption_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to validate caption request."""
    def wrapper(*args, **kwargs) -> Any:
        # Extract request data
        request_data = kwargs.get('request') or (args[0] if args else None)
        
        if request_data:
            try:
                if isinstance(request_data, dict):
                    validated_request = CaptionRequest(**request_data)
                elif isinstance(request_data, CaptionRequest):
                    validated_request = request_data
                else:
                    raise ValueError("Invalid request data type")
                
                # Replace with validated request
                if 'request' in kwargs:
                    kwargs['request'] = validated_request
                elif args:
                    args = (validated_request,) + args[1:]
                
                return func(*args, **kwargs)
            except Exception as e:
                return {"error": str(e), "is_valid": False}
        
        return func(*args, **kwargs)
    
    return wrapper

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ContentType",
    "ToneType",
    "CONTENT_LIMITS",
    
    # Models
    "BaseValidationModel",
    "CaptionRequest",
    "BatchCaptionRequest",
    "CaptionResponse",
    "SecurityValidationConfig",
    "UserInputValidation",
    
    # Utilities
    "validate_email",
    "validate_url",
    "validate_instagram_username",
    "sanitize_html",
    "validate_caption_content",
    
    # Decorators
    "validate_caption_request"
] 