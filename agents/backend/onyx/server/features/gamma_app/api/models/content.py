"""
Content Models
Content generation related Pydantic models
"""

from datetime import datetime
from typing import List, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from .enums import ContentType, OutputFormat, DesignStyle
from .base import BaseResponse
from .validators import sanitize_string_field

class ContentRequest(BaseModel):
    """Content generation request"""
    content_type: ContentType
    topic: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=1000)
    target_audience: str = Field(..., min_length=1, max_length=100)
    length: str = Field("medium", pattern="^(short|medium|long)$")
    style: DesignStyle = DesignStyle.PROFESSIONAL
    output_format: OutputFormat = OutputFormat.HTML
    include_images: bool = True
    include_charts: bool = False
    language: str = Field("en", pattern="^[a-z]{2}$")
    tone: str = Field("professional", pattern="^(professional|casual|friendly|formal|persuasive|informative|conversational|authoritative|creative|technical)$")
    keywords: List[str] = Field(default_factory=list, max_items=10)
    custom_instructions: str = Field("", max_length=2000)
    user_id: str = ""
    project_id: str = ""

    @field_validator('topic', 'description', 'target_audience', 'custom_instructions')
    @classmethod
    def sanitize_string_fields(cls, v: str) -> str:
        """Sanitize string fields by stripping whitespace"""
        return sanitize_string_field(v)

    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v):
        """Validate and sanitize keywords"""
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







