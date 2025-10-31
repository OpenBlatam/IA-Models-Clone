"""
Pydantic Schemas for Copywriting API
===================================

Clean, type-safe schemas following Pydantic v2 best practices.
"""

from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator
from enum import Enum


class CopywritingTone(str, Enum):
    """Copywriting tone options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    URGENT = "urgent"
    INSPIRATIONAL = "inspirational"


class CopywritingStyle(str, Enum):
    """Copywriting style options"""
    DIRECT_RESPONSE = "direct_response"
    BRAND_AWARENESS = "brand_awareness"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    STORYTELLING = "storytelling"
    TECHNICAL = "technical"


class CopywritingPurpose(str, Enum):
    """Copywriting purpose options"""
    SALES = "sales"
    LEAD_GENERATION = "lead_generation"
    ENGAGEMENT = "engagement"
    EDUCATION = "education"
    BRAND_BUILDING = "brand_building"
    CONVERSION = "conversion"


class CopywritingRequest(BaseModel):
    """Request schema for copywriting generation"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Core content
    topic: str = Field(..., min_length=1, max_length=500, description="Main topic or subject")
    target_audience: str = Field(..., min_length=1, max_length=200, description="Target audience description")
    key_points: List[str] = Field(default_factory=list, max_items=10, description="Key points to include")
    
    # Style and tone
    tone: CopywritingTone = Field(default=CopywritingTone.PROFESSIONAL, description="Desired tone")
    style: CopywritingStyle = Field(default=CopywritingStyle.DIRECT_RESPONSE, description="Writing style")
    purpose: CopywritingPurpose = Field(default=CopywritingPurpose.SALES, description="Content purpose")
    
    # Content specifications
    word_count: Optional[int] = Field(default=None, ge=50, le=2000, description="Desired word count")
    include_cta: bool = Field(default=True, description="Include call-to-action")
    language: str = Field(default="en", min_length=2, max_length=5, description="Content language")
    
    # Brand context
    brand_voice: Optional[str] = Field(default=None, max_length=300, description="Brand voice guidelines")
    brand_values: Optional[List[str]] = Field(default=None, max_items=5, description="Brand values to reflect")
    
    # Advanced options
    creativity_level: float = Field(default=0.7, ge=0.0, le=1.0, description="Creativity level (0-1)")
    variants_count: int = Field(default=3, ge=1, le=10, description="Number of variants to generate")
    
    @field_validator('key_points')
    @classmethod
    def validate_key_points(cls, v: List[str]) -> List[str]:
        if v:
            return [point.strip() for point in v if point.strip()]
        return v


class CopywritingVariant(BaseModel):
    """Individual copywriting variant"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=50, max_length=5000)
    word_count: int = Field(..., ge=1)
    cta: Optional[str] = Field(default=None, max_length=200)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CopywritingResponse(BaseModel):
    """Response schema for copywriting generation"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    request_id: UUID = Field(default_factory=uuid4)
    variants: List[CopywritingVariant] = Field(..., min_items=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: int = Field(..., ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def total_variants(self) -> int:
        return len(self.variants)
    
    @computed_field
    @property
    def best_variant(self) -> CopywritingVariant:
        return max(self.variants, key=lambda v: v.confidence_score)


class FeedbackRequest(BaseModel):
    """Request schema for providing feedback"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    variant_id: UUID = Field(..., description="ID of the variant to provide feedback for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(default=None, max_length=1000, description="Detailed feedback")
    improvements: Optional[List[str]] = Field(default=None, max_items=5, description="Suggested improvements")
    is_helpful: bool = Field(..., description="Whether the content was helpful")


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    feedback_id: UUID = Field(default_factory=uuid4)
    variant_id: UUID
    status: Literal["received", "processed", "error"] = Field(default="received")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchCopywritingRequest(BaseModel):
    """Request schema for batch copywriting generation"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    requests: List[CopywritingRequest] = Field(..., min_items=1, max_items=50)
    batch_id: Optional[UUID] = Field(default_factory=uuid4)
    
    @field_validator('requests')
    @classmethod
    def validate_requests(cls, v: List[CopywritingRequest]) -> List[CopywritingRequest]:
        if len(v) > 50:
            raise ValueError("Maximum 50 requests allowed per batch")
        return v


class BatchCopywritingResponse(BaseModel):
    """Response schema for batch copywriting generation"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    batch_id: UUID
    results: List[CopywritingResponse] = Field(..., min_items=1)
    failed_requests: List[Dict[str, Any]] = Field(default_factory=list)
    total_processing_time_ms: int = Field(..., ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @computed_field
    @property
    def success_count(self) -> int:
        return len(self.results)
    
    @computed_field
    @property
    def failure_count(self) -> int:
        return len(self.failed_requests)


class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    status: Literal["healthy", "degraded", "unhealthy"] = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="2.0.0")
    uptime_seconds: float = Field(..., ge=0)
    dependencies: Dict[str, str] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response schema"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[UUID] = Field(default=None, description="Request ID for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow)































