"""
Data Transfer Objects (DTOs) - API request/response models
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class AnalysisRequest(BaseModel):
    """Request DTO for content analysis"""
    content: str = Field(..., min_length=10, max_length=50000, description="Content to analyze")
    threshold: Optional[float] = Field(default=0.8, ge=0.0, le=1.0, description="Redundancy threshold")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class SimilarityRequest(BaseModel):
    """Request DTO for similarity check"""
    text1: str = Field(..., min_length=10, description="First text")
    text2: str = Field(..., min_length=10, description="Second text")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Similarity threshold")
    
    @validator('text1', 'text2')
    def validate_texts(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class QualityRequest(BaseModel):
    """Request DTO for quality assessment"""
    content: str = Field(..., min_length=10, max_length=50000, description="Content to assess")
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    include_readability: bool = Field(default=True, description="Include readability analysis")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class BatchRequest(BaseModel):
    """Request DTO for batch processing"""
    items: List[str] = Field(..., min_items=1, max_items=100, description="List of content items to analyze")
    threshold: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    parallel: bool = Field(default=True, description="Process items in parallel")
    
    @validator('items')
    def validate_items(cls, v):
        if not v:
            raise ValueError("Items list cannot be empty")
        return [item.strip() for item in v if item.strip()]


class ExportRequest(BaseModel):
    """Request DTO for data export"""
    data: Dict[str, Any] = Field(..., description="Data to export")
    format: str = Field(default="json", pattern="^(json|csv|pdf|xlsx)$", description="Export format")
    filename: Optional[str] = Field(None, description="Output filename")


class WebhookRequest(BaseModel):
    """Request DTO for webhook registration"""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
