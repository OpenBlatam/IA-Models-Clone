"""
API schemas for the Blaze AI module.

This module defines all request and response schemas for the FastAPI endpoints,
providing comprehensive validation and documentation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import base64


# Base request/response models
class BaseRequest(BaseModel):
    """Base request model with common fields."""
    timeout: Optional[float] = Field(default=30.0, ge=0.1, le=300.0, description="Request timeout in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(description="Whether the request was successful")
    message: str = Field(description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


# Text Generation Schemas
class TextGenerationRequest(BaseRequest):
    """Request schema for text generation."""
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for text generation")
    max_length: Optional[int] = Field(default=100, ge=1, le=2048, description="Maximum length of generated text")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    do_sample: Optional[bool] = Field(default=True, description="Whether to use sampling")
    num_return_sequences: Optional[int] = Field(default=1, ge=1, le=10, description="Number of sequences to generate")
    stop_sequences: Optional[List[str]] = Field(default_factory=list, description="Stop sequences")
    repetition_penalty: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Repetition penalty")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt is not empty and contains meaningful content."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class TextGenerationResponse(BaseResponse):
    """Response schema for text generation."""
    text: str = Field(description="Generated text")
    tokens_used: Optional[int] = Field(description="Number of tokens used")
    generation_time: Optional[float] = Field(description="Generation time in seconds")
    model_info: Optional[Dict[str, Any]] = Field(description="Model information")


# Image Generation Schemas
class ImageGenerationRequest(BaseRequest):
    """Request schema for image generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(default="", max_length=1000, description="Negative prompt")
    width: Optional[int] = Field(default=512, ge=64, le=2048, description="Image width")
    height: Optional[int] = Field(default=512, ge=64, le=2048, description="Image height")
    guidance_scale: Optional[float] = Field(default=7.5, ge=0.0, le=20.0, description="Classifier-free guidance scale")
    num_inference_steps: Optional[int] = Field(default=50, ge=1, le=100, description="Number of inference steps")
    num_images: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images to generate")
    seed: Optional[int] = Field(description="Random seed for reproducibility")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt is not empty."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()


class ImageGenerationResponse(BaseResponse):
    """Response schema for image generation."""
    images: List[str] = Field(description="Base64 encoded images")
    generation_time: Optional[float] = Field(description="Generation time in seconds")
    model_info: Optional[Dict[str, Any]] = Field(description="Model information")
    
    @validator('images')
    def validate_images(cls, v):
        """Validate that images are valid base64 strings."""
        for img in v:
            try:
                base64.b64decode(img)
            except Exception:
                raise ValueError("Invalid base64 image data")
        return v


# SEO Analysis Schemas
class SEOAnalysisRequest(BaseRequest):
    """Request schema for SEO analysis."""
    content: str = Field(..., min_length=10, max_length=50000, description="Content to analyze")
    url: Optional[str] = Field(description="URL of the content")
    title: Optional[str] = Field(max_length=200, description="Page title")
    meta_description: Optional[str] = Field(max_length=300, description="Meta description")
    target_keywords: Optional[List[str]] = Field(default_factory=list, description="Target keywords")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content has sufficient length for analysis."""
        if len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v.strip()


class SEOAnalysisResponse(BaseResponse):
    """Response schema for SEO analysis."""
    keywords: List[Dict[str, Any]] = Field(description="Extracted keywords with scores")
    readability_score: Optional[float] = Field(description="Readability score")
    seo_score: Optional[float] = Field(description="Overall SEO score")
    suggestions: List[str] = Field(description="SEO improvement suggestions")
    word_count: Optional[int] = Field(description="Word count")
    keyword_density: Optional[Dict[str, float]] = Field(description="Keyword density analysis")


# Brand Voice Schemas
class BrandVoiceRequest(BaseRequest):
    """Request schema for brand voice analysis/application."""
    content: str = Field(..., min_length=10, description="Content to analyze or transform")
    brand_name: str = Field(..., min_length=1, description="Brand name")
    action: str = Field(default="analyze", regex="^(analyze|apply|train)$", description="Action to perform")
    samples: Optional[List[str]] = Field(default_factory=list, description="Brand voice samples for training")
    
    @validator('content')
    def validate_content(cls, v):
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class BrandVoiceResponse(BaseResponse):
    """Response schema for brand voice operations."""
    transformed_content: Optional[str] = Field(description="Transformed content")
    brand_voice_score: Optional[float] = Field(description="Brand voice consistency score")
    voice_characteristics: Optional[Dict[str, Any]] = Field(description="Brand voice characteristics")
    suggestions: List[str] = Field(description="Brand voice improvement suggestions")


# Content Generation Schemas
class ContentGenerationRequest(BaseRequest):
    """Request schema for content generation."""
    content_type: str = Field(..., regex="^(blog|social|email|ad|landing_page)$", description="Type of content to generate")
    topic: str = Field(..., min_length=1, description="Content topic")
    brand_name: Optional[str] = Field(description="Brand name")
    tone: Optional[str] = Field(default="professional", description="Content tone")
    target_audience: Optional[str] = Field(description="Target audience")
    key_points: Optional[List[str]] = Field(default_factory=list, description="Key points to include")
    word_count: Optional[int] = Field(ge=50, le=5000, description="Target word count")
    
    @validator('topic')
    def validate_topic(cls, v):
        """Validate topic is not empty."""
        if not v.strip():
            raise ValueError("Topic cannot be empty")
        return v.strip()


class ContentGenerationResponse(BaseResponse):
    """Response schema for content generation."""
    content: str = Field(description="Generated content")
    word_count: Optional[int] = Field(description="Actual word count")
    content_structure: Optional[Dict[str, Any]] = Field(description="Content structure analysis")
    seo_optimization: Optional[Dict[str, Any]] = Field(description="SEO optimization suggestions")


# Batch Processing Schemas
class BatchRequest(BaseModel):
    """Individual batch request item."""
    request_type: str = Field(..., description="Type of request")
    data: Dict[str, Any] = Field(..., description="Request data")
    priority: Optional[int] = Field(default=0, ge=0, le=10, description="Request priority")


class BatchRequestWrapper(BaseRequest):
    """Request schema for batch processing."""
    requests: List[BatchRequest] = Field(..., min_items=1, max_items=100, description="List of requests to process")
    concurrency: Optional[int] = Field(default=5, ge=1, le=20, description="Number of concurrent requests")
    fail_fast: Optional[bool] = Field(default=False, description="Stop processing on first failure")


class BatchResponse(BaseResponse):
    """Response schema for batch processing."""
    results: List[Dict[str, Any]] = Field(description="Results for each request")
    successful_count: int = Field(description="Number of successful requests")
    failed_count: int = Field(description="Number of failed requests")
    total_time: float = Field(description="Total processing time")


# Health and Metrics Schemas
class HealthResponse(BaseResponse):
    """Response schema for health check."""
    status: str = Field(description="Overall system status")
    components: Dict[str, Dict[str, Any]] = Field(description="Health status of individual components")
    uptime: float = Field(description="System uptime in seconds")
    version: str = Field(description="System version")


class MetricsResponse(BaseResponse):
    """Response schema for system metrics."""
    system_metrics: Dict[str, Any] = Field(description="System-level metrics")
    engine_metrics: Dict[str, Any] = Field(description="Engine-specific metrics")
    service_metrics: Dict[str, Any] = Field(description="Service-specific metrics")
    performance_metrics: Dict[str, Any] = Field(description="Performance metrics")


# Error Response Schema
class ErrorResponse(BaseResponse):
    """Error response schema."""
    error_code: str = Field(description="Error code")
    error_type: str = Field(description="Error type")
    details: Optional[Dict[str, Any]] = Field(description="Error details")
    request_id: Optional[str] = Field(description="Request ID for tracking")


# Export all schemas
__all__ = [
    "BaseRequest", "BaseResponse",
    "TextGenerationRequest", "TextGenerationResponse",
    "ImageGenerationRequest", "ImageGenerationResponse",
    "SEOAnalysisRequest", "SEOAnalysisResponse",
    "BrandVoiceRequest", "BrandVoiceResponse",
    "ContentGenerationRequest", "ContentGenerationResponse",
    "BatchRequest", "BatchRequestWrapper", "BatchResponse",
    "HealthResponse", "MetricsResponse", "ErrorResponse"
]


