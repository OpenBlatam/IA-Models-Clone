from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Request Models
=============

Pydantic models for API request validation.
"""



class CopywritingRequest(BaseModel):
    """Copywriting generation request model"""
    
    prompt: str = Field(..., min_length=1, max_length=2000, description="The main prompt for copywriting")
    platform: str = Field(default="instagram", description="Target platform")
    content_type: str = Field(default="post", description="Type of content")
    tone: str = Field(default="professional", description="Tone of voice")
    target_audience: str = Field(default="general", description="Target audience")
    keywords: List[str] = Field(default=[], description="Keywords to include")
    brand_voice: str = Field(default="professional", description="Brand voice")
    num_variants: int = Field(default=3, ge=1, le=10, description="Number of variants to generate")
    max_tokens: int = Field(default=1024, ge=64, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    
    # Optional settings
    language: str = Field(default="en", description="Content language")
    style_guide: Optional[Dict[str, Any]] = Field(default=None, description="Style guide preferences")
    constraints: Optional[List[str]] = Field(default=None, description="Content constraints")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    
    @validator('platform')
    def validate_platform(cls, v) -> bool:
        valid_platforms = [
            'instagram', 'facebook', 'twitter', 'linkedin', 
            'email', 'blog', 'website', 'ad', 'general'
        ]
        if v.lower() not in valid_platforms:
            raise ValueError(f'Platform must be one of: {valid_platforms}')
        return v.lower()
    
    @validator('content_type')
    def validate_content_type(cls, v) -> bool:
        valid_types = [
            'post', 'caption', 'ad', 'email', 'article', 
            'headline', 'description', 'bio', 'general'
        ]
        if v.lower() not in valid_types:
            raise ValueError(f'Content type must be one of: {valid_types}')
        return v.lower()
    
    @validator('tone')
    def validate_tone(cls, v) -> bool:
        valid_tones = [
            'professional', 'casual', 'friendly', 'formal', 
            'conversational', 'persuasive', 'educational', 'humorous'
        ]
        if v.lower() not in valid_tones:
            raise ValueError(f'Tone must be one of: {valid_tones}')
        return v.lower()
    
    @validator('keywords')
    def validate_keywords(cls, v) -> bool:
        if len(v) > 20:
            raise ValueError('Maximum 20 keywords allowed')
        return [kw.strip().lower() for kw in v if kw.strip()]
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "prompt": "Create engaging content about digital marketing",
                "platform": "instagram",
                "content_type": "post",
                "tone": "professional",
                "target_audience": "entrepreneurs",
                "keywords": ["marketing", "digital", "growth"],
                "brand_voice": "professional",
                "num_variants": 3,
                "max_tokens": 1024,
                "temperature": 0.7
            }
        }


class BatchRequest(BaseModel):
    """Batch copywriting request model"""
    
    requests: List[CopywritingRequest] = Field(..., min_items=1, max_items=50, description="List of copywriting requests")
    batch_options: Optional[Dict[str, Any]] = Field(default=None, description="Batch processing options")
    
    @validator('requests')
    async def validate_requests(cls, v) -> bool:
        if not v:
            raise ValueError('At least one request is required')
        return v
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "requests": [
                    {
                        "prompt": "Create engaging content about digital marketing",
                        "platform": "instagram",
                        "content_type": "post"
                    },
                    {
                        "prompt": "Write a professional email about our services",
                        "platform": "email",
                        "content_type": "email"
                    }
                ],
                "batch_options": {
                    "priority": "normal",
                    "timeout": 120
                }
            }
        }


class OptimizationRequest(BaseModel):
    """Text optimization request model"""
    
    text: str = Field(..., min_length=1, max_length=5000, description="Text to optimize")
    platform: str = Field(default="instagram", description="Target platform")
    tone: str = Field(default="professional", description="Desired tone")
    target_audience: str = Field(default="general", description="Target audience")
    keywords: List[str] = Field(default=[], description="Keywords to include")
    
    # Optimization options
    optimization_type: str = Field(default="general", description="Type of optimization")
    max_length: Optional[int] = Field(default=None, ge=10, le=5000, description="Maximum text length")
    preserve_meaning: bool = Field(default=True, description="Preserve original meaning")
    enhance_engagement: bool = Field(default=True, description="Enhance engagement")
    improve_seo: bool = Field(default=False, description="Improve SEO")
    
    # Style preferences
    style_preferences: Optional[Dict[str, Any]] = Field(default=None, description="Style preferences")
    
    @validator('optimization_type')
    def validate_optimization_type(cls, v) -> bool:
        valid_types = ['general', 'engagement', 'seo', 'clarity', 'conversion']
        if v.lower() not in valid_types:
            raise ValueError(f'Optimization type must be one of: {valid_types}')
        return v.lower()
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "text": "Check out our amazing product!",
                "platform": "instagram",
                "tone": "professional",
                "target_audience": "entrepreneurs",
                "keywords": ["product", "amazing", "check"],
                "optimization_type": "engagement",
                "enhance_engagement": True
            }
        }


class AnalysisRequest(BaseModel):
    """Content analysis request model"""
    
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    analysis_types: List[str] = Field(default=["sentiment", "readability"], description="Types of analysis")
    
    # Analysis options
    include_suggestions: bool = Field(default=True, description="Include improvement suggestions")
    detailed_report: bool = Field(default=False, description="Generate detailed report")
    
    @validator('analysis_types')
    def validate_analysis_types(cls, v) -> bool:
        valid_types = ['sentiment', 'readability', 'tone', 'engagement', 'seo', 'grammar']
        for analysis_type in v:
            if analysis_type.lower() not in valid_types:
                raise ValueError(f'Analysis type must be one of: {valid_types}')
        return [t.lower() for t in v]
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "text": "Our product is the best solution for your needs!",
                "analysis_types": ["sentiment", "readability", "engagement"],
                "include_suggestions": True
            }
        } 