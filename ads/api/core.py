"""
Core API endpoints for ads generation and management.

This module consolidates the basic ads functionality from the original api.py file,
following Clean Architecture principles and using the new domain and application layers.
"""

from typing import Any, List, Dict, Optional, Union
from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
import base64
import io
from PIL import Image
import numpy as np
import cv2
import logging
from datetime import datetime
import asyncio

from ..domain.entities import Ad, AdCampaign, AdGroup
from ..domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria
from ..application.dto import (
    CreateAdRequest, CreateAdResponse, 
    CreateCampaignRequest, CreateCampaignResponse,
    ErrorResponse
)
from ..application.use_cases import (
    CreateAdUseCase, CreateCampaignUseCase
)
try:
    from ..infrastructure.repositories import (
        AdRepository, CampaignRepository
    )
except Exception:  # pragma: no cover - optional in tests
    AdRepository = CampaignRepository = object  # type: ignore
from ..core import get_current_user, format_response, handle_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/core", tags=["ads-core"])

# Request Models (using application DTOs)
class AdsRequest(BaseModel):
    """Request model for ads generation."""
    url: str = Field(..., description="Source URL for ads generation")
    type: str = Field(..., pattern=r"^(ads|brand-kit|custom)$", description="Type of content to generate")
    prompt: Optional[str] = Field(None, max_length=2000, description="Custom prompt for generation")
    target_audience: Optional[str] = Field(None, max_length=500, description="Target audience description")
    context: Optional[str] = Field(None, max_length=1000, description="Additional context")
    keywords: Optional[List[str]] = Field(None, max_items=20, description="Keywords for targeting")
    max_length: Optional[int] = Field(10000, ge=100, le=50000, description="Maximum content length")

class BrandVoiceRequest(BaseModel):
    """Brand voice settings request."""
    tone: str = Field("professional", description="Brand tone")
    style: str = Field("conversational", description="Brand style")
    personality_traits: List[str] = Field(default_factory=list, description="Personality traits")
    industry_specific_terms: List[str] = Field(default_factory=list, description="Industry terms")
    brand_guidelines: Optional[Dict[str, Any]] = Field(None, description="Brand guidelines")

class AudienceProfileRequest(BaseModel):
    """Audience profile request."""
    demographics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "age_range": None,
            "gender": None,
            "location": None,
            "occupation": None,
            "income_level": None
        },
        description="Demographic information"
    )
    interests: List[str] = Field(default_factory=list, description="Audience interests")
    pain_points: List[str] = Field(default_factory=list, description="Pain points")
    goals: List[str] = Field(default_factory=list, description="Audience goals")
    buying_behavior: Optional[Dict[str, Any]] = Field(None, description="Buying behavior")
    customer_stage: str = Field("awareness", description="Customer journey stage")

class ContentSourceRequest(BaseModel):
    """Content source request."""
    type: str = Field(..., description="Source type")
    content: str = Field(..., description="Source content")
    priority: int = Field(1, ge=1, le=10, description="Source priority")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")

class ProjectContextRequest(BaseModel):
    """Project context request."""
    project_name: str = Field(..., max_length=200, description="Project name")
    project_description: str = Field(..., max_length=1000, description="Project description")
    industry: str = Field(..., max_length=100, description="Industry")
    key_messages: List[str] = Field(default_factory=list, description="Key messages")
    brand_assets: List[str] = Field(default_factory=list, description="Brand assets")
    content_sources: List[ContentSourceRequest] = Field(default_factory=list, description="Content sources")
    custom_variables: Dict[str, Any] = Field(default_factory=dict, description="Custom variables")

# Response Models
class AdsResponse(BaseModel):
    """Response model for ads generation."""
    type: str
    content: Any
    metadata: Optional[Dict[str, Any]] = None
    generated_at: datetime = Field(default_factory=datetime.now)

class BrandVoiceResponse(BaseModel):
    """Brand voice analysis response."""
    tone: str
    style: str
    personality_traits: List[str]
    industry_specific_terms: List[str]
    brand_guidelines: Optional[Dict[str, Any]]
    analysis_score: float
    recommendations: List[str]

class AudienceProfileResponse(BaseModel):
    """Audience profile response."""
    demographics: Dict[str, Any]
    interests: List[str]
    pain_points: List[str]
    goals: List[str]
    buying_behavior: Optional[Dict[str, Any]]
    customer_stage: str
    targeting_score: float
    recommendations: List[str]

class ContentSourceResponse(BaseModel):
    """Content source response."""
    type: str
    content: str
    priority: int
    relevance_score: Optional[float]
    analysis: Dict[str, Any]

class ProjectContextResponse(BaseModel):
    """Project context response."""
    project_name: str
    project_description: str
    industry: str
    key_messages: List[str]
    brand_assets: List[str]
    content_sources: List[ContentSourceResponse]
    custom_variables: Dict[str, Any]
    context_score: float
    recommendations: List[str]

# Core endpoints
@router.post("/generate", response_model=AdsResponse)
async def generate_ads(
    request: AdsRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate ads based on the provided request."""
    try:
        # Convert request to domain DTO
        create_ad_request = CreateAdRequest(
            prompt=request.prompt or f"Generate {request.type} content from {request.url}",
            ad_type=AdType(request.type),
            target_audience=request.target_audience,
            context=request.context,
            keywords=request.keywords or [],
            max_length=request.max_length
        )
        
        # Execute use case
        use_case = CreateAdUseCase()
        result = await use_case.execute(create_ad_request)
        
        # Convert to response
        return AdsResponse(
            type=request.type,
            content=result.content,
            metadata={
                "generation_id": result.id,
                "target_audience": request.target_audience,
                "keywords": request.keywords,
                "max_length": request.max_length
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating ads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/brand-voice", response_model=BrandVoiceResponse)
async def analyze_brand_voice(
    request: BrandVoiceRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze and optimize brand voice settings."""
    try:
        # Analyze brand voice (placeholder implementation)
        analysis_score = 0.85
        recommendations = [
            "Consider adding more specific industry terminology",
            "Balance professional tone with approachability",
            "Include more personality traits for brand differentiation"
        ]
        
        return BrandVoiceResponse(
            tone=request.tone,
            style=request.style,
            personality_traits=request.personality_traits,
            industry_specific_terms=request.industry_specific_terms,
            brand_guidelines=request.brand_guidelines,
            analysis_score=analysis_score,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error analyzing brand voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audience-profile", response_model=AudienceProfileResponse)
async def analyze_audience_profile(
    request: AudienceProfileRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze and optimize audience profile."""
    try:
        # Analyze audience profile (placeholder implementation)
        targeting_score = 0.78
        recommendations = [
            "Consider expanding age range for broader appeal",
            "Add more specific pain points for better targeting",
            "Include buying behavior patterns for conversion optimization"
        ]
        
        return AudienceProfileResponse(
            demographics=request.demographics,
            interests=request.interests,
            pain_points=request.pain_points,
            goals=request.goals,
            buying_behavior=request.buying_behavior,
            customer_stage=request.customer_stage,
            targeting_score=targeting_score,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error analyzing audience profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/content-source", response_model=ContentSourceResponse)
async def analyze_content_source(
    request: ContentSourceRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze content source for relevance and quality."""
    try:
        # Analyze content source (placeholder implementation)
        analysis = {
            "readability_score": 0.82,
            "sentiment": "positive",
            "key_topics": ["content", "quality", "relevance"],
            "suggested_improvements": ["Add more specific examples", "Include data points"]
        }
        
        return ContentSourceResponse(
            type=request.type,
            content=request.content,
            priority=request.priority,
            relevance_score=request.relevance_score or 0.75,
            analysis=analysis
        )
        
    except Exception as e:
        logger.error(f"Error analyzing content source: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/project-context", response_model=ProjectContextResponse)
async def analyze_project_context(
    request: ProjectContextRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze project context for optimization opportunities."""
    try:
        # Analyze project context (placeholder implementation)
        context_score = 0.91
        recommendations = [
            "Consider adding competitor analysis",
            "Include more specific industry benchmarks",
            "Add customer journey mapping"
        ]
        
        # Convert content sources to responses
        content_sources = [
            ContentSourceResponse(
                type=source.type,
                content=source.content,
                priority=source.priority,
                relevance_score=source.relevance_score,
                analysis={"placeholder": "analysis"}
            )
            for source in request.content_sources
        ]
        
        return ProjectContextResponse(
            project_name=request.project_name,
            project_description=request.project_description,
            industry=request.industry,
            key_messages=request.key_messages,
            brand_assets=request.brand_assets,
            content_sources=content_sources,
            custom_variables=request.custom_variables,
            context_score=context_score,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error analyzing project context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/capabilities")
async def get_capabilities():
    """Get available capabilities."""
    return {
        "features": [
            "ads_generation",
            "brand_voice_analysis", 
            "audience_profile_analysis",
            "content_source_analysis",
            "project_context_analysis"
        ],
        "supported_types": ["ads", "brand-kit", "custom"],
        "max_content_length": 50000,
        "supported_platforms": ["facebook", "instagram", "twitter", "linkedin"]
    }
