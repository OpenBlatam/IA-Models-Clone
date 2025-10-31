"""
AI Enhancement API Endpoints
============================

REST API endpoints for AI-powered content enhancement.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from ..services.ai_enhancement_service import (
    AIEnhancementService, 
    EnhancementType, 
    EnhancementRequest, 
    EnhancementResult,
    ContentQuality
)
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/enhancement", tags=["AI Enhancement"])

# Pydantic models
class EnhancementRequestModel(BaseModel):
    content: str = Field(..., description="Content to enhance")
    enhancement_type: EnhancementType = Field(..., description="Type of enhancement to apply")
    target_audience: Optional[str] = Field(None, description="Target audience for the content")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Business context information")
    quality_requirements: Optional[Dict[str, Any]] = Field(None, description="Quality requirements")
    language: str = Field("en", description="Content language")
    tone: Optional[str] = Field(None, description="Desired tone")
    keywords: Optional[List[str]] = Field(None, description="Keywords to optimize for")
    max_length: Optional[int] = Field(None, description="Maximum length of enhanced content")
    min_length: Optional[int] = Field(None, description="Minimum length of enhanced content")

class BatchEnhancementRequestModel(BaseModel):
    requests: List[EnhancementRequestModel] = Field(..., description="List of enhancement requests")

class EnhancementSuggestionRequestModel(BaseModel):
    content: str = Field(..., description="Content to analyze for suggestions")
    business_context: Optional[Dict[str, Any]] = Field(None, description="Business context information")

class EnhancementResultModel(BaseModel):
    original_content: str
    enhanced_content: str
    enhancement_type: str
    quality_score: float
    improvements: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float

class ContentAnalysisModel(BaseModel):
    readability_score: float
    sentiment_score: float
    keyword_density: Dict[str, float]
    structure_score: float
    grammar_score: float
    overall_quality: str
    recommendations: List[str]

# Global enhancement service instance
enhancement_service = None

def get_enhancement_service() -> AIEnhancementService:
    """Get global enhancement service instance."""
    global enhancement_service
    if enhancement_service is None:
        enhancement_service = AIEnhancementService({})
    return enhancement_service

# API Endpoints

@router.post("/enhance", response_model=EnhancementResultModel)
async def enhance_content(
    request: EnhancementRequestModel,
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Enhance content using AI."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Convert request to service model
        enhancement_request = EnhancementRequest(
            content=request.content,
            enhancement_type=request.enhancement_type,
            target_audience=request.target_audience,
            business_context=request.business_context,
            quality_requirements=request.quality_requirements,
            language=request.language,
            tone=request.tone,
            keywords=request.keywords,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        # Enhance content
        result = await enhancement_service.enhance_content(enhancement_request)
        
        return EnhancementResultModel(
            original_content=result.original_content,
            enhanced_content=result.enhanced_content,
            enhancement_type=result.enhancement_type.value,
            quality_score=result.quality_score,
            improvements=result.improvements,
            suggestions=result.suggestions,
            metadata=result.metadata,
            processing_time=result.processing_time,
            confidence_score=result.confidence_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content enhancement failed: {str(e)}")

@router.post("/enhance/batch", response_model=List[EnhancementResultModel])
async def batch_enhance_content(
    request: BatchEnhancementRequestModel,
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Enhance multiple content pieces in batch."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Convert requests to service models
        enhancement_requests = []
        for req in request.requests:
            enhancement_requests.append(EnhancementRequest(
                content=req.content,
                enhancement_type=req.enhancement_type,
                target_audience=req.target_audience,
                business_context=req.business_context,
                quality_requirements=req.quality_requirements,
                language=req.language,
                tone=req.tone,
                keywords=req.keywords,
                max_length=req.max_length,
                min_length=req.min_length
            ))
        
        # Batch enhance content
        results = await enhancement_service.batch_enhance_content(enhancement_requests)
        
        return [
            EnhancementResultModel(
                original_content=result.original_content,
                enhanced_content=result.enhanced_content,
                enhancement_type=result.enhancement_type.value,
                quality_score=result.quality_score,
                improvements=result.improvements,
                suggestions=result.suggestions,
                metadata=result.metadata,
                processing_time=result.processing_time,
                confidence_score=result.confidence_score
            )
            for result in results
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch content enhancement failed: {str(e)}")

@router.post("/analyze", response_model=ContentAnalysisModel)
async def analyze_content(
    content: str = Field(..., description="Content to analyze"),
    current_user: User = Depends(require_permission("content:analyze"))
):
    """Analyze content quality and characteristics."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Analyze content
        analysis = await enhancement_service._analyze_content(content)
        
        return ContentAnalysisModel(
            readability_score=analysis.readability_score,
            sentiment_score=analysis.sentiment_score,
            keyword_density=analysis.keyword_density,
            structure_score=analysis.structure_score,
            grammar_score=analysis.grammar_score,
            overall_quality=analysis.overall_quality.value,
            recommendations=analysis.recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content analysis failed: {str(e)}")

@router.post("/suggestions")
async def get_enhancement_suggestions(
    request: EnhancementSuggestionRequestModel,
    current_user: User = Depends(require_permission("content:analyze"))
):
    """Get suggestions for content enhancement."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Get enhancement suggestions
        suggestions = await enhancement_service.get_enhancement_suggestions(
            request.content,
            request.business_context
        )
        
        return {
            "content": request.content,
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement suggestions: {str(e)}")

@router.get("/types")
async def get_enhancement_types():
    """Get available enhancement types."""
    
    enhancement_types = [
        {
            "type": enhancement_type.value,
            "name": enhancement_type.value.replace("_", " ").title(),
            "description": get_enhancement_description(enhancement_type)
        }
        for enhancement_type in EnhancementType
    ]
    
    return {
        "enhancement_types": enhancement_types,
        "total_count": len(enhancement_types)
    }

def get_enhancement_description(enhancement_type: EnhancementType) -> str:
    """Get description for enhancement type."""
    
    descriptions = {
        EnhancementType.CONTENT_OPTIMIZATION: "Optimize content for better engagement and clarity",
        EnhancementType.SEO_ENHANCEMENT: "Optimize content for search engines and SEO",
        EnhancementType.TONE_ADJUSTMENT: "Adjust content tone and style",
        EnhancementType.STRUCTURE_IMPROVEMENT: "Improve content structure and organization",
        EnhancementType.GRAMMAR_CORRECTION: "Correct grammar and punctuation errors",
        EnhancementType.READABILITY_ENHANCEMENT: "Enhance content readability and comprehension",
        EnhancementType.PERSONALIZATION: "Personalize content for specific audiences",
        EnhancementType.TRANSLATION: "Translate content to different languages",
        EnhancementType.SUMMARIZATION: "Create concise summaries of content",
        EnhancementType.EXPANSION: "Expand content with additional details and insights"
    }
    
    return descriptions.get(enhancement_type, "Enhance content quality and effectiveness")

@router.get("/quality-levels")
async def get_quality_levels():
    """Get available content quality levels."""
    
    quality_levels = [
        {
            "level": quality.value,
            "name": quality.value.title(),
            "description": get_quality_description(quality)
        }
        for quality in ContentQuality
    ]
    
    return {
        "quality_levels": quality_levels,
        "total_count": len(quality_levels)
    }

def get_quality_description(quality: ContentQuality) -> str:
    """Get description for quality level."""
    
    descriptions = {
        ContentQuality.EXCELLENT: "Content meets the highest quality standards",
        ContentQuality.GOOD: "Content is well-written with minor improvements possible",
        ContentQuality.FAIR: "Content is acceptable but could benefit from enhancements",
        ContentQuality.POOR: "Content needs significant improvement"
    }
    
    return descriptions.get(quality, "Content quality assessment")

@router.post("/optimize-seo")
async def optimize_seo(
    content: str = Field(..., description="Content to optimize for SEO"),
    keywords: List[str] = Field(..., description="Target keywords for SEO"),
    business_context: Optional[Dict[str, Any]] = Field(None, description="Business context"),
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Optimize content specifically for SEO."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Create SEO enhancement request
        enhancement_request = EnhancementRequest(
            content=content,
            enhancement_type=EnhancementType.SEO_ENHANCEMENT,
            keywords=keywords,
            business_context=business_context
        )
        
        # Enhance content for SEO
        result = await enhancement_service.enhance_content(enhancement_request)
        
        return {
            "original_content": result.original_content,
            "seo_optimized_content": result.enhanced_content,
            "keywords_used": keywords,
            "improvements": result.improvements,
            "suggestions": result.suggestions,
            "quality_score": result.quality_score,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SEO optimization failed: {str(e)}")

@router.post("/improve-readability")
async def improve_readability(
    content: str = Field(..., description="Content to improve readability"),
    target_reading_level: str = Field("intermediate", description="Target reading level"),
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Improve content readability."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Create readability enhancement request
        enhancement_request = EnhancementRequest(
            content=content,
            enhancement_type=EnhancementType.READABILITY_ENHANCEMENT,
            business_context={"reading_level": target_reading_level}
        )
        
        # Enhance content readability
        result = await enhancement_service.enhance_content(enhancement_request)
        
        return {
            "original_content": result.original_content,
            "readability_improved_content": result.enhanced_content,
            "target_reading_level": target_reading_level,
            "improvements": result.improvements,
            "suggestions": result.suggestions,
            "quality_score": result.quality_score,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Readability improvement failed: {str(e)}")

@router.post("/adjust-tone")
async def adjust_tone(
    content: str = Field(..., description="Content to adjust tone"),
    desired_tone: str = Field(..., description="Desired tone (professional, casual, friendly, etc.)"),
    target_audience: Optional[str] = Field(None, description="Target audience"),
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Adjust content tone."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Create tone adjustment request
        enhancement_request = EnhancementRequest(
            content=content,
            enhancement_type=EnhancementType.TONE_ADJUSTMENT,
            tone=desired_tone,
            target_audience=target_audience
        )
        
        # Adjust content tone
        result = await enhancement_service.enhance_content(enhancement_request)
        
        return {
            "original_content": result.original_content,
            "tone_adjusted_content": result.enhanced_content,
            "desired_tone": desired_tone,
            "target_audience": target_audience,
            "improvements": result.improvements,
            "suggestions": result.suggestions,
            "quality_score": result.quality_score,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tone adjustment failed: {str(e)}")

@router.post("/summarize")
async def summarize_content(
    content: str = Field(..., description="Content to summarize"),
    target_length: str = Field("medium", description="Target summary length (short, medium, long)"),
    key_points: Optional[List[str]] = Field(None, description="Key points to include"),
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Create a summary of content."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Create summarization request
        enhancement_request = EnhancementRequest(
            content=content,
            enhancement_type=EnhancementType.SUMMARIZATION,
            business_context={
                "target_length": target_length,
                "key_points": key_points or []
            }
        )
        
        # Summarize content
        result = await enhancement_service.enhance_content(enhancement_request)
        
        return {
            "original_content": result.original_content,
            "summary": result.enhanced_content,
            "target_length": target_length,
            "key_points": key_points,
            "improvements": result.improvements,
            "suggestions": result.suggestions,
            "quality_score": result.quality_score,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content summarization failed: {str(e)}")

@router.post("/expand")
async def expand_content(
    content: str = Field(..., description="Content to expand"),
    expansion_focus: str = Field("general", description="Focus area for expansion"),
    target_length: str = Field("long", description="Target expanded length"),
    current_user: User = Depends(require_permission("content:enhance"))
):
    """Expand content with additional details."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Create expansion request
        enhancement_request = EnhancementRequest(
            content=content,
            enhancement_type=EnhancementType.EXPANSION,
            business_context={
                "expansion_focus": expansion_focus,
                "target_length": target_length
            }
        )
        
        # Expand content
        result = await enhancement_service.enhance_content(enhancement_request)
        
        return {
            "original_content": result.original_content,
            "expanded_content": result.enhanced_content,
            "expansion_focus": expansion_focus,
            "target_length": target_length,
            "improvements": result.improvements,
            "suggestions": result.suggestions,
            "quality_score": result.quality_score,
            "confidence_score": result.confidence_score,
            "processing_time": result.processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content expansion failed: {str(e)}")

@router.get("/health")
async def enhancement_health_check():
    """Health check for enhancement service."""
    
    enhancement_service = get_enhancement_service()
    
    try:
        # Test enhancement service
        test_content = "This is a test content for health check."
        analysis = await enhancement_service._analyze_content(test_content)
        
        return {
            "status": "healthy",
            "service": "AI Enhancement Service",
            "timestamp": datetime.now().isoformat(),
            "test_analysis": {
                "readability_score": analysis.readability_score,
                "overall_quality": analysis.overall_quality.value
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "AI Enhancement Service",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }





























