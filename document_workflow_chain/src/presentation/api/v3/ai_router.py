"""
AI Router
=========

FastAPI router for AI operations and content generation.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ...shared.services.ai_service import (
    AIProvider,
    AIModelType,
    AIQualityLevel,
    generate_content,
    analyze_text,
    batch_generate,
    get_available_models,
    get_model_info,
    get_usage_statistics,
    summarize_text,
    extract_entities,
    analyze_sentiment,
    translate_text,
    generate_code
)
from ...shared.middleware.auth import get_current_user_optional
from ...shared.middleware.rate_limiter import rate_limit
from ...shared.middleware.metrics_middleware import record_ai_metrics
from ...shared.utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ai", tags=["AI Operations"])


# Request/Response models
class ContentGenerationRequest(BaseModel):
    """Content generation request"""
    prompt: str = Field(..., description="The prompt for content generation", min_length=1, max_length=10000)
    model: Optional[str] = Field(None, description="Specific model to use")
    provider: Optional[AIProvider] = Field(None, description="AI provider to use")
    model_type: AIModelType = Field(AIModelType.TEXT_GENERATION, description="Type of AI model")
    quality_level: AIQualityLevel = Field(AIQualityLevel.BALANCED, description="Quality level for generation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate", ge=1, le=32000)
    temperature: float = Field(0.7, description="Temperature for generation", ge=0.0, le=2.0)
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ContentGenerationResponse(BaseModel):
    """Content generation response"""
    id: str
    content: str
    model: str
    provider: str
    tokens_used: int
    processing_time: float
    quality_score: Optional[float]
    confidence_score: Optional[float]
    metadata: Dict[str, Any]


class TextAnalysisRequest(BaseModel):
    """Text analysis request"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=50000)
    analysis_type: AIModelType = Field(AIModelType.TEXT_ANALYSIS, description="Type of analysis")
    provider: Optional[AIProvider] = Field(None, description="AI provider to use")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchGenerationRequest(BaseModel):
    """Batch generation request"""
    prompts: List[str] = Field(..., description="List of prompts", min_items=1, max_items=10)
    model: Optional[str] = Field(None, description="Specific model to use")
    provider: Optional[AIProvider] = Field(None, description="AI provider to use")
    max_concurrent: int = Field(5, description="Maximum concurrent requests", ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    name: str
    provider: str
    model_type: str
    max_tokens: int
    cost_per_token: float
    quality_level: str
    capabilities: List[str]
    is_active: bool


class UsageStatisticsResponse(BaseModel):
    """Usage statistics response"""
    total_requests: int
    total_responses: int
    success_rate: float
    provider_usage: Dict[str, int]
    model_usage: Dict[str, int]
    total_tokens: int
    average_tokens: float
    average_processing_time: float
    available_models: int
    active_models: int


# AI Operations endpoints
@router.post("/generate", response_model=ContentGenerationResponse)
@rate_limit(requests=10, window=60)  # 10 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def generate_content_endpoint(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> ContentGenerationResponse:
    """Generate content using AI"""
    try:
        # Generate content
        response = await generate_content(
            prompt=request.prompt,
            model=request.model,
            provider=request.provider,
            model_type=request.model_type,
            quality_level=request.quality_level,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            user_id=current_user.get("id") if current_user else None,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        # Return response
        return ContentGenerationResponse(
            id=response.id,
            content=response.content,
            model=response.model,
            provider=response.provider.value,
            tokens_used=response.tokens_used,
            processing_time=response.processing_time,
            quality_score=response.quality_score,
            confidence_score=response.confidence_score,
            metadata=response.metadata
        )
    
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")


@router.post("/analyze", response_model=ContentGenerationResponse)
@rate_limit(requests=15, window=60)  # 15 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def analyze_text_endpoint(
    request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> ContentGenerationResponse:
    """Analyze text using AI"""
    try:
        # Analyze text
        response = await analyze_text(
            text=request.text,
            analysis_type=request.analysis_type,
            provider=request.provider,
            user_id=current_user.get("id") if current_user else None
        )
        
        # Return response
        return ContentGenerationResponse(
            id=response.id,
            content=response.content,
            model=response.model,
            provider=response.provider.value,
            tokens_used=response.tokens_used,
            processing_time=response.processing_time,
            quality_score=response.quality_score,
            confidence_score=response.confidence_score,
            metadata=response.metadata
        )
    
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")


@router.post("/batch-generate", response_model=List[ContentGenerationResponse])
@rate_limit(requests=5, window=60)  # 5 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def batch_generate_endpoint(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> List[ContentGenerationResponse]:
    """Generate content for multiple prompts in batch"""
    try:
        # Batch generate content
        responses = await batch_generate(
            prompts=request.prompts,
            model=request.model,
            provider=request.provider,
            max_concurrent=request.max_concurrent
        )
        
        # Convert responses
        result = []
        for response in responses:
            result.append(ContentGenerationResponse(
                id=response.id,
                content=response.content,
                model=response.model,
                provider=response.provider.value,
                tokens_used=response.tokens_used,
                processing_time=response.processing_time,
                quality_score=response.quality_score,
                confidence_score=response.confidence_score,
                metadata=response.metadata
            ))
        
        return result
    
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


# Specialized AI operations
@router.post("/summarize")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def summarize_text_endpoint(
    text: str = Body(..., description="Text to summarize"),
    max_length: int = Body(200, description="Maximum length of summary", ge=50, le=1000),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Summarize text"""
    try:
        summary = await summarize_text(
            text=text,
            max_length=max_length,
            user_id=current_user.get("id") if current_user else None
        )
        
        return {"summary": summary}
    
    except Exception as e:
        logger.error(f"Text summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text summarization failed: {str(e)}")


@router.post("/extract-entities")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def extract_entities_endpoint(
    text: str = Body(..., description="Text to extract entities from"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Extract entities from text"""
    try:
        entities = await extract_entities(
            text=text,
            user_id=current_user.get("id") if current_user else None
        )
        
        return {"entities": entities}
    
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.post("/analyze-sentiment")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def analyze_sentiment_endpoint(
    text: str = Body(..., description="Text to analyze sentiment"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Analyze sentiment of text"""
    try:
        sentiment = await analyze_sentiment(
            text=text,
            user_id=current_user.get("id") if current_user else None
        )
        
        return {"sentiment": sentiment}
    
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.post("/translate")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def translate_text_endpoint(
    text: str = Body(..., description="Text to translate"),
    target_language: str = Body("English", description="Target language"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Translate text"""
    try:
        translation = await translate_text(
            text=text,
            target_language=target_language,
            user_id=current_user.get("id") if current_user else None
        )
        
        return {"translation": translation}
    
    except Exception as e:
        logger.error(f"Text translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text translation failed: {str(e)}")


@router.post("/generate-code")
@rate_limit(requests=10, window=60)  # 10 requests per minute
@record_ai_metrics
@log_execution
@measure_performance
async def generate_code_endpoint(
    description: str = Body(..., description="Code description"),
    language: str = Body("python", description="Programming language"),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Generate code from description"""
    try:
        code = await generate_code(
            description=description,
            language=language,
            user_id=current_user.get("id") if current_user else None
        )
        
        return {"code": code}
    
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


# Model and system information endpoints
@router.get("/models", response_model=List[ModelInfoResponse])
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
async def get_available_models_endpoint(
    provider: Optional[AIProvider] = Query(None, description="Filter by provider")
) -> List[ModelInfoResponse]:
    """Get available AI models"""
    try:
        models = get_available_models(provider)
        
        result = []
        for model in models:
            result.append(ModelInfoResponse(
                name=model.name,
                provider=model.provider.value,
                model_type=model.model_type.value,
                max_tokens=model.max_tokens,
                cost_per_token=model.cost_per_token,
                quality_level=model.quality_level.value,
                capabilities=model.capabilities,
                is_active=model.is_active
            ))
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
@rate_limit(requests=30, window=60)  # 30 requests per minute
@log_execution
async def get_model_info_endpoint(model_name: str) -> ModelInfoResponse:
    """Get specific model information"""
    try:
        model = get_model_info(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return ModelInfoResponse(
            name=model.name,
            provider=model.provider.value,
            model_type=model.model_type.value,
            max_tokens=model.max_tokens,
            cost_per_token=model.cost_per_token,
            quality_level=model.quality_level.value,
            capabilities=model.capabilities,
            is_active=model.is_active
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.get("/usage-statistics", response_model=UsageStatisticsResponse)
@rate_limit(requests=10, window=60)  # 10 requests per minute
@log_execution
async def get_usage_statistics_endpoint() -> UsageStatisticsResponse:
    """Get AI usage statistics"""
    try:
        stats = get_usage_statistics()
        
        return UsageStatisticsResponse(
            total_requests=stats["total_requests"],
            total_responses=stats["total_responses"],
            success_rate=stats["success_rate"],
            provider_usage=stats["provider_usage"],
            model_usage=stats["model_usage"],
            total_tokens=stats["total_tokens"],
            average_tokens=stats["average_tokens"],
            average_processing_time=stats["average_processing_time"],
            available_models=stats["available_models"],
            active_models=stats["active_models"]
        )
    
    except Exception as e:
        logger.error(f"Failed to get usage statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage statistics: {str(e)}")


# Health check endpoint
@router.get("/health")
@log_execution
async def ai_health_check() -> Dict[str, Any]:
    """AI service health check"""
    try:
        # Check if AI service is running
        models = get_available_models()
        
        return {
            "status": "healthy",
            "available_models": len(models),
            "active_models": len([m for m in models if m.is_active]),
            "providers": list(set(m.provider.value for m in models)),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }
    
    except Exception as e:
        logger.error(f"AI health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }




