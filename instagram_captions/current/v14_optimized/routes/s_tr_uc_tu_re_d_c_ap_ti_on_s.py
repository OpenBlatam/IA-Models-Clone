from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import logging
import time
from dependencies import (
from .factory import RouteBuilder, RouteDecorator
from core.blocking_operations_limiter import limit_blocking_operations, OperationType
from core.exceptions import ValidationError, AIGenerationError
from core.schemas import CaptionRequest, CaptionResponse, BatchRequest, BatchResponse
        import asyncio
from typing import Any, List, Dict, Optional
"""
Structured Captions Routes for Instagram Captions API v14.0

Well-organized captions routes demonstrating:
- Clear dependency injection
- Consistent patterns and conventions
- Proper error handling
- Performance monitoring
- Easy testing and maintenance
"""


# Import dependencies
    CoreDependencies, AdvancedDependencies,
    require_authentication, validate_content_length
)

# Import route factory components

# Import core components

logger = logging.getLogger(__name__)

# Create router using factory
router = APIRouter(prefix="/captions", tags=["captions"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class StructuredCaptionRequest(BaseModel):
    """Structured caption generation request"""
    content_description: str = Field(..., min_length=10, max_length=1000, description="Content description")
    style: str = Field(default="casual", description="Caption style")
    tone: str = Field(default="friendly", description="Caption tone")
    hashtag_count: int = Field(default=15, ge=0, le=30, description="Number of hashtags")
    language: str = Field(default="en", description="Language code")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "content_description": "Beautiful sunset over mountains with golden light",
                "style": "casual",
                "tone": "friendly",
                "hashtag_count": 15,
                "language": "en"
            }
        }


class StructuredCaptionResponse(BaseModel):
    """Structured caption generation response"""
    caption: str = Field(..., description="Generated caption")
    hashtags: List[str] = Field(..., description="Generated hashtags")
    style: str = Field(..., description="Applied style")
    tone: str = Field(..., description="Applied tone")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="AI model used")
    confidence_score: float = Field(..., description="Confidence score")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "caption": "Golden hour magic ✨ Nature's perfect lighting show",
                "hashtags": ["#sunset", "#mountains", "#goldenhour", "#nature"],
                "style": "casual",
                "tone": "friendly",
                "processing_time": 1.23,
                "model_used": "gpt-3.5-turbo",
                "confidence_score": 0.95
            }
        }


# =============================================================================
# ROUTE HANDLERS WITH CLEAR DEPENDENCIES
# =============================================================================

@router.post("/generate", response_model=StructuredCaptionResponse)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="structured_caption_generation",
    user_id_param="user_id"
)
async def generate_structured_caption(
    request: StructuredCaptionRequest,
    deps: CoreDependencies = Depends(),
    request_context: Request = Depends()
) -> StructuredCaptionResponse:
    """
    Generate structured caption with clear dependencies
    
    Demonstrates:
    - Clear dependency injection
    - Input validation
    - Error handling
    - Performance monitoring
    """
    
    start_time = time.time()
    
    try:
        # Validate content length
        validated_content = await validate_content_length(
            request.content_description, 
            max_length=1000
        )
        
        # Generate caption using AI engine
        caption_result = await deps.ai_engine.generate_caption_optimized(
            content_description=validated_content,
            style=request.style,
            tone=request.tone,
            hashtag_count=request.hashtag_count,
            language=request.language
        )
        
        # Process result
        processing_time = time.time() - start_time
        
        # Create response
        response = StructuredCaptionResponse(
            caption=caption_result.caption,
            hashtags=caption_result.hashtags,
            style=request.style,
            tone=request.tone,
            processing_time=processing_time,
            model_used=caption_result.model_used,
            confidence_score=caption_result.confidence_score
        )
        
        # Log success
        logger.info(
            f"Caption generated successfully for user {deps.user['id']} "
            f"in {processing_time:.3f}s"
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except AIGenerationError as e:
        logger.error(f"AI generation error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate-advanced", response_model=StructuredCaptionResponse)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="advanced_caption_generation",
    user_id_param="user_id"
)
async def generate_advanced_caption(
    request: StructuredCaptionRequest,
    deps: AdvancedDependencies = Depends(),
    request_context: Request = Depends()
) -> StructuredCaptionResponse:
    """
    Generate advanced caption with full dependency set
    
    Demonstrates:
    - Advanced dependency injection
    - Database operations
    - Caching
    - Performance monitoring
    """
    
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"caption:{hash(request.content_description)}:{request.style}:{request.tone}"
        cached_result = await deps.cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for caption generation")
            return StructuredCaptionResponse(**cached_result)
        
        # Generate caption using AI engine
        caption_result = await deps.ai_engine.generate_caption_optimized(
            content_description=request.content_description,
            style=request.style,
            tone=request.tone,
            hashtag_count=request.hashtag_count,
            language=request.language
        )
        
        # Save to database
        await deps.db_pool.execute_query(
            query="""
                INSERT INTO caption_history (user_id, content, caption, style, tone, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            params=(
                deps.user["id"],
                request.content_description,
                caption_result.caption,
                request.style,
                request.tone
            )
        )
        
        # Cache result
        response_data = {
            "caption": caption_result.caption,
            "hashtags": caption_result.hashtags,
            "style": request.style,
            "tone": request.tone,
            "processing_time": time.time() - start_time,
            "model_used": caption_result.model_used,
            "confidence_score": caption_result.confidence_score
        }
        
        await deps.cache_manager.set(cache_key, response_data, ttl=3600)
        
        # Record performance metrics
        if deps.io_monitor:
            deps.io_monitor.record_operation(
                operation_type="advanced_caption_generation",
                duration=time.time() - start_time,
                success=True
            )
        
        return StructuredCaptionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Advanced caption generation failed: {e}")
        raise HTTPException(status_code=500, detail="Caption generation failed")


@router.post("/batch-generate", response_model=List[StructuredCaptionResponse])
@limit_blocking_operations(
    operation_type=OperationType.BATCH_OPERATION,
    identifier="batch_caption_generation",
    user_id_param="user_id"
)
async def batch_generate_captions(
    requests: List[StructuredCaptionRequest],
    deps: AdvancedDependencies = Depends(),
    max_concurrent: int = 5
) -> List[StructuredCaptionResponse]:
    """
    Batch generate captions with concurrency control
    
    Demonstrates:
    - Batch processing
    - Concurrency control
    - Error handling for batch operations
    """
    
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 requests per batch")
    
    start_time = time.time()
    results = []
    
    try:
        # Process requests in batches
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async async def process_single_request(req: StructuredCaptionRequest) -> StructuredCaptionResponse:
            async with semaphore:
                return await generate_structured_caption(req, deps)
        
        # Execute batch processing
        tasks = [process_single_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                # Return error response for failed items
                processed_results.append(
                    StructuredCaptionResponse(
                        caption=f"Error: {str(result)}",
                        hashtags=[],
                        style=requests[i].style,
                        tone=requests[i].tone,
                        processing_time=0.0,
                        model_used="error",
                        confidence_score=0.0
                    )
                )
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed: {len(processed_results)} items in {time.time() - start_time:.3f}s")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")


# =============================================================================
# ROUTE BUILDER PATTERN EXAMPLE
# =============================================================================

# Create route builder for monitoring endpoints
monitoring_builder = RouteBuilder(router)

@monitoring_builder.with_dependencies(Depends(require_authentication))
@monitoring_builder.with_tags("monitoring")
@monitoring_builder.with_description("Get caption generation statistics")
@monitoring_builder.build_route("/stats")
async def get_caption_stats(
    deps: AdvancedDependencies = Depends()
) -> Dict[str, Any]:
    """Get caption generation statistics"""
    
    try:
        # Get database statistics
        db_stats = await deps.db_pool.execute_query(
            query="""
                SELECT 
                    COUNT(*) as total_captions,
                    AVG(LENGTH(caption)) as avg_caption_length,
                    COUNT(DISTINCT user_id) as unique_users,
                    MAX(created_at) as last_generation
                FROM caption_history
            """
        )
        
        # Get cache statistics
        cache_stats = await deps.cache_manager.get_stats()
        
        # Get AI engine statistics
        engine_stats = deps.ai_engine.get_stats()
        
        return {
            "database": dict(db_stats[0]) if db_stats else {},
            "cache": cache_stats,
            "ai_engine": engine_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get caption stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")


# =============================================================================
# UTILITY ROUTES
# =============================================================================

@router.get("/styles")
async def get_available_styles() -> Dict[str, List[str]]:
    """Get available caption styles"""
    return {
        "styles": [
            "casual", "formal", "creative", "professional",
            "funny", "inspirational", "minimal", "detailed"
        ],
        "tones": [
            "friendly", "professional", "casual", "enthusiastic",
            "calm", "energetic", "sophisticated", "playful"
        ]
    }


@router.get("/health")
async def caption_service_health() -> Dict[str, Any]:
    """Health check for caption service"""
    return {
        "service": "caption-generation",
        "status": "healthy",
        "version": "14.0.0",
        "timestamp": time.time()
    }


# =============================================================================
# ERROR HANDLING MIDDLEWARE
# =============================================================================

@router.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return {
        "error": "validation_error",
        "message": str(exc),
        "timestamp": time.time()
    }


@router.exception_handler(AIGenerationError)
async def ai_generation_exception_handler(request: Request, exc: AIGenerationError):
    """Handle AI generation errors"""
    logger.error(f"AI generation error: {exc}")
    return {
        "error": "ai_generation_error",
        "message": str(exc),
        "timestamp": time.time()
    }


# Export router
__all__ = ["router"] 