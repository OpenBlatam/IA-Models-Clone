from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Optional, Dict, Any
from fastapi import (
from fastapi.responses import JSONResponse
import logging
import time
import uuid
from models.fastapi_best_practices import (
from dependencies import CoreDependencies, AdvancedDependencies, require_authentication
from core.blocking_operations_limiter import limit_blocking_operations, OperationType
from core.exceptions import ValidationError, AIGenerationError, CacheError
        import asyncio
from typing import Any, List, Dict, Optional
"""
FastAPI Best Practices - Path Operations

This module implements path operations following FastAPI best practices:
- Proper HTTP methods and status codes
- Comprehensive response models
- Proper error handling and validation
- Clear documentation and examples
- Performance optimization
- Security best practices
"""

    APIRouter, HTTPException, Depends, Request, Response,
    status, Query, Path, Header, BackgroundTasks
)

# Import models
    CaptionGenerationRequest, CaptionGenerationResponse,
    BatchCaptionRequest, BatchCaptionResponse,
    UserPreferences, ErrorResponse, HealthResponse,
    CaptionAnalytics, ServiceStatus
)

# Import dependencies

# Import core components

logger = logging.getLogger(__name__)

# Create router with proper configuration
router = APIRouter(
    prefix="/api/v14",
    tags=["fastapi-best-practices"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"}
    }
)


# =============================================================================
# PATH OPERATIONS WITH BEST PRACTICES
# =============================================================================

@router.post(
    "/captions/generate",
    response_model=CaptionGenerationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Instagram Caption",
    description="""
    Generate a high-quality Instagram caption based on content description.
    
    **Features:**
    - Multiple caption styles and tones
    - Customizable hashtag count
    - Multi-language support
    - Emoji integration
    - Performance metrics
    
    **Rate Limits:**
    - 100 requests per minute per user
    - 1000 requests per hour per user
    """,
    response_description="Successfully generated caption with metadata",
    openapi_extra={
        "examples": {
            "casual_caption": {
                "summary": "Casual Caption Example",
                "description": "Generate a casual, friendly caption",
                "value": {
                    "content_description": "Beautiful sunset over mountains with golden light",
                    "style": "casual",
                    "tone": "friendly",
                    "hashtag_count": 15,
                    "language": "en",
                    "include_emoji": True
                }
            },
            "professional_caption": {
                "summary": "Professional Caption Example", 
                "description": "Generate a professional, formal caption",
                "value": {
                    "content_description": "Modern office space with natural lighting",
                    "style": "professional",
                    "tone": "professional",
                    "hashtag_count": 10,
                    "language": "en",
                    "include_emoji": False
                }
            }
        }
    }
)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="fastapi_caption_generation",
    user_id_param="user_id"
)
async def generate_caption(
    request: CaptionGenerationRequest,
    background_tasks: BackgroundTasks,
    deps: CoreDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID"),
    user_agent: Optional[str] = Header(default=None, alias="User-Agent")
) -> CaptionGenerationResponse:
    """
    Generate Instagram caption with best practices implementation.
    
    This endpoint demonstrates FastAPI best practices including:
    - Proper request/response models
    - Background tasks for non-critical operations
    - Request ID tracking
    - Performance monitoring
    - Comprehensive error handling
    - Rate limiting and validation
    """
    
    start_time = time.time()
    
    try:
        # Log request for monitoring
        logger.info(
            f"Caption generation request started - "
            f"User: {deps.user['id']}, "
            f"Style: {request.style}, "
            f"Language: {request.language}, "
            f"Request-ID: {request_id}"
        )
        
        # Generate caption using AI engine
        caption_result = await deps.ai_engine.generate_caption_optimized(
            content_description=request.content_description,
            style=request.style.value,
            tone=request.tone.value,
            hashtag_count=request.hashtag_count,
            language=request.language.value,
            include_emoji=request.include_emoji,
            max_length=request.max_length
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = CaptionGenerationResponse(
            caption=caption_result.caption,
            hashtags=caption_result.hashtags,
            style=request.style,
            tone=request.tone,
            language=request.language,
            processing_time=processing_time,
            model_used=caption_result.model_used,
            confidence_score=caption_result.confidence_score,
            character_count=len(caption_result.caption),
            word_count=len(caption_result.caption.split())
        )
        
        # Add background task for analytics
        background_tasks.add_task(
            log_caption_generation_analytics,
            user_id=deps.user["id"],
            request=request,
            response=response,
            processing_time=processing_time
        )
        
        # Log success
        logger.info(
            f"Caption generated successfully - "
            f"User: {deps.user['id']}, "
            f"Processing time: {processing_time:.3f}s, "
            f"Confidence: {caption_result.confidence_score:.2f}"
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error in caption generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except AIGenerationError as e:
        logger.error(f"AI generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Unexpected error in caption generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post(
    "/captions/batch-generate",
    response_model=BatchCaptionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Batch Generate Captions",
    description="""
    Generate multiple captions in a single request with concurrency control.
    
    **Features:**
    - Concurrent processing with configurable limits
    - Individual error handling for each request
    - Batch performance metrics
    - Progress tracking
    
    **Limits:**
    - Maximum 50 requests per batch
    - Maximum 20 concurrent operations
    """,
    response_description="Batch processing results with performance metrics"
)
@limit_blocking_operations(
    operation_type=OperationType.BATCH_OPERATION,
    identifier="fastapi_batch_caption_generation",
    user_id_param="user_id"
)
async def batch_generate_captions(
    batch_request: BatchCaptionRequest,
    background_tasks: BackgroundTasks,
    deps: AdvancedDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID")
) -> BatchCaptionResponse:
    """
    Batch generate captions with concurrency control and error handling.
    
    Demonstrates:
    - Batch processing best practices
    - Concurrency control with semaphores
    - Individual error handling
    - Performance monitoring
    - Background task processing
    """
    
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(batch_request.requests) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 requests per batch"
            )
        
        # Process requests with concurrency control
        semaphore = asyncio.Semaphore(batch_request.max_concurrent)
        
        async async def process_single_request(req: CaptionGenerationRequest) -> CaptionGenerationResponse:
            async with semaphore:
                try:
                    return await generate_caption(req, background_tasks, deps, request_id)
                except Exception as e:
                    logger.error(f"Batch item failed: {e}")
                    # Return error response for failed items
                    return CaptionGenerationResponse(
                        caption=f"Error: {str(e)}",
                        hashtags=[],
                        style=req.style,
                        tone=req.tone,
                        language=req.language,
                        processing_time=0.0,
                        model_used="error",
                        confidence_score=0.0,
                        character_count=0,
                        word_count=0
                    )
        
        # Execute batch processing
        tasks = [process_single_request(req) for req in batch_request.requests]
        results = await asyncio.gather(*tasks)
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful_count = sum(1 for r in results if r.model_used != "error")
        failed_count = len(results) - successful_count
        
        # Create response
        response = BatchCaptionResponse(
            results=results,
            total_processing_time=total_time,
            successful_count=successful_count,
            failed_count=failed_count
        )
        
        # Log batch completion
        logger.info(
            f"Batch processing completed - "
            f"User: {deps.user['id']}, "
            f"Total: {len(results)}, "
            f"Successful: {successful_count}, "
            f"Failed: {failed_count}, "
            f"Time: {total_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch processing failed"
        )


@router.get(
    "/captions/{caption_id}",
    response_model=CaptionGenerationResponse,
    summary="Get Caption by ID",
    description="Retrieve a previously generated caption by its ID",
    response_description="Caption details and metadata"
)
async def get_caption_by_id(
    caption_id: str = Path(..., description="Unique caption identifier", example="cap_123456789"),
    deps: CoreDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID")
) -> CaptionGenerationResponse:
    """
    Retrieve caption by ID with caching and validation.
    
    Demonstrates:
    - Path parameter validation
    - Caching best practices
    - Error handling for not found
    - Request tracking
    """
    
    try:
        # Check cache first
        cache_key = f"caption:{caption_id}"
        cached_result = await deps.cache_manager.get(cache_key)
        
        if cached_result:
            logger.info(f"Cache hit for caption {caption_id}")
            return CaptionGenerationResponse(**cached_result)
        
        # Get from database (simplified for demo)
        # In real implementation, this would query the database
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Caption with ID {caption_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving caption {caption_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving caption"
        )


@router.get(
    "/captions",
    response_model=List[CaptionGenerationResponse],
    summary="List User Captions",
    description="Retrieve paginated list of user's generated captions",
    response_description="List of captions with pagination metadata"
)
async def list_user_captions(
    skip: int = Query(default=0, ge=0, description="Number of records to skip", example=0),
    limit: int = Query(default=10, ge=1, le=100, description="Number of records to return", example=10),
    style: Optional[str] = Query(default=None, description="Filter by caption style", example="casual"),
    language: Optional[str] = Query(default=None, description="Filter by language", example="en"),
    deps: CoreDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID")
) -> List[CaptionGenerationResponse]:
    """
    List user captions with pagination and filtering.
    
    Demonstrates:
    - Query parameter validation
    - Pagination best practices
    - Filtering capabilities
    - Performance optimization
    """
    
    try:
        # Build query parameters
        query_params = {
            "user_id": deps.user["id"],
            "skip": skip,
            "limit": limit
        }
        
        if style:
            query_params["style"] = style
        if language:
            query_params["language"] = language
        
        # Get captions from database (simplified for demo)
        # In real implementation, this would query the database with filters
        
        # Return empty list for demo
        return []
        
    except Exception as e:
        logger.error(f"Error listing captions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving captions"
        )


@router.put(
    "/captions/{caption_id}",
    response_model=CaptionGenerationResponse,
    summary="Update Caption",
    description="Update an existing caption (regenerate with new parameters)",
    response_description="Updated caption with new parameters"
)
async def update_caption(
    caption_id: str = Path(..., description="Caption ID to update", example="cap_123456789"),
    request: CaptionGenerationRequest = ...,
    deps: CoreDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID")
) -> CaptionGenerationResponse:
    """
    Update caption by regenerating with new parameters.
    
    Demonstrates:
    - PUT method for updates
    - Idempotent operations
    - Validation of existing resources
    - Cache invalidation
    """
    
    try:
        # Check if caption exists (simplified for demo)
        # In real implementation, this would check the database
        
        # Regenerate caption with new parameters
        response = await generate_caption(request, BackgroundTasks(), deps, request_id)
        
        # Update cache
        cache_key = f"caption:{caption_id}"
        await deps.cache_manager.set(cache_key, response.model_dump(), ttl=3600)
        
        return response
        
    except Exception as e:
        logger.error(f"Error updating caption {caption_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating caption"
        )


@router.delete(
    "/captions/{caption_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Caption",
    description="Delete a caption by ID",
    response_description="Caption successfully deleted"
)
async def delete_caption(
    caption_id: str = Path(..., description="Caption ID to delete", example="cap_123456789"),
    deps: CoreDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID")
) -> None:
    """
    Delete caption by ID.
    
    Demonstrates:
    - DELETE method for resource removal
    - Proper status codes
    - Cache invalidation
    - Soft delete considerations
    """
    
    try:
        # Check if caption exists (simplified for demo)
        # In real implementation, this would check the database
        
        # Delete from database (simplified for demo)
        # In real implementation, this would delete from database
        
        # Invalidate cache
        cache_key = f"caption:{caption_id}"
        await deps.cache_manager.delete(cache_key)
        
        logger.info(f"Caption {caption_id} deleted by user {deps.user['id']}")
        
        # Return 204 No Content
        return None
        
    except Exception as e:
        logger.error(f"Error deleting caption {caption_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting caption"
        )


# =============================================================================
# HEALTH AND MONITORING ENDPOINTS
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of all services",
    response_description="Comprehensive health status of all services"
)
async def health_check(
    deps: AdvancedDependencies = Depends()
) -> HealthResponse:
    """
    Comprehensive health check endpoint.
    
    Demonstrates:
    - Health check best practices
    - Service dependency monitoring
    - Performance metrics
    - Error aggregation
    """
    
    try:
        # Check individual services
        services = {}
        
        # Check AI engine
        try:
            ai_status = await check_ai_engine_health(deps)
            services["ai_engine"] = ai_status
        except Exception as e:
            services["ai_engine"] = ServiceStatus(
                service="ai_engine",
                status="unhealthy",
                version="unknown",
                uptime=0.0
            )
        
        # Check cache
        try:
            cache_status = await check_cache_health(deps)
            services["cache"] = cache_status
        except Exception as e:
            services["cache"] = ServiceStatus(
                service="cache",
                status="unhealthy",
                version="unknown",
                uptime=0.0
            )
        
        # Check database
        try:
            db_status = await check_database_health(deps)
            services["database"] = db_status
        except Exception as e:
            services["database"] = ServiceStatus(
                service="database",
                status="unhealthy",
                version="unknown",
                uptime=0.0
            )
        
        # Determine overall status
        overall_status = "healthy"
        if any(service.status == "unhealthy" for service in services.values()):
            overall_status = "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            version="14.0.0",
            services=services
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="14.0.0",
            services={}
        )


@router.get(
    "/analytics",
    response_model=CaptionAnalytics,
    summary="Get Analytics",
    description="Retrieve caption generation analytics and metrics",
    response_description="Comprehensive analytics data"
)
async def get_analytics(
    deps: AdvancedDependencies = Depends(),
    request_id: str = Header(default_factory=lambda: str(uuid.uuid4()), alias="X-Request-ID")
) -> CaptionAnalytics:
    """
    Get comprehensive analytics data.
    
    Demonstrates:
    - Analytics endpoint best practices
    - Data aggregation
    - Performance metrics
    - Caching for expensive operations
    """
    
    try:
        # Check cache for analytics
        cache_key = "analytics:summary"
        cached_analytics = await deps.cache_manager.get(cache_key)
        
        if cached_analytics:
            return CaptionAnalytics(**cached_analytics)
        
        # Generate analytics (simplified for demo)
        analytics = CaptionAnalytics(
            total_captions_generated=1000,
            average_processing_time=1.5,
            most_popular_style="casual",
            most_popular_tone="friendly",
            language_distribution={"en": 800, "es": 150, "fr": 50},
            success_rate=98.5,
            cache_hit_rate=75.2
        )
        
        # Cache analytics for 5 minutes
        await deps.cache_manager.set(cache_key, analytics.model_dump(), ttl=300)
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error retrieving analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving analytics"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def log_caption_generation_analytics(
    user_id: str,
    request: CaptionGenerationRequest,
    response: CaptionGenerationResponse,
    processing_time: float
) -> None:
    """Background task to log analytics data"""
    try:
        # Log analytics data (simplified for demo)
        logger.info(
            f"Analytics - User: {user_id}, "
            f"Style: {request.style}, "
            f"Language: {request.language}, "
            f"Processing time: {processing_time:.3f}s, "
            f"Confidence: {response.confidence_score:.2f}"
        )
    except Exception as e:
        logger.error(f"Error logging analytics: {e}")


async def check_ai_engine_health(deps: AdvancedDependencies) -> ServiceStatus:
    """Check AI engine health"""
    try:
        # Check AI engine status (simplified for demo)
        return ServiceStatus(
            service="ai_engine",
            status="healthy",
            version="1.0.0",
            uptime=3600.0
        )
    except Exception as e:
        raise e


async def check_cache_health(deps: AdvancedDependencies) -> ServiceStatus:
    """Check cache health"""
    try:
        # Check cache status (simplified for demo)
        return ServiceStatus(
            service="cache",
            status="healthy",
            version="1.0.0",
            uptime=3600.0
        )
    except Exception as e:
        raise e


async def check_database_health(deps: AdvancedDependencies) -> ServiceStatus:
    """Check database health"""
    try:
        # Check database status (simplified for demo)
        return ServiceStatus(
            service="database",
            status="healthy",
            version="1.0.0",
            uptime=3600.0
        )
    except Exception as e:
        raise e


# Export router
__all__ = ["router"] 