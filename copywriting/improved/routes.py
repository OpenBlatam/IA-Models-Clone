"""
API Routes for Copywriting Service
=================================

Clean, async routes following FastAPI best practices with proper error handling.
"""

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from fastapi_limiter.depends import RateLimiter
from fastapi_cache2.decorator import cache

from .schemas import (
    CopywritingRequest,
    CopywritingResponse,
    FeedbackRequest,
    FeedbackResponse,
    BatchCopywritingRequest,
    BatchCopywritingResponse,
    HealthCheckResponse,
    ErrorResponse
)
from .services import get_copywriting_service, CopywritingService
from .exceptions import (
    CopywritingException,
    ValidationError,
    ContentGenerationError,
    RateLimitExceededError,
    ResourceNotFoundError
)
from .config import get_api_settings, get_cache_settings

logger = logging.getLogger(__name__)
api_settings = get_api_settings()
cache_settings = get_cache_settings()

# Create router
router = APIRouter(
    prefix="/api/v2/copywriting",
    tags=["copywriting"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    }
)


def create_error_response(
    error: CopywritingException,
    status_code: int,
    request_id: UUID = None
) -> JSONResponse:
    """Create standardized error response"""
    error_response = ErrorResponse(
        error_code=error.error_code,
        error_message=error.message,
        error_details=error.details,
        request_id=request_id
    )
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


@router.post(
    "/generate",
    response_model=CopywritingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate copywriting content",
    description="Generate copywriting content based on the provided request parameters"
)
async def generate_copywriting(
    request: CopywritingRequest,
    service: CopywritingService = Depends(get_copywriting_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=api_settings.rate_limit_requests, seconds=api_settings.rate_limit_window))
) -> CopywritingResponse:
    """
    Generate copywriting content.
    
    This endpoint creates copywriting content based on the provided parameters including
    topic, target audience, tone, style, and purpose.
    """
    try:
        logger.info(f"Generating copywriting for topic: {request.topic}")
        response = await service.generate_copywriting(request)
        logger.info(f"Successfully generated copywriting with {response.total_variants} variants")
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(e, status.HTTP_400_BAD_REQUEST).body
        )
    except ContentGenerationError as e:
        logger.error(f"Content generation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=create_error_response(e, status.HTTP_422_UNPROCESSABLE_ENTITY).body
        )
    except RateLimitExceededError as e:
        logger.warning(f"Rate limit exceeded: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=create_error_response(e, status.HTTP_429_TOO_MANY_REQUESTS).body
        )
    except Exception as e:
        logger.error(f"Unexpected error in generate_copywriting: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                ContentGenerationError("Internal server error"),
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ).body
        )


@router.post(
    "/generate/batch",
    response_model=BatchCopywritingResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate copywriting content in batch",
    description="Generate copywriting content for multiple requests in a single batch operation"
)
async def generate_batch_copywriting(
    request: BatchCopywritingRequest,
    service: CopywritingService = Depends(get_copywriting_service),
    rate_limiter: RateLimiter = Depends(RateLimiter(times=5, seconds=60))  # Lower rate limit for batch
) -> BatchCopywritingResponse:
    """
    Generate copywriting content in batch.
    
    This endpoint processes multiple copywriting requests in a single operation,
    providing better performance for bulk content generation.
    """
    try:
        logger.info(f"Processing batch copywriting with {len(request.requests)} requests")
        response = await service.generate_batch_copywriting(request)
        logger.info(f"Batch processing completed: {response.success_count} successful, {response.failure_count} failed")
        return response
        
    except ValidationError as e:
        logger.warning(f"Batch validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(e, status.HTTP_400_BAD_REQUEST).body
        )
    except Exception as e:
        logger.error(f"Unexpected error in generate_batch_copywriting: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                ContentGenerationError("Internal server error"),
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ).body
        )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback",
    description="Submit feedback for a copywriting variant to improve future generations"
)
async def submit_feedback(
    request: FeedbackRequest,
    service: CopywritingService = Depends(get_copywriting_service)
) -> FeedbackResponse:
    """
    Submit feedback for a copywriting variant.
    
    This endpoint allows users to provide feedback on generated content,
    which helps improve the quality of future generations.
    """
    try:
        logger.info(f"Submitting feedback for variant: {request.variant_id}")
        response = await service.submit_feedback(request)
        logger.info(f"Feedback submitted successfully for variant: {request.variant_id}")
        return response
        
    except ValidationError as e:
        logger.warning(f"Feedback validation error: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(e, status.HTTP_400_BAD_REQUEST).body
        )
    except ResourceNotFoundError as e:
        logger.warning(f"Variant not found: {request.variant_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e, status.HTTP_404_NOT_FOUND).body
        )
    except Exception as e:
        logger.error(f"Unexpected error in submit_feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                ContentGenerationError("Internal server error"),
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ).body
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check",
    description="Check the health status of the copywriting service"
)
@cache(expire=cache_settings.default_ttl)
async def health_check(
    service: CopywritingService = Depends(get_copywriting_service)
) -> HealthCheckResponse:
    """
    Health check endpoint.
    
    This endpoint provides information about the service health,
    including database and Redis connectivity status.
    """
    try:
        logger.debug("Performing health check")
        response = await service.get_health_status()
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Return unhealthy status if health check fails
        return HealthCheckResponse(
            status="unhealthy",
            uptime_seconds=0,
            dependencies={"error": str(e)}
        )


@router.get(
    "/variants/{variant_id}",
    response_model=dict,
    summary="Get variant details",
    description="Get detailed information about a specific copywriting variant"
)
async def get_variant(
    variant_id: UUID,
    service: CopywritingService = Depends(get_copywriting_service)
) -> dict:
    """
    Get variant details.
    
    This endpoint retrieves detailed information about a specific
    copywriting variant by its ID.
    """
    try:
        logger.info(f"Retrieving variant: {variant_id}")
        # This would typically query the database for the variant
        # For now, return a placeholder response
        raise ResourceNotFoundError(
            message="Variant not found",
            resource_type="variant",
            resource_id=str(variant_id)
        )
        
    except ResourceNotFoundError as e:
        logger.warning(f"Variant not found: {variant_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(e, status.HTTP_404_NOT_FOUND).body
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_variant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                ContentGenerationError("Internal server error"),
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ).body
        )


@router.get(
    "/stats",
    response_model=dict,
    summary="Get service statistics",
    description="Get statistics about the copywriting service usage and performance"
)
@cache(expire=300)  # Cache for 5 minutes
async def get_stats(
    service: CopywritingService = Depends(get_copywriting_service)
) -> dict:
    """
    Get service statistics.
    
    This endpoint provides statistics about the service usage,
    performance metrics, and other relevant information.
    """
    try:
        logger.debug("Retrieving service statistics")
        # This would typically query the database for statistics
        # For now, return placeholder data
        return {
            "total_requests": 0,
            "total_variants_generated": 0,
            "average_processing_time_ms": 0,
            "success_rate": 1.0,
            "last_updated": "2024-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                ContentGenerationError("Internal server error"),
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ).body
        )


# Add middleware for request logging
@router.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response






























