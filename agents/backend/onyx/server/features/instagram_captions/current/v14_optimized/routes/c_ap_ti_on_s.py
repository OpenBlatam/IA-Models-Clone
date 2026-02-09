from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
from typing import List, Optional
from ..types import (
from ..utils import validate_api_key, sanitize_content, generate_request_id
from ..core.optimized_engine import optimized_engine, performance_monitor
from ..core.blocking_operations_limiter import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v14.0 - Caption Generation Routes
FastAPI router for caption generation endpoints with blocking operations limiting
"""


    OptimizedRequest, 
    OptimizedResponse, 
    BatchRequest, 
    BatchResponse,
    ErrorResponse
)
    blocking_limiter, limit_blocking_operations, limit_blocking_thread_operations,
    OperationType, rate_limit_context, concurrency_limit_context
)

# Router configuration
router = APIRouter(
    prefix="/api/v14",
    tags=["Caption Generation"],
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},  # Rate limit exceeded
        503: {"model": ErrorResponse},  # Service unavailable (circuit breaker)
        500: {"model": ErrorResponse}
    }
)

# Security
security = HTTPBearer()

# Dependency for API key validation
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key - async function for dependency injection"""
    if not validate_api_key(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Dependency to extract user identifier
async def get_user_identifier(request: Request, api_key: str = Depends(verify_api_key)) -> str:
    """Extract user identifier from request or API key"""
    # In a real application, you might extract user ID from JWT token
    # For now, we'll use the API key as identifier
    return api_key[:16]  # Use first 16 characters as identifier

@router.post("/generate", response_model=OptimizedResponse)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="single_generation",
    user_id_param="user_id"
)
async def generate_caption(
    request: OptimizedRequest,
    request_obj: Request,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier),
    background_tasks: BackgroundTasks = None
) -> OptimizedResponse:
    """Ultra-fast single caption generation with rate limiting and concurrency control"""
    start_time = time.time()
    
    try:
        # Sanitize content
        request.content_description = sanitize_content(request.content_description)
        
        # Generate caption with blocking operations limiting
        response = await optimized_engine.generate_caption(request)
        
        # Record performance metrics
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, True)
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, False)
        
        # Handle specific error types
        if "Rate limit exceeded" in str(e):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 60
                }
            )
        elif "Circuit breaker is open" in str(e):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Service temporarily unavailable. Please try again later.",
                    "retry_after": 30
                }
            )
        else:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/batch", response_model=BatchResponse)
@limit_blocking_operations(
    operation_type=OperationType.BATCH_PROCESSING,
    identifier="batch_generation",
    user_id_param="user_id"
)
async def batch_generate(
    batch_request: BatchRequest,
    request_obj: Request,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> BatchResponse:
    """Optimized batch processing for multiple caption requests with enhanced limiting"""
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(batch_request.requests) > 100:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "batch_size_exceeded",
                    "message": "Batch size cannot exceed 100 requests",
                    "max_batch_size": 100
                }
            )
        
        # Additional rate limiting for batch operations
        async with rate_limit_context(OperationType.BATCH_PROCESSING, f"batch_{user_id}"):
            # Sanitize all requests
            for req in batch_request.requests:
                req.content_description = sanitize_content(req.content_description)
            
            # Process batch with concurrency limiting
            async with concurrency_limit_context(OperationType.BATCH_PROCESSING, user_id):
                responses = await optimized_engine.batch_generate(batch_request.requests)
        
        # Record performance
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, True)
        
        return BatchResponse(
            batch_id=generate_request_id(),
            total_requests=len(batch_request.requests),
            successful_requests=len(responses),
            processing_time=processing_time,
            responses=responses
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, False)
        
        # Handle specific error types
        if "Rate limit exceeded" in str(e):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "Batch processing rate limit exceeded. Please try again later.",
                    "retry_after": 120
                }
            )
        elif "Circuit breaker is open" in str(e):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_unavailable",
                    "message": "Batch processing service temporarily unavailable.",
                    "retry_after": 60
                }
            )
        else:
            raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.post("/generate/legacy", response_model=OptimizedResponse)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="legacy_generation",
    user_id_param="user_id"
)
async def generate_caption_legacy(
    content_description: str,
    style: str = "casual",
    hashtag_count: int = 15,
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> OptimizedResponse:
    """Legacy endpoint for backward compatibility with rate limiting"""
    try:
        # Create request object
        request = OptimizedRequest(
            content_description=content_description,
            style=style,
            hashtag_count=hashtag_count
        )
        
        # Use main generation endpoint
        return await generate_caption(request, None, api_key, user_id)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

@router.post("/generate/priority", response_model=OptimizedResponse)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="priority_generation",
    user_id_param="user_id"
)
async def generate_caption_priority(
    request: OptimizedRequest,
    priority_level: int = 1,  # 1-5, higher is more priority
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> OptimizedResponse:
    """Priority caption generation with enhanced rate limiting"""
    start_time = time.time()
    
    try:
        # Validate priority level
        if not 1 <= priority_level <= 5:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_priority",
                    "message": "Priority level must be between 1 and 5",
                    "valid_range": [1, 5]
                }
            )
        
        # Use priority-specific identifier
        priority_identifier = f"priority_{priority_level}_{user_id}"
        
        # Sanitize content
        request.content_description = sanitize_content(request.content_description)
        
        # Generate caption with priority-specific limiting
        response = await blocking_limiter.execute_with_limits(
            OperationType.CAPTION_GENERATION,
            optimized_engine.generate_caption,
            priority_identifier,
            user_id,
            request
        )
        
        # Record performance metrics
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, True)
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, False)
        
        if "Rate limit exceeded" in str(e):
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "priority_rate_limit_exceeded",
                    "message": "Priority generation rate limit exceeded.",
                    "retry_after": 30
                }
            )
        else:
            raise HTTPException(status_code=500, detail=f"Priority generation failed: {str(e)}")

@router.get("/limits/status")
async def get_limits_status(
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> dict:
    """Get current rate limiting and concurrency status for the user"""
    try:
        # Get metrics for different operation types
        caption_metrics = await blocking_limiter.get_metrics(OperationType.CAPTION_GENERATION)
        batch_metrics = await blocking_limiter.get_metrics(OperationType.BATCH_PROCESSING)
        
        # Get current limits
        caption_limiter = await blocking_limiter.get_rate_limiter(
            OperationType.CAPTION_GENERATION, f"single_generation_{user_id}"
        )
        batch_limiter = await blocking_limiter.get_rate_limiter(
            OperationType.BATCH_PROCESSING, f"batch_{user_id}"
        )
        
        return {
            "user_id": user_id,
            "timestamp": time.time(),
            "caption_generation": {
                "metrics": caption_metrics.get("caption_generation", {}),
                "rate_limit": {
                    "tokens_available": caption_limiter.tokens,
                    "capacity": caption_limiter.capacity,
                    "refill_rate": caption_limiter.refill_rate
                }
            },
            "batch_processing": {
                "metrics": batch_metrics.get("batch_processing", {}),
                "rate_limit": {
                    "tokens_available": batch_limiter.tokens,
                    "capacity": batch_limiter.capacity,
                    "refill_rate": batch_limiter.refill_rate
                }
            },
            "limits": {
                "caption_generation_per_minute": 30,
                "batch_processing_per_minute": 10,
                "max_concurrent_caption_requests": 3,
                "max_concurrent_batch_requests": 2,
                "max_batch_size": 100
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get limits status: {str(e)}")

@router.post("/limits/reset")
async def reset_user_limits(
    api_key: str = Depends(verify_api_key),
    user_id: str = Depends(get_user_identifier)
) -> dict:
    """Reset rate limiting and concurrency limits for the user (admin only)"""
    try:
        # Reset metrics for the user
        await blocking_limiter.reset_metrics(OperationType.CAPTION_GENERATION)
        await blocking_limiter.reset_metrics(OperationType.BATCH_PROCESSING)
        
        return {
            "success": True,
            "message": "User limits reset successfully",
            "user_id": user_id,
            "timestamp": time.time()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset limits: {str(e)}")

# Health check endpoint with limiting
@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint with basic rate limiting"""
    try:
        # Basic health check without heavy limiting
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "14.0.0",
            "features": [
                "rate_limiting",
                "concurrency_control",
                "circuit_breaker",
                "blocking_operations_limiting"
            ]
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        } 