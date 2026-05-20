from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import time
import logging
import uuid
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from ..types.schemas import (
from ..core.prioritized_engine import PrioritizedAIEngine
from ..middleware import create_middleware_stack
from ..utils.error_handling import ValidationEngine, error_tracker
from ..utils.performance_monitor import PerformanceMonitor
from ..utils.cache_manager import SmartCacheManager
from ..utils.rate_limiter import RateLimiter
from ..types.exceptions import (
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Instagram Captions API v14.0 - FastAPI Implementation
Ultra-optimized FastAPI application with comprehensive Pydantic validation
"""



# Import our comprehensive schemas
    CaptionGenerationRequest,
    CaptionGenerationResponse,
    BatchCaptionRequest,
    BatchCaptionResponse,
    CaptionOptimizationRequest,
    APIErrorResponse,
    PerformanceMetrics,
    HealthCheckResponse,
    create_error_response,
    validate_request_data,
    sanitize_input_data
)

# Import our optimized components

# Import exception types
    ValidationError,
    ContentValidationError,
    StyleValidationError,
    HashtagCountError,
    AIGenerationError,
    RateLimitError,
    CacheError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("🚀 Starting Instagram Captions API v14.0")
    
    # Initialize components
    global optimized_engine, cache_manager, rate_limiter, performance_monitor, validation_engine
    
    try:
        # Initialize core components
        cache_manager = SmartCacheManager()
        rate_limiter = RateLimiter()
        performance_monitor = PerformanceMonitor()
        validation_engine = ValidationEngine()
        optimized_engine = PrioritizedAIEngine()
        
        # Warm up cache and models
        await cache_manager.initialize()
        await optimized_engine.initialize()
        
        logger.info("✅ All components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Instagram Captions API v14.0")
    
    try:
        # Cleanup resources
        await cache_manager.cleanup()
        await optimized_engine.cleanup()
        logger.info("✅ Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Instagram Captions API v14.0",
    description="Ultra-optimized Instagram caption generation with comprehensive validation",
    version="14.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add comprehensive middleware stack
middleware_stack = create_middleware_stack()
for middleware in middleware_stack:
    app.add_middleware(middleware)

# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

async async def get_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())

async async def validate_api_key_dependency(request: Request) -> str:
    """Validate API key from headers"""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    return api_key

async def rate_limit_check(request: Request, api_key: str = Depends(validate_api_key_dependency)) -> None:
    """Check rate limits"""
    client_ip = request.client.host
    if not await rate_limiter.check_rate_limit(client_ip, api_key):
        raise RateLimitError(
            message="Rate limit exceeded",
            details={"client_ip": client_ip, "api_key": api_key[:8] + "..."},
            path=str(request.url.path),
            method=request.method
        )

# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    request_id = str(uuid.uuid4())
    
    error_response = create_error_response(
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details={"validation_errors": exc.errors()},
        request_id=request_id,
        path=str(request.url.path),
        method=request.method
    )
    
    # Log validation error
    error_tracker.create_error(
        category="validation",
        priority="medium",
        message=f"Validation error: {exc.errors()}",
        context={"request_id": request_id, "path": str(request.url.path)},
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump()
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    request_id = str(uuid.uuid4())
    
    error_response = create_error_response(
        error_code="HTTP_ERROR",
        message=exc.detail,
        status_code=exc.status_code,
        request_id=request_id,
        path=str(request.url.path),
        method=request.method
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    request_id = str(uuid.uuid4())
    
    error_response = create_error_response(
        error_code="INTERNAL_ERROR",
        message="Internal server error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        details={"error_type": type(exc).__name__},
        request_id=request_id,
        path=str(request.url.path),
        method=request.method
    )
    
    # Log unexpected error
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    error_tracker.create_error(
        category="unexpected",
        priority="high",
        message=f"Unexpected error: {str(exc)}",
        context={"request_id": request_id, "path": str(request.url.path)},
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump()
    )

# =============================================================================
# CORE API ENDPOINTS
# =============================================================================

@app.post("/api/v14/generate", response_model=CaptionGenerationResponse)
async def generate_caption(
    request: CaptionGenerationRequest,
    api_key: str = Depends(validate_api_key_dependency),
    request_id: str = Depends(get_request_id),
    rate_limit: None = Depends(rate_limit_check)
):
    """Generate Instagram caption with comprehensive validation"""
    start_time = time.time()
    
    try:
        # Additional validation using our validation engine
        is_valid, validation_errors = await validation_engine.validate_request(
            request.model_dump(), request_id
        )
        
        if not is_valid:
            raise ValidationError(
                message="Request validation failed",
                details={"validation_errors": validation_errors},
                request_id=request_id,
                path="/api/v14/generate",
                method="POST"
            )
        
        # Generate caption using optimized engine
        response = await optimized_engine.generate_caption(request, request_id)
        
        # Record performance metrics
        performance_monitor.record_request(
            response_time=response.processing_time,
            is_success=True,
            cache_hit=response.cache_hit
        )
        
        logger.info(f"Caption generated successfully: {request_id}")
        return response
        
    except (ValidationError, ContentValidationError, StyleValidationError, HashtagCountError) as e:
        # Validation errors - return 400
        response_time = time.time() - start_time
        performance_monitor.record_request(
            response_time=response_time,
            is_success=False,
            cache_hit=False
        )
        
        error_response = create_error_response(
            error_code=e.error_code,
            message=e.message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=e.details,
            request_id=request_id,
            path="/api/v14/generate",
            method="POST"
        )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response.model_dump()
        )
        
    except (AIGenerationError, CacheError) as e:
        # AI/Cache errors - return 503
        response_time = time.time() - start_time
        performance_monitor.record_request(
            response_time=response_time,
            is_success=False,
            cache_hit=False
        )
        
        error_response = create_error_response(
            error_code=e.error_code,
            message=e.message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=e.details,
            request_id=request_id,
            path="/api/v14/generate",
            method="POST"
        )
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_response.model_dump()
        )
        
    except Exception as e:
        # Unexpected errors - return 500
        response_time = time.time() - start_time
        performance_monitor.record_request(
            response_time=response_time,
            is_success=False,
            cache_hit=False
        )
        
        logger.error(f"Unexpected error in caption generation: {e}", exc_info=True)
        
        error_response = create_error_response(
            error_code="INTERNAL_ERROR",
            message="Internal server error during caption generation",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__},
            request_id=request_id,
            path="/api/v14/generate",
            method="POST"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


@app.post("/api/v14/batch", response_model=BatchCaptionResponse)
async def generate_batch_captions(
    request: BatchCaptionRequest,
    api_key: str = Depends(validate_api_key_dependency),
    request_id: str = Depends(get_request_id),
    rate_limit: None = Depends(rate_limit_check)
):
    """Generate multiple captions in batch with comprehensive validation"""
    start_time = time.time()
    
    try:
        # Process batch requests
        responses = []
        errors = []
        
        for i, caption_request in enumerate(request.requests):
            try:
                # Validate individual request
                is_valid, validation_errors = await validation_engine.validate_request(
                    caption_request.model_dump(), f"{request_id}-{i}"
                )
                
                if not is_valid:
                    errors.append({
                        "index": i,
                        "error": "Validation failed",
                        "details": validation_errors
                    })
                    continue
                
                # Generate caption
                response = await optimized_engine.generate_caption(
                    caption_request, f"{request_id}-{i}"
                )
                responses.append(response)
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "details": {"error_type": type(e).__name__}
                })
        
        # Create batch response
        processing_time = time.time() - start_time
        successful_requests = len(responses)
        failed_requests = len(errors)
        
        batch_response = BatchCaptionResponse(
            batch_id=request_id,
            total_requests=len(request.requests),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            processing_time=processing_time,
            responses=responses,
            errors=errors,
            average_processing_time=processing_time / len(request.requests) if request.requests else 0,
            cache_hit_rate=performance_monitor.get_cache_hit_rate()
        )
        
        # Record performance metrics
        performance_monitor.record_batch_request(
            total_requests=len(request.requests),
            successful_requests=successful_requests,
            processing_time=processing_time
        )
        
        logger.info(f"Batch caption generation completed: {request_id}")
        return batch_response
        
    except Exception as e:
        logger.error(f"Error in batch caption generation: {e}", exc_info=True)
        
        error_response = create_error_response(
            error_code="BATCH_ERROR",
            message="Batch processing failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__},
            request_id=request_id,
            path="/api/v14/batch",
            method="POST"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


@app.post("/api/v14/optimize", response_model=CaptionGenerationResponse)
async def optimize_caption(
    request: CaptionOptimizationRequest,
    api_key: str = Depends(validate_api_key_dependency),
    request_id: str = Depends(get_request_id),
    rate_limit: None = Depends(rate_limit_check)
):
    """Optimize existing caption with comprehensive validation"""
    start_time = time.time()
    
    try:
        # Validate optimization request
        is_valid, validation_errors = await validation_engine.validate_request(
            request.model_dump(), request_id
        )
        
        if not is_valid:
            raise ValidationError(
                message="Optimization request validation failed",
                details={"validation_errors": validation_errors},
                request_id=request_id,
                path="/api/v14/optimize",
                method="POST"
            )
        
        # Optimize caption using optimized engine
        response = await optimized_engine.optimize_caption(request, request_id)
        
        # Record performance metrics
        performance_monitor.record_request(
            response_time=response.processing_time,
            is_success=True,
            cache_hit=response.cache_hit
        )
        
        logger.info(f"Caption optimized successfully: {request_id}")
        return response
        
    except Exception as e:
        response_time = time.time() - start_time
        performance_monitor.record_request(
            response_time=response_time,
            is_success=False,
            cache_hit=False
        )
        
        logger.error(f"Error in caption optimization: {e}", exc_info=True)
        
        error_response = create_error_response(
            error_code="OPTIMIZATION_ERROR",
            message="Caption optimization failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__},
            request_id=request_id,
            path="/api/v14/optimize",
            method="POST"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


# =============================================================================
# MONITORING ENDPOINTS
# =============================================================================

@app.get("/api/v14/health", response_model=HealthCheckResponse)
async def health_check():
    """Comprehensive health check with performance metrics"""
    try:
        # Get performance metrics
        metrics = performance_monitor.get_metrics()
        
        # Check component health
        components = {
            "ai_engine": "healthy" if optimized_engine.is_ready() else "degraded",
            "cache": "healthy" if cache_manager.is_healthy() else "degraded",
            "rate_limiter": "healthy" if rate_limiter.is_healthy() else "degraded",
            "validation_engine": "healthy" if validation_engine.is_healthy() else "degraded"
        }
        
        # Determine overall status
        overall_status = "healthy"
        if any(status == "unhealthy" for status in components.values()):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in components.values()):
            overall_status = "degraded"
        
        health_response = HealthCheckResponse(
            status=overall_status,
            version="14.0.0",
            uptime=performance_monitor.get_uptime(),
            components=components,
            performance=metrics
        )
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        
        error_response = create_error_response(
            error_code="HEALTH_CHECK_ERROR",
            message="Health check failed",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"error_type": type(e).__name__}
        )
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_response.model_dump()
        )


@app.get("/api/v14/metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        metrics = performance_monitor.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        
        error_response = create_error_response(
            error_code="METRICS_ERROR",
            message="Failed to retrieve performance metrics",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


@app.get("/api/v14/info")
async def get_api_info():
    """Get API information and capabilities"""
    try:
        info = {
            "api_name": "Instagram Captions API v14.0",
            "version": "14.0.0",
            "description": "Ultra-optimized Instagram caption generation with comprehensive validation",
            "features": [
                "Comprehensive Pydantic validation",
                "Advanced error handling",
                "Performance monitoring",
                "Smart caching",
                "Rate limiting",
                "Batch processing",
                "Caption optimization"
            ],
            "endpoints": {
                "generate": "/api/v14/generate",
                "batch": "/api/v14/batch",
                "optimize": "/api/v14/optimize",
                "health": "/api/v14/health",
                "metrics": "/api/v14/metrics"
            },
            "performance_features": [
                "Async processing",
                "Multi-level caching",
                "Connection pooling",
                "Batch optimization",
                "Lazy loading"
            ]
        }
        
        return info
        
    except Exception as e:
        logger.error(f"API info failed: {e}")
        
        error_response = create_error_response(
            error_code="INFO_ERROR",
            message="Failed to retrieve API information",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error_type": type(e).__name__}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.model_dump()
        )


# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.post("/api/v14/validate")
async def validate_request_endpoint(request_data: Dict[str, Any]):
    """Validate request data against schemas"""
    try:
        # Determine schema type based on request data
        if "requests" in request_data:
            validated_data = validate_request_data(request_data, BatchCaptionRequest)
        elif "caption" in request_data:
            validated_data = validate_request_data(request_data, CaptionOptimizationRequest)
        else:
            validated_data = validate_request_data(request_data, CaptionGenerationRequest)
        
        return {
            "valid": True,
            "data": validated_data.model_dump(),
            "schema_type": type(validated_data).__name__
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.post("/api/v14/sanitize")
async def sanitize_input_endpoint(request_data: Dict[str, Any]):
    """Sanitize input data"""
    try:
        sanitized_data = sanitize_input_data(request_data)
        return {
            "sanitized": True,
            "data": sanitized_data
        }
        
    except Exception as e:
        error_response = create_error_response(
            error_code="SANITIZATION_ERROR",
            message="Failed to sanitize input data",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"error_type": type(e).__name__}
        )
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response.model_dump()
        )


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Instagram Captions API v14.0",
        "version": "14.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v14/health"
    }


# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    
    uvicorn.run(
        "fast_api_v14:app",
        host="0.0.0.0",
        port=8140,
        reload=True,
        log_level="info"
    ) 