"""
Improved Video Processing API

FastAPI-based API following best practices:
- Early returns and guard clauses for error handling
- Lifespan context managers instead of startup/shutdown events
- Modular route organization with APIRouter
- Enhanced type hints and Pydantic models
- Performance optimizations with caching and async operations
- Comprehensive input validation and sanitization
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any, Annotated
from fastapi import FastAPI, HTTPException, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
import time
import asyncio
import uuid
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

# Import improved modules
from .models import (
    VideoClipRequest,
    VideoClipResponse,
    VideoClipBatchRequest,
    VideoClipBatchResponse,
    ViralVideoRequest,
    ViralVideoResponse,
    LangChainRequest,
    LangChainResponse,
    HealthResponse,
    ErrorResponse
)
from .dependencies import (
    get_video_processor,
    get_viral_processor,
    get_langchain_processor,
    get_batch_processor,
    get_current_user,
    get_request_id
)
from .validation import (
    validate_video_request,
    validate_batch_request,
    validate_viral_request,
    validate_langchain_request,
    sanitize_youtube_url,
    validate_system_health
)
from .error_handling import (
    VideoProcessingError,
    ValidationError,
    SecurityError,
    ResourceError,
    create_error_response,
    handle_processing_errors
)
from .cache import CacheManager
from .monitoring import PerformanceMonitor, HealthChecker

# Configure structured logging
logger = structlog.get_logger("video_api")

# Security
security = HTTPBearer(auto_error=False)

# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Video Processing API", version="3.1.0")
    
    try:
        # Initialize cache manager
        app.state.cache = CacheManager()
        await app.state.cache.initialize()
        
        # Initialize performance monitor
        app.state.monitor = PerformanceMonitor()
        await app.state.monitor.start()
        
        # Initialize health checker
        app.state.health_checker = HealthChecker()
        await app.state.health_checker.initialize()
        
        # Validate system health
        health_status = await app.state.health_checker.check_system_health()
        if not health_status.is_healthy:
            logger.warning("System health issues detected", issues=health_status.issues)
        
        logger.info("Video Processing API startup completed successfully")
        yield
        
    except Exception as e:
        logger.error("Failed to start Video Processing API", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Video Processing API")
        
        if hasattr(app.state, 'cache'):
            await app.state.cache.close()
        
        if hasattr(app.state, 'monitor'):
            await app.state.monitor.stop()
        
        if hasattr(app.state, 'health_checker'):
            await app.state.health_checker.close()
        
        logger.info("Video Processing API shutdown completed")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Video Processing API",
    description="Advanced video processing with LangChain integration for intelligent content optimization",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ROUTERS
# =============================================================================

# Create routers for modular organization
video_router = APIRouter(prefix="/api/v1/video", tags=["video"])
viral_router = APIRouter(prefix="/api/v1/viral", tags=["viral"])
langchain_router = APIRouter(prefix="/api/v1/langchain", tags=["langchain"])
config_router = APIRouter(prefix="/api/v1/config", tags=["config"])
utils_router = APIRouter(prefix="/api/v1/utils", tags=["utils"])

# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Add request context and performance monitoring."""
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Set request context in logger
    logger.set_request_id(request_id)
    
    # Start performance monitoring
    start_time = time.perf_counter()
    
    # Log request start
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        duration = time.perf_counter() - start_time
        
        # Log request completion
        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration=duration
        )
        
        # Add performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = str(duration)
        
        # Update performance metrics
        if hasattr(app.state, 'monitor'):
            await app.state.monitor.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration
            )
        
        return response
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(
            "Request failed",
            error=str(e),
            duration=duration
        )
        raise

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check(
    health_checker: Annotated[HealthChecker, Depends(lambda: app.state.health_checker)]
):
    """Enhanced health check endpoint with system monitoring."""
    try:
        # Check system health
        health_status = await health_checker.check_system_health()
        
        return HealthResponse(
            status=health_status.status,
            version="3.1.0",
            features=[
                "video_processing",
                "viral_optimization", 
                "langchain_integration",
                "batch_processing",
                "performance_monitoring",
                "caching",
                "async_processing"
            ],
            system_health=health_status.system_metrics,
            gpu_health=health_status.gpu_metrics,
            warnings=health_status.issues,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            version="3.1.0",
            error="Health check failed",
            timestamp=time.time()
        )

# =============================================================================
# VIDEO PROCESSING ROUTES
# =============================================================================

@video_router.post("/process", response_model=VideoClipResponse)
@handle_processing_errors
async def process_video(
    request: VideoClipRequest,
    processor: Annotated[Any, Depends(get_video_processor)],
    req: Annotated[Request, Depends(get_request_id)],
    cache: Annotated[CacheManager, Depends(lambda: app.state.cache)]
):
    """Process a single video clip with enhanced error handling and caching."""
    # Guard clauses - early returns for error conditions
    if not request:
        raise ValidationError("Request object is required")
    
    if not request.youtube_url or not request.youtube_url.strip():
        raise ValidationError("YouTube URL is required and cannot be empty")
    
    if not request.language or not request.language.strip():
        raise ValidationError("Language is required and cannot be empty")
    
    # Security validation - early return
    if not sanitize_youtube_url(request.youtube_url):
        raise SecurityError("Invalid or potentially malicious YouTube URL")
    
    # System health validation - early return
    health_status = await app.state.health_checker.check_system_health()
    if not health_status.is_healthy:
        raise ResourceError("System is not healthy for processing")
    
    # Check cache first
    cache_key = f"video:{request.youtube_url}:{request.language}:{request.max_clip_length}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        logger.info("Returning cached result", cache_key=cache_key)
        return VideoClipResponse(**cached_result)
    
    # Validate request data - early return
    validation_result = validate_video_request(request)
    if not validation_result.is_valid:
        raise ValidationError(f"Request validation failed: {validation_result.errors}")
    
    # Happy path: Process video
    start_time = time.perf_counter()
    
    try:
        response = await processor.process_video_async(request)
        processing_time = time.perf_counter() - start_time
        
        # Cache successful result
        await cache.set(cache_key, response.dict(), ttl=3600)  # 1 hour TTL
        
        logger.info(
            "Video processing completed",
            youtube_url=request.youtube_url,
            processing_time=processing_time,
            success=response.success
        )
        
        return response
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error(
            "Video processing failed", 
            error=str(e),
            processing_time=processing_time
        )
        raise VideoProcessingError(f"Video processing failed: {str(e)}")

@video_router.post("/batch", response_model=VideoClipBatchResponse)
@handle_processing_errors
async def process_video_batch(
    request: VideoClipBatchRequest,
    processor: Annotated[Any, Depends(get_batch_processor)],
    req: Annotated[Request, Depends(get_request_id)],
    cache: Annotated[CacheManager, Depends(lambda: app.state.cache)]
):
    """Process multiple video clips in batch with early error handling."""
    # Guard clauses - early returns
    if not request:
        raise ValidationError("Batch request object is required")
    
    if not request.requests or not isinstance(request.requests, list):
        raise ValidationError("Requests list is required and must be a list")
    
    if len(request.requests) == 0:
        raise ValidationError("Batch cannot be empty")
    
    if len(request.requests) > 100:
        raise ValidationError("Batch size exceeds maximum limit of 100")
    
    # System health validation - early return
    health_status = await app.state.health_checker.check_system_health()
    if not health_status.is_healthy:
        raise ResourceError("System is not healthy for batch processing")
    
    # Validate each request in batch - early return
    for i, req_item in enumerate(request.requests):
        if not req_item:
            raise ValidationError(f"Request at index {i} is null")
        
        if not req_item.youtube_url or not req_item.youtube_url.strip():
            raise ValidationError(f"YouTube URL is required for request at index {i}")
        
        if not sanitize_youtube_url(req_item.youtube_url):
            raise SecurityError(f"Invalid YouTube URL at index {i}")
    
    # Validate batch request data - early return
    validation_result = validate_batch_request(request)
    if not validation_result.is_valid:
        raise ValidationError(f"Batch validation failed: {validation_result.errors}")
    
    # Happy path: Process batch
    start_time = time.perf_counter()
    
    try:
        response = await processor.process_batch_async(request.requests)
        processing_time = time.perf_counter() - start_time
        
        successful_count = len([r for r in response.responses if r.success])
        
        logger.info(
            "Batch video processing completed",
            batch_size=len(request.requests),
            processing_time=processing_time,
            successful_count=successful_count
        )
        
        return VideoClipBatchResponse(
            responses=response.responses,
            processing_time=processing_time,
            total_requests=len(request.requests),
            successful_requests=successful_count
        )
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error(
            "Batch video processing failed", 
            error=str(e),
            processing_time=processing_time
        )
        raise VideoProcessingError(f"Batch video processing failed: {str(e)}")

# =============================================================================
# VIRAL PROCESSING ROUTES
# =============================================================================

@viral_router.post("/process", response_model=ViralVideoResponse)
@handle_processing_errors
async def process_viral_variants(
    request: ViralVideoRequest,
    processor: Annotated[Any, Depends(get_viral_processor)],
    req: Annotated[Request, Depends(get_request_id)],
    cache: Annotated[CacheManager, Depends(lambda: app.state.cache)]
):
    """Generate viral video variants with early error handling."""
    # Guard clauses - early returns
    if not request:
        raise ValidationError("Request object is required")
    
    if not request.youtube_url or not request.youtube_url.strip():
        raise ValidationError("YouTube URL is required and cannot be empty")
    
    if request.n_variants <= 0 or request.n_variants > 50:
        raise ValidationError("n_variants must be between 1 and 50")
    
    # Security validation - early return
    if not sanitize_youtube_url(request.youtube_url):
        raise SecurityError("Invalid or potentially malicious YouTube URL")
    
    # System health validation - early return
    health_status = await app.state.health_checker.check_system_health()
    if not health_status.is_healthy:
        raise ResourceError("System is not healthy for viral processing")
    
    # Validate request data - early return
    validation_result = validate_viral_request(request)
    if not validation_result.is_valid:
        raise ValidationError(f"Viral request validation failed: {validation_result.errors}")
    
    # Check cache first
    cache_key = f"viral:{request.youtube_url}:{request.n_variants}:{request.use_langchain}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        logger.info("Returning cached viral result", cache_key=cache_key)
        return ViralVideoResponse(**cached_result)
    
    # Happy path: Process viral variants
    start_time = time.perf_counter()
    
    try:
        response = await processor.process_viral_variants_async(request)
        processing_time = time.perf_counter() - start_time
        
        # Cache successful result
        await cache.set(cache_key, response.dict(), ttl=1800)  # 30 minutes TTL
        
        logger.info(
            "Viral processing completed",
            youtube_url=request.youtube_url,
            variants_generated=response.successful_variants,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error("Viral processing failed", error=str(e), processing_time=processing_time)
        raise VideoProcessingError(f"Viral processing failed: {str(e)}")

# =============================================================================
# LANGCHAIN ROUTES
# =============================================================================

@langchain_router.post("/analyze", response_model=LangChainResponse)
@handle_processing_errors
async def analyze_content_with_langchain(
    request: LangChainRequest,
    processor: Annotated[Any, Depends(get_langchain_processor)],
    req: Annotated[Request, Depends(get_request_id)],
    cache: Annotated[CacheManager, Depends(lambda: app.state.cache)]
):
    """Analyze video content using LangChain with early error handling."""
    # Guard clauses - early returns
    if not request:
        raise ValidationError("Request object is required")
    
    if not request.youtube_url or not request.youtube_url.strip():
        raise ValidationError("YouTube URL is required and cannot be empty")
    
    # Security validation - early return
    if not sanitize_youtube_url(request.youtube_url):
        raise SecurityError("Invalid or potentially malicious YouTube URL")
    
    # System health validation - early return
    health_status = await app.state.health_checker.check_system_health()
    if not health_status.is_healthy:
        raise ResourceError("System is not healthy for LangChain analysis")
    
    # Validate request data - early return
    validation_result = validate_langchain_request(request)
    if not validation_result.is_valid:
        raise ValidationError(f"LangChain request validation failed: {validation_result.errors}")
    
    # Check cache first
    cache_key = f"langchain:{request.youtube_url}:{request.analysis_type}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        logger.info("Returning cached LangChain result", cache_key=cache_key)
        return LangChainResponse(**cached_result)
    
    # Happy path: Analyze with LangChain
    start_time = time.perf_counter()
    
    try:
        response = await processor.analyze_content_async(request)
        analysis_time = time.perf_counter() - start_time
        
        # Cache successful result
        await cache.set(cache_key, response.dict(), ttl=7200)  # 2 hours TTL
        
        logger.info(
            "LangChain analysis completed",
            youtube_url=request.youtube_url,
            analysis_type=request.analysis_type,
            analysis_time=analysis_time
        )
        
        return response
        
    except Exception as e:
        analysis_time = time.perf_counter() - start_time
        logger.error("LangChain analysis failed", error=str(e), analysis_time=analysis_time)
        raise VideoProcessingError(f"LangChain analysis failed: {str(e)}")

# =============================================================================
# CONFIGURATION ROUTES
# =============================================================================

@config_router.get("/langchain")
async def get_langchain_config():
    """Get current LangChain configuration."""
    return {
        "model_name": "gpt-4",
        "enable_content_analysis": True,
        "enable_engagement_analysis": True,
        "enable_viral_analysis": True,
        "enable_title_optimization": True,
        "enable_caption_optimization": True,
        "enable_timing_optimization": True,
        "batch_size": 5,
        "max_retries": 3,
        "use_agents": True,
        "use_memory": True
    }

@config_router.get("/viral")
async def get_viral_config():
    """Get current viral processing configuration."""
    return {
        "max_variants": 10,
        "min_viral_score": 0.3,
        "enable_langchain": True,
        "enable_screen_division": True,
        "enable_transitions": True,
        "enable_effects": True,
        "enable_animations": True
    }

@config_router.get("/video")
async def get_video_config():
    """Get current video processing configuration."""
    return {
        "max_workers": 4,
        "batch_size": 5,
        "enable_audit_logging": True,
        "enable_performance_tracking": True
    }

# =============================================================================
# UTILITY ROUTES
# =============================================================================

@utils_router.post("/validate")
async def validate_video_request_endpoint(request: VideoClipRequest):
    """Validate a video processing request."""
    try:
        validation_result = validate_video_request(request)
        
        return {
            "valid": validation_result.is_valid,
            "message": "Request is valid" if validation_result.is_valid else "Request validation failed",
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "request": request
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": str(e),
            "request": request
        }

@utils_router.get("/content-types")
async def get_content_types():
    """Get available content types for LangChain analysis."""
    return {
        "content_types": ["educational", "entertainment", "news", "tutorial", "review", "vlog"],
        "engagement_types": ["high", "medium", "low"],
        "platforms": ["youtube", "tiktok", "instagram", "twitter", "linkedin"]
    }

@utils_router.post("/estimate-processing-time")
async def estimate_processing_time(
    n_variants: int = 5,
    use_langchain: bool = True,
    batch_size: int = 1
):
    """Estimate processing time for video generation."""
    try:
        # Base processing time
        base_time = 2.0  # seconds per variant
        
        # LangChain overhead
        langchain_overhead = 5.0 if use_langchain else 0.0
        
        # Batch processing efficiency
        batch_efficiency = 0.8 if batch_size > 1 else 1.0
        
        # Calculate estimated time
        estimated_time = (base_time * n_variants + langchain_overhead) * batch_efficiency
        
        return {
            "estimated_time_seconds": estimated_time,
            "estimated_time_minutes": estimated_time / 60,
            "n_variants": n_variants,
            "use_langchain": use_langchain,
            "batch_size": batch_size,
            "factors": {
                "base_processing": base_time * n_variants,
                "langchain_overhead": langchain_overhead,
                "batch_efficiency": batch_efficiency
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Validation exception handler with request ID tracking."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        "Validation error",
        error=str(exc),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=400,
        content=create_error_response(
            error_code="VALIDATION_ERROR",
            message=str(exc),
            request_id=request_id
        )
    )

@app.exception_handler(SecurityError)
async def security_exception_handler(request: Request, exc: SecurityError):
    """Security exception handler with threat response."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        "Security error",
        error=str(exc),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=403,
        content=create_error_response(
            error_code="SECURITY_ERROR",
            message="Security violation detected",
            request_id=request_id
        )
    )

@app.exception_handler(ResourceError)
async def resource_exception_handler(request: Request, exc: ResourceError):
    """Resource exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "Resource error",
        error=str(exc),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=507,
        content=create_error_response(
            error_code="RESOURCE_ERROR",
            message=str(exc),
            request_id=request_id
        )
    )

@app.exception_handler(VideoProcessingError)
async def video_processing_exception_handler(request: Request, exc: VideoProcessingError):
    """Video processing exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "Video processing error",
        error=str(exc),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=422,
        content=create_error_response(
            error_code="PROCESSING_ERROR",
            message=str(exc),
            request_id=request_id
        )
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with request ID tracking."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "Unhandled exception",
        error=str(exc),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            error_code="INTERNAL_ERROR",
            message="An internal error occurred",
            request_id=request_id
        )
    )

# =============================================================================
# ROUTER REGISTRATION
# =============================================================================

# Include routers
app.include_router(video_router)
app.include_router(viral_router)
app.include_router(langchain_router)
app.include_router(config_router)
app.include_router(utils_router)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "improved_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
