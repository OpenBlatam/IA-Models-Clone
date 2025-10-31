"""
Video Processing API

FastAPI-based API for video processing with LangChain integration.
Enhanced with intelligent content analysis and optimization for short-form videos.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
import asyncio
import uuid
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from .error_handling import (
    ErrorHandler, 
    ErrorCode, 
    ValidationError, 
    ProcessingError, 
    ExternalServiceError,
    ResourceError,
    CriticalSystemError,
    SecurityError,
    ConfigurationError,
    validate_request,
    handle_processing_errors
)
from .validation import (
    validate_video_request_data,
    validate_batch_request_data,
    validate_and_sanitize_url,
    validate_system_health,
    validate_gpu_health,
    check_system_resources,
    check_gpu_availability
)

from .models.video_models import (
    VideoClipRequest,
    VideoClipResponse,
    VideoClipBatchRequest,
    VideoClipBatchResponse
)
from .models.viral_models import (
    ViralVideoVariant,
    ViralVideoBatchResponse,
    ViralCaptionConfig,
    LangChainAnalysis,
    ContentOptimization,
    ShortVideoOptimization,
    ContentType,
    EngagementType,
    create_default_caption_config
)
from .processors.video_processor import VideoProcessor, VideoProcessorConfig
from .processors.viral_processor import ViralVideoProcessor, ViralProcessorConfig
from .processors.langchain_processor import (
    LangChainVideoProcessor,
    LangChainConfig,
    create_langchain_processor,
    create_optimized_langchain_processor
)
from .processors.batch_processor import BatchVideoProcessor, BatchProcessorConfig
from .utils.parallel_utils import HybridParallelProcessor, ParallelConfig

# Import enhanced logging
try:
    from .logging_config import EnhancedLogger, ErrorMessages, log_error_with_context
    logger = EnhancedLogger("api")
except ImportError:
    logger = structlog.get_logger()

# Import error factories
try:
    from .error_factories import (
        error_factory, context_manager,
        create_validation_error, create_processing_error, create_encoding_error,
        create_inference_error, create_resource_error, create_api_error,
        create_security_error, create_error_context
    )
except ImportError:
    error_factory = None
    context_manager = None

error_handler = ErrorHandler()

# =============================================================================
# REQUEST MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests for tracking with enhanced logging and error context."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Set request ID in logger
    if hasattr(logger, 'set_request_id'):
        logger.set_request_id(request_id)
    
    # Set request context for error tracking
    if context_manager:
        context_manager.set_request_context(request_id)
        error_handler.set_request_context(request_id)
    
    # Log request start
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    start_time = time.perf_counter()
    
    try:
        response = await call_next(request)
        
        # Log request completion
        duration = time.perf_counter() - start_time
        logger.info(
            "Request completed",
            status_code=response.status_code,
            duration=duration
        )
        
        response.headers["X-Request-ID"] = request_id
        return response
        
    except Exception as e:
        # Log request error
        duration = time.perf_counter() - start_time
        logger.error(
            "Request failed",
            error=e,
            duration=duration
        )
        raise

# =============================================================================
# API CONFIGURATION
# =============================================================================

app = FastAPI(
    title="Video Processing API",
    description="Advanced video processing with LangChain integration for intelligent content optimization",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
# DEPENDENCY INJECTION
# =============================================================================

def get_video_processor() -> VideoProcessor:
    """Get video processor instance."""
    config = VideoProcessorConfig(
        max_workers=4,
        batch_size=5,
        enable_audit_logging=True
    )
    return VideoProcessor(config)

def get_viral_processor() -> ViralVideoProcessor:
    """Get viral processor instance."""
    config = ViralProcessorConfig(
        max_variants=10,
        enable_langchain=True,
        enable_screen_division=True,
        enable_transitions=True,
        enable_effects=True
    )
    return ViralVideoProcessor(config)

def get_langchain_processor() -> LangChainVideoProcessor:
    """Get LangChain processor instance."""
    config = LangChainConfig(
        model_name="gpt-4",
        enable_content_analysis=True,
        enable_engagement_analysis=True,
        enable_viral_analysis=True,
        enable_title_optimization=True,
        enable_caption_optimization=True,
        enable_timing_optimization=True,
        batch_size=5,
        max_retries=3,
        use_agents=True,
        use_memory=True
    )
    return LangChainVideoProcessor(config)

def get_batch_processor() -> BatchVideoProcessor:
    """Get batch processor instance."""
    config = BatchProcessorConfig(
        max_workers=8,
        batch_size=10,
        enable_parallel_processing=True
    )
    return BatchVideoProcessor(config)

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with system monitoring."""
    try:
        # Check system resources
        system_health = check_system_resources()
        gpu_health = check_gpu_availability()
        
        # Determine overall health status
        health_status = "healthy"
        warnings = []
        
        # Check for critical conditions
        if system_health.get("memory_critical"):
            health_status = "degraded"
            warnings.append("Critical memory usage")
        
        if system_health.get("disk_critical"):
            health_status = "critical"
            warnings.append("Critical disk space")
        
        if system_health.get("cpu_critical"):
            health_status = "degraded"
            warnings.append("High CPU usage")
        
        if gpu_health.get("memory_critical"):
            health_status = "degraded"
            warnings.append("Critical GPU memory usage")
        
        if not gpu_health["available"]:
            health_status = "degraded"
            warnings.append("GPU not available")
        
        return {
            "status": health_status,
            "version": "3.0.0",
            "features": [
                "video_processing",
                "viral_optimization", 
                "langchain_integration",
                "batch_processing",
                "parallel_processing",
                "system_monitoring"
            ],
            "system_health": system_health,
            "gpu_health": gpu_health,
            "warnings": warnings,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "version": "3.0.0",
            "error": "Health check failed",
            "timestamp": time.time()
        }

# =============================================================================
# VIDEO PROCESSING ENDPOINTS
# =============================================================================

@app.post("/api/v1/video/process", response_model=VideoClipResponse)
@handle_processing_errors
async def process_video(
    request: VideoClipRequest,
    processor: VideoProcessor = Depends(get_video_processor),
    req: Request = None
):
    """Process a single video clip with enhanced error handling and system monitoring."""
    # ERROR HANDLING: Extract request ID first
    request_id = getattr(req.state, 'request_id', None) if req else None
    
    # ERROR HANDLING: Validate request object - early return
    if not request:
        raise ValidationError("Request object is required", "request", None, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Check for None/empty YouTube URL - early return
    if not request.youtube_url or not request.youtube_url.strip():
        raise ValidationError("YouTube URL is required and cannot be empty", "youtube_url", request.youtube_url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Check for None/empty language - early return
    if not request.language or not request.language.strip():
        raise ValidationError("Language is required and cannot be empty", "language", request.language, ErrorCode.INVALID_LANGUAGE_CODE)
    
    # ERROR HANDLING: System health validation (critical) - early return
    try:
        validate_system_health()
    except CriticalSystemError as e:
        logger.critical("System health validation failed during video processing", 
                       error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: GPU health validation - early return
    try:
        validate_gpu_health()
    except ResourceError as e:
        logger.warning("GPU health validation failed, falling back to CPU", 
                      error=str(e), request_id=request_id)
        # Continue with CPU processing instead of failing
    
    # ERROR HANDLING: Security validation (high priority) - early return
    malicious_patterns = ["javascript:", "data:", "vbscript:", "file://", "ftp://", "eval(", "exec(", "system("]
    if any(pattern in request.youtube_url.lower() for pattern in malicious_patterns):
        logger.warning("Malicious input detected in YouTube URL", 
                      url=request.youtube_url, request_id=request_id)
        raise SecurityError(
            "Malicious input detected in YouTube URL",
            "malicious_input",
            {"url": request.youtube_url}
        )
    
    # ERROR HANDLING: Request validation (medium priority) - early return
    try:
        validate_video_request_data(
            youtube_url=request.youtube_url,
            language=request.language,
            max_clip_length=getattr(request, 'max_clip_length', None),
            min_clip_length=getattr(request, 'min_clip_length', None),
            audience_profile=getattr(request, 'audience_profile', None)
        )
    except ValidationError as e:
        logger.warning("Request validation failed", 
                      error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: URL sanitization - early return
    try:
        request.youtube_url = validate_and_sanitize_url(request.youtube_url)
    except ValidationError as e:
        logger.warning("URL sanitization failed", 
                      error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: Processor validation - early return
    if not processor:
        raise ConfigurationError("Video processor is not available", "video_processor", ErrorCode.MISSING_CONFIG)
    
    # Start processing timer after all validations
    start_time = time.perf_counter()
    
    # HAPPY PATH: Process video and return response
    try:
        # Processing with monitoring
        response = processor.process_video(request)
        
        processing_time = time.perf_counter() - start_time
        
        logger.info(
            "Video processing completed",
            youtube_url=request.youtube_url,
            processing_time=processing_time,
            success=response.success,
            request_id=getattr(req.state, 'request_id', None) if req else None
        )
        
        return response
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error(
            "Video processing failed", 
            error=str(e),
            processing_time=processing_time,
            request_id=getattr(req.state, 'request_id', None) if req else None
        )
        raise ProcessingError(f"Video processing failed: {str(e)}", "process_video")

@app.post("/api/v1/video/batch", response_model=VideoClipBatchResponse)
@handle_processing_errors
async def process_video_batch(
    request: VideoClipBatchRequest,
    processor: BatchVideoProcessor = Depends(get_batch_processor),
    req: Request = None
):
    """Process multiple video clips in batch with early error handling."""
    # ERROR HANDLING: Extract request ID first
    request_id = getattr(req.state, 'request_id', None) if req else None
    
    # ERROR HANDLING: Validate request object - early return
    if not request:
        raise ValidationError("Batch request object is required", "request", None, ErrorCode.INVALID_BATCH_SIZE)
    
    # ERROR HANDLING: Validate requests list - early return
    if not request.requests or not isinstance(request.requests, list):
        raise ValidationError("Requests list is required and must be a list", "requests", request.requests, ErrorCode.INVALID_BATCH_SIZE)
    
    # ERROR HANDLING: Check for empty batch - early return
    if len(request.requests) == 0:
        raise ValidationError("Batch cannot be empty", "requests", [], ErrorCode.INVALID_BATCH_SIZE)
    
    # ERROR HANDLING: Check batch size limits - early return
    if len(request.requests) > 100:  # Reasonable batch size limit
        raise ValidationError("Batch size exceeds maximum limit of 100", "batch_size", len(request.requests), ErrorCode.INVALID_BATCH_SIZE)
    
    # ERROR HANDLING: System health validation - early return
    try:
        validate_system_health()
    except CriticalSystemError as e:
        logger.critical("System health validation failed during batch processing", 
                       error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: Validate each request in batch - early return
    for i, req_item in enumerate(request.requests):
        if not req_item:
            raise ValidationError(f"Request at index {i} is null", f"requests[{i}]", None, ErrorCode.INVALID_YOUTUBE_URL)
        
        if not req_item.youtube_url or not req_item.youtube_url.strip():
            raise ValidationError(f"YouTube URL is required for request at index {i}", f"requests[{i}].youtube_url", req_item.youtube_url, ErrorCode.INVALID_YOUTUBE_URL)
        
        if not req_item.language or not req_item.language.strip():
            raise ValidationError(f"Language is required for request at index {i}", f"requests[{i}].language", req_item.language, ErrorCode.INVALID_LANGUAGE_CODE)
        
        # ERROR HANDLING: Security check for each URL - early return
        malicious_patterns = ["javascript:", "data:", "vbscript:", "file://", "ftp://", "eval(", "exec(", "system("]
        if any(pattern in req_item.youtube_url.lower() for pattern in malicious_patterns):
            logger.warning("Malicious input detected in batch request", 
                          url=req_item.youtube_url, index=i, request_id=request_id)
            raise SecurityError(
                f"Malicious input detected in YouTube URL at index {i}",
                "malicious_input",
                {"url": req_item.youtube_url, "index": i}
            )
    
    # ERROR HANDLING: Processor validation - early return
    if not processor:
        raise ConfigurationError("Batch processor is not available", "batch_processor", ErrorCode.MISSING_CONFIG)
    
    # ERROR HANDLING: Validate batch request data - early return
    try:
        validate_batch_request_data(
            requests=request.requests,
            batch_size=len(request.requests)
        )
    except ValidationError as e:
        logger.warning("Batch request validation failed", 
                      error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: Sanitize URLs - early return
    try:
        for req_item in request.requests:
            req_item.youtube_url = validate_and_sanitize_url(req_item.youtube_url)
    except ValidationError as e:
        logger.warning("URL sanitization failed in batch", 
                      error=str(e), request_id=request_id)
        raise
    
    # Start processing timer after all validations
    start_time = time.perf_counter()
    
    # HAPPY PATH: Process batch and return response
    try:
        response = processor.process_batch(request.requests)
        
        processing_time = time.perf_counter() - start_time
        
        successful_count = len([r for r in response if r.success])
        
        logger.info(
            "Batch video processing completed",
            batch_size=len(request.requests),
            processing_time=processing_time,
            successful_count=successful_count,
            request_id=getattr(req.state, 'request_id', None) if req else None
        )
        
        return VideoClipBatchResponse(
            responses=response,
            processing_time=processing_time,
            total_requests=len(request.requests),
            successful_requests=successful_count
        )
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error(
            "Batch video processing failed", 
            error=str(e),
            processing_time=processing_time,
            request_id=getattr(req.state, 'request_id', None) if req else None
        )
        raise ProcessingError(f"Batch video processing failed: {str(e)}", "process_batch")

# =============================================================================
# VIRAL PROCESSING ENDPOINTS
# =============================================================================

@app.post("/api/v1/viral/process", response_model=ViralVideoBatchResponse)
async def process_viral_variants(
    request: VideoClipRequest,
    n_variants: int = 5,
    audience_profile: Optional[Dict[str, Any]] = None,
    use_langchain: bool = True,
    processor: ViralVideoProcessor = Depends(get_viral_processor),
    req: Request = None
):
    """Generate viral video variants with optional LangChain optimization and early error handling."""
    # ERROR HANDLING: Extract request ID first
    request_id = getattr(req.state, 'request_id', None) if req else None
    
    # ERROR HANDLING: Validate request object - early return
    if not request:
        raise ValidationError("Request object is required", "request", None, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Validate n_variants parameter - early return
    if not isinstance(n_variants, int):
        raise ValidationError("n_variants must be an integer", "n_variants", n_variants, ErrorCode.INVALID_BATCH_SIZE)
    
    if n_variants <= 0:
        raise ValidationError("n_variants must be positive", "n_variants", n_variants, ErrorCode.INVALID_BATCH_SIZE)
    
    if n_variants > 50:  # Reasonable limit for viral variants
        raise ValidationError("n_variants cannot exceed 50", "n_variants", n_variants, ErrorCode.INVALID_BATCH_SIZE)
    
    # ERROR HANDLING: Validate use_langchain parameter - early return
    if not isinstance(use_langchain, bool):
        raise ValidationError("use_langchain must be a boolean", "use_langchain", use_langchain, ErrorCode.INVALID_CONFIG)
    
    # ERROR HANDLING: Validate audience_profile if provided - early return
    if audience_profile is not None and not isinstance(audience_profile, dict):
        raise ValidationError("audience_profile must be a dictionary", "audience_profile", audience_profile, ErrorCode.INVALID_AUDIENCE_PROFILE)
    
    # ERROR HANDLING: Check for None/empty YouTube URL - early return
    if not request.youtube_url or not request.youtube_url.strip():
        raise ValidationError("YouTube URL is required and cannot be empty", "youtube_url", request.youtube_url, ErrorCode.INVALID_YOUTUBE_URL)
    
    # ERROR HANDLING: Check for None/empty language - early return
    if not request.language or not request.language.strip():
        raise ValidationError("Language is required and cannot be empty", "language", request.language, ErrorCode.INVALID_LANGUAGE_CODE)
    
    # ERROR HANDLING: System health validation - early return
    try:
        validate_system_health()
    except CriticalSystemError as e:
        logger.critical("System health validation failed during viral processing", 
                       error=str(e), request_id=request_id)
        raise
    
    # ERROR HANDLING: Security validation - early return
    malicious_patterns = ["javascript:", "data:", "vbscript:", "file://", "ftp://", "eval(", "exec(", "system("]
    if any(pattern in request.youtube_url.lower() for pattern in malicious_patterns):
        logger.warning("Malicious input detected in YouTube URL", 
                      url=request.youtube_url, request_id=request_id)
        raise SecurityError(
            "Malicious input detected in YouTube URL",
            "malicious_input",
            {"url": request.youtube_url}
        )
    
    # ERROR HANDLING: Processor validation - early return
    if not processor:
        raise ConfigurationError("Viral processor is not available", "viral_processor", ErrorCode.MISSING_CONFIG)
    
    # ERROR HANDLING: URL sanitization - early return
    try:
        request.youtube_url = validate_and_sanitize_url(request.youtube_url)
    except ValidationError as e:
        logger.warning("URL sanitization failed", 
                      error=str(e), request_id=request_id)
        raise
    
    # Start processing timer after all validations
    start_time = time.perf_counter()
    
    # HAPPY PATH: Process viral variants and return response
    try:
        response = processor.process_viral_variants(
            request=request,
            n_variants=n_variants,
            audience_profile=audience_profile,
            use_langchain=use_langchain
        )
        
        processing_time = time.perf_counter() - start_time
        
        logger.info(
            "Viral processing completed",
            youtube_url=request.youtube_url,
            variants_generated=response.successful_variants,
            average_viral_score=response.average_viral_score,
            processing_time=processing_time,
            langchain_used=use_langchain
        )
        
        return response
        
    except Exception as e:
        logger.error("Viral processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/viral/batch", response_model=List[ViralVideoBatchResponse])
async def process_viral_batch(
    requests: List[VideoClipRequest],
    n_variants_per_request: int = 5,
    audience_profiles: Optional[List[Dict[str, Any]]] = None,
    use_langchain: bool = True,
    processor: ViralVideoProcessor = Depends(get_viral_processor)
):
    """Process multiple videos for viral variants in batch."""
    try:
        start_time = time.perf_counter()
        
        responses = processor.process_batch(
            requests=requests,
            n_variants_per_request=n_variants_per_request,
            audience_profiles=audience_profiles
        )
        
        processing_time = time.perf_counter() - start_time
        
        logger.info(
            "Viral batch processing completed",
            total_requests=len(requests),
            variants_per_request=n_variants_per_request,
            processing_time=processing_time,
            successful_requests=len([r for r in responses if r.success]),
            langchain_used=use_langchain
        )
        
        return responses
        
    except Exception as e:
        logger.error("Viral batch processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LANGCHAIN ENDPOINTS
# =============================================================================

@app.post("/api/v1/langchain/analyze", response_model=LangChainAnalysis)
async def analyze_content_with_langchain(
    request: VideoClipRequest,
    audience_profile: Optional[Dict[str, Any]] = None,
    processor: LangChainVideoProcessor = Depends(get_langchain_processor)
):
    """Analyze video content using LangChain for intelligent insights."""
    try:
        start_time = time.perf_counter()
        
        # Create a temporary response to get analysis
        temp_response = processor.process_video_with_langchain(
            request=request,
            n_variants=1,
            audience_profile=audience_profile
        )
        
        analysis_time = time.perf_counter() - start_time
        
        if not temp_response.variants or not temp_response.variants[0].langchain_analysis: raise HTTPException(status_code=500, detail="LangChain analysis failed")
        
        analysis = temp_response.variants[0].langchain_analysis
        
        logger.info(
            "LangChain analysis completed",
            youtube_url=request.youtube_url,
            content_type=analysis.content_type.value,
            viral_potential=analysis.viral_potential,
            engagement_score=analysis.engagement_score,
            analysis_time=analysis_time
        )
        
        return analysis
        
    except Exception as e:
        logger.error("LangChain analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/langchain/optimize", response_model=ContentOptimization)
async def optimize_content_with_langchain(
    request: VideoClipRequest,
    audience_profile: Optional[Dict[str, Any]] = None,
    processor: LangChainVideoProcessor = Depends(get_langchain_processor)
):
    """Optimize video content using LangChain for maximum engagement."""
    try:
        start_time = time.perf_counter()
        
        # Create a temporary response to get optimization
        temp_response = processor.process_video_with_langchain(
            request=request,
            n_variants=1,
            audience_profile=audience_profile
        )
        
        optimization_time = time.perf_counter() - start_time
        
        if not temp_response.variants or not temp_response.variants[0].content_optimization: raise HTTPException(status_code=500, detail="LangChain optimization failed")
        
        optimization = temp_response.variants[0].content_optimization
        
        logger.info(
            "LangChain optimization completed",
            youtube_url=request.youtube_url,
            optimal_title=optimization.optimal_title,
            optimal_tags_count=len(optimization.optimal_tags),
            optimal_hashtags_count=len(optimization.optimal_hashtags),
            optimization_time=optimization_time
        )
        
        return optimization
        
    except Exception as e:
        logger.error("LangChain optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/langchain/short-video", response_model=ShortVideoOptimization)
async def optimize_short_video_with_langchain(
    request: VideoClipRequest,
    audience_profile: Optional[Dict[str, Any]] = None,
    processor: LangChainVideoProcessor = Depends(get_langchain_processor)
):
    """Optimize specifically for short-form video platforms."""
    try:
        start_time = time.perf_counter()
        
        # Create a temporary response to get short video optimization
        temp_response = processor.process_video_with_langchain(
            request=request,
            n_variants=1,
            audience_profile=audience_profile
        )
        
        optimization_time = time.perf_counter() - start_time
        
        if not temp_response.variants or not temp_response.variants[0].short_video_optimization: raise HTTPException(status_code=500, detail="Short video optimization failed")
        
        short_opt = temp_response.variants[0].short_video_optimization
        
        logger.info(
            "Short video optimization completed",
            youtube_url=request.youtube_url,
            optimal_clip_length=short_opt.optimal_clip_length,
            hook_duration=short_opt.hook_duration,
            vertical_format=short_opt.vertical_format,
            optimization_time=optimization_time
        )
        
        return short_opt
        
    except Exception as e:
        logger.error("Short video optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/langchain/process", response_model=ViralVideoBatchResponse)
async def process_with_langchain(
    request: VideoClipRequest,
    n_variants: int = 5,
    audience_profile: Optional[Dict[str, Any]] = None,
    processor: LangChainVideoProcessor = Depends(get_langchain_processor)
):
    """Process video with full LangChain optimization pipeline."""
    try:
        start_time = time.perf_counter()
        
        response = processor.process_video_with_langchain(
            request=request,
            n_variants=n_variants,
            audience_profile=audience_profile
        )
        
        processing_time = time.perf_counter() - start_time
        
        logger.info(
            "LangChain processing completed",
            youtube_url=request.youtube_url,
            variants_generated=response.successful_variants,
            average_viral_score=response.average_viral_score,
            ai_enhancement_score=response.ai_enhancement_score,
            langchain_analysis_time=response.langchain_analysis_time,
            content_optimization_time=response.content_optimization_time,
            total_processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error("LangChain processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/langchain/batch", response_model=List[ViralVideoBatchResponse])
async def process_langchain_batch(
    requests: List[VideoClipRequest],
    n_variants_per_request: int = 3,
    audience_profiles: Optional[List[Dict[str, Any]]] = None,
    processor: LangChainVideoProcessor = Depends(get_langchain_processor)
):
    """Process multiple videos with LangChain optimization in batch."""
    try:
        start_time = time.perf_counter()
        
        responses = []
        for i, request in enumerate(requests):
            audience_profile = audience_profiles[i] if audience_profiles and i < len(audience_profiles) else None
            
            response = processor.process_video_with_langchain(
                request=request,
                n_variants=n_variants_per_request,
                audience_profile=audience_profile
            )
            responses.append(response)
        
        processing_time = time.perf_counter() - start_time
        
        logger.info(
            "LangChain batch processing completed",
            total_requests=len(requests),
            variants_per_request=n_variants_per_request,
            processing_time=processing_time,
            successful_requests=len([r for r in responses if r.success])
        )
        
        return responses
        
    except Exception as e:
        logger.error("LangChain batch processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/api/v1/config/langchain")
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

@app.get("/api/v1/config/viral")
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

@app.get("/api/v1/config/video")
async def get_video_config():
    """Get current video processing configuration."""
    return {
        "max_workers": 4,
        "batch_size": 5,
        "enable_audit_logging": True,
        "enable_performance_tracking": True
    }

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================

@app.post("/api/v1/utils/validate")
async def validate_video_request(request: VideoClipRequest):
    """Validate a video processing request."""
    try:
        # Basic validation
        if not request.youtube_url: raise ValueError("YouTube URL is required")
        if request.max_clip_length <= 0: raise ValueError("Max clip length must be positive")
        if request.max_clip_length > 600: raise ValueError("Max clip length cannot exceed 10 minutes")  # 10 minutes
        
        return {
            "valid": True,
            "message": "Request is valid",
            "request": request
        }
        
    except Exception as e:
        return {
            "valid": False,
            "message": str(e),
            "request": request
        }

@app.get("/api/v1/utils/content-types")
async def get_content_types():
    """Get available content types for LangChain analysis."""
    return {
        "content_types": [ct.value for ct in ContentType],
        "engagement_types": [et.value for et in EngagementType]
    }

@app.post("/api/v1/utils/estimate-processing-time")
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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with request ID tracking."""
    request_id = getattr(request.state, 'request_id', None)
    error_response = error_handler.handle_unknown_error(exc, request_id)
    
    return JSONResponse(
        status_code=500,
        content=error_response.to_dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler with request ID tracking."""
    request_id = getattr(request.state, 'request_id', None)
    
    logger.error(
        "HTTP exception", 
        status_code=exc.status_code, 
        detail=exc.detail,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time(),
                "request_id": request_id
            }
        }
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Validation exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    error_response = error_handler.handle_validation_error(exc, request_id)
    
    return JSONResponse(
        status_code=400,
        content=error_response.to_dict()
    )

@app.exception_handler(ProcessingError)
async def processing_exception_handler(request: Request, exc: ProcessingError):
    """Processing exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    error_response = error_handler.handle_processing_error(exc, request_id)
    
    return JSONResponse(
        status_code=422,
        content=error_response.to_dict()
    )

@app.exception_handler(ExternalServiceError)
async def external_service_exception_handler(request: Request, exc: ExternalServiceError):
    """External service exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    error_response = error_handler.handle_external_service_error(exc, request_id)
    
    return JSONResponse(
        status_code=503,
        content=error_response.to_dict()
    )

@app.exception_handler(ResourceError)
async def resource_exception_handler(request: Request, exc: ResourceError):
    """Resource exception handler."""
    request_id = getattr(request.state, 'request_id', None)
    error_response = error_handler.handle_resource_error(exc, request_id)
    
    return JSONResponse(
        status_code=507,
        content=error_response.to_dict()
    )

@app.exception_handler(CriticalSystemError)
async def critical_system_exception_handler(request: Request, exc: CriticalSystemError):
    """Handle critical system errors with immediate alerting."""
    request_id = getattr(request.state, 'request_id', None)
    
    error_response = error_handler.handle_critical_system_error(exc, request_id)
    
    return JSONResponse(
        status_code=500,  # Internal Server Error
        content=error_response.to_dict()
    )

@app.exception_handler(SecurityError)
async def security_exception_handler(request: Request, exc: SecurityError):
    """Handle security errors with threat response."""
    request_id = getattr(request.state, 'request_id', None)
    
    error_response = error_handler.handle_security_error(exc, request_id)
    
    return JSONResponse(
        status_code=403,  # Forbidden
        content=error_response.to_dict()
    )

@app.exception_handler(ConfigurationError)
async def configuration_exception_handler(request: Request, exc: ConfigurationError):
    """Handle configuration errors with fallback strategies."""
    request_id = getattr(request.state, 'request_id', None)
    
    error_response = error_handler.handle_configuration_error(exc, request_id)
    
    return JSONResponse(
        status_code=500,  # Internal Server Error
        content=error_response.to_dict()
    )

# =============================================================================
# STARTUP AND SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event with enhanced system validation."""
    logger.info("Video Processing API starting up", version="3.0.0")
    
    try:
        # PRIORITY 1: System health check
        logger.info("Checking system health...")
        system_health = check_system_resources()
        gpu_health = check_gpu_availability()
        
        logger.info("System health status", 
                   system_health=system_health, 
                   gpu_health=gpu_health)
        
        # PRIORITY 2: Validate critical resources
        if system_health.get("disk_critical"):
            logger.critical("Critical disk space detected during startup")
            raise CriticalSystemError(
                "Insufficient disk space for video processing",
                "disk",
                {"usage_percent": system_health["disk_usage"]}
            )
        
        if not gpu_health["available"]:
            logger.warning("GPU not available - falling back to CPU processing")
        
        # PRIORITY 3: Initialize processors
        logger.info("Initializing processors...")
        video_processor = get_video_processor()
        viral_processor = get_viral_processor()
        langchain_processor = get_langchain_processor()
        batch_processor = get_batch_processor()
        
        logger.info("All processors initialized successfully")
        
        # PRIORITY 4: System readiness check
        logger.info("Video Processing API ready for requests")
        
    except CriticalSystemError:
        logger.critical("Critical system error during startup - shutting down")
        raise
    except Exception as e:
        logger.error("Failed to initialize system", error=str(e))
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Video Processing API shutting down")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 