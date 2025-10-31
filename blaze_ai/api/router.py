"""
FastAPI router for the Blaze AI module.

This module provides comprehensive API endpoints with:
- Request validation and sanitization
- Rate limiting and security
- Comprehensive error handling
- Response formatting and logging
- Health monitoring and metrics
- Batch processing capabilities
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ..core.interfaces import CoreConfig, SystemHealth
from ..engines import EngineManager, get_engine_manager
from ..services import ServiceRegistry, get_service_registry
from ..utils.logging import get_logger
from ..utils.metrics import Timer
from .schemas import (
    TextGenerationRequest, TextGenerationResponse,
    ImageGenerationRequest, ImageGenerationResponse,
    SEOAnalysisRequest, SEOAnalysisResponse,
    BrandVoiceRequest, BrandVoiceResponse,
    ContentGenerationRequest, ContentGenerationResponse,
    BatchRequest, BatchResponse,
    HealthResponse, MetricsResponse
)
from .responses import (
    ErrorResponse, SuccessResponse,
    create_error_response, create_success_response
)

# Create router
router = APIRouter(prefix="/blaze", tags=["blaze-ai"])

# Global instances
_engine_manager: Optional[EngineManager] = None
_service_registry: Optional[ServiceRegistry] = None
_system_health: Optional[SystemHealth] = None
_logger: Optional[logging.Logger] = None

def get_dependencies():
    """Get global dependencies."""
    global _engine_manager, _service_registry, _system_health, _logger
    
    if _engine_manager is None:
        _engine_manager = get_engine_manager()
    
    if _service_registry is None:
        _service_registry = get_service_registry()
    
    if _system_health is None:
        _system_health = SystemHealth()
    
    if _logger is None:
        _logger = get_logger("api.router")
    
    return _engine_manager, _service_registry, _system_health, _logger

# Rate limiting middleware
class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [req_time for req_time in self.requests[client_id] 
                                      if req_time > minute_ago]
        else:
            self.requests[client_id] = []
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True

_rate_limiter = RateLimiter()

async def check_rate_limit(request: Request):
    """Check rate limit for the request."""
    client_id = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

# Request logging middleware
async def log_request(request: Request, call_next):
    """Log incoming requests."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    start_time = time.time()
    client_id = request.client.host if request.client else "unknown"
    
    logger.info(f"Request started: {request.method} {request.url.path} from {client_id}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        logger.info(f"Request completed: {request.method} {request.url.path} "
                   f"from {client_id} - {response.status_code} in {duration:.3f}s")
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} "
                    f"from {client_id} - {str(e)} in {duration:.3f}s")
        raise

# Health check endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Get system health status."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        # Get engine health
        engine_status = engine_manager.get_engine_status()
        
        # Get service health
        service_status = service_registry.get_service_status()
        
        # Get overall system health
        overall_health = system_health.get_health_report()
        
        return HealthResponse(
            status="healthy" if system_health.is_healthy() else "unhealthy",
            engines=engine_status,
            services=service_status,
            system=overall_health,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# Metrics endpoint
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        # Get engine metrics
        engine_metrics = engine_manager.get_system_metrics()
        
        # Get service metrics
        service_metrics = service_registry.get_service_status()
        
        return MetricsResponse(
            engines=engine_metrics,
            services=service_metrics,
            system=system_health.get_health_report(),
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics retrieval failed")

# Text generation endpoint
@router.post("/llm/generate", response_model=TextGenerationResponse)
async def generate_text(
    request: TextGenerationRequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Generate text using LLM engine."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        with Timer() as timer:
            result = await engine_manager.dispatch(
                "llm", "generate", request.dict()
            )
        
        # Log successful request
        logger.info(f"Text generation completed in {timer.elapsed:.3f}s")
        
        # Add cleanup task
        background_tasks.add_task(log_request_metrics, "llm", "generate", timer.elapsed, True)
        
        return TextGenerationResponse(
            text=result.get("text", ""),
            metadata=result.get("metadata", {}),
            processing_time=timer.elapsed
        )
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        background_tasks.add_task(log_request_metrics, "llm", "generate", timer.elapsed, False)
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

# Image generation endpoint
@router.post("/diffusion/generate", response_model=ImageGenerationResponse)
async def generate_image(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Generate image using diffusion engine."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        with Timer() as timer:
            result = await engine_manager.dispatch(
                "diffusion", "generate", request.dict()
            )
        
        # Log successful request
        logger.info(f"Image generation completed in {timer.elapsed:.3f}s")
        
        # Add cleanup task
        background_tasks.add_task(log_request_metrics, "diffusion", "generate", timer.elapsed, True)
        
        return ImageGenerationResponse(
            image_data=result.get("image_data", ""),
            metadata=result.get("metadata", {}),
            processing_time=timer.elapsed
        )
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        background_tasks.add_task(log_request_metrics, "diffusion", "generate", timer.elapsed, False)
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

# SEO analysis endpoint
@router.post("/seo/analyze", response_model=SEOAnalysisResponse)
async def analyze_seo(
    request: SEOAnalysisRequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Analyze content for SEO optimization."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        with Timer() as timer:
            result = await engine_manager.dispatch(
                "router", "seo_analysis", request.dict()
            )
        
        # Log successful request
        logger.info(f"SEO analysis completed in {timer.elapsed:.3f}s")
        
        # Add cleanup task
        background_tasks.add_task(log_request_metrics, "router", "seo_analysis", timer.elapsed, True)
        
        return SEOAnalysisResponse(
            keywords=result.get("keywords", []),
            readability_score=result.get("readability_score", 0.0),
            suggestions=result.get("suggestions", []),
            metadata=result.get("metadata", {}),
            processing_time=timer.elapsed
        )
        
    except Exception as e:
        logger.error(f"SEO analysis failed: {e}")
        background_tasks.add_task(log_request_metrics, "router", "seo_analysis", timer.elapsed, False)
        raise HTTPException(status_code=500, detail=f"SEO analysis failed: {str(e)}")

# Brand voice endpoint
@router.post("/brand/apply", response_model=BrandVoiceResponse)
async def apply_brand_voice(
    request: BrandVoiceRequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Apply brand voice to content."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        with Timer() as timer:
            result = await engine_manager.dispatch(
                "router", "brand_voice", request.dict()
            )
        
        # Log successful request
        logger.info(f"Brand voice application completed in {timer.elapsed:.3f}s")
        
        # Add cleanup task
        background_tasks.add_task(log_request_metrics, "router", "brand_voice", timer.elapsed, True)
        
        return BrandVoiceResponse(
            content=result.get("content", ""),
            changes_made=result.get("changes_made", []),
            confidence_score=result.get("confidence_score", 0.0),
            metadata=result.get("metadata", {}),
            processing_time=timer.elapsed
        )
        
    except Exception as e:
        logger.error(f"Brand voice application failed: {e}")
        background_tasks.add_task(log_request_metrics, "router", "brand_voice", timer.elapsed, False)
        raise HTTPException(status_code=500, detail=f"Brand voice application failed: {str(e)}")

# Content generation endpoint
@router.post("/content/generate", response_model=ContentGenerationResponse)
async def generate_content(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Generate content of specified type."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        with Timer() as timer:
            result = await engine_manager.dispatch(
                "router", "content_generation", request.dict()
            )
        
        # Log successful request
        logger.info(f"Content generation completed in {timer.elapsed:.3f}s")
        
        # Add cleanup task
        background_tasks.add_task(log_request_metrics, "router", "content_generation", timer.elapsed, True)
        
        return ContentGenerationResponse(
            content=result.get("content", ""),
            content_type=result.get("content_type", ""),
            metadata=result.get("metadata", {}),
            processing_time=timer.elapsed
        )
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        background_tasks.add_task(log_request_metrics, "router", "content_generation", timer.elapsed, False)
        raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

# Batch processing endpoint
@router.post("/process/batch", response_model=BatchResponse)
async def process_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    rate_limit: None = Depends(check_rate_limit)
):
    """Process multiple requests in batch."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        with Timer() as timer:
            # Prepare batch requests
            batch_requests = []
            for item in request.requests:
                batch_requests.append({
                    'engine': item.engine,
                    'operation': item.operation,
                    'params': item.params
                })
            
            # Process batch
            results = await engine_manager.dispatch_batch(batch_requests)
        
        # Log successful request
        logger.info(f"Batch processing completed in {timer.elapsed:.3f}s for {len(request.requests)} requests")
        
        # Add cleanup task
        background_tasks.add_task(log_request_metrics, "batch", "process", timer.elapsed, True)
        
        return BatchResponse(
            results=results,
            total_requests=len(request.requests),
            successful_requests=len([r for r in results if not isinstance(r, dict) or 'error' not in r]),
            failed_requests=len([r for r in results if isinstance(r, dict) and 'error' in r]),
            processing_time=timer.elapsed
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        background_tasks.add_task(log_request_metrics, "batch", "process", timer.elapsed, False)
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Error handlers
@router.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    logger.warning(f"Validation error: {exc.errors()}")
    
    return ORJSONResponse(
        status_code=422,
        content=create_error_response(
            error_type="validation_error",
            message="Request validation failed",
            details=exc.errors()
        )
    )

@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return ORJSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_type="http_error",
            message=exc.detail,
            status_code=exc.status_code
        )
    )

@router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return ORJSONResponse(
        status_code=500,
        content=create_error_response(
            error_type="internal_error",
            message="An unexpected error occurred",
            details=str(exc) if logger.isEnabledFor(logging.DEBUG) else None
        )
    )

# Background tasks
async def log_request_metrics(engine: str, operation: str, duration: float, success: bool):
    """Log request metrics in background."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    try:
        # Update system health
        status = "success" if success else "error"
        system_health.update_component_health(
            component=f"{engine}.{operation}",
            status=status,
            message=f"Request {status}",
            details={
                "duration": duration,
                "success": success,
                "timestamp": time.time()
            }
        )
        
        # Log metrics
        logger.debug(f"Request metrics: {engine}.{operation} - {duration:.3f}s - {success}")
        
    except Exception as e:
        logger.error(f"Failed to log request metrics: {e}")

# Startup and shutdown events
@router.on_event("startup")
async def startup_event():
    """Initialize API on startup."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    logger.info("Blaze AI API starting up")
    
    # Initialize components
    try:
        # Pre-warm engines
        await engine_manager.dispatch("llm", "health_check", {})
        await engine_manager.dispatch("diffusion", "health_check", {})
        
        logger.info("Blaze AI API startup completed")
        
    except Exception as e:
        logger.error(f"Blaze AI API startup failed: {e}")
        raise

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    engine_manager, service_registry, system_health, logger = get_dependencies()
    
    logger.info("Blaze AI API shutting down")
    
    try:
        # Shutdown components
        await engine_manager.shutdown()
        await service_registry.shutdown_all()
        
        logger.info("Blaze AI API shutdown completed")
        
    except Exception as e:
        logger.error(f"Blaze AI API shutdown error: {e}")

# Export router
__all__ = ["router"]


